"""Run image/video demos and explicitly labeled development/validation/test evaluation."""
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import cv2
import numpy as np
from lane_core import Config, LABELS, Tracker, detect, panel, read_image, write_image


def config_from(path):
    cfg = Config()
    if path:
        values = json.loads(path.read_text(encoding="utf-8"))
        unknown = set(values)-set(asdict(cfg))
        if unknown:
            raise ValueError(f"Unknown configuration keys: {unknown}")
        cfg = Config(**values)
    if not 160 <= cfg.width <= 1920:
        raise ValueError("width must be 160..1920")
    if not 0 < cfg.threshold_ratio <= 1 or not 0 < cfg.threshold_min <= cfg.threshold_max <= 255:
        raise ValueError("Invalid threshold settings")
    if not 0 <= cfg.roi_top < .5 or not 0 < cfg.min_area_ratio < .5:
        raise ValueError("Invalid ROI/area settings")
    if not 0 < cfg.min_height_ratio <= 1 or not 0 < cfg.bottom_reach <= 1:
        raise ValueError("Invalid component geometry settings")
    if not 0 < cfg.curve_threshold < 5 or not .1 <= cfg.branch_width_ratio <= .95:
        raise ValueError("Invalid curve or branch threshold")
    return cfg


def new_output(path):
    if path.exists():
        raise ValueError(f"Output already exists: {path}. Use a new name to preserve earlier results.")
    path.mkdir(parents=True)


def save_json(path, value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding="utf-8")


def csv_write(path, rows, fields):
    with path.open("w",newline="",encoding="utf-8-sig") as f:
        wr=csv.DictWriter(f,fieldnames=fields);wr.writeheader();wr.writerows(rows)


def image_demo(args,cfg):
    image=read_image(args.input)
    new_output(args.output)
    start=time.perf_counter()
    frame,gray,raw,mask,result=detect(image,cfg)
    result["detector_ms"]=(time.perf_counter()-start)*1000
    result["source_image"]=str(args.input)
    result["coordinate_system"]="resized processing frame; x right, y down; not original-resolution coordinates"
    view=panel(frame,gray,raw,mask,result)
    write_image(args.output/"pipeline.jpg",view)
    write_image(args.output/"tape_mask.png",mask)
    write_image(args.output/"dark_candidates.png",raw)
    write_image(args.output/"input_processing_size.jpg",frame)
    save_json(args.output/"result.json",result)
    save_json(args.output/"config_used.json",asdict(cfg))
    print(f"Result: {result['label']}; status={result['status']}; detector={result['detector_ms']:.1f} ms")
    print(f"Saved pipeline.jpg, tape_mask.png, and full result.json in {args.output}")
    if args.show:
        cv2.imshow("Lane detection - press any key to close",view)
        cv2.waitKey(0);cv2.destroyAllWindows()


def video_demo(args,cfg):
    source=int(args.input) if args.camera else args.input
    cap=cv2.VideoCapture(source)
    if not cap.isOpened():
        raise ValueError("Cannot open video/camera. Verify path, camera permissions, and codec.")
    try:
        new_output(args.output)
    except Exception:
        cap.release();raise
    fps=cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0 or fps > 240:
        fps=30.0
    tracker=Tracker()
    fields=["frame","source_time_seconds","label","detected","status","offset_normalized",
            "tracked_offset_normalized","curvature","row_coverage","detector_ms"]
    writer=None
    n=0
    durations=[]
    try:
        with (args.output/"video_predictions.csv").open("w",newline="",encoding="utf-8-sig") as f:
            table=csv.DictWriter(f,fieldnames=fields);table.writeheader()
            while True:
                ok,image=cap.read()
                if not ok:
                    break
                start=time.perf_counter()
                frame,gray,raw,mask,result=detect(image,cfg)
                tracker.update(result)
                dt=(time.perf_counter()-start)*1000
                durations.append(dt)
                view=panel(frame,gray,raw,mask,result)
                if writer is None:
                    writer=cv2.VideoWriter(str(args.output/"demo.avi"),cv2.VideoWriter_fourcc(*"MJPG"),fps,
                                           (view.shape[1],view.shape[0]))
                    if not writer.isOpened():
                        raise ValueError("Could not create MJPEG AVI output")
                writer.write(view)
                table.writerow(dict(frame=n,source_time_seconds=f"{n/fps:.3f}",
                    **{k:result[k] for k in fields if k in result},detector_ms=f"{dt:.3f}"))
                n+=1
                if args.show:
                    cv2.imshow("Lane pipeline - Q/Esc quits",view)
                    if cv2.waitKey(max(1,round(1000/fps-dt))) & 0xff in (27,ord('q')):
                        break
                if args.max_frames and n>=args.max_frames:
                    break
    finally:
        cap.release()
        if writer is not None: writer.release()
        if args.show: cv2.destroyAllWindows()
    if not n:
        raise ValueError("No frames decoded")
    save_json(args.output/"config_used.json",asdict(cfg))
    save_json(args.output/"video_summary.json",dict(frames=n,source_nominal_fps=fps,
        mean_detector_ms=float(np.mean(durations)),detector_only_fps=float(1000/np.mean(durations)),
        notes="Timing excludes decoding, drawing, writing, and display. Source time uses nominal FPS; approximate for VFR.",
        tracking="Exponential smoothing of near-path offset; resets on loss. No optical flow or route choice."))
    print(f"Processed {n} frames. See {args.output}/demo.avi and video_predictions.csv")


def confusion_image(cm):
    size=125
    image=np.full((190+size*len(LABELS),250+size*len(LABELS),3),255,np.uint8)
    cv2.putText(image,"TRUE rows / PREDICTED columns",(15,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,0),1)
    abbreviations=["straight","left","right","cross","no path"]
    for i,name in enumerate(abbreviations):
        cv2.putText(image,name,(250+i*size,160),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,0),1)
        cv2.putText(image,LABELS[i],(8,190+i*size+65),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),1)
        for j in range(len(LABELS)):
            fraction=cm[i,j]/max(1,int(cm[i].sum()))
            color=(255,int(255-120*fraction),int(255-180*fraction))
            cv2.rectangle(image,(250+j*size,190+i*size),(250+(j+1)*size,190+(i+1)*size),color,-1)
            cv2.rectangle(image,(250+j*size,190+i*size),(250+(j+1)*size,190+(i+1)*size),(120,120,120),1)
            cv2.putText(image,str(cm[i,j]),(285+j*size,260+i*size),cv2.FONT_HERSHEY_SIMPLEX,.75,(0,0,0),2)
    return image


def evaluate(args,cfg):
    root=args.data.resolve()
    files=[]
    for label in LABELS:
        folder=root/label
        if not folder.is_dir():
            raise ValueError(f"Missing category folder: {folder}")
        files.extend((p,label) for p in sorted(folder.iterdir()) if p.suffix.lower() in {".jpg",".jpeg",".png"})
    if not files:
        raise ValueError("No images found")
    if args.output.resolve().is_relative_to(root):
        raise ValueError("Evaluation output must be outside the dataset folder")
    # Enforce the requested evaluation stage when using the standard split directory names.
    expected={"train":"development","val":"validation","test":"test"}
    if root.name in expected and args.name != expected[root.name]:
        raise ValueError(f"Folder {root.name} must be reported as {expected[root.name]}")
    new_output(args.output)
    cm=np.zeros((len(LABELS),len(LABELS)),np.int64)
    rows=[]; failures=[]; errors=[]; samples=[]; sample_counts={}
    for p,true in files:
        try:
            image=read_image(p)
            start=time.perf_counter()
            frame,gray,raw,mask,result=detect(image,cfg)
            elapsed=(time.perf_counter()-start)*1000
            pred=result["label"]
            cm[LABELS.index(true),LABELS.index(pred)]+=1
            row=dict(image_path=p.relative_to(root).as_posix(),true_label=true,predicted_label=pred,
                     correct=pred==true,status=result["status"],curvature=result["curvature"],
                     row_coverage=result["row_coverage"],detector_ms=elapsed)
            rows.append(row)
            n=sample_counts.get(true,0)
            if n<2:
                view=panel(frame,gray,raw,mask,result)
                write_image(args.output/"examples"/f"{true}_{n+1}.jpg",view)
                sample_counts[true]=n+1
            if pred!=true:
                failures.append(row)
                write_image(args.output/"errors"/true/(p.stem+".jpg"),panel(frame,gray,raw,mask,result))
        except (ValueError,OSError,cv2.error) as exc:
            errors.append(dict(image_path=str(p),error=str(exc)))
    if not rows:
        raise ValueError("No images could be evaluated")
    per_class=[]
    for i,label in enumerate(LABELS):
        tp=int(cm[i,i]);support=int(cm[i].sum());predicted=int(cm[:,i].sum())
        precision=tp/predicted if predicted else 0.0
        recall=tp/support if support else 0.0
        f1=2*precision*recall/(precision+recall) if precision+recall else 0.0
        per_class.append(dict(label=label,support=support,precision=precision,recall=recall,f1=f1))
    truth_path=np.array([r['true_label']!='no_visible_path' for r in rows])
    pred_path=np.array([r['predicted_label']!='no_visible_path' for r in rows])
    tp=int(np.sum(truth_path & pred_path));fp=int(np.sum(~truth_path & pred_path));fn=int(np.sum(truth_path & ~pred_path))
    dataset_hash=hashlib.sha256()
    for p,true in files:
        dataset_hash.update(p.relative_to(root).as_posix().encode());dataset_hash.update(hashlib.sha256(p.read_bytes()).digest())
    summary=dict(stage=args.name,input_images=len(files),evaluated_images=len(rows),unreadable_images=len(errors),
        accuracy=float(np.trace(cm)/cm.sum()),macro_f1=float(np.mean([x['f1'] for x in per_class])),
        per_class=per_class,confusion_label_order=LABELS,confusion_matrix=cm.tolist(),
        path_presence=dict(tp=tp,fp=fp,fn=fn,precision=tp/(tp+fp) if tp+fp else 0.0,recall=tp/(tp+fn) if tp+fn else 0.0),
        mean_detector_ms=float(np.mean([r['detector_ms'] for r in rows])),config=asdict(cfg),
        dataset_fingerprint=dataset_hash.hexdigest(),environment=dict(python=platform.python_version(),opencv=cv2.__version__,platform=platform.platform()),
        limitations=["Development results are resubstitution measurements, not validation/test performance.",
          "No manual masks or centerline ground truth provided: segmentation IoU and localization error NOT measured.",
          "Frames from the same clip/setup are correlated. Image count is not independent scene count.",
          "Classical rule parameters were developed on the uploaded training images.",
          "Timing excludes image decoding, visualization, and output writing."])
    save_json(args.output/"metrics.json",summary)
    save_json(args.output/"config_used.json",asdict(cfg))
    fields=list(rows[0])
    csv_write(args.output/"predictions.csv",rows,fields)
    csv_write(args.output/"misclassified.csv",failures,fields)
    csv_write(args.output/"unreadable.csv",errors,["image_path","error"])
    csv_write(args.output/"per_class.csv",per_class,list(per_class[0]))
    csv_write(args.output/"confusion_matrix.csv",[dict(true_label=label,**{l:int(cm[i,j]) for j,l in enumerate(LABELS)}) for i,label in enumerate(LABELS)],['true_label']+LABELS)
    write_image(args.output/"confusion_matrix.png",confusion_image(cm))
    print(f"{args.name.upper()} only: {len(rows)} images; accuracy={summary['accuracy']:.4f}; macro F1={summary['macro_f1']:.4f}")
    print("No geometric accuracy measured without independent ground truth.")
    return 1 if errors else 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('image','video','evaluate'):
        p=sub.add_parser(name)
        p.add_argument('--config',type=Path)
        p.add_argument('--output',type=Path,required=True)
        if name=='evaluate':
            p.add_argument('--data',type=Path,required=True)
            p.add_argument('--name',choices=['development','validation','test'],required=True)
        else:
            p.add_argument('--input',type=Path if name=='image' else str,required=True)
            p.add_argument('--show',action='store_true')
        if name=='video':
            p.add_argument('--camera',action='store_true',help='Treat input as numeric camera ID')
            p.add_argument('--max-frames',type=int,default=0)
    args=parser.parse_args()
    try:
        cfg=config_from(args.config)
        if args.command=='image': image_demo(args,cfg)
        elif args.command=='video': video_demo(args,cfg)
        else: return evaluate(args,cfg)
        return 0
    except (ValueError,OSError,cv2.error) as exc:
        print(f'ERROR: {exc}',file=sys.stderr);return 2


if __name__=='__main__':
    sys.exit(main())
