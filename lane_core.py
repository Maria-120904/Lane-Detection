"""Classical tape segmentation and geometric scene recognition; no learned weights."""
from dataclasses import dataclass, asdict
import cv2
import numpy as np

LABELS = ["straight", "left_curve", "right_curve", "intersection", "no_visible_path"]


@dataclass
class Config:
    width: int = 640
    threshold_ratio: float = 0.60
    threshold_min: int = 45
    threshold_max: int = 110
    roi_top: float = 0.025
    min_area_ratio: float = 0.001
    min_height_ratio: float = 0.42
    bottom_reach: float = 0.82
    curve_threshold: float = 0.065
    branch_width_ratio: float = 0.38


def read_image(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def write_image(path, image):
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, data = cv2.imencode(path.suffix, image)
    if not ok:
        raise ValueError(f"Cannot encode image: {path}")
    data.tofile(str(path))


def runs(row):
    xs = np.flatnonzero(row)
    if not len(xs):
        return []
    blocks = np.split(xs, np.flatnonzero(np.diff(xs) > 1) + 1)
    return [(int(b[0]), int(b[-1])) for b in blocks if len(b) >= 2]


def detect(image, config=None):
    cfg = config or Config()
    if image is None or image.size == 0:
        raise ValueError("Empty image")
    h0, w0 = image.shape[:2]
    if min(h0, w0) < 16:
        raise ValueError("Image is too small")
    frame = cv2.resize(image, (cfg.width, max(16, round(h0 * cfg.width / w0))))
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    y0 = int(h * cfg.roi_top)
    threshold = float(np.clip(np.median(gray[int(h*.35):int(h*.95)]) * cfg.threshold_ratio,
                              cfg.threshold_min, cfg.threshold_max))
    raw = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)[1]
    raw[:y0] = 0
    raw[:, :int(w*.01)] = 0
    raw[:, int(w*.99):] = 0
    clean = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, components, stats, centroids = cv2.connectedComponentsWithStats(clean, 8)
    candidates = []
    for k in range(1, n):
        x, y, cw, ch, area = stats[k]
        if (area >= h*w*cfg.min_area_ratio and ch >= h*cfg.min_height_ratio
                and y+ch >= h*cfg.bottom_reach):
            score = ch/h + min(area/(h*w), .15) - .1*abs(centroids[k][0]-w/2)/w
            candidates.append((score, k))
    mask = np.zeros_like(raw)
    result = {"label": "no_visible_path", "detected": False, "status": "no_valid_path_component",
              "threshold": threshold, "curvature": None, "offset_normalized": None,
              "row_coverage": 0.0, "fit_residual": None, "centerline": [],
              "intersection_segments": [], "processing_width": w, "processing_height": h}
    if not candidates:
        return frame, gray, raw, mask, result
    k = max(candidates)[1]
    mask[components == k] = 255
    # Follow narrow tape runs from near to far; ignore broad crossing rows.
    points, last_x, old_x, last_y = [], None, None, None
    sampled = list(range(int(h*.96), max(y0, int(h*.035)), -max(2, h//100)))
    for y in sampled:
        row = np.mean(mask[max(0,y-1):min(h,y+2)], axis=0) >= 85
        segments = [(a,b) for a,b in runs(row) if (b-a+1) < w*.16]
        if not segments:
            continue
        target = w/2 if last_x is None else last_x
        if last_x is not None and old_x is not None:
            target += float(np.clip(last_x-old_x, -w*.02, w*.02))
        a,b = min(segments, key=lambda ab: abs((ab[0]+ab[1])/2-target))
        cx = (a+b)/2
        if last_x is not None and abs(cx-last_x) > w*.12:
            continue
        points.append((float(cx), int(y)))
        old_x, last_x, last_y = last_x, cx, y
    result["row_coverage"] = len(points)/max(1,len(sampled))
    result["centerline"] = [[x, y] for x,y in points]
    if len(points) < 12 or max(p[1] for p in points)-min(p[1] for p in points) < h*.35:
        result["status"] = "insufficient_centerline_support"
        return frame, gray, raw, mask, result
    xy = np.array(points)
    t, xn = (h-1-xy[:,1])/h, xy[:,0]/w
    coef = np.polyfit(t, xn, 2)
    # Reject isolated run-selection outliers then refit.
    residual = np.abs(xn - np.polyval(coef,t))
    inliers = residual <= max(.008, 3*float(np.median(residual)))
    if inliers.sum() >= 12:
        coef = np.polyfit(t[inliers],xn[inliers],2)
        xy = xy[inliers]
        t, xn = t[inliers], xn[inliers]
        result["centerline"] = xy.tolist()
    curvature = float(coef[0])
    result.update(detected=True, label="straight", status="path_detected",
                  curvature=curvature, offset_normalized=float((xy[0,0]-w/2)/(w/2)),
                  fit_residual=float(np.mean(np.abs(xn-np.polyval(coef,t)))))
    if curvature < -cfg.curve_threshold:
        result["label"] = "left_curve"
    elif curvature > cfg.curve_threshold:
        result["label"] = "right_curve"
    # Long near-horizontal segments on the SELECTED tape component signal a crossing.
    lines = cv2.HoughLinesP(mask, 1, np.pi/180, threshold=max(25,w//12),
                           minLineLength=int(w*cfg.branch_width_ratio), maxLineGap=12)
    branches = []
    if lines is not None:
        for x1,y1,x2,y2 in lines[:,0,:]:
            if abs(y2-y1) <= .20*max(1,abs(x2-x1)):
                cy = (y1+y2)/2
                if cy < h*.05 or cy > h*.94:
                    continue
                # Require tape backbone above AND below the putative crossbar.
                above = xy[xy[:,1] < cy-h*.025]
                below = xy[xy[:,1] > cy+h*.025]
                if len(above) >= 3 and len(below) >= 3:
                    px = float(np.polyval(coef,(h-1-cy)/h)*w)
                    if min(x1,x2)-10 <= px <= max(x1,x2)+10:
                        branches.append([int(x1),int(y1),int(x2),int(y2)])
    if branches:
        result["label"] = "intersection"
        result["status"] = "crossing_detected_route_not_selected"
        result["intersection_segments"] = branches[:12]
    return frame, gray, raw, mask, result


class Tracker:
    """Exponential smoothing of near-path offset across frames, reset on path loss.

    This is lightweight temporal association, not optical flow or metric tracking.
    It never propagates an old path as a new detection and never chooses a route.
    """
    def __init__(self, alpha=.35):
        self.alpha = alpha
        self.offset = None

    def update(self, result):
        current = result["offset_normalized"]
        if not result["detected"] or current is None:
            self.offset = None
        elif self.offset is None or abs(current-self.offset) > .5:
            self.offset = current
        else:
            self.offset = self.alpha*current + (1-self.alpha)*self.offset
        result["tracked_offset_normalized"] = self.offset
        return result


def panel(frame, gray, raw, mask, result):
    h,w = frame.shape[:2]
    overlay = frame.copy()
    tint = overlay.copy()
    tint[mask > 0] = (0,180,0)
    overlay = cv2.addWeighted(overlay,.65,tint,.35,0)
    cv2.line(overlay,(w//2,0),(w//2,h-1),(200,160,30),1)
    if result["detected"]:
        pts = np.array(result["centerline"],np.int32)
        if len(pts)>1:
            cv2.polylines(overlay,[pts],False,(0,255,255),2)
        for x1,y1,x2,y2 in result["intersection_segments"]:
            cv2.line(overlay,(x1,y1),(x2,y2),(255,160,0),2)
    detail = "NO PATH / LOST" if not result["detected"] else f"offset={result['offset_normalized']:+.3f}  curve={result['curvature']:+.3f}"
    if result.get("tracked_offset_normalized") is not None:
        detail += f"  tracked={result['tracked_offset_normalized']:+.3f}"
    def tile(im,title,subtitle=""):
        if im.ndim==2:
            im=cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
        header=np.full((60,w,3),20,np.uint8)
        cv2.putText(header,title,(8,23),cv2.FONT_HERSHEY_SIMPLEX,.60,(255,255,255),1)
        if subtitle:
            cv2.putText(header,subtitle,(8,47),cv2.FONT_HERSHEY_SIMPLEX,.46,(230,230,230),1)
        return np.vstack([header,im])
    return np.vstack([np.hstack([tile(frame,"INPUT"),tile(gray,"GRAYSCALE")]),
                      np.hstack([tile(raw,"DARK PIXEL CANDIDATES"),tile(overlay,"RESULT: "+result['label'],detail)])])
