# Lane Detection and Path Scene Recognition

A classical computer-vision project that detects a black tape path on a
light-colored floor and classifies the visible path scene into one of
five categories. Developed as a midterm prototype for a Computer Vision
course, the project can serve as a starting point for later integration
with a lane-following robot.

> **Scope:** This repository focuses on visual path detection and scene
> recognition. It does not, by itself, implement a fully autonomous
> robot or guarantee reliable operation under every floor, lighting, or
> camera condition.

## Table of Contents

-   [Overview](#overview)
-   [Features](#features)
-   [Scene Categories](#scene-categories)
-   [Methodology](#methodology)
-   [Dataset and Evaluation](#dataset-and-evaluation)
-   [Reported Results](#reported-results)
-   [Installation](#installation)
-   [Usage](#usage)
-   [Project Structure](#project-structure)
-   [Limitations](#limitations)
-   [Future Improvements](#future-improvements)
-   [Technologies](#technologies)

## Overview

The system processes an image or video frame to find a dark tape path
against a comparatively light floor. It analyzes the detected path and
its geometry, classifies the visible scene, and produces visual output
for inspection.

The prototype uses a **classical computer-vision algorithm** rather than
a trained deep-learning model. This approach makes the processing
pipeline inspectable and is suitable for an initial prototype with a
constrained path appearance and environment.

## Features

-   Detects a black tape path on a light-colored floor.
-   Recognizes five path-scene categories.
-   Processes still images and video using separate CLI commands.
-   Produces visual output for inspecting detections.
-   Supports evaluation using the project's labeled dataset.
-   Provides a foundation for future robot integration.

## Scene Categories

  -----------------------------------------------------------------------
  Category                            Description
  ----------------------------------- -----------------------------------
  `straight`                          A path that proceeds generally
                                      straight ahead.

  `left_curve`                        A path that bends to the left.

  `right_curve`                       A path that bends to the right.

  `intersection`                      A visible path region where paths
                                      branch or intersect.

  `no_visible_path`                   No path is detected clearly enough
                                      to assign a path category.
  -----------------------------------------------------------------------

The output category describes the scene visible to the camera. It is not
the same as controlling a robot's motors or completing an autonomous
navigation task.

## Methodology

The system follows a rule-based image-processing pipeline:

1.  **Input acquisition:** Read a still image or extract a frame from a
    video.
2.  **Region of interest (ROI):** Focus processing on the floor area
    where the path is expected.
3.  **Grayscale conversion:** Convert the image to grayscale for
    intensity-based processing.
4.  **Thresholding:** Separate dark tape pixels from the lighter floor
    to create a binary mask.
5.  **Noise filtering and mask cleanup:** Reduce unwanted regions before
    path analysis.
6.  **Path and geometry analysis:** Analyze the detected region's
    position, shape, or centerline.
7.  **Scene classification:** Apply geometric decision rules to classify
    the scene into one of the five categories.
8.  **Visual output:** Display the detection and classification results
    for inspection.

### Simplified pipeline

``` text
Image / Video
      |
      v
Region of Interest (ROI)
      |
      v
Grayscale Conversion
      |
      v
Thresholding
      |
      v
Noise Filtering / Mask Cleanup
      |
      v
Path and Geometry Analysis
      |
      v
Scene Classification
      |
      v
Visual Output
```

### Why use an algorithm instead of a trained model?

The current prototype uses explicit image-processing operations and
decision rules, not neural-network weights learned from a training set.
Its performance depends on settings such as thresholding, camera angle,
lighting, floor appearance, and path geometry. A trained model could be
explored later if the project needs to handle a wider range of scenes.

## Dataset and Evaluation

The dataset was created from recorded clips of the tape path, with
variations in camera position, height, distance, and starting position.
Selected frames were labeled with one of the five scene categories.

The evaluation setup previously reported for this project was:

  -----------------------------------------------------------------------------
  Split                                       Images Purpose
  --------------------- ---------------------------- --------------------------
  Development                                    600 Develop and tune the
                                                     detection/classification
                                                     approach.

  Validation                                     100 Check behavior during
                                                     development.

  Testing                                        100 Evaluate on the held-out
                                                     test set.

  **Total**                                  **800** 
  -----------------------------------------------------------------------------

The reported split corresponds to 120 development images and 20
validation plus 20 test images per category. Verify these counts against
the final dataset before publishing if the split has changed.

### Evaluation measures

-   **Accuracy:** proportion of evaluated samples classified correctly.
-   **Precision:** proportion of predictions for a category that are
    correct.
-   **Recall:** proportion of ground-truth samples in a category that
    are correctly identified.
-   **F1-score:** harmonic mean of precision and recall.
-   **FPS (frames per second):** processing throughput during video
    evaluation, where reported by the demo.
-   **Visual inspection:** checks whether the detected path and overlays
    align with the visible tape.

## Reported Results

The results previously recorded for the current prototype evaluation
were:

  Evaluation split     Correct predictions   Reported accuracy
  ------------------ --------------------- -------------------
  Development set                593 / 600              98.83%
  Validation set                 100 / 100                100%
  Test set                       100 / 100                100%

The reported validation and test precision, recall, and F1-score were
also **1.00 for each of the five categories** in that evaluation.

### Interpreting the results

These results describe performance on the evaluated dataset; they do
**not** establish 100% accuracy in all real-world settings. Each of the
validation and test sets contained 100 images, so a small number of
errors could noticeably change the reported percentages. Frames from
similar clips may also be correlated. A stronger generalization test
should use recordings from separate sessions, camera positions, floor
areas, and lighting conditions, and ensure near-duplicate frames do not
cross dataset splits.

The system may behave differently with faded tape, dark or patterned
floors, strong shadows, reflections, camera tilt, or path shapes not
represented in the dataset.

## Installation

### 1. Clone the repository

Replace the placeholders with your actual GitHub repository URL and
folder name:

``` bash
git clone <YOUR-REPOSITORY-URL>
cd <YOUR-REPOSITORY-FOLDER>
```

### 2. (Recommended) Create and activate a virtual environment

``` bash
python -m venv .venv
```

**Windows PowerShell:**

``` powershell
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux:**

``` bash
source .venv/bin/activate
```

### 3. Install dependencies

Run this from the project root, where `requirements.txt` is located:

``` bash
python -m pip install -r requirements.txt
```

## Usage

Run commands from the **project root directory**. The examples below use
the command structure supported by this project: `image` or `video`,
followed by `--input`, `--output`, and optionally `--show`.

### Process an image

``` bash
python lane_demo.py image --input dataset/images/test/straight/50f76b5ec199_frame0000056_t00002.002.jpg --output results/images/straight_test1 --show
```

Process an intersection example:

``` bash
python lane_demo.py image --input dataset/images/test/intersection/69a64b82619f_frame0000224_t00008.008.jpg --output results/images/intersection_test1 --show
```

Process an image at another location (Windows example):

``` bash
python lane_demo.py image --input "C:/your/path/image.jpg" --output results/images/my_image --show
```

### Process a video

``` bash
python lane_demo.py video --input "C:/your/path/clip.mp4" --output results/videos/video01 --show
```

Replace `C:/your/path/clip.mp4` with the real path to your video file.
If the file is inside the repository, you can use a relative path, for
example:

``` bash
python lane_demo.py video --input dataset/videos/test/intersection/clip01.mp4 --output results/videos/intersection_video01 --show
```

**Notes:**

-   `--input` must point to an existing image or video file.
-   The input path and output path are examples; change them to match
    your files.
-   `--show` requests display of the demo output. Depending on your
    environment, a graphical display may be required.
-   Results are written under the specified output path according to the
    script's implementation.
-   Create any required input/output directories if the script does not
    create them automatically.
-   If a command fails, run `python lane_demo.py --help` and check the
    script's argument names.

## Project Structure

The core files and folders discussed during development include:

``` text
.
├── lane_core.py
├── lane_demo.py
├── requirements.txt
├── dataset/
│   ├── images/
│   │   ├── test/
│   │   ├── train/
│   │   └── val/
│   └── videos/
│       └── test/
├── results/
└── README.md
```

This is an illustrative structure. Update it to match the files actually
included in the GitHub repository. The `results/` directory may be
generated at runtime rather than committed.

## Limitations

-   Designed primarily for high-contrast black tape on a light-colored
    floor.
-   Threshold-based detection may be sensitive to lighting, shadows,
    reflections, and changes in floor or tape color.
-   Recognition depends on camera viewpoint and the path shapes
    represented during development.
-   A small evaluation dataset and perfect scores do not guarantee
    generalization.
-   The perception prototype does not itself provide motor control,
    obstacle avoidance, or a complete autonomous robot mission.
-   Runtime and FPS depend on the computer, resolution, and processing
    configuration.

## Future Improvements

-   Evaluate on larger datasets from independent recording sessions.
-   Report a confusion matrix and per-class metrics.
-   Measure FPS under a documented hardware and resolution setup.
-   Test robustness to shadows, tape gaps, floor markings, and
    camera-angle changes.
-   Integrate the perception output with a robot controller using safe,
    controlled tests.
-   Compare with a trained computer-vision model if broader scene
    variability warrants it.

## Technologies

-   **Python** --- implementation language
-   **OpenCV** --- image and video processing
-   **NumPy** --- numerical and array operations
-   **Git and GitHub** --- version control and project hosting


