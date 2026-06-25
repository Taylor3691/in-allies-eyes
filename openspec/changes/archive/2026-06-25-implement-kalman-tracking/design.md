## Context

The Re-ID demo app requires linear target tracking to stabilize bounding boxes. This design specifies how we implement the Kalman Filter matrices and integrate the OpenCV face detector into the Gradio webcam streaming loop.

## Goals / Non-Goals

**Goals:**
- Implement linear Kalman Filter mathematical equations in [tracker.py](file:///c:/Users/VICTUS/Documents/RecognitionCourse/in-allies-eyes/src/Kalman_filter/tracker.py) for bounding boxes.
- Integrate OpenCV Haar Cascade Face Detector to provide real-time target bounding box detections.
- Implement the double bounding box overlay rendering (Red for raw detection, Green for Kalman state) in [app.py](file:///c:/Users/VICTUS/Documents/RecognitionCourse/in-allies-eyes/app.py).

**Non-Goals:**
- Implementing deep learning-based detectors (like YOLO) which add deployment and startup overhead.
- Implementing feature-level matching (re-identification) within the tracking loop itself.

## Decisions

### Decision 1: Kalman Filter State Representation
We choose a 8-dimensional state vector and 4-dimensional measurement vector (Standard in DeepSORT):
* **State vector:** $x = [c_x, c_y, a, h, v_{cx}, v_{cy}, v_a, v_h]^T$ where $(c_x, c_y)$ is bounding box center, $a$ is aspect ratio, $h$ is height, and $v$ represents their velocities.
* **Measurement vector:** $z = [c_x, c_y, a, h]^T$
* **Rationale:** Modeling aspect ratio and height rather than width and height directly handles aspect ratio stability better when people/faces move closer or further from the camera.

### Decision 2: OpenCV Haar Cascade Face Detector
To avoid external dependencies and keep frame inference under 10ms on CPU, we use OpenCV's built-in `CascadeClassifier` with the default frontal face xml file.

### Decision 3: Streaming Frame Processing in Gradio
Gradio's `gr.Image(sources="webcam", streaming=True)` will be used. It sends frames from the webcam to the Python backend. The backend processes the frame, runs the Kalman state update, overlays the bounding boxes, and returns the modified image.

## Risks / Trade-offs

* **[Risk]** Bounding box association failure (matching a detection to the tracker when multiple faces are in view).
  * **Mitigation:** Implement a simple Intersection-over-Union (IoU) distance check. If the detection's IoU with the tracker's predicted box is above a threshold (e.g. 0.3), associate them. Otherwise, treat it as a new track or ignore.
