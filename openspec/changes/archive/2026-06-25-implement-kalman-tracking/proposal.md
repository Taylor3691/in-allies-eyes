## Why

Webcam bounding boxes are currently jittery and sensitive to target occlusion. Implementing full Kalman Filter tracking equations and hooking them up to the live webcam feed stabilizes target bounding boxes and provides smoothed crops for feature extraction.

## What Changes

- Implement linear Kalman Filter motion equations (Predict/Update state cycles) inside [src/Kalman_filter/tracker.py](file:///c:/Users/VICTUS/Documents/RecognitionCourse/in-allies-eyes/src/Kalman_filter/tracker.py).
- Integrate a face or person detector inside the webcam processing loop in [app.py](file:///c:/Users/VICTUS/Documents/RecognitionCourse/in-allies-eyes/app.py).
- Render double overlays (Red detector box and Green Kalman box) dynamically onto the Gradio video interface.

## Capabilities

### New Capabilities
- `realtime-webcam-tracking`: Requirements for processing webcam frame streams, overlaying tracking boxes, and updating detector states.

### Modified Capabilities
- `kalman-tracker-placeholder`: Adding linear motion state transition, measurement covariance matrices, and actual state calculation functions.

## Impact

- Modifies [tracker.py](file:///c:/Users/VICTUS/Documents/RecognitionCourse/in-allies-eyes/src/Kalman_filter/tracker.py) implementation.
- Refactors [app.py](file:///c:/Users/VICTUS/Documents/RecognitionCourse/in-allies-eyes/app.py) callback loops for live video processing.
