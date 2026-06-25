## 1. Kalman Filter Implementation

- [x] 1.1 Define state transition and covariance matrices ($F, H, P, Q, R$) in the `KalmanTracker.__init__` method in [tracker.py](file:///c:/Users/VICTUS/Documents/RecognitionCourse/in-allies-eyes/src/Kalman_filter/tracker.py).
- [x] 1.2 Implement the constant velocity state prediction steps in the `KalmanTracker.predict` method in [tracker.py](file:///c:/Users/VICTUS/Documents/RecognitionCourse/in-allies-eyes/src/Kalman_filter/tracker.py).
- [x] 1.3 Implement the linear correction updates in the `KalmanTracker.update` method in [tracker.py](file:///c:/Users/VICTUS/Documents/RecognitionCourse/in-allies-eyes/src/Kalman_filter/tracker.py).

## 2. Webcam and Face Detector Integration

- [x] 2.1 Initialize the OpenCV Cascade face detector at the top of [app.py](file:///c:/Users/VICTUS/Documents/RecognitionCourse/in-allies-eyes/app.py).
- [x] 2.2 Create the webcam frame callback in [app.py](file:///c:/Users/VICTUS/Documents/RecognitionCourse/in-allies-eyes/app.py) that processes video frames, runs the face detector, and updates the tracker state.
- [x] 2.3 Integrate bounding box drawings (Red for detector, Green for Kalman) on frames, and wire the webcam UI components to the callback function.
