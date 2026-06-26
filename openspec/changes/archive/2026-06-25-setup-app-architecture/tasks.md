## 1. Scaffold app.py UI Structure

- [x] 1.1 Create [app.py](file:///c:/Users/VICTUS/Documents/RecognitionCourse/in-allies-eyes/app.py) in the project root and add Gradio imports.
- [x] 1.2 Implement the 3-tab layout containing "Realtime Capture / Kalman Demo", "Gallery Search / CA-Jaccard Demo", and "Four-Mode Top-5 Comparison" using Gradio Blocks.
- [x] 1.3 Add visual placeholders (mock video streams, empty output grids) to ensure the UI starts up correctly without loading deep models.

## 2. Set Up Kalman Tracker Module

- [x] 2.1 Create [src/Kalman_filter/tracker.py](file:///c:/Users/VICTUS/Documents/RecognitionCourse/in-allies-eyes/src/Kalman_filter/tracker.py) inside the tracker module path.
- [x] 2.2 Define the `KalmanTracker` class in [tracker.py](file:///c:/Users/VICTUS/Documents/RecognitionCourse/in-allies-eyes/src/Kalman_filter/tracker.py) with basic state fields (state, status).
- [x] 2.3 Define placeholder `predict` and `update` methods in `KalmanTracker` to allow importing and testing class behavior.
