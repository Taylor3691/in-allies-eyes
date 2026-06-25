## ADDED Requirements

### Requirement: Face Detection on Webcam Stream
The system SHALL run OpenCV's Haar Cascade classifier to find target faces on incoming webcam frames.

#### Scenario: Face detection overlay
- **WHEN** a face is in the webcam frame and Kalman tracking is disabled
- **THEN** it draws a Red bounding box around the detected coordinates in the Gradio image output.

### Requirement: Live Kalman Filter Overlay
The system SHALL run the `KalmanTracker` on the detector coordinates when the Kalman checkbox is enabled.

#### Scenario: Kalman-smoothed overlay
- **WHEN** a face is detected and Kalman tracking is enabled
- **THEN** it draws a Green bounding box representing the Kalman state, alongside the Red detector box.

### Requirement: Kalman State Switch
The system SHALL switch tracker status between "Found" (when a detection matches the active track) and "Missing" (when the target is lost, running in "Predict only" mode).

#### Scenario: Occlusion handling
- **WHEN** the webcam face is blocked or lost by the detector
- **THEN** the Green box continues showing predicted positions in "Predict only" mode for up to 30 frames before resetting the tracking state.
