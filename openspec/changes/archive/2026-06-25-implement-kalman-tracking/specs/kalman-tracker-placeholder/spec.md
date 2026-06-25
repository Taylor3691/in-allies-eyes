## MODIFIED Requirements

### Requirement: Kalman Filter Class Definition
The system SHALL expose a `KalmanTracker` class under the [src/Kalman_filter](file:///c:/Users/VICTUS/Documents/RecognitionCourse/in-allies-eyes/src/Kalman_filter) module structure. Upon initialization, it SHALL set up state variables for an 8-dimensional state vector $[c_x, c_y, a, h, v_{cx}, v_{cy}, v_a, v_h]^T$ and standard linear Kalman Filter covariance matrices (state transition $F$, measurement projection $H$, covariance $P$, process noise $Q$, and measurement noise $R$).

#### Scenario: Verify tracker class instantiation
- **WHEN** the `KalmanTracker` is instantiated
- **THEN** it initializes with empty tracking variables, a status of "Missing", and all state matrices correctly sized.

### Requirement: Standard Tracking Interface
The `KalmanTracker` class MUST expose two methods: `predict()` for motion estimation and `update(bbox)` for coordinate correction.
* `predict()` SHALL apply the constant velocity motion model to update the state vector $x$ and covariance matrix $P$.
* `update(bbox)` SHALL apply linear measurement corrections using Kalman gain equations to adjust the state vector $x$ and covariance matrix $P$.

#### Scenario: Call tracking updates on stub
- **WHEN** the user calls the `predict` or `update` methods on the tracker with measurements
- **THEN** the state vector and covariance matrices are mathematically updated and the status switches to "Found".
