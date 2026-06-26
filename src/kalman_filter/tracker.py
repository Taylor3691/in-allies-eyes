import numpy as np

class KalmanTracker:
    """
    Linear Kalman Filter Tracker for 2D bounding boxes.
    Uses constant velocity motion model.
    """
    def __init__(self):
        # State vector: [cx, cy, a, h, v_cx, v_cy, v_a, v_h]^T
        # cx, cy: center coordinates of bounding box
        # a: aspect ratio (w / h)
        # h: height
        # v_*: velocity of each component
        self.state = None
        self.status = "Missing"  # Can be "Found" or "Missing"
        self.mode = "Predict only"  # Can be "Update" or "Predict only"

        # F: State transition matrix (predicts next state based on dt = 1)
        self.F = np.eye(8)
        for i in range(4):
            self.F[i, i + 4] = 1.0

        # H: Measurement matrix (projects 8D state vector into 4D measurement vector)
        self.H = np.zeros((4, 8))
        for i in range(4):
            self.H[i, i] = 1.0

        # P: Covariance matrix (uncertainty of estimate)
        self.P = np.eye(8) * 10.0

        # Q: Process noise covariance (uncertainty of motion model)
        self.Q = np.eye(8)
        # Give higher process noise uncertainty to velocities
        self.Q[4:8, 4:8] *= 0.5
        self.Q[0:4, 0:4] *= 0.05

        # R: Measurement noise covariance (uncertainty of detection model)
        self.R = np.eye(4) * 2.0

    def predict(self):
        """
        Estimate state vector and error covariance in the next frame.
        """
        if self.state is None:
            return
        
        # x_k|k-1 = F * x_k-1|k-1
        self.state = np.dot(self.F, self.state)
        # P_k|k-1 = F * P_k-1|k-1 * F^T + Q
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        self.mode = "Predict only"

    def update(self, bbox):
        """
        Correct state and covariance using the new measurement.
        
        Args:
            bbox: tuple or list [x, y, w, h] of top-left coordinates and size.
        """
        x, y, w, h = bbox
        cx = x + w / 2.0
        cy = y + h / 2.0
        a = float(w) / float(h) if h > 0 else 0.0
        z = np.array([[cx], [cy], [a], [h]])

        if self.state is None:
            # Initialization on first detection
            self.state = np.zeros((8, 1))
            self.state[0:4] = z
            self.P = np.eye(8) * 10.0
            self.status = "Found"
            self.mode = "Update"
            return

        # y = z - H * x
        y_residual = z - np.dot(self.H, self.state)
        # S = H * P * H^T + R
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        # K = P * H^T * inv(S)
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        # x = x + K * y
        self.state = self.state + np.dot(K, y_residual)
        # P = (I - K * H) * P
        I = np.eye(8)
        self.P = np.dot(I - np.dot(K, self.H), self.P)

        self.status = "Found"
        self.mode = "Update"

    def get_rect(self):
        """
        Convert tracking state [cx, cy, a, h] back to top-left rect [x, y, w, h].
        """
        if self.state is None:
            return None
        cx, cy, a, h = self.state[0:4, 0]
        w = a * h
        x = cx - w / 2.0
        y = cy - h / 2.0
        return int(x), int(y), int(w), int(h)
