import numpy as np

class KalmanFilterReg:
    def __init__(self):
        """Initializes the Kalman Filter for dynamic hedge ratio estimation."""
        self.x = np.array([1, 1.1])  # Initial hedge ratio
        self.A = np.eye(2)  # State transition matrix
        self.Q = np.eye(2) * 10  # Process noise covariance
        self.R = np.array([[0.1]])  # Measurement noise covariance
        self.P = np.eye(2) * 10  # Error covariance

    def predict(self):
        """Predicts the next hedge ratio."""
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, x, y):
        """Updates the hedge ratio based on new observations."""
        C = np.array([[1, x]])
        S = C @ self.P @ C.T + self.R
        K = self.P @ C.T @ np.linalg.inv(S)
        self.P = (np.eye(2) - K @ C) @ self.P
        self.x = self.x + K @ (y - C @ self.x)

    def run_kalman_filter(self, x_series, y_series):
        """Applies the Kalman Filter to estimate the hedge ratio dynamically."""
        hedge_ratios = []
        for x, y in zip(x_series, y_series):
            self.predict()
            self.update(x, y)
            params = self.x
            hedge_ratios.append(params[0] + params[1] * x)  # Store hedge ratio
        return np.array(hedge_ratios)
