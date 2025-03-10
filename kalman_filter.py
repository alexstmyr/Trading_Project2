import numpy as np

class KalmanFilterReg:
    def __init__(self):
        """Initializes the Kalman Filter for dynamic hedge ratio estimation."""
        self.x = np.array([1, 1.1089])  # Initial hedge ratio
        self.A = np.eye(2)  # State transition matrix
        self.Q = np.eye(2) * 0.001  # Process noise covariance
        self.R = np.array([[100]])  # Measurement noise covariance
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
        for i in range(len(x_series)):
            self.predict()
            x_val = x_series[i]
            y_val = y_series[i]
            self.update(x_val, y_val)
            params = self.x
            hedge_ratios.append(params[1])
        return np.array(hedge_ratios)
