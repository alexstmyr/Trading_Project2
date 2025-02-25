import numpy as np

class KalmanFilterReg:
    def __init__(self):
        """Initializes the Kalman Filter for Hedge Ratio Estimation."""
        self.x = np.array([1, 1])  # Initial hedge ratio guess (intercept & slope)
        self.A = np.eye(2)  # Transition matrix (identity)
        self.Q = np.eye(2) * 0.001  # Covariance in estimation
        self.R = np.array([[1]]) * 10  # Observation noise (adjust to avoid overfitting)
        self.P = np.eye(2) * 1000  # Initial error covariance

    def predict(self):
        """Predicts the next hedge ratio."""
        self.P = self.A @ self.P @ self.A.T + self.Q  # Covariance prediction

    def update(self, x, y):
        """Updates the hedge ratio estimate based on new observations."""
        C = np.array([[1, x]])  # Observation matrix
        S = C @ self.P @ C.T + self.R  # Innovation covariance
        K = self.P @ C.T @ np.linalg.inv(S)  # Kalman gain
        self.P = (np.eye(2) - K @ C) @ self.P  # Update covariance matrix
        self.x = self.x + K @ (y - C @ self.x)  # Update state estimate

    def run_kalman_filter(self, x_series, y_series):
        """Applies the Kalman Filter over time to estimate the hedge ratio."""
        kalman_preds = []
        hedge_ratios = []

        for x, y in zip(x_series, y_series):
            self.predict()
            self.update(x, y)
            hedge_ratios.append(self.x[1])  # Store hedge ratio (slope)
            kalman_preds.append(self.x[0] + self.x[1] * x)  # Predicted spread

        return np.array(hedge_ratios), np.array(kalman_preds)
