import numpy as np
import matplotlib.pyplot as plt

class KalmanFilterReg():
    def __init__(self):
        self.x = np.array([1,1]) #initial observation (hedge-ratio)
        self.A = np.eye(2) # transition matrix
        self.Q = np.eye(2) * 0.001 # covariance matrix in estimations 
        self.R = np.array([[1]]) * 10 # errors in observations - si el ruido es muy grande se atonta, si es muy chico hace overfit
        self.P = np.eye(2) * 1000 # predicted error covariance matrix

    def predict(self):
        self.P = self.A @ self.P @self.A.T +self.Q

    def update(self, x , y):
        C = np.array([[1, x]]) #observation (1,2)
        S = C @ self.P @ C.T + self.R
        K = self.P @ C.T @ np.linalg.inv(S) # kalman gain
        self.P = (np.eye(2) - K @ C) @ self.P
        self.x = self.x + K @ (y - C @ self.x)