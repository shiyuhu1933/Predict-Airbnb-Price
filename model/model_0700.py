from typing import Tuple
import numpy as np


class Model:
    # Modify your model, default is a linear regression model with random weights
    ID_DICT = {"Shiyu Hu": "", "BU_ID": "U88890700", "BU_EMAIL": "Shiyuhu@bu.edu"}

    def __init__(self):
        self.theta = None

    def preprocess(self, X: np.array, y: np.array) -> Tuple[np.array, np.array]:
        ###############################################
        ####      add preprocessing code here      ####
        ###############################################
        return X, y

    def train(self, X_train: np.array, y_train: np.array):
        """
        Train model with training data
        """
        ###############################################
        ####   initialize and train your model     ####
        ###############################################
        X_train = np.vstack((np.ones((X_train.shape[0],)), X_train.T)).T
        step1 = np.dot(X_train.T, X_train)
        step2 = np.linalg.pinv(step1)
        step3 = np.dot(step2, X_train.T)
        self.theta = np.dot(step3, y_train)
        print("This is theta", self)
        return self

    def predict(self, X_val: np.array) -> np.array:
        """
        Predict with model and given feature
        """
        ###############################################
        ####      add model prediction code here   ####
        ###############################################
        X_val = np.vstack((np.ones((X_val.shape[0],)), X_val.T)).T
        return np.dot(X_val, self.theta)
