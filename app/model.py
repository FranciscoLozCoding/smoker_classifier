import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import load
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

class SmokerModel:
    """
    Smoker Model Class that can predict new instances

    INPUTS
    ---
    model_path: the path to the model file
    scaler_path: the path to the min max scaler file
    """
    def __init__(self, model_path, scaler_path):
        self.model = load(model_path)
        self.scaler = load(scaler_path)
        self.labels = ["non-smoker", "smoker"]
        
    def scale(self, X):
        """
        Apply the scaler used to train the model to the new data

        INPUT
        -----
        X: the data to be scaled
        
        OUTPUT
        ------
        returns the scaled data
        """

        new_data_scaled = self.scaler.transform(X)

        return new_data_scaled

    def predict(self, X: pd.DataFrame) -> str:
        """
        Make a prediction on one sample using the loaded model.

        INPUT
        -----
        X: the data to predict a label for

        OUTPUT
        ------
        predicted label
        """

        # scale the data
        X_scaled = self.scale(X)

        #check array only has one sample
        if X.shape[0] != 1:
            raise ValueError("Input array must contain only one sample, but {} samples were found".format(X.shape[0]))
            return

        # Now, use the scaled data to make predictions using the loaded model
        array = self.model.predict(X_scaled)

        #predict
        predicted_label = array[0]
        str_label = self.labels[predicted_label]
        return str_label