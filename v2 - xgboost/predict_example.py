import os
import numpy as np
import hsfs
import joblib
import xgboost as xgb

class Predict(object):

    def __init__(self):
        """ Initializes the serving state, reads a trained model"""        
        # Get feature store handle
        fs_conn = hsfs.connection()
        self.fs = fs_conn.get_feature_store()

        # Load the model from the JSON file
        self.model = xgb.XGBRegressor()
        self.model.load_model(os.environ["ARTIFACT_FILES_PATH"] + "/xgboost_model.json")
        print("Initialization Complete")

    def predict(self, inputs):
        """ Serves a prediction request usign a trained model"""
        return self.model.predict(inputs).tolist()
