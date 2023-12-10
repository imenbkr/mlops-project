from operator import index
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
import sklearn
from fastapi import FastAPI, File, UploadFile
import uvicorn
import sys  
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import mlflow
from fastapi.middleware.cors import CORSMiddleware
import os

import uvicorn
from mlflow.tracking import MlflowClient

from sklearn.preprocessing import StandardScaler 
from processing_startup import drop_features, time_series_data, process_dataset
import joblib

dataset=pd.read_csv("C:/Users/IMEN/Mlops-project/startup data.csv",\
                    converters={'status': lambda x: int(x == 'acquired')},parse_dates=['founded_at','first_funding_at','last_funding_at'])
dataset.rename(columns={'status':'is_acquired'}, inplace=True)
y=dataset["is_acquired"]
X= dataset.loc[:, dataset.columns != 'is_acquired']
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2, random_state=42)


# Load your model
def load_model():
    model_path = "C:/Users/IMEN/Mlops-project/best_model.pkl"  # Replace with the actual path to your model
    return joblib.load(model_path)

model = load_model()

os.environ['GIT_PYTHON_GIT_EXECUTABLE'] = 'C:/Program Files/Git/bin/git.exe'

app = FastAPI()
origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#DagsHub_username = os.environ["DAGSHUB_USERNAME"]
#DagsHub_token= os.environ["DAGSHUB_TOKEN"]



# Initialize MLflow
mlflow.set_tracking_uri("https://dagshub.com/imenbkr/mlops-project.mlflow")
os.environ['MLFLOW_TRACKING_USERNAME'] = "imenbkr"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "47203520d6f0e2662b976c3ca77ade4e57156990"

# Start MLflow run
with mlflow.start_run():

    """# Log parameters
    mlflow.log_param("parameter_name", "value")

    # Log preprocessing information or any other relevant information
    mlflow.log_param("preprocessing_info", "StandardScaler used for numeric features")"""

    """# Log metrics 
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("precision", precision_score(y_test, y_pred))
    mlflow.log_metric("recall", recall_score(y_test, y_pred))"""

    # this endpoint receives data in the form of a csv file (startup data)
    @app.post("/predict/csv")
    def return_predictions(file: UploadFile = File(...)):
        # Read the uploaded CSV file
        data = pd.read_csv(file.file)

        # Preprocess the data using functions from processing_startup.py
        data = process_dataset(data)
        data = time_series_data(data)
        data = drop_features(data)

        # Make predictions
        predictions = model.predict(data)

        # Log predictions as an artifact
        mlflow.log_artifact("predictions.csv", data)

        return {"predictions": predictions.tolist()}

    if __name__ == "__main__":
        uvicorn.run("main:app", host="0.0.0.0", port=8080)