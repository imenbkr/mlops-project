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
import os
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

mlflow.set_tracking_uri("https://dagshub.com/imenbkr/mlops-project.mlflow")
os.environ['MLFLOW_TRACKING_USERNAME']="imenbkr"
os.environ["MLFLOW_TRACKING_PASSWORD"]="47203520d6f0e2662b976c3ca77ade4e57156990" 


#setup mlflow
mlflow.start_run()
mlflow.log_param("parameter name", "value")
mlflow.log_metric("metric name", 1)
mlflow.end_run()


@app.get("/")
def read_root():
    return {"Hello": "to Success Prediction App"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080)