# Startup Success Prediction with Ensemble Modeling.

## Introduction

This project aims to predict the success or failure of startups based on a dataset with various features. We will use ensemble modeling techniques, MLflow for tracking experiments, Docker for containerization, and create a Streamlit dashboard for interactive visualization.

This readme file is still under more development as the project goes on..████▒▒▒▒  50%

## Table of Contents

- [Dataset](#dataset)
- [Requirements](#requirements)
- [Ensemble Modeling](#ensemble-modeling)
- [MLflow Integration](#mlflow-integration)
- [Dockerization](#dockerization)
- [Streamlit Dashboard](#streamlit-dashboard)

## Dataset

The dataset is from kaggle : [Link](https://www.kaggle.com/datasets/manishkc06/startup-success-prediction/data)
## Requirements

- Python 3.9
- Scikit-Learn
- MLflow
- Docker
- Streamlit
- Pandas
- NumPy

I will be uploading the requierements file soon so that installations can be made with a simple command : 
```bash pip install -r requierements.txt```

## Ensemble Modeling
We will experiment with various machine learning algorithms to predict startup success.

## MLflow Integration
We will use MLflow for experiment tracking. All experiments are logged in the mlflow_runs directory.

to access the MLflow UI at ```http://localhost:5000``` to view and compare experiments.

## Dockerization
We will include a Dockerfile to containerize the project.

## Streamlit Dashboard
we will use a Streamlit dashboard that can be found in app.py. to start the dashboard :

```streamlit run app.py```

The dashboard will be available at http://localhost:8501 in the web browser.

