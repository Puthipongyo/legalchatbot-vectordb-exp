import mlflow
from pathlib import Path


def setup_mlflow(ExperimentName: str): 
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment(ExperimentName)
