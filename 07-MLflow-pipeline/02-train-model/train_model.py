import json
import os

import argparse
from pathlib import Path

from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

import mlflow
from mlflow.models.signature import infer_signature
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.sklearn import save_model
from mlflow.tracking.client import MlflowClient

os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://172.7.0.45:30234"    # "http://minio-service.kubeflow.svc:9000"
os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"


mlflow.set_tracking_uri("http://172.7.0.45:32677")    # "http://mlflow-service.mlflow-system.svc:5000"
experiment = mlflow.get_experiment(experiment_id="0")
mlflow.set_experiment(experiment.name)

def train_model(config):
    # Open and reads file "data"
    with open(config.data) as data_file:
        data = json.load(data_file)
    
    # Data type is 'dict', however since the file was loaded as a json object, it is first loaded as a string
    # thus we need to load again from such string in order to get the dict-type object.
    data = json.loads(data)

    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    
    # mlflow.start_run(run_name='test')
    with mlflow.start_run(experiment_id=0) as run:
        print(mlflow.active_run().info.run_id)
        # Initialize and train the model
        model = XGBClassifier()
        model.fit(X_train, y_train)

        # Get predictions
        y_pred = model.predict(X_test)
        
        # Get accuracy
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)

        # Save output into file
        with open(config.acc, 'w') as accuracy_file:
            accuracy_file.write(str(accuracy))

        mlflow.pytorch.log_model(model, "xgb")
        mlflow.end_run()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    
    # Input argument: data
    # Output argument: accuracy
    p.add_argument('--data', type=str, default='./raw_data.json')
    p.add_argument('--acc', type=str)

    config = p.parse_args()
    
    train_model(config)