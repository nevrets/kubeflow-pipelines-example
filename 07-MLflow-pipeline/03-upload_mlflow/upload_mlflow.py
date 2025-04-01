import os
import json
import argparse
from pathlib import Path

import torch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.pytorch
from mlflow.models.signature import infer_signature

# kubeflow - minio-service - 9000:30234/TCP  
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://172.7.0.45:30234"    # "http://minio-service.kubeflow.svc:9000"
os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"


def upload_model(config):
    # Open and reads file "data"
    with open(config.data) as data_file:
        data = json.load(data_file)
    
    # Open and reads file "data"
    with open(config.model) as model:
        model.load_state_dict(torch.load('best_model.pth'))
    
    data = json.loads(data)

    X_test_tensor = data['X_test']
    # y_test_tensor = data['y_test']
    
    # Infer the signature of the model
    sample_input = X_test_tensor[:1]
    model.eval()
    with torch.no_grad():
        sample_output = model(sample_input)
    signature = infer_signature(sample_input.numpy(), sample_output.numpy())

    print("Model signature:", signature)

    mlflow.set_tracking_uri("http://172.7.0.45:32677")    # "http://mlflow-service.mlflow-system.svc:5000"
    # mlflow-system  mlflow-service  5000:32677/TCP

    experiment_id = mlflow.create_experiment('mlflow-pipeline')

    with mlflow.start_run(experiment_id=experiment_id) as run:
        print(mlflow.active_run().info.run_id)
        mlflow.log_param('max_iter', 500)

        # Log model artifact to S3
        artifact_path = "07-Kubeflow-pipeline-MLflow/best-model"      
        mlflow.log_artifact('best_model.pth', artifact_path=artifact_path)
        
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="sklearn-model",
            signature=signature,
            registered_model_name='SimpleNN',
        )
    
    model_uri = f"runs:/{run.info.run_id}/sklearn-model"
    
    # Register model linked to S3 artifact location          
    mlflow.register_model(
        model_uri,
        'SimpleNN'
    )

    return {"artifact_path": artifact_path, "artifact_uri": run.info.artifact_uri}


if __name__ == '__main__':

    # Defining and parsing the command-line arguments
    p = argparse.ArgumentParser(description='My program description')
    
    # Input argument: data
    # Output argument: accuracy
    p.add_argument('--data', type=str)
    p.add_argument('--model', type=str)

    config = p.parse_args()
    upload_model(config)