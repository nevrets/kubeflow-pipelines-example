import kfp
from kfp import dsl
from kfp.components import func_to_container_op


@dsl.pipeline(name='ML Models Pipeline', 
              description='MLflow pipeline process.')
def mlflow_upload_pipeline():
    load_data = dsl.ContainerOp(
        name="load-data",
        image="harbor.euso.kr/mlflow-test/load-data:1",
        command=['python', 'load_data.py'],
        arguments=[
            '--data', './data/raw_data.json'
        ],
        file_outputs={'raw_data': '/raw_data.json'}
    )
    
    train_model = dsl.ContainerOp(
        name="train-model",
        image="harbor.euso.kr/mlflow-test/augment-data:11",
        command=['python', 'train_model.py'],
        arguments=[
            '--data_path', load_data.outputs['raw_data']
        ],
        file_outputs={'aug_data': '/shotplan_aug_sample.csv'}
    )

    train_model.after(load_data)

if __name__ == '__main__':
    # ml_models_pipeline()
    kfp.compiler.Compiler().compile(mlflow_upload_pipeline, "07-MLflow-pipeline/MLflow-upload-pipeline.yaml")
    
    import requests

    USERNAME = "test@euclidsoft.co.kr"
    PASSWORD = "dbzmfflem1!"
    NAMESPACE = "intent-analysis"
    HOST = "https://kubeflow.euso.kr" # istio-ingressgateway's external-ip created by the load balancer.
    
    session = requests.Session()
    response = session.get(HOST)

    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
    }

    data = {"login": USERNAME, "password": PASSWORD}
    session.post(response.url, headers=headers, data=data)
    session_cookie = session.cookies.get_dict()["authservice_session"]

    # Submit the pipeline for execution
    pipeline_func = mlflow_upload_pipeline
    client = kfp.Client(
        host=f"{HOST}/pipeline",
        namespace=f"{NAMESPACE}",
        cookies=f"authservice_session={session_cookie}",
    )
    
    print(client.list_pipelines())
    
    run = client.create_run_from_pipeline_func(pipeline_func, arguments={})