import kfp
from kfp import dsl
from config import CFG


@dsl.pipeline(
    name='nevret-iris',
    description='nevret iris test'
)
def iris_pipeline():
    add_p = dsl.ContainerOp(
        name="load iris data pipeline",
        image="nevret/nevret-iris-preprocessing:0.5",
        command=['python', 'load_data.py'],
        arguments=[
            '--data_path', './Iris.csv'
        ],
        file_outputs={'iris' : '/iris.csv'}
    )
    
    ml = dsl.ContainerOp(
        name="training pipeline",
        image="nevret/nevret-iris-training:0.5",
        command=['python', 'train.py'],
        arguments=[
            '--data', add_p.outputs['iris']
        ]
    )

    ml.after(add_p)
    
    
    
if __name__ == "__main__":
    import kfp.compiler as compiler
    compiler.Compiler().compile(iris_pipeline, "02-Iris/pipeline.yaml")
    
    import kfp
    import requests

    USERNAME = CFG.kf_username
    PASSWORD = CFG.kf_password
    NAMESPACE = CFG.kf_namespace
    HOST = CFG.kf_host

    session = requests.Session()
    response = session.get(HOST)

    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
    }

    data = {"login": USERNAME, "password": PASSWORD}
    session.post(response.url, headers=headers, data=data)
    session_cookie = session.cookies.get_dict()["authservice_session"]    

    
    # Submit the pipeline for execution
    pipeline_func = iris_pipeline
    client = kfp.Client(
        host=f"{HOST}/pipeline",
        namespace=f"{NAMESPACE}",
        cookies=f"authservice_session={session_cookie}",
    )
    print(client.list_pipelines())
    
    run = client.create_run_from_pipeline_func(pipeline_func, arguments={})
