import kfp
from kfp import dsl
from kfp.components import func_to_container_op
from config import CFG
@func_to_container_op
def show_results(decision_tree:float, 
                 logistic_regression:float, 
                 svm:float, 
                 naive_bayes:float, 
                 xgb:float) -> None:
    
    # Given the outputs from decision_tree, logistic regression, svm, naive_bayes, xgboost components
    print(f"Decision tree (accuracy): {decision_tree}")
    print(f"Logistic regression (accuracy): {logistic_regression}")
    print(f"SVM (SVC) (accuracy): {svm}")
    print(f"Naive Bayes (Gaussian) (accuracy): {naive_bayes}")
    print(f"XGBoost (accuracy): {xgb}")


@dsl.pipeline(name='ML Models Pipeline', description='Applies Decision Tree, Logistic Regression, SVM, Naive Bayes, XGBoost for classification problem.')
def ml_models_pipeline():
    # Loads the yaml manifest for each component
    download = kfp.components.load_component_from_file('/data/nevret/kubeflow_pipelines/03-ML-model-project/01-download_data/download_data.yaml')
    decision_tree = kfp.components.load_component_from_file('/data/nevret/kubeflow_pipelines/03-ML-model-project/02-decision_tree/decision_tree.yaml')
    logistic_regression = kfp.components.load_component_from_file('/data/nevret/kubeflow_pipelines/03-ML-model-project/03-logistic_regression/logistic_regression.yaml')
    svm = kfp.components.load_component_from_file('/data/nevret/kubeflow_pipelines/03-ML-model-project/04-svm/svm.yaml')
    naive_bayes = kfp.components.load_component_from_file('/data/nevret/kubeflow_pipelines/03-ML-model-project/05-naive_bayes/naive_bayes.yaml')
    xgb = kfp.components.load_component_from_file('/data/nevret/kubeflow_pipelines/03-ML-model-project/06-xgb/xgb.yaml')
    
    # Run download_data task
    download_task = download()
    
    # Run ML models tasks with input data
    decision_tree_task = decision_tree(download_task.output)
    logistic_regression_task = logistic_regression(download_task.output)
    svm_task = svm(download_task.output)
    naive_bayes_task = naive_bayes(download_task.output)
    xgb_task = xgb(download_task.output)
    
    # Given the outputs from ML models tasks
    # the component "show_results" is called to print the results.
    show_results(decision_tree_task.output, logistic_regression_task.output, svm_task.output, naive_bayes_task.output, xgb_task.output)



if __name__ == '__main__':
    # ml_models_pipeline()
    kfp.compiler.Compiler().compile(ml_models_pipeline, "03-ML-model-project/ML-model-pipeline.yaml")
    
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
    pipeline_func = ml_models_pipeline
    client = kfp.Client(
        host=f"{HOST}/pipeline",
        namespace=f"{NAMESPACE}",
        cookies=f"authservice_session={session_cookie}",
    )
    
    print(client.list_pipelines())
    
    run = client.create_run_from_pipeline_func(pipeline_func, arguments={})
    
    print('')