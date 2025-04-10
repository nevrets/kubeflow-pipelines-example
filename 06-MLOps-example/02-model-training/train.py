import argparse
import pandas as pd
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from mlflow.models.signature import infer_signature

import mlflow


def split_train_test_data(iris):
    train, test = train_test_split(iris, test_size=0.2, random_state=42, stratify=iris['Species'])

    X_train = train.drop(columns=['Species'], axis=1)
    y_train = train['Species']
    X_test = test.drop(columns=['Species'], axis=1)
    y_test = test['Species']

    return X_train, X_test, y_train, y_test


def main(args):
    # Check download folder
    os.makedirs('/mnt/outputs', exist_ok=True)
    
    experiment = mlflow.get_experiment_by_name("[DEMO]Iris_Training")
    if experiment is None:
        experiment_id = mlflow.create_experiment("[DEMO]Iris_Training")
        logger.info(f"Created new experiment with ID: {experiment_id}")
    else:
        experiment_id = experiment.experiment_id
        logger.info(f"Using existing experiment with ID: {experiment_id}")
        
    mlflow.set_experiment("[DEMO]Iris_Training")
    
    # Run 이름 설정
    run_name = f"LogisticRegression_iter{args.max_iter}_random{args.random_state}"
    with mlflow.start_run(run_name=run_name) as run:
        # run_id를 output parameter로 저장
        with open('/mnt/outputs/mlflow_run_id.txt', 'w') as f:
            f.write(run.info.run_id)

        # 하이퍼파라미터 설정
        hyperparameters = {
            "max_iter": args.max_iter,
            "multi_class": args.multi_class,
            "random_state": args.random_state,
        }
        
        # 모델 파라미터 로깅
        mlflow.log_params(hyperparameters)

        iris = pd.read_csv(os.path.join(args.data_path, 'iris.csv'))
        iris = iris.drop(columns=['Id']).reset_index(drop=True)
        X_train, X_test, y_train, y_test = split_train_test_data(iris)
        
        logger.info(f"X_train shape: {X_train.shape}")
        logger.info(f"X_test shape: {X_test.shape}")
        logger.info(f"y_train shape: {y_train.shape}")
        logger.info(f"y_test shape: {y_test.shape}")
        
        # 모델 학습
        model = LogisticRegression(**hyperparameters)
        model.fit(X_train, y_train)
        
        # 예측 및 평가
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # 성능 메트릭 계산
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='weighted'),
            "recall": recall_score(y_test, y_pred, average='weighted'),
            "f1": f1_score(y_test, y_pred, average='weighted')
        }
        
        logger.info(f"metrics: {metrics}")
        
        # 성능 메트릭 로깅
        mlflow.log_metrics(metrics)
        
        # 혼동 행렬 시각화
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('/mnt/confusion_matrix.png')
        mlflow.log_artifact('/mnt/confusion_matrix.png')

        
        # 분류 보고서 저장
        report = classification_report(y_test, y_pred, output_dict=True)
        with open('/mnt/outputs/classification_report.json', 'w') as f:
            json.dump(report, f)
        mlflow.log_artifact('/mnt/outputs/classification_report.json')
        
        # 모델 저장 (시그니처 추가)
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            model, 
            "model",
            signature=signature,
            input_example=X_train.iloc[:5]
        )
        
        # 모델 등록
        model_name = "Iris_logistic_regression"
        version = mlflow.register_model(f"runs:/{run.info.run_id}/model", model_name)
        
        print(f"Run Name: {run_name}")
        print(f"하이퍼파라미터:")
        for param, value in hyperparameters.items():
            print(f"- {param}: {value}")
        print(f"\n성능 메트릭:")
        for metric, value in metrics.items():
            print(f"- {metric}: {value:.4f}")
        print(f"\n모델 등록 완료: {model_name} (버전 {version.version})")



if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('--data_path', type=str, default=".")
    argument_parser.add_argument('--random_state', type=int, default=42)
    argument_parser.add_argument('--max_iter', type=int, default=1000)
    argument_parser.add_argument('--multi_class', type=str, default="multinomial")
    args = argument_parser.parse_args()

    main(args)