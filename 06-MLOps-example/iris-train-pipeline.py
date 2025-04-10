import os
import kfp
from kfp import compiler, dsl

from utils import add_nfs_volume, add_configmap, KubeflowClient, add_sharedmemory
from config import CFG


@kfp.dsl.pipeline(
    name="IRIS Model Training Pipeline",
    description="",
)
def iris_train_pipeline(
    lakefs_date: str = "2025-04-08-104352/",
    random_state: int = 42,
    max_iter: int = 1000,
    multi_class: str = "multinomial",
):
    # ----- Temporary volume create ----- #
    vop = dsl.VolumeOp(
        name="Temporary volume create",
        storage_class="k8s-nfs",
        resource_name="tmp-volume",
        modes=dsl.VOLUME_MODE_RWM,     # ReadWriteMany
        size="1Gi",
    ).add_pod_annotation(name="pipelines.kubeflow.org/max_cache_staleness", value="P0D")


    download_datasets_op = (
        dsl.ContainerOp(
            name="Download Datasets",
            image=CFG.hb_iris_demo_function_image,
            command=["python", "lakefs_download.py"],
            arguments=[
                "--root", "/mnt/preprocessed",
                "--repo", "demo-iris",
                "--date", lakefs_date,
            ],
            pvolumes={"/mnt": vop.volume},
        )
        .add_pod_annotation(name="pipelines.kubeflow.org/max_cache_staleness", value="P0D")
        .after(vop)
    )


    train_model_op = (
        dsl.ContainerOp(
            name="Train Model",
            image=CFG.hb_iris_demo_model_image,
            command=["python", "train.py"],
            arguments=[
                "--data_path", "/mnt/preprocessed", 
                "--random_state", random_state,
                "--max_iter", max_iter,
                "--multi_class", multi_class,
            ],
            pvolumes={"/mnt": vop.volume},
        )
        .add_pod_annotation(name="pipelines.kubeflow.org/max_cache_staleness", value="P0D")
        .after(download_datasets_op)
    )
    
    


if __name__ == "__main__":
    # ----- build ----- #
    pipeconf = kfp.dsl.PipelineConf()

    file_name = os.path.splitext(os.path.basename(__file__))[0]
    compiler.Compiler().compile(iris_train_pipeline, f"{file_name}.tar.gz", pipeline_conf=pipeconf)

    try:
        client = KubeflowClient(
            endpoint=CFG.kf_host,
            username=CFG.kf_username,
            password=CFG.kf_password,
            namespace=CFG.kf_namespace,
        )
        res = client.upload_pipeline(f"{file_name}.tar.gz", "iris-train-pipeline")
        print("업로드 결과:", res)
    except Exception as e:
        print("에러 발생:", str(e))
        
    print("파이프라인 목록:", client.client.list_pipelines())