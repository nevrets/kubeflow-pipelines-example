FROM kubeflownotebookswg/jupyter-pytorch-cuda-full:latest
WORKDIR /app
COPY requirements.txt /app
RUN pip install -r requirements.txt
COPY upload_mlflow.py /app