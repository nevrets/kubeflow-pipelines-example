FROM python:3.10-slim
ENV PYTHONUNBUFFERED 1
# RUN mkdir /workspace
WORKDIR /workspace
COPY . /workspace/
RUN pip install --no-cache-dir -r requirements.txt