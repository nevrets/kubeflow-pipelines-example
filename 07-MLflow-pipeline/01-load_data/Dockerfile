FROM python:3.8-slim                                                
ENV PYTHONUNBUFFERED 1

WORKDIR /app
COPY . /app

RUN apt-get update
RUN pip install -r requirements.txt

# ENTRYPOINT ["python"]
# CMD ["run.py"]