# Dockerfile for the prediction FastAPI endpoint
FROM python:3.9

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

RUN pip install torch==2.0.1+cpu --index-url https://download.pytorch.org/whl/cpu

RUN pip install -r requirements.txt

COPY . /app

CMD exec uvicorn main:app --reload --host 0.0.0.0 --port $PORT
