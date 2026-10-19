FROM python:3.13-slim

WORKDIR /app

COPY Backend/api/requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY Backend/api/ ./api/
COPY models/ ./models/
COPY mlruns/ ./mlruns/
COPY mlflow.db ./mlflow.db
COPY mlflow_models.py ./mlflow_models.py
COPY mlflow_tracking.py ./mlflow_tracking.py

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]