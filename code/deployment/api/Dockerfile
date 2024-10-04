# FROM python:3.10-slim
FROM python:3.9

WORKDIR /api

RUN pip install uvicorn fastapi pandas mlflow scikit-learn

COPY . /api

# RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "code.deployment.api.api:app", "--host", "0.0.0.0", "--port", "8000"]
