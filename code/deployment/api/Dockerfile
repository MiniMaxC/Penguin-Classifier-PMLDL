FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements-model.txt /app/requirements-model.txt
COPY code/deployment/api/requirements.txt /app/requirements-api.txt
RUN pip install --no-cache-dir -r requirements-model.txt -r requirements-api.txt
COPY penguins.py /app/penguins.py
COPY code/deployment/api/main.py /app/main.py
COPY models/model.joblib models/metadata.json /app/models/
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
