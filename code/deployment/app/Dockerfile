FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY code/deployment/app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY code/deployment/app/app.py /app/app.py
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--browser.gatherUsageStats=false"]
