FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY sentiment_processor/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY sentiment_processor/process_sentiment.py .

CMD ["python", "process_sentiment.py"]
