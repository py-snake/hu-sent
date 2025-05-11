FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY model/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model file and application code
COPY model/best_model.pt /app/
COPY model/app.py /app/

CMD ["python", "app.py"]
