FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY importer/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create data directory and copy files
RUN mkdir -p /app/data
COPY data/*.json /app/data/
COPY importer/import_comments.py /app/

CMD ["python", "/app/import_comments.py"]
