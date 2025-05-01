FROM python:3.9-slim

WORKDIR /app

# Install necessary system dependencies
RUN apt-get update && apt-get install -y gcc python3-dev curl wget && rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt specific to the model service
COPY model/requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the rest of the model code to the container's /app directory
COPY model /app

# Set the command to run the application (ensure the app.py is at /app/app.py)
CMD ["python", "/app/app.py"]
