# syntax=docker/dockerfile:1

# Stage 1: Capture versions
FROM python:3.9-slim as version-capturer

WORKDIR /app
COPY *.txt ./

RUN pip install --no-cache-dir -r file1_requirements.txt -r file2_requirements.txt && \
    pip freeze > /tmp/requirements_versions.txt

# Stage 2: Web server to display versions
FROM python:3.9-slim

WORKDIR /app
COPY --from=version-capturer /tmp/requirements_versions.txt .

# Install minimal web server
RUN pip install --no-cache-dir flask

# Create simple web server
COPY <<EOF ./version_server.py
from flask import Flask, send_file
app = Flask(__name__)

@app.route('/')
def show_versions():
    return send_file('requirements_versions.txt', mimetype='text/plain')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
EOF

EXPOSE 5000
CMD ["python", "version_server.py"]
