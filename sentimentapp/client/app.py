from flask import Flask, render_template, request, jsonify
import requests
import os
import time

app = Flask(__name__)
API_URL = os.getenv('API_URL', 'http://sentiment-api:5000/predict')


def call_api_with_retry(text, max_retries=3, delay=2):
    for i in range(max_retries):
        try:
            response = requests.post(API_URL, json={'text': text}, timeout=10)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            if i == max_retries - 1:
                raise
            time.sleep(delay)
    return None


@app.route('/', methods=['GET', 'POST'])
def index():
    sentiment = None
    error = None

    if request.method == 'POST':
        text = request.form['text']
        try:
            response = call_api_with_retry(text)
            if response.status_code == 200:
                sentiment = response.json()
        except Exception as e:
            error = f"Failed to analyze sentiment: {str(e)}"

    return render_template('index.html', sentiment=sentiment, error=error)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

