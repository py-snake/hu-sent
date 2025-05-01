from flask import Flask, request, render_template, jsonify
import requests
import os

app = Flask(__name__)
API_URL = os.getenv("API_URL", "http://sentiment-api:5000/predict")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form.get('text')
    try:
        response = requests.post(API_URL, json={'text': text})
        return jsonify(response.json())
    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

