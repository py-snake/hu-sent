from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import os

app = Flask(__name__)

# Configuration
MODEL_NAME = os.getenv("MODEL_NAME", "SZTAKI-HLT/hubert-base-cc")
MAX_LENGTH = int(os.getenv("MAX_LENGTH", 128))
LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SentimentClassifier(nn.Module):
    def __init__(self, model_name, num_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        return self.fc(pooled_output)


# Load model at startup
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = SentimentClassifier(MODEL_NAME, num_classes=3).to(DEVICE)
model.load_state_dict(torch.load("best_model.pt", map_location=DEVICE))
model.eval()


@app.route('/predict', methods=['POST'])
def predict():
    text = request.json.get('text', '')

    encoding = tokenizer(
        text,
        max_length=MAX_LENGTH,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(DEVICE)
    attention_mask = encoding['attention_mask'].to(DEVICE)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        _, preds = torch.max(outputs, dim=1)

    return jsonify({
        "sentiment": LABEL_MAP[preds.cpu().item()],
        "text": text
    })


@app.route('/health')
def health():
    return "OK", 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

