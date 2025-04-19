import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

# Configuration
MODEL_NAME = "SZTAKI-HLT/hubert-base-cc"  # Same as training
MAX_LENGTH = 128  # Same as training
LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}  # Reverse mapping
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


def load_model(model_path):
    # Initialize model with same architecture as training
    model = SentimentClassifier(MODEL_NAME, num_classes=3).to(DEVICE)

    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model


def predict_sentiment(text, model, tokenizer):
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

    return LABEL_MAP[preds.cpu().item()]


def main():
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = load_model("best_model.pt")

    # Example predictions
    texts = [
        "Ez a film fantasztikus volt!",
        "Nem tetszett a könyv.",
        "Átlagos élmény volt, semmi különös."
    ]

    for text in texts:
        sentiment = predict_sentiment(text, model, tokenizer)
        print(f"Text: {text}\nSentiment: {sentiment}\n")


if __name__ == "__main__":
    main()

