import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

# Configuration - Using a publicly available Hungarian model
MODEL_NAME = "SZTAKI-HLT/hubert-base-cc"  # Public Hungarian BERT model
BATCH_SIZE = 16
MAX_LENGTH = 128
EPOCHS = 3
LEARNING_RATE = 2e-5
NUM_CLASSES = 3  # negative, neutral, positive
LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2} # Create label mapping
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 1. Load and prepare dataset
class HungarianSentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = [LABEL_MAP[label] for label in labels]  # Convert string labels to numbers
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


def load_and_prepare_data():
    # Load dataset
    dataset = load_dataset("NYTK/HuSST")

    # Prepare data
    train_texts = dataset['train']['sentence']
    train_labels = dataset['train']['label']
    val_texts = dataset['validation']['sentence']
    val_labels = dataset['validation']['label']

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Create datasets
    train_dataset = HungarianSentimentDataset(
        train_texts, train_labels, tokenizer, MAX_LENGTH
    )
    val_dataset = HungarianSentimentDataset(
        val_texts, val_labels, tokenizer, MAX_LENGTH
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False
    )

    return train_loader, val_loader, tokenizer

# 2. Define model
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


# 3. Training functions
def train_epoch(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0
    correct_predictions = 0

    for batch in tqdm(data_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(outputs, dim=1)
        correct_predictions += torch.sum(preds == labels)

    accuracy = correct_predictions.double() / len(data_loader.dataset)
    avg_loss = total_loss / len(data_loader)

    return avg_loss, accuracy


def eval_model(model, data_loader, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluation"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = nn.CrossEntropyLoss()(outputs, labels)

            total_loss += loss.item()
            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == labels)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct_predictions.double() / len(data_loader.dataset)
    avg_loss = total_loss / len(data_loader)

    return avg_loss, accuracy, all_preds, all_labels


# 4. Main training loop
def main():
    # Load data
    train_loader, val_loader, tokenizer = load_and_prepare_data()

    # Initialize model
    model = SentimentClassifier(MODEL_NAME, NUM_CLASSES).to(DEVICE)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    best_accuracy = 0
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print("-" * 30)

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, DEVICE
        )
        print(f"Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")

        val_loss, val_acc, val_preds, val_labels = eval_model(
            model, val_loader, DEVICE
        )
        print(f"Val Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

        # Save best model
        if val_acc > best_accuracy:
            torch.save(model.state_dict(), "best_model.pt")
            best_accuracy = val_acc
            print("Saved new best model")

        # Classification report
        print("\nClassification Report:")
        print(classification_report(
            val_labels, val_preds,
            target_names=["negative", "neutral", "positive"]
        ))

    print(f"\nTraining complete. Best validation accuracy: {best_accuracy:.4f}")


# 5. Prediction function (to use after training)
def predict_sentiment(text, model, tokenizer, device):
    encoding = tokenizer(
        text,
        max_length=MAX_LENGTH,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        _, preds = torch.max(outputs, dim=1)

    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    return label_map[preds.cpu().item()]


if __name__ == "__main__":
    main()
