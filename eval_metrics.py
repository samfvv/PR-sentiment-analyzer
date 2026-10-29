import os
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. Load fine-tuned PyTorch DistilBERT model
MODEL_PATH = "./my_pytorch_pr_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# 2. Load the held-out test set saved by fine_tune.py
DATA_FILE = "pr_test_split.csv"

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"Could not find '{DATA_FILE}'. Run fine_tune.py first.")

print(f"Loading held-out test set from '{DATA_FILE}'...")
test_df = pd.read_csv(DATA_FILE)

label_col = "label"
text_col = "text"

texts = test_df[text_col].astype(str).tolist()
true_labels = test_df[label_col].astype(int).tolist()

# 3. Batch Inference Loop
batch_size = 32
predictions = []

print(f"Evaluating DistilBERT predictions on {len(texts)} test samples...")

with torch.no_grad():
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(
            batch_texts, 
            padding=True, 
            truncation=True, 
            max_length=128, 
            return_tensors="pt"
        ).to(device)
        
        outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        predictions.extend(preds)

# 4. Generate Final Metrics Report
label_names = ["Negative (Risk)", "Neutral", "Positive (Growth)"]

print("\n" + "="*50)
print("             MODEL EVALUATION METRICS             ")
print("="*50 + "\n")

print(classification_report(true_labels, predictions, target_names=label_names, digits=4))

acc = accuracy_score(true_labels, predictions)
macro_f1 = f1_score(true_labels, predictions, average="macro")

print(f"Overall Accuracy: {acc * 100:.2f}%")
print(f"Macro F1-Score:   {macro_f1 * 100:.2f}%")

# Majority-class baseline: accuracy from always guessing the most common class
baseline = test_df[label_col].value_counts(normalize=True).max()
print(f"Majority-class baseline: {baseline * 100:.2f}%")