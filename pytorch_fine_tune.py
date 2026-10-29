import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split  # NEW

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"🚀 Using computing device: {device.type.upper()}")

print("--- STEP 1: Preparing Data for PyTorch ---")
df = pd.read_csv('pr_master_dataset.csv')

# hold out 20% for testing. The model never trains on these rows.
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)
test_df.to_csv('pr_test_split.csv', index=False)  # NEW: the evaluation script loads this file
print(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows")

# build dataset from the training rows only
dataset = Dataset.from_pandas(train_df.reset_index(drop=True))

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# PyTorch DataLoaders handle shuffling and batching the training data
train_loader = DataLoader(tokenized_datasets, batch_size=16, shuffle=True)

print("--- STEP 2: Initializing Model and Optimizer ---")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)
model.to(device)  # Push the model's brain weights onto your hardware chip

# The Optimizer is the mathematical engine that tweaks weights based on errors
optimizer = AdamW(model.parameters(), lr=2e-5)

print("\n🏋️ STARTING RAW PYTORCH TRAINING LOOP 🏋️")
model.train()  # Put model in training mode

epochs = 1  # Decide this BEFORE looking at test results (see note below)
for epoch in range(epochs):
    print(f"\n--- Epoch {epoch+1} ---")

    for step, batch in enumerate(train_loader):
        # Move this batch of data to your Mac's processing chip
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        # 1. Clear out old mathematical gradients from the last guess
        optimizer.zero_grad()

        # 2. Forward Pass: Let the AI guess
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        # 3. Grab the Loss (a float number representing how wrong the AI's guess was)
        loss = outputs.loss

        # 4. Backward Pass: Backpropagation (PyTorch calculates the calculus for how to fix the error)
        loss.backward()

        # 5. Optimization Step: Actually tweak the weights by a tiny fraction
        optimizer.step()

        # Print out the exact math tracking every 50 steps
        if step % 50 == 0:
            print(f"Step {step:<4} | Current Mathematical Loss (Error Rate): {loss.item():.4f}")

print("\n--- STEP 3: Saving PyTorch-Trained Model ---")
model.save_pretrained("./my_pytorch_pr_model")
tokenizer.save_pretrained("./my_pytorch_pr_model")
print("🎉 PyTorch training complete! Saved in 'my_pytorch_pr_model'")