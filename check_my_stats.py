import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. Load your newly trained custom model
model_path = "./my_pr_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Put the model in evaluation mode
model.eval()

# 2. Enter ANY draft here (test your corporate risk or your new Twitter slang!)
test_text = "We need to make sure the model is not overfitted"

print("\n📊 ANALYZING MODEL PREDICTIONS...")

# 3. Process the text through the AI's math engine
inputs = tokenizer(test_text, return_tensors="pt", truncation=True, padding=True, max_length=128)

with torch.no_grad():
    outputs = model(**inputs)
    # Extract the raw brain signals (logits)
    logits = outputs.logits
    # Convert those raw signals into clean percentages
    probabilities = F.softmax(logits, dim=1).flatten().tolist()

# 4. Map the percentages to our PR categories
labels = ["NEGATIVE (Risk)", "NEUTRAL (Standard)", "POSITIVE (Growth)"]

print("\n================ CUSTOM AI RISK REPORT ================")
print(f"Draft Text: \"{test_text}\"")
print("-------------------------------------------------------")
print("Confidence Breakdown:")
for label, prob in zip(labels, probabilities):
    # Create a simple visual loading bar using characters
    bar = "█" * int(prob * 20)
    print(f"{label:<20} : [{bar:<20}] {prob:.2%}")
print("=======================================================\n")