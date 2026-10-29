from flask import Flask, render_template, request, jsonify
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = Flask(__name__)

# Load custom PyTorch model
model_path = "./my_pytorch_pr_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

# Route to website homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle AI calculations when click "Analyze"
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
        
    # Run the text through PyTorch pipeline
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = F.softmax(outputs.logits, dim=1).flatten().tolist()
    
    # Send the raw percentages back to the webpage JavaScript
    return jsonify({
        'negative': probabilities[0] * 100,
        'neutral': probabilities[1] * 100,
        'positive': probabilities[2] * 100
    })

if __name__ == '__main__':
    # Start the server locally on port 5001
    app.run(debug=True, port=5001)