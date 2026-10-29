# PR Risk Sentiment Analyzer

A three-class text classifier that scores PR drafts and social posts as **negative (risk)**, **neutral**, or **positive (growth)**. I fine-tuned DistilBERT in a custom PyTorch training loop, evaluated it on a held-out test set against a majority-class baseline, and built two interfaces around it: a Flask web app and a Gradio chat copilot that uses the Gemini API to explain the scores and suggest safer rewrites. This project was designed to learn the systematics of training a LLM model while incorporating an interface that users can converse with to see training results.


## Results

Evaluated on a held-out test set of 7,220 samples (20% stratified split, never used for training).

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Negative (Risk) | 0.708 | 0.834 | 0.766 | 1,943 |
| Neutral | 0.748 | 0.683 | 0.714 | 2,907 |
| Positive (Growth) | 0.823 | 0.791 | 0.807 | 2,370 |

- **Accuracy:** 75.90%
- **Macro-F1:** 76.22%
- **Majority-class baseline:** 40.26% (always predicting Neutral)

The model reaches nearly twice the baseline accuracy after a single epoch. Recall on the negative (risk) class is the highest of the three, which suits a tool meant to catch risky text. Neutral is the weakest class.

### Evaluation note

An earlier evaluation of this project reported higher numbers because the training script used the full dataset, so the test rows overlapped with the training data. I fixed this by splitting the data before training, saving the held-out rows to `pr_test_split.csv`, and re-evaluating on only those rows. The numbers above are from the corrected pipeline.

## How it works

```
Public datasets + hand-written examples
        |  load_and_mix.py
        v
pr_master_dataset.csv  (~36,100 rows)
        |  fine_tune.py   (80/20 stratified split)
        v
my_pytorch_pr_model/   +   pr_test_split.csv
        |                          |
        |                  eval_metrics.py  -> report
        v
app.py (Flask)   /   copilot_app.py (Gradio + Gemini)
```

### Data

`load_and_mix.py` builds one dataset from three sources and standardizes the labels (0 = negative, 1 = neutral, 2 = positive):

- A multiclass sentiment dataset of social media text from Hugging Face (`Sp1786/multiclass-sentiment-analysis-dataset`, train split)
- A financial news sentiment dataset from Kaggle (`financial.csv`), with its text labels mapped to the same 0/1/2 scheme
- 18 hand-written examples of Twitter-style slang (a very small share of the data)
- make sure to download these csv files from Kaggle and Hugging Face to be loaded in training

Rows with missing values are dropped, and the result is shuffled with a fixed seed. The labels come from the source datasets, not from my own annotation.

### Training

`fine_tune.py` fine-tunes `distilbert-base-uncased` with a new 3-class classification head, using a hand-written PyTorch loop:

- Stratified 80/20 train/test split (`random_state=42`): 28,880 training rows and 7,220 test rows
- Tokenization truncated/padded to 128 tokens
- AdamW optimizer, learning rate 2e-5, batch size 16, 1 epoch
- Each step: zero gradients, forward pass, cross-entropy loss, backward pass, optimizer step
- Uses Apple Silicon (MPS) when available, otherwise CPU

### Evaluation

`eval_metrics.py` loads the saved model and `pr_test_split.csv`, runs batched inference, and prints a per-class precision/recall/F1 report, overall accuracy, macro-F1, and the majority-class baseline.

### Apps

- **Flask web app (`app.py`):** serves a page that sends text to a `/predict` endpoint and returns the three class percentages as JSON. Runs on port 5001.
- **Gradio copilot (`copilot_app.py`):** runs the classifier, then sends the draft and the class probabilities to the Gemini API, which explains the score and proposes revised options. Runs on port 7860.

## Getting started

```bash
pip install torch transformers datasets pandas scikit-learn flask gradio google-genai
```

Run the pipeline in order:

```bash
python load_and_mix.py     # builds pr_master_dataset.csv
python fine_tune.py        # trains the model, saves ./my_pytorch_pr_model and pr_test_split.csv
python eval_metrics.py     # prints the held-out evaluation report
```

Then start either interface:

```bash
python app.py              # Flask app on http://localhost:5001
```

```bash
export GEMINI_API_KEY="your-key-here"
python copilot_app.py      # Gradio app on http://localhost:7860
```

## Project structure

```
.
├── load_and_mix.py        # builds the combined dataset
├── fine_tune.py           # PyTorch training loop
├── eval_metrics.py        # held-out evaluation
├── app.py                 # Flask web app
├── templates/index.html   # Flask front end
├── copilot_app.py         # Gradio + Gemini copilot
└── README.md
```

## Limitations

- **"PR risk" is a sentiment proxy.** The labels are sentiment labels, and I treat negative text as risk. The model has not been validated against real PR outcomes.
- **Mixed domains.** Social media posts and financial news differ a lot in style, and a single overall score can hide per-domain differences.
- **Limited training.** One epoch, no validation set, no hyperparameter tuning, and a single run, so the test metrics carry some statistical uncertainty.
- **Possible duplicates.** Near-duplicate sentences across the train and test splits would inflate scores slightly; the data has not been deduplicated.
- **LLM output is not evaluated.** Gemini's explanations are plausible rationales based on the text and the scores, not an interpretation of the model's internals, and the suggested rewrites have not been measured for safety.
