import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import gradio as gr
from google import genai
from dotenv import load_dotenv

# 1. Load the fine-tuned PyTorch DistilBERT model
MODEL_PATH = "./my_pytorch_pr_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

# 2. Load environment variables
load_dotenv()

# 3. Gemini client. The key must come from the environment, never from the code.
api_key=os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("Set the GEMINI_API_KEY environment variable before starting the app.")
client = genai.Client(api_key=api_key)
GEMINI_MODEL = "gemini-3.6-flash"  # keep whichever model name works for you


def run_classifier(text):
    """Return class probabilities (as fractions) from the fine-tuned model."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1).flatten().tolist()
    return {
        "Negative (risk)": probs[0],
        "Neutral": probs[1],
        "Positive (growth)": probs[2],
    }


def score_block(scores):
    """Plain-text bar readout shown above every reply."""
    width = 20
    lines = []
    for name, p in scores.items():
        filled = round(p * width)
        bar = "█" * filled + "░" * (width - filled)
        lines.append(f"{name:<18}{bar}  {p * 100:5.1f}%")
    return "```\n" + "\n".join(lines) + "\n```"


def pr_copilot_chat(user_message, history):
    if not user_message.strip():
        return "Paste a draft or post to review."

    scores = run_classifier(user_message)
    readout = score_block(scores)

    prompt = f"""
You are a senior PR communications advisor reviewing a draft before it goes out.

Draft:
\"\"\"{user_message}\"\"\"

A fine-tuned sentiment classifier scored the draft as follows:
- Negative (risk): {scores['Negative (risk)'] * 100:.1f}%
- Neutral: {scores['Neutral'] * 100:.1f}%
- Positive (growth): {scores['Positive (growth)'] * 100:.1f}%

The scores are already shown to the user, so do not repeat them.

Write a short reply in plain, direct language with no emoji and these two sections:
1. Why it reads this way: point to the specific phrasing, tone, or brand risks that explain the scores.
2. Two safer options: two rewritten versions of the draft that keep its intent.
Keep it under 200 words.
"""

    try:
        response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        return f"{readout}\n\n{response.text}"
    except Exception as e:
        print(f"Gemini error: {e}")  # full detail stays in the terminal
        return (
            f"{readout}\n\n"
            "The explanation service is unavailable right now. "
            "The classifier scores above are still valid."
        )


# 3. Visual design: quiet, editorial, one neutral family. Risk is conveyed by the scores, not by color.
THEME = gr.themes.Base(
    primary_hue="slate",
    neutral_hue="slate",
    radius_size=gr.themes.sizes.radius_sm,
    font=(gr.themes.GoogleFont("Public Sans"), "system-ui", "sans-serif"),
).set(
    body_background_fill="#F6F8FA",
    body_background_fill_dark="#11181F",
    block_shadow="none",
)

CSS = """
.gradio-container { max-width: 860px !important; margin: 0 auto !important; }
.gradio-container h1 {
    font-family: "Iowan Old Style", "Palatino Linotype", Georgia, serif;
    font-weight: 600;
    font-size: 2rem;
    letter-spacing: -0.01em;
    text-align: left !important;
}
.gradio-container .prose p { text-align: left !important; }
"""

EXAMPLES = [
    "We're thrilled to announce our new feature! Honestly, if you don't like it, that's on you.",
    "Thanks for the feedback. We're pausing the rollout while we look into the issues you've raised.",
    "Our latest update is live. See the release notes for what changed.",
]

# Gradio 6 moved theme and css from the constructor to launch(); Gradio 5 expects the opposite.
GRADIO_MAJOR = int(gr.__version__.split(".")[0])
style = {"theme": THEME, "css": CSS}

demo = gr.ChatInterface(
    fn=pr_copilot_chat,
    title="PR Risk Analyzer",
    description=(
        "Paste a draft or post. A fine-tuned DistilBERT model scores its tone, "
        "and Gemini explains the score and suggests safer wording."
    ),
    textbox=gr.Textbox(placeholder="Paste a draft or post…", container=False, scale=7),
    examples=EXAMPLES,
    cache_examples=False,
    **({} if GRADIO_MAJOR >= 6 else style),
)

if __name__ == "__main__":
    demo.launch(server_port=7860, **(style if GRADIO_MAJOR >= 6 else {}))