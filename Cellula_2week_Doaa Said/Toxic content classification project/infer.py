import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -------------------------
# 1. Define category labels
# -------------------------
id2label = {
    0: "Safe",
    1: "Violent Crimes",
    2: "Non-Violent Crimes",
    3: "Unsafe",
    4: "Unknown S-Type",
    5: "Sex-Related Crimes",
    6: "Suicide & Self-Harm",
    7: "Elections",
    8: "Child Sexual Exploitation"
}
label2id = {v: k for k, v in id2label.items()}

# -------------------------
# 2. Load model + tokenizer
# -------------------------
MODEL_PATH = "./saved_model"   # update this with your trained model path

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

# -------------------------
# 3. Prediction function
# -------------------------
def classify_text(text: str) -> str:
    """Classify a single text input into one of the toxic categories."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
    return id2label[prediction]
