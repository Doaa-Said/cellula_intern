import spacy
import numpy as np
import joblib
import json
import tensorflow as tf

# Load spaCy
nlp = spacy.load("en_core_web_lg")

# Load model + scaler + labels
model = tf.keras.models.load_model("Toxic_content_classification_project/lstm_model.h5")
scaler = joblib.load("Toxic_content_classification_project/scaler.pkl")

with open("Toxic_content_classification_project/config.json") as f:
    config = json.load(f)
category_map = config["category_map"]
id2label = {v: k for k, v in category_map.items()}

# Inference function (extended)
def classify(query=None, image_desc=None):
    if query is None:
        query_vec = np.zeros(300)   # dummy zeros if no text
    else:
        query_vec = nlp(query).vector

    if image_desc is None:
        image_vec = np.zeros(300)   # dummy zeros if no image
    else:
        image_vec = nlp(image_desc).vector

    # Concatenate → always 600 dims
    x = np.concatenate([query_vec, image_vec]).reshape(1, -1)

    # Scale
    x = scaler.transform(x)
    timesteps = 20
    features_per_step = x.shape[1] // timesteps
    x = x.reshape(-1, timesteps, features_per_step)

    preds = model.predict(x)
    pred_idx = np.argmax(preds, axis=1)[0]
    return id2label[pred_idx]
