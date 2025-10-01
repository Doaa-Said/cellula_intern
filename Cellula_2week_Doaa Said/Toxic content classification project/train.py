import pandas as pd
import numpy as np
import spacy
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
import joblib

# Load dataset
df = pd.read_csv("cellula toxic data  (1).csv")

# Normalize labels
df['Toxic Category'] = (
    df['Toxic Category']
    .str.strip()
    .str.replace("-", "-", regex=False)
    .str.title()
)

category_map = {
    'Safe': 0,
    'Violent Crimes': 1,
    'Non-Violent Crimes': 2,
    'Unsafe': 3,
    'Unknown S-Type': 4,
    'Sex-Related Crimes': 5,
    'Suicide & Self-Harm': 6,
    'Elections': 7,
    'Child Sexual Exploitation': 8
}
df['Toxic Category'] = df['Toxic Category'].map(category_map)

# Load spaCy embeddings
nlp = spacy.load("en_core_web_lg")
df['query'] = df['query'].apply(lambda t: nlp(t).vector)
df['image descriptions'] = df['image descriptions'].apply(lambda t: nlp(t).vector)

# Combine embeddings
X = np.array([np.concatenate([q, d]) for q, d in zip(df["query"], df["image descriptions"])])
y = df["Toxic Category"].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2022)

# Scale features
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape for LSTM
timesteps = 20
features_per_step = X_train.shape[1] // timesteps
X_train = X_train.reshape(-1, timesteps, features_per_step)
X_test = X_test.reshape(-1, timesteps, features_per_step)

# One-hot labels
num_classes = len(np.unique(y))
y_train_cat = to_categorical(y_train, num_classes=num_classes)
y_test_cat = to_categorical(y_test, num_classes=num_classes)

# Build LSTM model
model = Sequential([
    Input(shape=(timesteps, features_per_step)),
    LSTM(128),
    Dropout(0.5),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(num_classes, activation="softmax")
])

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

# Train
history = model.fit(X_train, y_train_cat, validation_data=(X_test, y_test_cat), epochs=10, batch_size=32)

# Evaluate
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test_cat, axis=1)

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=range(num_classes),
            yticklabels=range(num_classes))
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
# Ensure directory exists
import os
import json
import joblib
import matplotlib.pyplot as plt
os.makedirs("Toxic_content_classification_project", exist_ok=True)

# Save confusion matrix
plt.savefig("Toxic_content_classification_project/confusion_matrix.png")

# Save model
model.save("Toxic_content_classification_project/lstm_model.h5")

# Save scaler
joblib.dump(scaler, "Toxic_content_classification_project/scaler.pkl")

# Save category map
with open("Toxic_content_classification_project/config.json", "w") as f:
    json.dump({"category_map": category_map}, f)