# Import needed libraries
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from datasets import DatasetDict

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import nltk
nltk.download('stopwords')

nltk.download('wordnet')
# Read the file
df = pd.read_csv("cellula toxic data  (1).csv")

# Show the columns names
print(df.columns.tolist())

df.drop_duplicates(subset="query", inplace=True) # Remove duplicates
img_desc = df.drop_duplicates(subset="image descriptions")
df = df[["query","Toxic Category"]]

img_desc = img_desc[["image descriptions","Toxic Category"]]
img_desc = img_desc.rename(columns={"image descriptions": "query"})

df = pd.concat([df, img_desc], ignore_index=True)

df.dropna(inplace=True) # Remove Null entries

df = df.reset_index(drop=True) # Reset the indexing

wordnet=WordNetLemmatizer() # Create an object of the Lemmatizer class

for i in range(len(df['query'])):
    query = re.sub('[^a-zA-Z]', ' ', df['query'][i]) # Remove regex
    query = query.lower() # Lower case all the text
    query = ' '.join(query.split()) # Remove extra whitespace
    query_tokens = query.split() # Split the query into words to be able to process them
    query_tokens = [word for word in query_tokens if not word in set(stopwords.words('english'))] # removing stop words
    query_tokens = [wordnet.lemmatize(word) for word in query_tokens] # Lemmatize the words
    query = ' '.join(query_tokens) # Join the words back into a string
    df['query'][i] = query # Add the string to the list
# Encode multi-class labels
encoder = LabelEncoder() # Initialize encoder

df['Output'] = encoder.fit_transform(df['Toxic Category'])         # Label encode Toxic Category

# Split Data
df = df.rename(columns={"Output": "label","query":"text"})  # rename label column

train_df, __df = train_test_split(df[["text", "label"]], test_size=0.4, stratify=df["label"], random_state=42)
test_df, val_df = train_test_split(__df, test_size=0.5, random_state=42)

del __df
# Tokenize input text
from transformers import AutoTokenizer
from datasets import Dataset

train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))
val_dataset = Dataset.from_pandas(val_df.reset_index(drop=True))

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length")

tokenized_train = train_dataset.map(tokenize, batched=True)
tokenized_test = test_dataset.map(tokenize, batched=True)
tokenized_val = val_dataset.map(tokenize, batched=True)

tokenized_train = tokenized_train.remove_columns(['text'])
tokenized_test = tokenized_test.remove_columns(['text'])
tokenized_val = tokenized_val.remove_columns(['text'])
# Set up parameter-efficient fine-tuning (LoRA)
from peft import get_peft_model, LoraConfig,TaskType

config = LoraConfig(
    task_type=TaskType.SEQ_CLS,  # Sequence classification task
    r=8,                         # Low-rank dimension
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_lin", "v_lin"]
)

# Initialize DistilBERT with pre-trained weights
from transformers import DistilBertForSequenceClassification

L_model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=9
)
L_model = get_peft_model(L_model,config)
L_model.print_trainable_parameters()

# Configure training parameters for multi-class classification
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./lora-distilbert",
   
    learning_rate=2e-4,
    per_device_train_batch_size=4,
    num_train_epochs=4,
    save_strategy="epoch",
    # logging_dir='./logs',
)
# Begin initial fine-tuning experiments
trainer = Trainer(
    model=L_model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val
)

trainer.train()
# Evaluating Performance
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

predictions = trainer.predict(tokenized_test) 
predicted_labels = np.argmax(predictions.predictions, axis=1)
true_labels = tokenized_test['label']

DistilBERT_classification = classification_report(true_labels, predicted_labels)
print(DistilBERT_classification)
# Get labels in the correct order
label_order = np.arange(len(encoder.classes_))

# Compute confusion matrix with fixed label order
cm = confusion_matrix(true_labels, predicted_labels, labels=label_order)

# Plot
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=encoder.classes_,
            yticklabels=encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()







from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

# ==========================
# 1. Load dataset
# ==========================
df = pd.read_csv("cellula toxic data  (1).csv")  # change filename if needed
print("Dataset head:\n", df.head())

# Assuming columns: "query", "image descriptions", "Toxic Category"
# We'll combine query + image descriptions as input text
df["text"] = df["query"].astype(str) + " " + df["image descriptions"].astype(str)

# Encode labels (convert to numbers)
label2id = {label: i for i, label in enumerate(df["Toxic Category"].unique())}
id2label = {i: label for label, i in label2id.items()}
df["label"] = df["Toxic Category"].map(label2id)

# ==========================
# 2. Train/Validation Split
# ==========================
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

dataset = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset
})

# ==========================
# 3. Tokenizer
# ==========================
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

tokenized_dataset = dataset.map(tokenize, batched=True)

# Remove unused columns for Trainer
tokenized_dataset = tokenized_dataset.remove_columns(["query", "image descriptions", "Toxic Category", "text", "__index_level_0__"])
tokenized_dataset.set_format("torch")

# ==========================
# 4. Model
# ==========================
num_labels = len(label2id)
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)

# ==========================
# 5. Metrics (Classification Report)
# ==========================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    report = classification_report(labels, preds, target_names=list(label2id.keys()), output_dict=True)
    return {
        "accuracy": report["accuracy"],
        "precision_macro": report["macro avg"]["precision"],
        "recall_macro": report["macro avg"]["recall"],
        "f1_macro": report["macro avg"]["f1-score"]
    }

# ==========================
# 6. Training Args
# ==========================
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    report_to="none"   # <--- disable TensorBoard
)
# ==========================
# 7. Trainer
# ==========================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# ==========================
# 8. Train & Evaluate
# ==========================
trainer.train()

print("\n🔎 Final Evaluation Report:")
predictions = trainer.predict(tokenized_dataset["validation"])
preds = np.argmax(predictions.predictions, axis=-1)
print(classification_report(predictions.label_ids, preds, target_names=list(label2id.keys())))
# ==========================
# 9. Save Model & Tokenizer
# ==========================
save_dir = "./toxic_lora_model"
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

print(f"\n✅ Model and tokenizer saved to {save_dir}")
