
import pandas as pd
import os

DB_FILE = "database.csv"

def init_db():
    if not os.path.exists(DB_FILE):
        df = pd.DataFrame(columns=["input", "classification"])
        df.to_csv(DB_FILE, index=False)

def add_entry(user_input: str, classification: str):
    df = pd.DataFrame([[user_input, classification]], columns=["input", "classification"])
    df.to_csv(DB_FILE, mode="a", header=False, index=False)

def get_all_entries():
    return pd.read_csv(DB_FILE)
