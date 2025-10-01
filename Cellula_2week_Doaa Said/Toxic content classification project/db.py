import sqlite3

DB_PATH = "classification.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            input_text TEXT,
            prediction TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def insert_record(input_text, prediction):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO records (input_text, prediction) VALUES (?, ?)", (input_text, prediction))
    conn.commit()
    conn.close()

def fetch_all():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM records ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    return rows
