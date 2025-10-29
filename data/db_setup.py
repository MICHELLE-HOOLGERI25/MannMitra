import sqlite3
import os

DB_PATH = os.path.join("data", "mannmitra.db")

def get_connection():
    # ensure /data exists
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return conn

def init_db():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS user_mood (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            date TEXT,
            who5_score INTEGER,
            reflection1 TEXT,
            reflection2 TEXT,
            reflection3 TEXT,
            inferred_mood TEXT,
            growth_score REAL,
            UNIQUE(username, date) ON CONFLICT REPLACE
        )
    """)
    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()
