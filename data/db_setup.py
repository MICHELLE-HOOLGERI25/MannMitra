import sqlite3
import os
import streamlit as st

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mannmitra.db")

@st.cache_resource
def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL;")      # ✅ allows concurrent reads/writes
    conn.execute("PRAGMA busy_timeout=30000;")    # ✅ waits before failing if locked
    conn.execute("PRAGMA synchronous=NORMAL;")    # ✅ balances safety & performance
    conn.execute("PRAGMA foreign_keys=ON;")
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

if __name__ == "__main__":
    init_db()
