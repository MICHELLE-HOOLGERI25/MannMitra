import sqlite3
import os
import streamlit as st

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mannmitra.db")

@st.cache_resource
def get_connection():
    # Create a cached, thread-safe connection
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=30000;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def init_db():
    """Create tables if they don't exist (safe on Streamlit Cloud)."""
    conn = get_connection()
    cur = conn.cursor()

    # --- example tables; replace/add your own schema as before ---
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        pin TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS user_mood (
        username TEXT,
        date TEXT,
        who5_score INTEGER,
        reflection1 TEXT,
        reflection2 TEXT,
        reflection3 TEXT,
        inferred_mood TEXT,
        growth_score REAL,
        PRIMARY KEY (username, date)
    )
    """)

    conn.commit()
