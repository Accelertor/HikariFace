import sqlite3
import numpy as np
from datetime import datetime  # you forgot to import this!

DB_PATH = "face_embeddings.db"

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    return conn

def create_tables():
    conn = get_connection()
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS embeddings (
                    roll_number TEXT PRIMARY KEY,
                    name TEXT,
                    embedding BLOB,
                    present_absent TEXT DEFAULT 'absent',
                    last_seen TEXT
                )''')
    conn.commit()
    conn.close()

def add_or_update_user(roll, name, embedding):
    conn = get_connection()
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO embeddings (roll_number, name, embedding) VALUES (?, ?, ?)",
              (roll, name, embedding.astype(np.float32).tobytes()))
    conn.commit()
    conn.close()

def get_embedding(roll):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT embedding FROM embeddings WHERE roll_number = ?", (roll,))
    row = c.fetchone()
    conn.close()
    if row:
        return np.frombuffer(row[0], dtype=np.float32)
    else:
        return None

def mark_attendance(roll, status):
    conn = get_connection()
    c = conn.cursor()
    now = datetime.now().isoformat(timespec='seconds')
    c.execute("""
        UPDATE embeddings
        SET present_absent = ?, last_seen = ?
        WHERE roll_number = ?
    """, (status, now, roll))
    conn.commit()
    conn.close()

def get_all_users():
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT roll_number, name, present_absent, last_seen FROM embeddings")
    rows = c.fetchall()
    conn.close()
    return rows

def get_attendance_for_today(roll):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT present_absent, last_seen FROM embeddings WHERE roll_number = ?", (roll,))
    row = c.fetchone()
    conn.close()

    if row:
        status, last_seen = row
        if status == "present" and last_seen:
            # Check if last_seen date is today
            last_seen_date = datetime.fromisoformat(last_seen).date()
            if last_seen_date == datetime.now().date():
                return {"status": status, "timestamp": last_seen}
    return None

# Call this once at start
create_tables()
