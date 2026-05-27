import sqlite3
import json
import uuid
import os
from typing import Dict, List, Optional, Any

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
METADATA_DB_PATH = os.path.join(BASE_DIR, "metadata.db")

def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(METADATA_DB_PATH, timeout=60.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db():
    """Initializes the central SQLite database for users, files, and queries."""
    os.makedirs(os.path.dirname(METADATA_DB_PATH), exist_ok=True)
    conn = get_connection()
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
    except sqlite3.OperationalError:
        pass
    cursor = conn.cursor()
    
    # 1. Users Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        active_token TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    
    # Safely add active_token if it doesn't exist (for existing databases)
    cursor.execute("PRAGMA table_info(users);")
    columns = [row["name"] for row in cursor.fetchall()]
    if "active_token" not in columns:
        cursor.execute("ALTER TABLE users ADD COLUMN active_token TEXT;")
    
    # 2. Files Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS files (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        file_name TEXT NOT NULL,
        table_name TEXT NOT NULL,
        row_count INTEGER NOT NULL,
        columns_json TEXT NOT NULL,
        sample_rows_json TEXT NOT NULL,
        uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
    );
    """)
    
    # 3. Query History Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS query_history (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        file_id TEXT NOT NULL,
        question TEXT NOT NULL,
        sql_query TEXT NOT NULL,
        explanation TEXT NOT NULL,
        visualization_config_json TEXT NOT NULL,
        executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
        FOREIGN KEY (file_id) REFERENCES files (id) ON DELETE CASCADE
    );
    """)
    
    # 4. Chat/Conversational History Table (for follow-up context)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chat_history (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        file_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
        FOREIGN KEY (file_id) REFERENCES files (id) ON DELETE CASCADE
    );
    """)
    
    conn.commit()
    conn.close()

# --- User Management ---

def create_user(email: str, password_hash: str) -> str:
    user_id = str(uuid.uuid4())
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO users (id, email, password_hash) VALUES (?, ?, ?)",
        (user_id, email.lower().strip(), password_hash)
    )
    conn.commit()
    conn.close()
    return user_id

def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE email = ?", (email.lower().strip(),))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def get_user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def update_user_active_token(user_id: str, token: str):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET active_token = ? WHERE id = ?", (token, user_id))
    conn.commit()
    conn.close()

def get_user_active_token(user_id: str) -> Optional[str]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT active_token FROM users WHERE id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    return row["active_token"] if row else None

# --- File Management ---

def add_file(user_id: str, file_name: str, table_name: str, row_count: int, columns: List[str], sample_rows: List[Dict[str, Any]], custom_file_id: Optional[str] = None) -> str:
    file_id = custom_file_id or str(uuid.uuid4())
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO files (id, user_id, file_name, table_name, row_count, columns_json, sample_rows_json)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (file_id, user_id, file_name, table_name, row_count, json.dumps(columns), json.dumps(sample_rows))
    )
    conn.commit()
    conn.close()
    return file_id

def get_files_by_user(user_id: str) -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM files WHERE user_id = ? ORDER BY uploaded_at DESC", (user_id,))
    rows = cursor.fetchall()
    conn.close()
    
    result = []
    for r in rows:
        item = dict(r)
        item["columns"] = json.loads(item["columns_json"])
        item["sample_rows"] = json.loads(item["sample_rows_json"])
        result.append(item)
    return result

def get_file_by_id(file_id: str, user_id: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM files WHERE id = ? AND user_id = ?", (file_id, user_id))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        item = dict(row)
        item["columns"] = json.loads(item["columns_json"])
        item["sample_rows"] = json.loads(item["sample_rows_json"])
        return item
    return None

def delete_file(file_id: str, user_id: str) -> bool:
    """Deletes metadata and allows caller to clean up the associated physical database file."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM files WHERE id = ? AND user_id = ?", (file_id, user_id))
    rows_affected = cursor.rowcount
    conn.commit()
    conn.close()
    return rows_affected > 0

# --- Query History ---

def add_query_history(user_id: str, file_id: str, question: str, sql_query: str, explanation: str, viz_config: Dict[str, Any]):
    query_id = str(uuid.uuid4())
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO query_history (id, user_id, file_id, question, sql_query, explanation, visualization_config_json)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (query_id, user_id, file_id, question, sql_query, explanation, json.dumps(viz_config))
    )
    conn.commit()
    conn.close()
    return query_id

def get_query_history(user_id: str, file_id: Optional[str] = None) -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    if file_id:
        cursor.execute(
            "SELECT * FROM query_history WHERE user_id = ? AND file_id = ? ORDER BY executed_at DESC LIMIT 50",
            (user_id, file_id)
        )
    else:
        cursor.execute(
            "SELECT * FROM query_history WHERE user_id = ? ORDER BY executed_at DESC LIMIT 50",
            (user_id,)
        )
    rows = cursor.fetchall()
    conn.close()
    
    result = []
    for r in rows:
        item = dict(r)
        item["visualization_config"] = json.loads(item["visualization_config_json"])
        result.append(item)
    return result

# --- Chat History Management (Conversational Thread Context) ---

def add_chat_message(user_id: str, file_id: str, role: str, content: str):
    msg_id = str(uuid.uuid4())
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO chat_history (id, user_id, file_id, role, content) VALUES (?, ?, ?, ?, ?)",
        (msg_id, user_id, file_id, role, content)
    )
    conn.commit()
    conn.close()

def get_chat_history(user_id: str, file_id: str) -> List[Dict[str, str]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT role, content FROM chat_history WHERE user_id = ? AND file_id = ? ORDER BY created_at ASC",
        (user_id, file_id)
    )
    rows = cursor.fetchall()
    conn.close()
    return [{"role": r["role"], "content": r["content"]} for r in rows]

def clear_chat_history(user_id: str, file_id: str):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "DELETE FROM chat_history WHERE user_id = ? AND file_id = ?",
        (user_id, file_id)
    )
    conn.commit()
    conn.close()
