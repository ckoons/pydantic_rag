"""
Database module for the Pydantic RAG application.
Handles database setup, connection, and query operations.
"""

import os
import sqlite3
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

# Setup database path
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pydantic_docs_simple.db")

def setup_database() -> str:
    """Create a simple SQLite database for storing documentation"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create docs table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS docs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        url TEXT NOT NULL,
        title TEXT,
        content TEXT NOT NULL,
        embedding BLOB
    )
    ''')
    
    conn.commit()
    conn.close()
    return f"Database setup completed at {DB_PATH}"

def store_document(
    url: str, 
    title: str, 
    content: str, 
    embedding: List[float]
) -> None:
    """Store a document in the database with its embedding"""
    # Convert embedding to bytes for storage
    embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()
    
    # Store in database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO docs (url, title, content, embedding) VALUES (?, ?, ?, ?)",
        (url, title, content, embedding_bytes)
    )
    conn.commit()
    conn.close()

def get_all_documents() -> List[Dict[str, str]]:
    """Retrieve all documents from the database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT url, title FROM docs")
    results = cursor.fetchall()
    conn.close()
    
    docs = []
    for url, title in results:
        docs.append({"url": url, "title": title or url})
    
    return docs

def get_documents_with_embeddings() -> List[Tuple]:
    """Retrieve all documents with their embeddings"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, url, title, content, embedding FROM docs")
    results = cursor.fetchall()
    conn.close()
    return results