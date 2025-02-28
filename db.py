"""
Database module for the Pydantic RAG application.
Handles database setup, connection, and query operations.
"""

import os
import sqlite3
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Setup database path - use environment variable or default to project directory
DB_PATH = os.getenv("DATABASE_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "pydantic_docs_simple.db"))

# Create directory for database if it doesn't exist
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

def setup_database(reset=False) -> str:
    """Create a simple SQLite database for storing documentation"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Option to reset the database completely
    if reset:
        cursor.execute("DROP TABLE IF EXISTS docs")
        
    # Create docs table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS docs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        url TEXT NOT NULL,
        title TEXT,
        content TEXT NOT NULL,
        embedding BLOB,
        crawl_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    conn.commit()
    conn.close()
    return f"Database setup completed at {DB_PATH}"
    
def reset_database() -> str:
    """Reset the database by dropping and recreating the tables"""
    return setup_database(reset=True)

def store_document(
    url: str, 
    title: str, 
    content: str, 
    embedding: List[float]
) -> None:
    """Store a document in the database with its embedding. 
    Updates existing documents with the same URL."""
    # Convert embedding to bytes for storage
    embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()
    
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check if this URL already exists
    cursor.execute("SELECT id FROM docs WHERE url = ?", (url,))
    existing_doc = cursor.fetchone()
    
    if existing_doc:
        # Update existing document
        cursor.execute(
            "UPDATE docs SET title = ?, content = ?, embedding = ? WHERE url = ?",
            (title, content, embedding_bytes, url)
        )
    else:
        # Insert new document
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