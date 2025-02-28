"""
Tests for the database module.
"""

import pytest
import os
import sqlite3
import numpy as np
from unittest.mock import patch

from db import setup_database, store_document, get_all_documents, get_documents_with_embeddings

def test_setup_database(temp_db_path):
    """Test database setup functionality"""
    with patch('db.DB_PATH', temp_db_path):
        result = setup_database()
        assert 'Database setup completed' in result
        
        # Verify the table exists
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='docs'")
        assert cursor.fetchone() is not None
        conn.close()

def test_store_document(temp_db_path):
    """Test storing a document in the database"""
    test_url = "https://example.com"
    test_title = "Example Document"
    test_content = "This is example content"
    test_embedding = [0.5] * 1536  # Simple test embedding
    
    with patch('db.DB_PATH', temp_db_path):
        # Store the document
        store_document(test_url, test_title, test_content, test_embedding)
        
        # Verify it was stored
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT url, title, content FROM docs WHERE url = ?", (test_url,))
        result = cursor.fetchone()
        conn.close()
        
        assert result is not None
        assert result[0] == test_url
        assert result[1] == test_title
        assert result[2] == test_content

def test_get_all_documents(temp_db_path):
    """Test retrieving all documents from the database"""
    # Add a test document
    test_url = "https://example2.com"
    test_title = "Another Document"
    test_content = "More test content"
    test_embedding = [0.7] * 1536
    
    with patch('db.DB_PATH', temp_db_path):
        # Store the document
        store_document(test_url, test_title, test_content, test_embedding)
        
        # Get all documents
        docs = get_all_documents()
        
        # Check the results
        assert len(docs) >= 1  # At least our document should be there
        assert any(doc['url'] == test_url for doc in docs)
        assert any(doc['title'] == test_title for doc in docs)

def test_get_documents_with_embeddings(temp_db_path):
    """Test retrieving documents with their embeddings"""
    # Add a test document
    test_url = "https://example3.com"
    test_title = "Document with Embedding"
    test_content = "Embedding test content"
    test_embedding = [0.9] * 1536
    
    with patch('db.DB_PATH', temp_db_path):
        # Store the document
        store_document(test_url, test_title, test_content, test_embedding)
        
        # Get documents with embeddings
        docs = get_documents_with_embeddings()
        
        # Check the results
        assert len(docs) >= 1
        
        # Find our document in the results
        found = False
        for doc in docs:
            _, url, title, content, embedding_bytes = doc
            if url == test_url:
                found = True
                assert title == test_title
                assert content == test_content
                # Check that embedding bytes can be converted back
                embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                assert len(embedding) > 0
        
        assert found, "Added document not found in results"