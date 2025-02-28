"""
Configuration file for pytest.
Contains fixtures and configuration for testing.
"""

import os
import sys
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import sqlite3
import tempfile
import numpy as np

# Ensure the project root is in the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

@pytest.fixture
def mock_openai_client():
    """Mock for the OpenAI client to avoid actual API calls during tests"""
    mock_client = AsyncMock()
    
    # Mock embedding response
    mock_embedding_data = AsyncMock()
    mock_embedding_data.data = [AsyncMock(embedding=[0.1] * 1536)]
    mock_client.embeddings.create.return_value = mock_embedding_data
    
    # Mock completion response
    mock_completion = AsyncMock()
    mock_completion.choices = [AsyncMock(message=AsyncMock(content="Mocked answer"))]
    mock_client.chat.completions.create.return_value = mock_completion
    
    return mock_client

@pytest.fixture
def temp_db_path():
    """Create a temporary database for testing"""
    # Create temporary file
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    
    # Set up the test database
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS docs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        url TEXT NOT NULL,
        title TEXT,
        content TEXT NOT NULL,
        embedding BLOB
    )
    ''')
    
    # Add some test data
    embedding_bytes = np.array([0.1] * 1536, dtype=np.float32).tobytes()
    cursor.execute(
        "INSERT INTO docs (url, title, content, embedding) VALUES (?, ?, ?, ?)",
        ("https://test.com", "Test Document", "This is test content", embedding_bytes)
    )
    conn.commit()
    conn.close()
    
    yield path
    
    # Clean up the temp file after tests
    os.unlink(path)

@pytest.fixture
def mock_requests_get():
    """Mock for requests.get to avoid actual HTTP requests during tests"""
    with patch('requests.get') as mock_get:
        mock_response = MagicMock()
        mock_response.text = """
        <html>
            <head><title>Test Page</title></head>
            <body>
                <p>Test content</p>
                <a href="https://test2.com">Link 1</a>
                <a href="/relative-link">Relative Link</a>
            </body>
        </html>
        """
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response
        yield mock_get