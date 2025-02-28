"""
Tests for the search module.
"""

import pytest
import numpy as np
from unittest.mock import patch, AsyncMock, MagicMock

from search import search_similar, prepare_context

@pytest.mark.asyncio
async def test_search_similar(mock_openai_client, temp_db_path):
    """Test searching for similar documents"""
    # Set up test data
    query = "Test query"
    mock_embedding = [0.1] * 1536
    
    # Mock the database and embedding functions
    with patch('search.get_embedding', return_value=AsyncMock(return_value=mock_embedding)), \
         patch('search.get_documents_with_embeddings') as mock_get_docs, \
         patch('search.calculate_similarity', return_value=0.8):
        
        # Mock document results
        doc1 = (1, "https://example1.com", "Test Doc 1", "Content 1", np.array([0.2] * 1536, dtype=np.float32).tobytes())
        doc2 = (2, "https://example2.com", "Test Doc 2", "Content 2", np.array([0.3] * 1536, dtype=np.float32).tobytes())
        mock_get_docs.return_value = [doc1, doc2]
        
        # Test the search function
        results = await search_similar(query, top_k=2)
        
        # Check results
        assert len(results) == 2
        # Each result should be (similarity, doc_id, url, title, content)
        assert results[0][0] == 0.8  # similarity
        assert results[0][1] in (1, 2)  # doc_id
        assert results[1][0] == 0.8  # similarity
        assert results[1][1] in (1, 2)  # doc_id

@pytest.mark.asyncio
async def test_search_similar_no_embedding(mock_openai_client):
    """Test search handling when embedding fails"""
    # Mock embedding function to return None (error case)
    with patch('search.get_embedding', AsyncMock(return_value=None)):
        results = await search_similar("Test query")
        assert results == []

@pytest.mark.asyncio
async def test_search_similar_empty_db(mock_openai_client):
    """Test search with an empty database"""
    # Mock empty database
    with patch('search.get_embedding', return_value=AsyncMock(return_value=[0.1] * 1536)), \
         patch('search.get_documents_with_embeddings', return_value=[]):
        
        results = await search_similar("Test query")
        assert results == []

def test_prepare_context():
    """Test context preparation from documents"""
    # Create test documents with the format (similarity, doc_id, url, title, content)
    doc1 = (0.9, 1, "https://example1.com", "Test Doc 1", "Content from document 1")
    doc2 = (0.8, 2, "https://example2.com", "Test Doc 2", "Content from document 2")
    
    # Test with multiple documents
    context = prepare_context([doc1, doc2])
    assert "Content from document 1" in context
    assert "Content from document 2" in context
    
    # Test with a single document
    context = prepare_context([doc1])
    assert context == "Content from document 1"
    
    # Test with empty list
    context = prepare_context([])
    assert context == ""