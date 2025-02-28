"""
Tests for the embeddings module.
"""

import pytest
import numpy as np
from unittest.mock import patch, AsyncMock

import embeddings

@pytest.mark.asyncio
async def test_get_embedding(mock_openai_client):
    """Test getting embeddings from the OpenAI API"""
    with patch('embeddings.openai_client', mock_openai_client):
        # Test with short text
        short_text = "This is a test"
        embedding = await embeddings.get_embedding(short_text)
        assert embedding is not None
        assert len(embedding) > 0  # Should have some values
        
        # Test with long text (should truncate)
        long_text = "a" * 10000
        embedding = await embeddings.get_embedding(long_text)
        assert embedding is not None
        
        # Test error handling
        mock_openai_client.embeddings.create.side_effect = Exception("API error")
        embedding = await embeddings.get_embedding("Error test")
        assert embedding is None

def test_calculate_similarity():
    """Test vector similarity calculation"""
    # Test identical vectors
    v1 = np.array([1.0, 0.0, 0.0])
    assert embeddings.calculate_similarity(v1, v1) == 1.0
    
    # Test orthogonal vectors
    v2 = np.array([0.0, 1.0, 0.0])
    assert embeddings.calculate_similarity(v1, v2) == 0.0
    
    # Test opposite vectors
    v3 = np.array([-1.0, 0.0, 0.0])
    assert embeddings.calculate_similarity(v1, v3) == -1.0
    
    # Test typical case
    v4 = np.array([0.5, 0.5, 0.0])
    expected = 0.5 / np.sqrt(0.5**2 + 0.5**2)  # cos(angle)
    assert abs(embeddings.calculate_similarity(v1, v4) - expected) < 1e-6

@pytest.mark.asyncio
async def test_get_answer(mock_openai_client):
    """Test generating an answer using the LLM"""
    with patch('embeddings.openai_client', mock_openai_client):
        # Test successful answer generation
        query = "What is Pydantic?"
        context = "Pydantic is a data validation library for Python."
        answer = await embeddings.get_answer(query, context)
        assert answer == "Mocked answer"
        
        # Check that the right parameters were passed to the API
        mock_openai_client.chat.completions.create.assert_called_once()
        call_args = mock_openai_client.chat.completions.create.call_args[1]
        assert call_args['model'] is not None
        assert len(call_args['messages']) == 2
        assert call_args['temperature'] == 0.5
        
        # Test error handling
        mock_openai_client.chat.completions.create.reset_mock()
        mock_openai_client.chat.completions.create.side_effect = Exception("API error")
        error_answer = await embeddings.get_answer(query, context)
        assert "Error generating answer" in error_answer