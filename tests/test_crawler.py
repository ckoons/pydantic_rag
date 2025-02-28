"""
Tests for the web crawler module.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from bs4 import BeautifulSoup

from crawler import extract_links, process_url

def test_extract_links():
    """Test extracting links from HTML content"""
    html = """
    <html>
        <body>
            <a href="https://example.com/page1">Link 1</a>
            <a href="/relative/path">Relative Link</a>
            <a href="not-a-url">Invalid Link</a>
            <a>No href</a>
        </body>
    </html>
    """
    soup = BeautifulSoup(html, 'html.parser')
    base_url = "https://example.com"
    
    links = extract_links(soup, base_url)
    
    assert len(links) == 2
    assert "https://example.com/page1" in links
    assert "https://example.com/relative/path" in links
    assert "not-a-url" not in links

@pytest.mark.asyncio
async def test_process_url_success(mock_openai_client, mock_requests_get, temp_db_path):
    """Test successful processing of a URL"""
    url = "https://example.com"
    
    # Set up mocks
    with patch('crawler.get_embedding', return_value=AsyncMock(return_value=[0.1] * 1536)), \
         patch('crawler.store_document', MagicMock()), \
         patch('db.DB_PATH', temp_db_path):
        
        # Test with no recursive crawling (depth=1)
        result = await process_url(url, depth=1, timeout=10)
        assert "Stored" in result
        assert url in result
        
        # Verify correct methods were called
        mock_requests_get.assert_called_once()
        
@pytest.mark.asyncio
async def test_process_url_already_processed():
    """Test that URLs are not processed more than once"""
    url = "https://example.com"
    
    # Call with a pre-populated processed_urls set
    processed_urls = {url}
    result = await process_url(url, processed_urls=processed_urls)
    
    assert "Already processed" in result
    assert url in result

@pytest.mark.asyncio
async def test_process_url_request_error(mock_requests_get):
    """Test handling of request errors"""
    url = "https://example-error.com"
    
    # Set up mock to raise an exception
    mock_requests_get.side_effect = Exception("Request failed")
    
    result = await process_url(url)
    
    assert "Error fetching" in result
    assert url in result

@pytest.mark.asyncio
async def test_process_url_embedding_error(mock_requests_get):
    """Test handling of embedding generation errors"""
    url = "https://example.com"
    
    # Set up mocks - embedding returns None (error)
    with patch('crawler.get_embedding', AsyncMock(return_value=None)):
        result = await process_url(url)
        
        assert "Skipping" in result
        assert url in result
        assert "no embedding generated" in result

@pytest.mark.asyncio
async def test_process_url_recursive(mock_openai_client, mock_requests_get, temp_db_path):
    """Test recursive crawling with depth > 1"""
    url = "https://example.com"
    
    # Mock successful embedding and storage
    with patch('crawler.get_embedding', return_value=AsyncMock(return_value=[0.1] * 1536)), \
         patch('crawler.store_document', MagicMock()), \
         patch('db.DB_PATH', temp_db_path), \
         patch('crawler.process_url', side_effect=AsyncMock(return_value="Recursive URL processed")) as mock_process:
        
        # Force the patched process_url to only be called for recursive calls
        original_process = process_url
        async def side_effect(url, **kwargs):
            if kwargs.get('depth', 1) < 2:  # For the initial call
                return await original_process(url, **kwargs)
            return "Recursive URL processed"
            
        mock_process.side_effect = side_effect
        
        # Test with recursive crawling (depth=2)
        result = await process_url(url, depth=2, timeout=10)
        
        assert "Stored" in result
        assert url in result