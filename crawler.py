"""
Web crawler module for the Pydantic RAG application.
Handles website crawling, content extraction, and processing.
"""

import requests
from typing import List, Set, Optional, Any
from bs4 import BeautifulSoup
from urllib.parse import urlparse

from embeddings import get_embedding
from db import store_document

def extract_links(soup: BeautifulSoup, base_url: str) -> List[str]:
    """Extract links from a BeautifulSoup parsed page"""
    links = []
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        # Handle relative URLs
        if href.startswith('/'):
            parsed_base = urlparse(base_url)
            href = f"{parsed_base.scheme}://{parsed_base.netloc}{href}"
        # Only include http/https URLs
        if href.startswith(('http://', 'https://')):
            links.append(href)
    return links

async def process_url(
    url: str, 
    depth: int = 1, 
    timeout: int = 30, 
    progress_bar: Optional[Any] = None, 
    processed_urls: Optional[Set[str]] = None
) -> str:
    """Process a single URL: fetch content, extract title, get embedding, store in db"""
    if processed_urls is None:
        processed_urls = set()
    
    # Skip if already processed
    if url in processed_urls:
        return f"Already processed: {url}"
    
    # Add to processed set
    processed_urls.add(url)
    
    status_msg = f"Processing: {url} (depth {depth}, timeout {timeout}s)"
    if progress_bar:
        progress_bar.progress(0.1, text=status_msg)
    else:
        print(status_msg)
    
    # Fetch content
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        html = response.text
        if progress_bar:
            progress_bar.progress(0.3, text="Content fetched")
    except Exception as e:
        error_msg = f"Error fetching {url}: {e}"
        if progress_bar:
            progress_bar.error(error_msg)
        else:
            print(error_msg)
        return error_msg
    
    # Parse with BeautifulSoup
    soup = BeautifulSoup(html, 'html.parser')
    if progress_bar:
        progress_bar.progress(0.4, text="Parsing content")
    
    # Extract title and content
    title = soup.title.string if soup.title else url
    
    # Get main content
    content = soup.get_text(separator="\n\n")
    if progress_bar:
        progress_bar.progress(0.5, text="Getting embedding")
    
    # Generate embedding
    embedding = await get_embedding(content)
    
    if embedding:
        if progress_bar:
            progress_bar.progress(0.8, text="Storing in database")
        
        # Store in database
        store_document(url, title, content, embedding)
        
        success_msg = f"Stored {url} in database"
        
        # Process linked pages if depth > 1
        if depth > 1:
            if progress_bar:
                progress_bar.progress(0.9, text=f"Crawling linked pages (depth {depth-1})")
                progress_bar.write(f"Depth is {depth}, crawling deeper")
            else:
                print(f"Depth is {depth}, crawling deeper")
            
            # Extract links
            links = extract_links(soup, url)
            
            # Process each link with reduced depth
            for link in links[:5]:  # Limit to first 5 links to avoid excessive crawling
                if progress_bar:
                    progress_bar.write(f"Following link: {link} at depth {depth-1}")
                    
                await process_url(
                    link, 
                    depth=depth-1, 
                    timeout=timeout, 
                    progress_bar=None, 
                    processed_urls=processed_urls
                )
        
        if progress_bar:
            progress_bar.progress(1.0, text="Complete!")
        else:
            print(success_msg)
        return success_msg
    else:
        error_msg = f"Skipping {url} - no embedding generated"
        if progress_bar:
            progress_bar.error(error_msg)
        else:
            print(error_msg)
        return error_msg