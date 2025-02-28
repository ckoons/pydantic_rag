"""
Search module for the Pydantic RAG application.
Handles vector similarity search and context preparation.
"""

import numpy as np
import uuid
from typing import List, Tuple, Dict, Any, Optional

from embeddings import get_embedding, calculate_similarity
from db import get_documents_with_embeddings
from vector_store import vector_store
from caching import get_cached_query_result, cache_query_result

# Flag to determine if we should use vector DB or SQLite search
USE_VECTOR_DB = True

async def search_similar_sqlite(query: str, top_k: int = 3) -> List[Tuple]:
    """Search for similar documents using SQLite and in-memory similarity calculation"""
    # Get query embedding
    query_embedding = await get_embedding(query)
    if not query_embedding:
        return []
    
    # Convert to numpy array
    query_vector = np.array(query_embedding, dtype=np.float32)
    
    # Get all documents
    results = get_documents_with_embeddings()
    if not results:
        return []
    
    # Calculate similarities
    similarities = []
    for doc_id, url, title, content, embedding_bytes in results:
        if embedding_bytes:
            try:
                # Convert stored bytes back to numpy array
                doc_vector = np.frombuffer(embedding_bytes, dtype=np.float32)
                
                # Calculate cosine similarity
                similarity = calculate_similarity(query_vector, doc_vector)
                similarities.append((similarity, doc_id, url, title, content))
            except Exception as e:
                print(f"Error calculating similarity for doc {doc_id}: {e}")
    
    # Sort by similarity (highest first)
    similarities.sort(reverse=True)
    
    # Return top k results
    return similarities[:top_k]

async def search_similar_vectordb(query: str, top_k: int = 3) -> List[Tuple]:
    """Search for similar documents using FAISS vector database"""
    # Get query embedding
    query_embedding = await get_embedding(query)
    if not query_embedding:
        return []
    
    # Search in vector store
    results = vector_store.search(query_embedding, top_k=top_k)
    if not results:
        return []
    
    # Convert to the format expected by the application
    similarities = []
    for item in results:
        doc_id = item['id']
        metadata = item['metadata']
        score = item['score']
        
        # Extract document details from metadata
        url = metadata.get('url', '')
        title = metadata.get('title', '')
        content = metadata.get('content', '')
        
        similarities.append((score, doc_id, url, title, content))
    
    return similarities

async def search_similar(query: str, top_k: int = 3) -> List[Tuple]:
    """Search for similar documents using either vector DB or SQLite"""
    # Try to get from cache first
    cache_key = f"search:{query}:{top_k}"
    cached_results = await get_cached_query_result(cache_key)
    if cached_results:
        print("Using cached search results")
        return cached_results
    
    # Not in cache, perform search
    if USE_VECTOR_DB:
        results = await search_similar_vectordb(query, top_k)
    else:
        results = await search_similar_sqlite(query, top_k)
    
    # Cache the results for 1 hour
    await cache_query_result(cache_key, results, ttl=3600)
    
    return results

async def ensure_document_in_vectordb(doc_id: int, url: str, title: str, content: str, embedding: List[float]) -> bool:
    """Ensure a document is stored in the vector database"""
    if not embedding:
        return False
        
    # Prepare metadata
    metadata = {
        'url': url,
        'title': title,
        'content': content,
        'source': 'db'
    }
    
    # Add to vector store
    doc_id_str = str(doc_id)
    return vector_store.add_document(doc_id_str, embedding, metadata)

def prepare_context(similar_docs: List[Tuple]) -> str:
    """Create context from similar documents for RAG"""
    # Clean up each document content
    cleaned_contents = []
    
    for _, _, url, title, content in similar_docs:
        # Include source information
        doc_header = f"Source: {title or url}"
        
        # Clean up the content
        import re
        # Remove any remaining "Skip to content" or navigation artifacts
        cleaned_content = re.sub(r'(?i)skip\s+to\s+content.*?\n', '', content)
        # Remove excessive whitespace
        cleaned_content = re.sub(r'\n{3,}', '\n\n', cleaned_content)
        cleaned_content = re.sub(r'\s{2,}', ' ', cleaned_content)
        # Remove any HTML tags that might have survived
        cleaned_content = re.sub(r'<[^>]+>', '', cleaned_content)
        
        # Add to the context
        cleaned_contents.append(f"{doc_header}\n\n{cleaned_content}")
    
    # Join all documents with clear separation
    return "\n\n---\n\n".join(cleaned_contents)