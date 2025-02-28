"""
Caching module for the Pydantic RAG application.
Handles caching of embeddings and other expensive operations.
"""

import os
import json
import hashlib
import pickle
import time
from typing import List, Dict, Any, Optional, Union, Callable
import diskcache
import aiofiles
import asyncio

# Cache configuration
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
EMBEDDING_CACHE_DIR = os.path.join(CACHE_DIR, "embeddings")
DOCUMENT_CACHE_DIR = os.path.join(CACHE_DIR, "documents")
QUERY_CACHE_DIR = os.path.join(CACHE_DIR, "queries")

# Initialize the cache
os.makedirs(EMBEDDING_CACHE_DIR, exist_ok=True)
os.makedirs(DOCUMENT_CACHE_DIR, exist_ok=True)
os.makedirs(QUERY_CACHE_DIR, exist_ok=True)

# Create diskcache instances
embedding_cache = diskcache.Cache(EMBEDDING_CACHE_DIR)
document_cache = diskcache.Cache(DOCUMENT_CACHE_DIR)
query_cache = diskcache.Cache(QUERY_CACHE_DIR)

def compute_hash(text: str) -> str:
    """
    Compute a stable hash for a text string.
    
    Args:
        text: Input text to hash
        
    Returns:
        Hash string
    """
    return hashlib.md5(text.encode('utf-8')).hexdigest()

async def get_cached_embedding(text: str, 
                              ttl: int = 86400 * 30,  # Default 30 days
                              embedding_func: Optional[Callable] = None) -> Optional[List[float]]:
    """
    Get an embedding from cache or compute it if not available.
    
    Args:
        text: Text to get embedding for
        ttl: Time-to-live in seconds
        embedding_func: Function to compute embedding if not in cache
        
    Returns:
        Embedding vector or None if computation fails
    """
    # Normalize and hash the text for cache key
    normalized_text = text.strip()
    text_hash = compute_hash(normalized_text)
    
    # Check if the embedding is in the cache
    embedding = embedding_cache.get(text_hash)
    
    if embedding is not None:
        # Update access time
        embedding_cache.touch(text_hash, expire=ttl)
        return embedding
    
    # Not in cache, compute if a function is provided
    if embedding_func:
        try:
            embedding = await embedding_func(normalized_text)
            if embedding:
                # Store in cache with TTL
                embedding_cache.set(text_hash, embedding, expire=ttl)
                return embedding
        except Exception as e:
            print(f"Error computing embedding: {e}")
    
    return None

def clear_embedding_cache() -> int:
    """
    Clear the embedding cache.
    
    Returns:
        Number of items cleared
    """
    count = len(embedding_cache)
    embedding_cache.clear()
    return count

def cache_stats() -> Dict[str, Any]:
    """
    Get statistics about the cache.
    
    Returns:
        Dictionary with cache statistics
    """
    return {
        "embedding_cache": {
            "size": len(embedding_cache),
            "directory": EMBEDDING_CACHE_DIR,
            "disk_size_bytes": sum(os.path.getsize(os.path.join(EMBEDDING_CACHE_DIR, f)) for f in os.listdir(EMBEDDING_CACHE_DIR) if os.path.isfile(os.path.join(EMBEDDING_CACHE_DIR, f)))
        },
        "document_cache": {
            "size": len(document_cache),
            "directory": DOCUMENT_CACHE_DIR,
            "disk_size_bytes": sum(os.path.getsize(os.path.join(DOCUMENT_CACHE_DIR, f)) for f in os.listdir(DOCUMENT_CACHE_DIR) if os.path.isfile(os.path.join(DOCUMENT_CACHE_DIR, f)))
        },
        "query_cache": {
            "size": len(query_cache),
            "directory": QUERY_CACHE_DIR,
            "disk_size_bytes": sum(os.path.getsize(os.path.join(QUERY_CACHE_DIR, f)) for f in os.listdir(QUERY_CACHE_DIR) if os.path.isfile(os.path.join(QUERY_CACHE_DIR, f)))
        }
    }

async def cache_document_chunks(document_id: str, chunks: List[str], ttl: int = 86400 * 7) -> bool:
    """
    Cache document chunks.
    
    Args:
        document_id: Unique ID for the document
        chunks: List of text chunks
        ttl: Time-to-live in seconds (default 7 days)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        document_cache.set(document_id, chunks, expire=ttl)
        return True
    except Exception as e:
        print(f"Error caching document chunks: {e}")
        return False

async def get_cached_document_chunks(document_id: str) -> Optional[List[str]]:
    """
    Get document chunks from cache.
    
    Args:
        document_id: Unique ID for the document
        
    Returns:
        List of text chunks or None if not in cache
    """
    return document_cache.get(document_id)

async def cache_query_result(query: str, result: Any, ttl: int = 3600) -> bool:
    """
    Cache a query result.
    
    Args:
        query: The query string
        result: The result to cache
        ttl: Time-to-live in seconds (default 1 hour)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        query_hash = compute_hash(query.strip())
        query_cache.set(query_hash, result, expire=ttl)
        return True
    except Exception as e:
        print(f"Error caching query result: {e}")
        return False

async def get_cached_query_result(query: str) -> Optional[Any]:
    """
    Get a query result from cache.
    
    Args:
        query: The query string
        
    Returns:
        Cached result or None if not in cache
    """
    query_hash = compute_hash(query.strip())
    return query_cache.get(query_hash)