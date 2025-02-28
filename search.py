"""
Search module for the Pydantic RAG application.
Handles vector similarity search and context preparation.
"""

import numpy as np
import uuid
import logging
from typing import List, Tuple, Dict, Any, Optional, Union

from embeddings import get_embedding, calculate_similarity
from db import get_documents_with_embeddings
from output_formats import SearchResult, AnswerWithSources
from caching import get_cached_query_result, cache_query_result

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flag to determine if we should use vector DB or SQLite search
USE_VECTOR_DB = False  # Default to SQLite for now

# Conditionally import vector_store if available
try:
    from vector_store import vector_store
    USE_VECTOR_DB = True
    logger.info("Vector database is available and will be used for search")
except ImportError:
    logger.warning("Vector database is not available, falling back to SQLite search")
    USE_VECTOR_DB = False

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
                logger.error(f"Error calculating similarity for doc {doc_id}: {e}")
    
    # Sort by similarity (highest first)
    similarities.sort(reverse=True)
    
    # Return top k results
    return similarities[:top_k]

async def search_similar_vectordb(query: str, top_k: int = 3) -> List[Tuple]:
    """Search for similar documents using FAISS vector database"""
    if not USE_VECTOR_DB:
        logger.warning("Vector database not available, falling back to SQLite search")
        return await search_similar_sqlite(query, top_k)
    
    # Get query embedding
    query_embedding = await get_embedding(query)
    if not query_embedding:
        return []
    
    # Search in vector store
    try:
        # Get results from vector store
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
    except Exception as e:
        logger.error(f"Error searching vector database: {e}")
        logger.info("Falling back to SQLite search")
        return await search_similar_sqlite(query, top_k)

async def search_similar(query: str, top_k: int = 3) -> List[Tuple]:
    """Search for similar documents using either vector DB or SQLite"""
    # Try to get from cache first
    cache_key = f"search:{query}:{top_k}"
    cached_results = await get_cached_query_result(cache_key)
    if cached_results:
        logger.info("Using cached search results")
        return cached_results
    
    # Not in cache, perform search
    if USE_VECTOR_DB:
        results = await search_similar_vectordb(query, top_k)
    else:
        results = await search_similar_sqlite(query, top_k)
    
    # Cache the results for 1 hour
    await cache_query_result(cache_key, results, ttl=3600)
    
    return results

async def search_similar_structured(query: str, top_k: int = 3) -> List[SearchResult]:
    """
    Search for similar documents and return structured results
    
    Args:
        query: Query string
        top_k: Number of results to return
        
    Returns:
        List of SearchResult objects
    """
    results = await search_similar(query, top_k)
    
    # Convert to SearchResult objects
    structured_results = []
    
    for similarity, doc_id, url, title, content in results:
        # Create a brief summary (first 200 chars)
        summary = content[:200] + "..." if len(content) > 200 else content
        
        # Create SearchResult
        result = SearchResult(
            title=title or url,
            url=url,
            summary=summary,
            relevance_score=float(similarity),
            metadata={"doc_id": doc_id}
        )
        
        structured_results.append(result)
        
    return structured_results

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

async def prepare_structured_context(query: str, top_k: int = 3) -> Tuple[str, List[SearchResult]]:
    """
    Prepare context from similar documents and return structured results
    
    Args:
        query: Query string
        top_k: Number of results to return
        
    Returns:
        Tuple of (context string, list of SearchResult objects)
    """
    # Search for similar documents
    search_results = await search_similar(query, top_k)
    
    # Prepare context
    context = prepare_context(search_results)
    
    # Convert to structured results
    structured_results = await search_similar_structured(query, top_k)
    
    return context, structured_results

async def get_answer_with_sources(
    query: str, 
    top_k: int = 3,
    provider: Optional[str] = None,
    model_name: Optional[str] = None
) -> AnswerWithSources:
    """
    Get a structured answer with sources
    
    Args:
        query: Query string
        top_k: Number of results to return
        provider: Optional provider name (defaults to environment variable)
        model_name: Optional model name (defaults to environment variable)
        
    Returns:
        AnswerWithSources object
    """
    from embeddings import get_structured_answer
    
    # Get context and structured sources
    context, sources = await prepare_structured_context(query, top_k)
    
    # If no context, return early
    if not context:
        return AnswerWithSources(
            answer="I don't have information about that in my knowledge base. Try crawling additional documentation pages first.",
            sources=[],
            confidence=0.0
        )
    
    # Generate structured answer with provider settings
    answer = await get_structured_answer(
        query, 
        context, 
        sources,
        provider=provider,
        model_name=model_name
    )
    
    return answer