"""
Search module for the Pydantic RAG application.
Handles vector similarity search and context preparation.
"""

import numpy as np
from typing import List, Tuple, Dict, Any

from embeddings import get_embedding, calculate_similarity
from db import get_documents_with_embeddings

async def search_similar(query: str, top_k: int = 3) -> List[Tuple]:
    """Search for similar documents using vector similarity"""
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