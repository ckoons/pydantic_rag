"""
Embeddings module for the Pydantic RAG application.
Handles generation and comparison of vector embeddings.
"""

import os
import numpy as np
from typing import List, Tuple, Optional, Callable, Any
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Import our caching system
from caching import get_cached_embedding, cache_query_result, get_cached_query_result

# Load environment variables
load_dotenv()

# Initialize OpenAI client
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def _generate_embedding(text: str) -> Optional[List[float]]:
    """Generate embedding directly from OpenAI (without caching)"""
    # Clean and prepare text for embeddings
    if not text or len(text.strip()) < 50:
        print(f"Warning: Text is too short ({len(text.strip()) if text else 0} chars)")
        # If text is too short, it's not useful for embeddings
        if len(text.strip()) < 20:
            print("Text too short for embedding, returning None")
            return None
    
    try:
        # Trim to avoid token limits but keep as much content as possible
        # The embedding model can handle ~8K tokens
        trimmed_text = text[:32000] if len(text) > 32000 else text
        
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=trimmed_text
        )
        
        # Check if we got a valid embedding
        embedding = response.data[0].embedding
        if not embedding or len(embedding) < 100:
            print(f"Warning: Received invalid embedding of length {len(embedding) if embedding else 0}")
            return None
            
        return embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

async def get_embedding(text: str) -> Optional[List[float]]:
    """Get embedding from cache or generate a new one"""
    # Try to get from cache first, with fallback to generating new embedding
    return await get_cached_embedding(
        text=text,
        ttl=86400 * 30,  # Cache for 30 days
        embedding_func=_generate_embedding
    )

def calculate_similarity(query_vector: np.ndarray, doc_vector: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors"""
    return np.dot(query_vector, doc_vector) / (np.linalg.norm(query_vector) * np.linalg.norm(doc_vector))

async def _generate_answer(query: str, context: str) -> str:
    """Generate answer directly from LLM (without caching)"""
    try:
        # Determine which model to use based on environment variables
        model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        
        # Create a more detailed system prompt
        system_prompt = """
        You are an AI assistant specialized in documentation about PydanticAI, Anthropic, and other topics in the knowledgebase. 
        Answer questions based ONLY on the provided context. 
        If the context doesn't contain relevant information, say "I don't have information about that in my knowledge base. Try crawling additional documentation pages first."
        Be concise but thorough in your answers.
        """
        
        # Format the user prompt clearly
        user_prompt = f"""
        CONTEXT INFORMATION:
        {context}
        
        USER QUESTION:
        {query}
        
        Please provide a helpful answer based on the context above.
        """
        
        # Make the API call with timeout and retry
        print(f"Sending request to OpenAI using model: {model}")
        response = await openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt.strip()}
            ],
            temperature=0.5,  # Lower temperature for more focused answers
            max_tokens=800    # Ensure we get a reasonably sized response
        )
        
        # Extract and return the response content
        answer = response.choices[0].message.content
        print(f"Received answer: {answer[:100]}...")
        return answer
        
    except Exception as e:
        error_msg = f"Error generating answer: {str(e)}"
        print(error_msg)
        return error_msg

async def get_answer(query: str, context: str) -> str:
    """Generate an answer using LLM with RAG and caching"""
    # Create a cache key that combines query and context hash
    from hashlib import md5
    
    # For caching purposes, we need to ensure the context isn't too large
    # We'll use a hash of the full context but only include a truncated version in the cache key
    context_hash = md5(context.encode()).hexdigest()
    cache_key = f"{query}::{context_hash}"
    
    # Try to get result from cache
    cached_result = await get_cached_query_result(cache_key)
    if cached_result:
        print("Using cached answer")
        return cached_result
    
    # Not in cache, generate a new answer
    answer = await _generate_answer(query, context)
    
    # Cache the result (1 hour TTL for LLM responses)
    await cache_query_result(cache_key, answer, ttl=3600)
    
    return answer