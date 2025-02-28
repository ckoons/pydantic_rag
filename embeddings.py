"""
Embeddings module for the Pydantic RAG application.
Handles generation and comparison of vector embeddings.
"""

import os
import numpy as np
from typing import List, Tuple, Optional
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def get_embedding(text: str) -> Optional[List[float]]:
    """Get embedding from OpenAI"""
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
        print(f"Error getting embedding: {e}")
        return None

def calculate_similarity(query_vector: np.ndarray, doc_vector: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors"""
    return np.dot(query_vector, doc_vector) / (np.linalg.norm(query_vector) * np.linalg.norm(doc_vector))

async def get_answer(query: str, context: str) -> str:
    """Generate an answer using GPT-4 with RAG"""
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