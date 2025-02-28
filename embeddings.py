"""
Embeddings module for the Pydantic RAG application.
Handles generation and comparison of vector embeddings.
"""

import os
import numpy as np
import logging
from typing import List, Tuple, Optional, Dict, Any, Union, Type

from dotenv import load_dotenv

# Import our provider abstraction
from llm_providers import (
    LLMProviderFactory, 
    EmbeddingModel, 
    LLM, 
    EmbeddingProvider,
    LLMProvider
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get the default embedding model from the factory
embedding_model = LLMProviderFactory.get_embedding_model()

async def get_embedding(
    text: str, 
    provider: Optional[str] = None,
    model_name: Optional[str] = None
) -> Optional[List[float]]:
    """
    Get embedding for text using the specified provider
    
    Args:
        text: Text to get embedding for
        provider: Optional provider name (defaults to environment variable)
        model_name: Optional model name (defaults to environment variable)
        
    Returns:
        Embedding vector or None if computation fails
    """
    # Get the embedding model (default or specified)
    model = embedding_model
    if provider or model_name:
        model = LLMProviderFactory.get_embedding_model(provider, model_name)
        
    # Get embedding from the model
    return await model.get_embedding(text)

def calculate_similarity(query_vector: np.ndarray, doc_vector: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors"""
    return np.dot(query_vector, doc_vector) / (np.linalg.norm(query_vector) * np.linalg.norm(doc_vector))

# Get the default LLM from the factory
llm = LLMProviderFactory.get_llm()

# Import output formatting
from output_formats import OutputFormatter, OutputFormat, PydanticOutputParser, AnswerWithSources

async def get_answer(
    query: str, 
    context: str,
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    format_type: OutputFormat = OutputFormat.TEXT
) -> Union[str, Dict[str, Any]]:
    """
    Generate an answer using LLM with RAG
    
    Args:
        query: User query
        context: Context for RAG
        provider: Optional provider name (defaults to environment variable)
        model_name: Optional model name (defaults to environment variable)
        format_type: Output format type
        
    Returns:
        Generated answer in the specified format
    """
    # Get the LLM (default or specified)
    model = llm
    if provider or model_name:
        model = LLMProviderFactory.get_llm(provider, model_name)
    
    # Create system prompt
    system_prompt = """
    You are an AI assistant specialized in documentation about PydanticAI, Anthropic, and other topics in the knowledgebase. 
    Answer questions based ONLY on the provided context. 
    If the context doesn't contain relevant information, say "I don't have information about that in my knowledge base. Try crawling additional documentation pages first."
    Be concise but thorough in your answers.
    """
    
    # Format user prompt
    user_prompt = f"""
    CONTEXT INFORMATION:
    {context}
    
    USER QUESTION:
    {query}
    
    Please provide a helpful answer based on the context above.
    """
    
    try:
        # Generate text response
        logger.info(f"Generating answer using {model.__class__.__name__}")
        response = await model.generate_text(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.5,
            max_tokens=800
        )
        
        # Format the output according to the specified format
        if format_type != OutputFormat.TEXT:
            return OutputFormatter.format_output(response, format_type)
        else:
            return response.get("text", "I couldn't generate an answer.")
            
    except Exception as e:
        error_msg = f"Error generating answer: {str(e)}"
        logger.error(error_msg)
        return error_msg

async def get_structured_answer(
    query: str, 
    context: str,
    sources: List[Dict[str, Any]],
    provider: Optional[str] = None,
    model_name: Optional[str] = None
) -> AnswerWithSources:
    """
    Generate a structured answer with sources using LLM with RAG
    
    Args:
        query: User query
        context: Context for RAG
        sources: Source documents used for context
        provider: Optional provider name (defaults to environment variable)
        model_name: Optional model name (defaults to environment variable)
        
    Returns:
        Structured answer with sources
    """
    try:
        # Get the LLM (default or specified)
        model = llm
        if provider or model_name:
            model = LLMProviderFactory.get_llm(provider, model_name)
        
        # Create system prompt
        system_prompt = """
        You are an AI assistant that provides well-structured answers with sources.
        Answer questions based ONLY on the provided context.
        If the context doesn't contain relevant information, say "I don't have information about that in my knowledge base."
        """
        
        # Format user prompt
        user_prompt = f"""
        CONTEXT INFORMATION:
        {context}
        
        USER QUESTION:
        {query}
        
        Please analyze the context and provide a helpful answer.
        """
        
        # Define the output schema
        output_schema = {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "The answer to the user's question based on the context"
                },
                "sources": {
                    "type": "array",
                    "description": "Sources used to generate the answer",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "url": {"type": "string"},
                            "summary": {"type": "string"}
                        },
                        "required": ["title", "url", "summary"]
                    }
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence level from a scale 0-1"
                }
            },
            "required": ["answer", "sources"]
        }
        
        # Generate structured response
        response = await model.generate_structured_output(
            prompt=user_prompt,
            output_schema=output_schema,
            system_prompt=system_prompt,
            temperature=0.5
        )
        
        # Parse to Pydantic model
        result = await PydanticOutputParser.parse_to_pydantic(response, AnswerWithSources)
        
        # If parsing succeeded, add model information
        if isinstance(result, AnswerWithSources):
            result.model_used = response.get("model", "unknown")
            
        return result
            
    except Exception as e:
        error_msg = f"Error generating structured answer: {str(e)}"
        logger.error(error_msg)
        return AnswerWithSources(
            answer=f"Error: {error_msg}",
            sources=[],
            confidence=0.0
        )