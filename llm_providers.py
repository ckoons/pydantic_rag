"""
LLM Provider module for the Pydantic RAG application.
Provides a unified interface for different LLM and embedding providers.
"""

import os
import json
import asyncio
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple, Callable
import importlib
from enum import Enum
import logging

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Provider Enums
class EmbeddingProvider(str, Enum):
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    SENTENCE_TRANSFORMERS = "sentence_transformers"

class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    LLAMAINDEX = "llamaindex"

# Abstract base classes
class EmbeddingModel(ABC):
    """Abstract base class for embedding models"""
    
    @abstractmethod
    async def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding for text"""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        pass

class LLM(ABC):
    """Abstract base class for large language models"""
    
    @abstractmethod
    async def generate_text(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        stop_sequences: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Generate text from prompt"""
        pass
    
    @abstractmethod
    async def generate_structured_output(
        self,
        prompt: str,
        output_schema: Dict[str, Any],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """Generate structured output from prompt with a given schema"""
        pass

# OpenAI Implementation
class OpenAIEmbedding(EmbeddingModel):
    """OpenAI embedding model"""
    
    def __init__(self, model_name: str = "text-embedding-3-small"):
        """Initialize OpenAI embedding model"""
        try:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.model_name = model_name
            
            # Dimension map for different models
            self.dimension_map = {
                "text-embedding-3-small": 1536,
                "text-embedding-3-large": 3072,
                "text-embedding-ada-002": 1536
            }
            
            if self.model_name not in self.dimension_map:
                logger.warning(f"Unknown model: {model_name}, assuming 1536 dimensions")
                self._dimension = 1536
            else:
                self._dimension = self.dimension_map[self.model_name]
                
        except ImportError:
            raise ImportError("OpenAI package not installed. Please install it with 'pip install openai'.")
    
    async def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding from OpenAI"""
        # Clean and prepare text for embeddings
        if not text or len(text.strip()) < 20:
            logger.warning(f"Text too short: {len(text.strip() if text else 0)} chars")
            return None
        
        try:
            # Trim text to avoid token limits
            trimmed_text = text[:32000] if len(text) > 32000 else text
            
            response = await self.client.embeddings.create(
                model=self.model_name,
                input=trimmed_text
            )
            
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting OpenAI embedding: {e}")
            return None
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self._dimension

class OpenAILLM(LLM):
    """OpenAI language model"""
    
    def __init__(self, model_name: str = "gpt-4o-mini"):
        """Initialize OpenAI language model"""
        try:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.model_name = model_name
        except ImportError:
            raise ImportError("OpenAI package not installed. Please install it with 'pip install openai'.")
    
    async def generate_text(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        stop_sequences: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Generate text from OpenAI"""
        try:
            messages = []
            
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt.strip()})
            
            messages.append({"role": "user", "content": prompt.strip()})
            
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop_sequences
            )
            
            return {
                "text": response.choices[0].message.content,
                "model": self.model_name,
                "provider": "openai",
                "finish_reason": response.choices[0].finish_reason,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
        except Exception as e:
            logger.error(f"Error generating text with OpenAI: {e}")
            return {"error": str(e), "text": "Error generating response"}
    
    async def generate_structured_output(
        self,
        prompt: str,
        output_schema: Dict[str, Any],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """Generate structured output from OpenAI"""
        try:
            # Build system prompt that includes structured output formatting
            full_system_prompt = system_prompt or "You are a helpful assistant."
            full_system_prompt += "\n\nYou will respond with JSON matching this schema:\n"
            full_system_prompt += json.dumps(output_schema, indent=2)
            
            messages = [
                {"role": "system", "content": full_system_prompt.strip()},
                {"role": "user", "content": prompt.strip()}
            ]
            
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"}
            )
            
            # Parse the JSON response
            try:
                content = response.choices[0].message.content
                structured_output = json.loads(content)
                
                return {
                    "data": structured_output,
                    "model": self.model_name,
                    "provider": "openai",
                    "finish_reason": response.choices[0].finish_reason,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    }
                }
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON from OpenAI: {e}\nContent: {content}")
                return {
                    "error": f"Invalid JSON: {str(e)}",
                    "raw_text": content,
                    "provider": "openai"
                }
                
        except Exception as e:
            logger.error(f"Error generating structured output with OpenAI: {e}")
            return {"error": str(e), "provider": "openai"}

# Anthropic Implementation
class AnthropicLLM(LLM):
    """Anthropic language model"""
    
    def __init__(self, model_name: str = "claude-3-sonnet-20240229"):
        """Initialize Anthropic language model"""
        try:
            import anthropic
            self.client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            self.model_name = model_name
        except ImportError:
            raise ImportError("Anthropic package not installed. Please install it with 'pip install anthropic'.")
    
    async def generate_text(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        stop_sequences: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Generate text from Anthropic"""
        try:
            system = system_prompt.strip() if system_prompt else None
            
            response = await self.client.messages.create(
                model=self.model_name,
                system=system,
                messages=[{"role": "user", "content": prompt.strip()}],
                temperature=temperature,
                max_tokens=max_tokens,
                stop_sequences=stop_sequences
            )
            
            return {
                "text": response.content[0].text,
                "model": self.model_name,
                "provider": "anthropic",
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                }
            }
        except Exception as e:
            logger.error(f"Error generating text with Anthropic: {e}")
            return {"error": str(e), "text": "Error generating response"}
    
    async def generate_structured_output(
        self,
        prompt: str,
        output_schema: Dict[str, Any],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """Generate structured output from Anthropic"""
        try:
            # Build system prompt that includes structured output formatting
            full_system_prompt = system_prompt or "You are a helpful assistant."
            full_system_prompt += "\n\nYou will respond with JSON matching this schema:\n"
            full_system_prompt += json.dumps(output_schema, indent=2)
            full_system_prompt += "\n\nEnsure your entire response is valid JSON that strictly adheres to this schema."
            
            response = await self.client.messages.create(
                model=self.model_name,
                system=full_system_prompt.strip(),
                messages=[{"role": "user", "content": prompt.strip()}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Parse the JSON response
            try:
                content = response.content[0].text
                
                # Remove any markdown code block formatting if present
                if content.startswith("```json"):
                    content = content.replace("```json", "", 1)
                    if content.endswith("```"):
                        content = content[:-3]
                elif content.startswith("```"):
                    content = content.replace("```", "", 1)
                    if content.endswith("```"):
                        content = content[:-3]
                
                content = content.strip()
                structured_output = json.loads(content)
                
                return {
                    "data": structured_output,
                    "model": self.model_name,
                    "provider": "anthropic",
                    "usage": {
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens
                    }
                }
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON from Anthropic: {e}\nContent: {content}")
                return {
                    "error": f"Invalid JSON: {str(e)}",
                    "raw_text": content,
                    "provider": "anthropic"
                }
                
        except Exception as e:
            logger.error(f"Error generating structured output with Anthropic: {e}")
            return {"error": str(e), "provider": "anthropic"}

# Hugging Face / Sentence Transformers Implementation
class HuggingFaceEmbedding(EmbeddingModel):
    """HuggingFace embedding model using sentence-transformers"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize HuggingFace embedding model"""
        try:
            from sentence_transformers import SentenceTransformer
            import torch
            
            self.model_name = model_name
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading Hugging Face model {model_name} on {self.device}")
            
            self.model = SentenceTransformer(model_name, device=self.device)
            self._dimension = self.model.get_sentence_embedding_dimension()
            
        except ImportError:
            raise ImportError("sentence-transformers package not installed. Please install it with 'pip install sentence-transformers transformers'.")
    
    async def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding from HuggingFace"""
        if not text or len(text.strip()) < 20:
            logger.warning(f"Text too short: {len(text.strip() if text else 0)} chars")
            return None
        
        try:
            # Execute in a thread to avoid blocking
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None, 
                lambda: self.model.encode(text, convert_to_numpy=True).tolist()
            )
            
            return embedding
        except Exception as e:
            logger.error(f"Error getting HuggingFace embedding: {e}")
            return None
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self._dimension

# LLM Provider Factory
class LLMProviderFactory:
    """Factory for creating LLM providers"""
    
    @staticmethod
    def get_embedding_model(provider: str = None, model_name: str = None) -> EmbeddingModel:
        """Get embedding model"""
        # Use environment variables if not specified
        if not provider:
            provider = os.getenv("EMBEDDING_PROVIDER", EmbeddingProvider.OPENAI.value)
        
        if provider == EmbeddingProvider.OPENAI.value:
            model = model_name or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
            return OpenAIEmbedding(model)
        elif provider == EmbeddingProvider.HUGGINGFACE.value or provider == EmbeddingProvider.SENTENCE_TRANSFORMERS.value:
            model = model_name or os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
            return HuggingFaceEmbedding(model)
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")
    
    @staticmethod
    def get_llm(provider: str = None, model_name: str = None) -> LLM:
        """Get LLM"""
        # Use environment variables if not specified
        if not provider:
            provider = os.getenv("LLM_PROVIDER", LLMProvider.OPENAI.value)
        
        if provider == LLMProvider.OPENAI.value:
            model = model_name or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            return OpenAILLM(model)
        elif provider == LLMProvider.ANTHROPIC.value:
            model = model_name or os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet-20240229")
            return AnthropicLLM(model)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

# Default instances
default_embedding_model = LLMProviderFactory.get_embedding_model()
default_llm = LLMProviderFactory.get_llm()