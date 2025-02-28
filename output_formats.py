"""
Output formats module for the Pydantic RAG application.
Handles different output formats for LLM responses.
"""

import json
import re
import logging
import datetime
from enum import Enum
from typing import Dict, Any, List, Optional, Union, Type
import markdown as md
from markdownify import markdownify
from pydantic import BaseModel, create_model, Field, ValidationError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OutputFormat(str, Enum):
    """Enum for output formats"""
    TEXT = "text"
    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"
    PYDANTIC = "pydantic"

class OutputFormatter:
    """
    Class for formatting LLM outputs into different formats.
    Supports plain text, markdown, HTML, and structured formats.
    """
    
    @staticmethod
    def format_output(
        output: Dict[str, Any], 
        format_type: OutputFormat = OutputFormat.TEXT
    ) -> Any:
        """
        Format LLM output
        
        Args:
            output: LLM output dictionary from generate_text
            format_type: Desired output format
            
        Returns:
            Formatted output
        """
        if "error" in output:
            logger.warning(f"Error in LLM output: {output['error']}")
            return {"error": output["error"]}
        
        text_content = output.get("text", "")
        
        if format_type == OutputFormat.TEXT:
            return text_content
            
        elif format_type == OutputFormat.MARKDOWN:
            # For markdown, we assume the text is already markdown or needs minimal conversion
            return text_content
            
        elif format_type == OutputFormat.HTML:
            # Convert markdown to HTML
            try:
                html_content = md.markdown(text_content)
                return html_content
            except Exception as e:
                logger.error(f"Error converting to HTML: {e}")
                return text_content
                
        elif format_type == OutputFormat.JSON:
            # Try to parse as JSON
            try:
                # If it already looks like JSON, parse it
                if text_content.strip().startswith('{') and text_content.strip().endswith('}'):
                    return json.loads(text_content)
                
                # If it contains a code block with JSON, extract and parse it
                match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text_content, re.DOTALL)
                if match:
                    return json.loads(match.group(1))
                
                # Return as is (may not be valid JSON)
                return {"content": text_content, "warning": "Could not parse as JSON"}
                
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON: {e}")
                return {"error": f"Invalid JSON: {str(e)}", "text": text_content}
                
        else:
            logger.warning(f"Unsupported format type: {format_type}")
            return text_content
    
    @staticmethod
    def convert_html_to_markdown(html_content: str) -> str:
        """
        Convert HTML to Markdown
        
        Args:
            html_content: HTML content
            
        Returns:
            Markdown content
        """
        try:
            return markdownify(html_content)
        except Exception as e:
            logger.error(f"Error converting HTML to Markdown: {e}")
            return html_content
    
    @staticmethod
    def create_pydantic_model(schema: Dict[str, Any], model_name: str = "DynamicModel") -> Type[BaseModel]:
        """
        Create a Pydantic model from a schema
        
        Args:
            schema: JSON Schema
            model_name: Name for the model
            
        Returns:
            Pydantic model class
        """
        field_definitions = {}
        
        for field_name, field_info in schema.get("properties", {}).items():
            field_type = field_info.get("type", "string")
            field_description = field_info.get("description", "")
            field_required = field_name in schema.get("required", [])
            
            # Map JSON schema types to Python types
            type_mapping = {
                "string": str,
                "integer": int,
                "number": float,
                "boolean": bool,
                "array": list,
                "object": dict
            }
            
            python_type = type_mapping.get(field_type, str)
            
            # Handle optional fields
            if not field_required:
                python_type = Optional[python_type]
                default_value = field_info.get("default", None)
                field_definitions[field_name] = (python_type, Field(default=default_value, description=field_description))
            else:
                field_definitions[field_name] = (python_type, Field(description=field_description))
        
        # Create the model dynamically
        return create_model(model_name, **field_definitions)

class PydanticOutputParser:
    """
    Class for parsing LLM outputs into Pydantic models
    """
    
    @staticmethod
    async def parse_to_pydantic(
        llm_output: Dict[str, Any],
        model_class: Type[BaseModel],
    ) -> Union[BaseModel, Dict[str, Any]]:
        """
        Parse LLM output to a Pydantic model
        
        Args:
            llm_output: Output from LLM
            model_class: Pydantic model class to parse to
            
        Returns:
            Pydantic model instance or error dictionary
        """
        # Check if output has a validation error
        if "error" in llm_output:
            return {"error": llm_output["error"]}
        
        # For structured output with data
        if "data" in llm_output:
            data = llm_output["data"]
            try:
                return model_class(**data)
            except ValidationError as e:
                logger.error(f"Validation error: {e}")
                return {
                    "error": f"Validation error: {str(e)}",
                    "raw_data": data
                }
        
        # For text output
        text = llm_output.get("text", "")
        
        # Try to parse as JSON
        try:
            # Clean up any markdown code block formatting
            if text.startswith("```json"):
                text = text.replace("```json", "", 1)
                if text.endswith("```"):
                    text = text[:-3]
            elif text.startswith("```"):
                text = text.replace("```", "", 1)
                if text.endswith("```"):
                    text = text[:-3]
            
            # Strip and parse
            text = text.strip()
            if text.startswith('{') and text.endswith('}'):
                data = json.loads(text)
                return model_class(**data)
            else:
                match = re.search(r'\{.*\}', text, re.DOTALL)
                if match:
                    json_str = match.group(0)
                    data = json.loads(json_str)
                    return model_class(**data)
                else:
                    return {"error": "Could not extract JSON from text output", "text": text}
        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(f"Error parsing to Pydantic model: {e}")
            return {"error": f"Failed to parse output to model: {str(e)}", "text": text}


# Common Pydantic models that can be used throughout the application

class SearchResult(BaseModel):
    """Search result model"""
    title: str
    url: str
    summary: str
    relevance_score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

class AnswerWithSources(BaseModel):
    """Answer with sources model"""
    answer: str
    sources: List[SearchResult]
    confidence: Optional[float] = None
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.now)
    model_used: Optional[str] = None
    
class ChatMessage(BaseModel):
    """Chat message model"""
    role: str
    content: str
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.now)
    
class ChatHistory(BaseModel):
    """Chat history model"""
    messages: List[ChatMessage]
    metadata: Optional[Dict[str, Any]] = None

class KeyPointsExtraction(BaseModel):
    """Key points extraction model"""
    main_topic: str
    key_points: List[str]
    summary: str
    related_topics: Optional[List[str]] = None

class DocumentAnalysis(BaseModel):
    """Document analysis model"""
    title: str
    author: Optional[str] = None
    summary: str
    key_points: List[str]
    sentiment: Optional[str] = None
    topics: List[str]
    recommendations: Optional[List[str]] = None