"""
Document processing module for the Pydantic RAG application.
Handles document loading, parsing, and chunking for various file types.
"""

import os
import re
import tempfile
from typing import List, Optional, Dict, Any, Tuple, BinaryIO
from urllib.parse import urlparse

# Document loaders
import PyPDF2
from docx import Document as DocxDocument
from bs4 import BeautifulSoup

# Text chunking
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Error handling
class DocumentProcessingError(Exception):
    """Exception raised for errors in the document processing module."""
    pass

def detect_file_type(file_path_or_buffer: Any) -> str:
    """
    Detect the type of file from path, URL, or buffer.
    
    Args:
        file_path_or_buffer: Path string, URL, or file-like object
        
    Returns:
        String indicating file type ('pdf', 'docx', 'html', 'text', or 'unknown')
    """
    # Handle string path or URL
    if isinstance(file_path_or_buffer, str):
        # Check if it's a URL
        if file_path_or_buffer.startswith(('http://', 'https://')):
            parsed_url = urlparse(file_path_or_buffer)
            path = parsed_url.path.lower()
        else:
            path = file_path_or_buffer.lower()
            
        # Check extensions
        if path.endswith('.pdf'):
            return 'pdf'
        elif path.endswith('.docx'):
            return 'docx'
        elif path.endswith(('.html', '.htm')):
            return 'html'
        elif path.endswith(('.txt', '.md', '.rst')):
            return 'text'
        else:
            return 'unknown'
    
    # For file-like objects, try to determine from metadata or content
    # This is a simplified approach - more robust detection would involve checking magic bytes
    try:
        if hasattr(file_path_or_buffer, 'name'):
            return detect_file_type(file_path_or_buffer.name)
    except:
        pass
    
    # Default to unknown if we can't determine
    return 'unknown'

def extract_text_from_pdf(file_path_or_buffer: Any) -> str:
    """
    Extract text from a PDF file.
    
    Args:
        file_path_or_buffer: File path or file-like object for the PDF
        
    Returns:
        Extracted text as a string
    """
    try:
        # Create PDF reader object
        reader = PyPDF2.PdfReader(file_path_or_buffer)
        
        # Extract text from each page
        text_parts = []
        for page in reader.pages:
            text_parts.append(page.extract_text())
        
        # Join all pages with double newlines
        full_text = "\n\n".join(text_parts)
        
        # Clean up the text
        # Remove excessive whitespace
        full_text = re.sub(r'\s+', ' ', full_text)
        # Fix broken sentences (common in PDFs)
        full_text = re.sub(r'(\w)- (\w)', r'\1\2', full_text)
        
        return full_text
    except Exception as e:
        raise DocumentProcessingError(f"Failed to extract text from PDF: {str(e)}")

def extract_text_from_docx(file_path_or_buffer: Any) -> str:
    """
    Extract text from a DOCX file.
    
    Args:
        file_path_or_buffer: File path or file-like object for the DOCX
        
    Returns:
        Extracted text as a string
    """
    try:
        # Create DOCX document object
        doc = DocxDocument(file_path_or_buffer)
        
        # Extract text from paragraphs
        text_parts = []
        for para in doc.paragraphs:
            if para.text.strip():
                text_parts.append(para.text)
        
        # Join all paragraphs with newlines
        full_text = "\n\n".join(text_parts)
        
        return full_text
    except Exception as e:
        raise DocumentProcessingError(f"Failed to extract text from DOCX: {str(e)}")

def extract_text_from_html(html_content: str) -> str:
    """
    Extract text from HTML content.
    
    Args:
        html_content: HTML content as a string
        
    Returns:
        Extracted text as a string
    """
    try:
        # Parse HTML with BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script_or_style in soup(['script', 'style', 'head', 'meta', 'link']):
            script_or_style.decompose()
        
        # Get text
        text = soup.get_text(separator='\n')
        
        # Clean up text
        # Remove blank lines
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        text = '\n'.join(lines)
        
        return text
    except Exception as e:
        raise DocumentProcessingError(f"Failed to extract text from HTML: {str(e)}")

def process_document(file_path_or_buffer: Any, document_type: Optional[str] = None) -> str:
    """
    Process a document and extract its text content.
    
    Args:
        file_path_or_buffer: Path, URL, or file-like object for the document
        document_type: Optional type override ('pdf', 'docx', 'html', 'text')
        
    Returns:
        Extracted text content
    """
    # Determine document type if not provided
    if not document_type:
        document_type = detect_file_type(file_path_or_buffer)
    
    # Extract text based on document type
    if document_type == 'pdf':
        return extract_text_from_pdf(file_path_or_buffer)
    elif document_type == 'docx':
        return extract_text_from_docx(file_path_or_buffer)
    elif document_type == 'html':
        # If a file path is provided, we need to read the file
        if isinstance(file_path_or_buffer, str) and os.path.isfile(file_path_or_buffer):
            with open(file_path_or_buffer, 'r', encoding='utf-8') as f:
                html_content = f.read()
            return extract_text_from_html(html_content)
        # If it's already content, process directly
        elif isinstance(file_path_or_buffer, str):
            return extract_text_from_html(file_path_or_buffer)
        # If file-like object, read and process
        else:
            html_content = file_path_or_buffer.read()
            if isinstance(html_content, bytes):
                html_content = html_content.decode('utf-8')
            return extract_text_from_html(html_content)
    elif document_type == 'text':
        # Handle text files
        if isinstance(file_path_or_buffer, str) and os.path.isfile(file_path_or_buffer):
            with open(file_path_or_buffer, 'r', encoding='utf-8') as f:
                return f.read()
        elif isinstance(file_path_or_buffer, str):
            return file_path_or_buffer
        else:
            content = file_path_or_buffer.read()
            if isinstance(content, bytes):
                content = content.decode('utf-8')
            return content
    else:
        raise DocumentProcessingError(f"Unsupported document type: {document_type}")

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks for processing.
    
    Args:
        text: The text to chunk
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of text chunks
    """
    # Use LangChain's RecursiveCharacterTextSplitter for intelligent chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_text(text)
    return chunks

def process_document_with_chunking(
    file_path_or_buffer: Any, 
    document_type: Optional[str] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[str]:
    """
    Process a document and split it into chunks.
    
    Args:
        file_path_or_buffer: Path, URL, or file-like object for the document
        document_type: Optional type override
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of document chunks
    """
    # Extract text from document
    text = process_document(file_path_or_buffer, document_type)
    
    # Split into chunks
    chunks = chunk_text(text, chunk_size, chunk_overlap)
    
    return chunks