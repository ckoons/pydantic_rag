"""
Streamlit UI module for the Pydantic RAG application.
Handles user interaction and display components.
"""

import streamlit as st
import os
import logging
from typing import List, Dict, Optional
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from db import get_all_documents, reset_database
from crawler import process_url
from search import search_similar, prepare_context, get_answer_with_sources
from embeddings import get_answer
from llm_providers import LLMProvider, EmbeddingProvider, LLMProviderFactory
from output_formats import OutputFormat, OutputFormatter

def show_crawled_docs() -> List[Dict[str, str]]:
    """Display all crawled documents in the database"""
    docs = get_all_documents()
    
    if not docs:
        st.info("No documents have been crawled yet.")
        return []
    
    return docs

async def show_crawler_ui():
    """Display the crawler UI tab"""
    st.header("Web Crawler")
    st.write("Enter a URL to crawl for documentation")
    
    # Embedding model settings in sidebar
    with st.sidebar:
        st.subheader("Embedding Settings")
        
        # Embedding provider selection
        if "embedding_provider" not in st.session_state:
            st.session_state.embedding_provider = os.getenv("EMBEDDING_PROVIDER", EmbeddingProvider.OPENAI.value)
            
        st.session_state.embedding_provider = st.selectbox(
            "Embedding Provider", 
            options=[
                EmbeddingProvider.OPENAI.value, 
                EmbeddingProvider.SENTENCE_TRANSFORMERS.value
            ],
            index=0 if st.session_state.embedding_provider == EmbeddingProvider.OPENAI.value else 1,
            help="Select which embedding provider to use for document embeddings"
        )
        
        # Model selection based on provider
        if st.session_state.embedding_provider == EmbeddingProvider.OPENAI.value:
            model_options = ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"]
            default_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        else:  # Sentence Transformers
            model_options = [
                "sentence-transformers/all-MiniLM-L6-v2", 
                "sentence-transformers/all-mpnet-base-v2",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            ]
            default_model = os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        
        if "embedding_model" not in st.session_state:
            st.session_state.embedding_model = default_model
            
        # Only update model if provider changes
        if st.session_state.embedding_model not in model_options:
            st.session_state.embedding_model = default_model
            
        st.session_state.embedding_model = st.selectbox(
            "Embedding Model", 
            options=model_options,
            index=model_options.index(st.session_state.embedding_model) if st.session_state.embedding_model in model_options else 0,
            help="Select which embedding model to use"
        )
    
    # Initialize session state for form values
    if "crawl_depth" not in st.session_state:
        st.session_state.crawl_depth = 1
    if "crawl_timeout" not in st.session_state:
        st.session_state.crawl_timeout = 30
        
    url_input = st.text_input("URL to crawl")
    
    # Add crawling options
    col1, col2 = st.columns(2)
    with col1:
        depth = st.slider("Crawl Depth", 1, 10, st.session_state.crawl_depth, 
                         help="How many links deep to follow from the starting URL",
                         key="depth_slider")
        # Update session state when slider changes
        st.session_state.crawl_depth = depth
    with col2:
        timeout = st.slider("Timeout (seconds)", 10, 120, st.session_state.crawl_timeout, 
                           help="Maximum time to wait for a page to load",
                           key="timeout_slider")
        # Update session state when slider changes
        st.session_state.crawl_timeout = timeout
    
    # URL filtering options
    st.subheader("URL Filtering")
    st.write("Optionally filter which URLs are crawled using regex patterns")
    
    # Initialize session state for patterns
    if "include_patterns" not in st.session_state:
        st.session_state.include_patterns = ""
    if "exclude_patterns" not in st.session_state:
        st.session_state.exclude_patterns = ""
    
    col1, col2 = st.columns(2)
    with col1:
        include_patterns = st.text_area("Include patterns (one per line)", 
                                        value=st.session_state.include_patterns,
                                        help="Only crawl URLs that match at least one of these regex patterns",
                                        key="include_patterns_input")
        st.session_state.include_patterns = include_patterns
    with col2:
        exclude_patterns = st.text_area("Exclude patterns (one per line)", 
                                        value=st.session_state.exclude_patterns,
                                        help="Skip URLs that match any of these regex patterns",
                                        key="exclude_patterns_input")
        st.session_state.exclude_patterns = exclude_patterns
    
    # Set up columns for buttons
    col1, col2 = st.columns(2)
    
    # Process include and exclude patterns (needed for both buttons)
    include_list = None
    if st.session_state.include_patterns.strip():
        include_list = [p.strip() for p in st.session_state.include_patterns.split('\n') if p.strip()]
    
    exclude_list = None
    if st.session_state.exclude_patterns.strip():
        exclude_list = [p.strip() for p in st.session_state.exclude_patterns.split('\n') if p.strip()]
    
    # Crawl new URL button
    with col1:
        if st.button("Crawl URL") and url_input:
            if include_list:
                st.write(f"Including URLs matching: {', '.join(include_list)}")
            
            if exclude_list:
                st.write(f"Excluding URLs matching: {', '.join(exclude_list)}")
            
            # Debug info
            st.write(f"Starting crawl with depth: {st.session_state.crawl_depth}, timeout: {st.session_state.crawl_timeout}")
            
            with st.status("Crawling...") as status:
                result = await process_url(
                    url_input, 
                    depth=st.session_state.crawl_depth, 
                    timeout=st.session_state.crawl_timeout, 
                    progress_bar=status,
                    include_patterns=include_list,
                    exclude_patterns=exclude_list,
                    embedding_provider=st.session_state.embedding_provider,
                    embedding_model=st.session_state.embedding_model
                )
                if "Error" in result:
                    st.error(result)
                else:
                    st.success(result)
    
    # Recrawl existing URLs button
    with col2:
        if st.button("Recrawl Existing URLs"):
            # Get all documents from the database
            docs = get_all_documents()
            
            if not docs:
                st.warning("No URLs found in the database to recrawl.")
            else:
                # Filter out test entries
                valid_docs = [doc for doc in docs if not (doc.get('title') and "Test Page" in doc.get('title'))]
                
                # Create a checkbox to select URLs to recrawl
                st.write(f"Found {len(valid_docs)} URLs in the database.")
                
                # Using a container for the recrawl status
                recrawl_container = st.container()
                
                with recrawl_container:
                    with st.status(f"Ready to recrawl {len(valid_docs)} URLs...") as status:
                        status.update(label=f"Select URLs to recrawl from the {len(valid_docs)} URLs found", state="running")
                        
                        # Create a dictionary to store the selection state
                        if "selected_urls" not in st.session_state:
                            st.session_state.selected_urls = {doc['url']: True for doc in valid_docs}
                        
                        # Select all / Deselect all buttons
                        select_col1, select_col2 = st.columns(2)
                        with select_col1:
                            if st.button("Select All"):
                                for url in st.session_state.selected_urls:
                                    st.session_state.selected_urls[url] = True
                        with select_col2:
                            if st.button("Deselect All"):
                                for url in st.session_state.selected_urls:
                                    st.session_state.selected_urls[url] = False
                        
                        # Display checkboxes for each URL
                        for doc in valid_docs:
                            url = doc['url']
                            if url not in st.session_state.selected_urls:
                                st.session_state.selected_urls[url] = True
                            st.session_state.selected_urls[url] = st.checkbox(
                                f"{doc.get('title') or url}", 
                                value=st.session_state.selected_urls[url],
                                key=f"url_{url}"
                            )
                        
                        # Get selected URLs
                        selected_urls = [url for url, selected in st.session_state.selected_urls.items() if selected]
                        
                        if st.button(f"Recrawl {len(selected_urls)} Selected URLs"):
                            if not selected_urls:
                                st.warning("No URLs selected for recrawling.")
                            else:
                                status.update(label=f"Recrawling {len(selected_urls)} URLs...", state="running")
                                
                                success_count = 0
                                error_count = 0
                                
                                for i, url in enumerate(selected_urls):
                                    try:
                                        status.write(f"Processing {i+1}/{len(selected_urls)}: {url}")
                                        
                                        result = await process_url(
                                            url, 
                                            depth=st.session_state.crawl_depth, 
                                            timeout=st.session_state.crawl_timeout, 
                                            progress_bar=None,  # Using the outer status instead
                                            include_patterns=include_list,
                                            exclude_patterns=exclude_list
                                        )
                                        
                                        if "Error" in result:
                                            error_count += 1
                                            status.error(f"Error recrawling {url}: {result}")
                                        else:
                                            success_count += 1
                                    except Exception as e:
                                        error_count += 1
                                        status.error(f"Exception while processing {url}: {str(e)}")
                                
                                status.update(
                                    label=f"Recrawl completed! {success_count} successes, {error_count} errors", 
                                    state="complete", 
                                    expanded=True
                                )
                                
                                if success_count > 0:
                                    st.success(f"Successfully recrawled {success_count} URLs")
                                if error_count > 0:
                                    st.warning(f"Failed to recrawl {error_count} URLs")
                            
    # Database management section
    st.subheader("Database Management")
    
    # Reset database button
    if st.button("Reset Database"):
        if st.session_state.get("confirm_reset") != True:
            st.session_state.confirm_reset = True
            st.warning("⚠️ This will delete ALL crawled documents! Click again to confirm.")
        else:
            with st.status("Resetting database...") as status:
                result = reset_database()
                status.update(label=result, state="complete")
                st.session_state.confirm_reset = False
                st.success("Database has been reset. All crawled documents have been removed.")
                
    # Section for crawling common documentation
    st.subheader("Crawl Common Documentation")
    
    doc_col1, doc_col2 = st.columns(2)
    with doc_col1:
        if st.button("Crawl Pydantic Documentation"):
            st.write("Crawling Pydantic documentation...")
            with st.status("Crawling Pydantic docs...") as status:
                # URLs to crawl for Pydantic
                pydantic_urls = [
                    "https://docs.pydantic.dev/latest/",
                    "https://docs.pydantic.dev/latest/concepts/models/",
                    "https://docs.pydantic.dev/latest/concepts/validators/",
                    "https://docs.pydantic.dev/latest/concepts/fields/",
                ]
                
                success_count = 0
                error_count = 0
                
                for i, url in enumerate(pydantic_urls):
                    try:
                        status.write(f"Processing {i+1}/{len(pydantic_urls)}: {url}")
                        
                        result = await process_url(
                            url, 
                            depth=2,  # Crawl one level deep for comprehensive docs
                            timeout=60,  # Longer timeout for these key pages
                            progress_bar=None,
                            include_patterns=["docs.pydantic.dev"],  # Only crawl pydantic docs
                            exclude_patterns=None,
                            embedding_provider=st.session_state.embedding_provider,
                            embedding_model=st.session_state.embedding_model
                        )
                        
                        if "Error" in result:
                            error_count += 1
                            status.error(f"Error crawling {url}: {result}")
                        else:
                            success_count += 1
                    except Exception as e:
                        error_count += 1
                        status.error(f"Exception processing {url}: {str(e)}")
                
                status.update(
                    label=f"Pydantic docs crawl completed! {success_count} successes, {error_count} errors", 
                    state="complete", 
                    expanded=True
                )
    
    with doc_col2:
        if st.button("Crawl FastAPI Documentation"):
            st.write("Crawling FastAPI documentation...")
            with st.status("Crawling FastAPI docs...") as status:
                # URLs to crawl for FastAPI
                fastapi_urls = [
                    "https://fastapi.tiangolo.com/",
                    "https://fastapi.tiangolo.com/tutorial/",
                    "https://fastapi.tiangolo.com/advanced/",
                ]
                
                success_count = 0
                error_count = 0
                
                for i, url in enumerate(fastapi_urls):
                    try:
                        status.write(f"Processing {i+1}/{len(fastapi_urls)}: {url}")
                        
                        result = await process_url(
                            url, 
                            depth=2,  # Crawl one level deep
                            timeout=60,  # Longer timeout
                            progress_bar=None,
                            include_patterns=["fastapi.tiangolo.com"],  # Only crawl FastAPI docs
                            exclude_patterns=None,
                            embedding_provider=st.session_state.embedding_provider,
                            embedding_model=st.session_state.embedding_model
                        )
                        
                        if "Error" in result:
                            error_count += 1
                            status.error(f"Error crawling {url}: {result}")
                        else:
                            success_count += 1
                    except Exception as e:
                        error_count += 1
                        status.error(f"Exception processing {url}: {str(e)}")
                
                status.update(
                    label=f"FastAPI docs crawl completed! {success_count} successes, {error_count} errors", 
                    state="complete", 
                    expanded=True
                )
    
    # Show crawled documents
    st.subheader("Crawled Documents")
    docs = show_crawled_docs()
    if docs:
        for doc in docs:
            # Skip entries with "Test" in the title (to remove test data)
            if not doc['title'] or "Test Page" in doc['title']:
                continue
            st.write(f"- [{doc['title']}]({doc['url']})")

async def show_chat_ui():
    """Display the chat UI tab"""
    st.header("Chat with Pydantic AI Documentation")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "llm_provider" not in st.session_state:
        st.session_state.llm_provider = os.getenv("LLM_PROVIDER", LLMProvider.OPENAI.value)
    if "output_format" not in st.session_state:
        st.session_state.output_format = OutputFormat.TEXT.value
    if "use_structured_output" not in st.session_state:
        st.session_state.use_structured_output = False
    
    # Settings sidebar
    with st.sidebar:
        st.subheader("Chat Settings")
        
        # LLM Provider selection
        st.session_state.llm_provider = st.selectbox(
            "LLM Provider", 
            options=[LLMProvider.OPENAI.value, LLMProvider.ANTHROPIC.value],
            index=0 if st.session_state.llm_provider == LLMProvider.OPENAI.value else 1,
            help="Select which LLM provider to use for generating answers"
        )
        
        # Model selection based on provider
        if st.session_state.llm_provider == LLMProvider.OPENAI.value:
            model_options = ["gpt-4o-mini", "gpt-4o", "gpt-4"]
            default_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        else:  # Anthropic
            model_options = ["claude-3-sonnet-20240229", "claude-3-opus-20240229", "claude-3-haiku-20240307"]
            default_model = os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet-20240229")
        
        if "model_name" not in st.session_state:
            st.session_state.model_name = default_model
            
        # Only update model if provider changes
        if st.session_state.model_name not in model_options:
            st.session_state.model_name = default_model
            
        st.session_state.model_name = st.selectbox(
            "Model", 
            options=model_options,
            index=model_options.index(st.session_state.model_name) if st.session_state.model_name in model_options else 0,
            help="Select which model to use for the selected provider"
        )
        
        # Output format selection
        output_format_options = [
            OutputFormat.TEXT.value,
            OutputFormat.MARKDOWN.value, 
            OutputFormat.HTML.value
        ]
        
        st.session_state.output_format = st.selectbox(
            "Output Format",
            options=output_format_options,
            index=output_format_options.index(st.session_state.output_format) if st.session_state.output_format in output_format_options else 0,
            help="Select the format for displaying answers"
        )
        
        # Toggle for structured output
        st.session_state.use_structured_output = st.toggle(
            "Use Structured Output", 
            value=st.session_state.use_structured_output,
            help="When enabled, generates structured answers with better source attribution"
        )
        
        # Number of sources to use
        if "top_k" not in st.session_state:
            st.session_state.top_k = 3
            
        st.session_state.top_k = st.slider(
            "Number of Sources", 
            min_value=1, 
            max_value=10, 
            value=st.session_state.top_k,
            help="How many similar documents to use as context"
        )
        
        # Clear chat button
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and "format" in message and message["format"] == OutputFormat.HTML.value:
                st.components.v1.html(message["content"], height=None, scrolling=True)
            else:
                st.write(message["content"])
    
    # Chat input
    query = st.chat_input("Ask a question about Pydantic AI")
    
    if query:
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": query})
        
        # Display user message
        with st.chat_message("user"):
            st.write(query)
        
        # Generate and display response
        with st.chat_message("assistant"):
            with st.spinner("Searching documentation..."):
                if st.session_state.use_structured_output:
                    # Use structured output with sources and pass provider settings
                    result = await get_answer_with_sources(
                        query, 
                        top_k=st.session_state.top_k,
                        provider=st.session_state.llm_provider,
                        model_name=st.session_state.model_name
                    )
                    
                    # Update provider info in the result if available
                    if hasattr(result, "model_used") and not result.model_used:
                        result.model_used = f"{st.session_state.llm_provider}/{st.session_state.model_name}"
                    
                    # Display answer
                    if st.session_state.output_format == OutputFormat.HTML.value:
                        # Convert to HTML if needed
                        html_content = OutputFormatter.format_output(
                            {"text": result.answer}, 
                            OutputFormat.HTML
                        )
                        st.components.v1.html(html_content, height=None, scrolling=True)
                        formatted_answer = html_content
                        output_format = OutputFormat.HTML.value
                    else:
                        st.markdown(result.answer)
                        formatted_answer = result.answer
                        output_format = st.session_state.output_format
                    
                    # Display sources in an expander with more detail
                    with st.expander("View sources"):
                        st.write("These are the sources used to answer your question:")
                        for i, source in enumerate(result.sources):
                            st.markdown(f"**Source {i+1}**: [{source.title}]({source.url})")
                            st.markdown(f"Relevance: {source.relevance_score:.2f}" if source.relevance_score else "")
                            st.markdown(f"**Summary**: {source.summary}")
                            st.markdown(f"[Visit this page]({source.url})")
                    
                    # Add assistant message to chat with format info
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": formatted_answer,
                        "format": output_format,
                        "sources": [s.dict() for s in result.sources] if result.sources else []
                    })
                    
                else:
                    # Traditional RAG approach
                    # Search for similar documents
                    similar_docs = await search_similar(query, top_k=st.session_state.top_k)
                    
                    # Prepare context from similar documents
                    if similar_docs:
                        # Show sources in an expander with more detail
                        with st.expander("View sources"):
                            st.write("These are the sources used to answer your question:")
                            for i, (similarity, doc_id, url, title, content) in enumerate(similar_docs):
                                st.markdown(f"**Source {i+1}**: [{title or url}]({url})")
                                st.markdown(f"Similarity: {similarity:.2f}")
                                
                                # Show a cleaner content preview with more detail
                                import re
                                preview_content = re.sub(r'\s+', ' ', content).strip()
                                preview_content = re.sub(r'<[^>]+>', '', preview_content)
                                
                                if len(preview_content) > 50:
                                    st.markdown(f"**Content preview**: {preview_content[:1000]}...")
                                else:
                                    st.markdown("**Note**: This source appears to have limited textual content.")
                                    
                                # Display a link to directly view the source
                                st.markdown(f"[Visit this page]({url})")
                        
                        # Create context from similar documents
                        context = prepare_context(similar_docs)
                        
                        # Generate answer
                        with st.spinner("Generating answer from context..."):
                            # Use the selected provider and model
                            answer = await get_answer(
                                query, 
                                context, 
                                provider=st.session_state.llm_provider,
                                model_name=st.session_state.model_name,
                                format_type=OutputFormat(st.session_state.output_format)
                            )
                            
                            # Log the provider and model used
                            logger.info(f"Generated answer using {st.session_state.llm_provider}/{st.session_state.model_name}")
                            
                        # Handle different output formats
                        if isinstance(answer, dict):
                            # Handle error
                            if "error" in answer:
                                st.error(f"Error: {answer['error']}")
                                formatted_answer = f"Error: {answer['error']}"
                                output_format = OutputFormat.TEXT.value
                            else:
                                # Display based on output format
                                if st.session_state.output_format == OutputFormat.HTML.value:
                                    st.components.v1.html(answer, height=None, scrolling=True)
                                    formatted_answer = answer
                                    output_format = OutputFormat.HTML.value
                                else:
                                    # Handle dictionary response for other formats
                                    content = answer.get("text", "") if isinstance(answer, dict) else answer
                                    st.markdown(content)
                                    formatted_answer = content
                                    output_format = st.session_state.output_format
                        else:
                            # Handle string response (default text format)
                            # Clean up any remaining HTML or navigation artifacts in the answer
                            import re
                            # Remove any HTML tags if text format
                            if st.session_state.output_format == OutputFormat.TEXT.value:
                                clean_answer = re.sub(r'<[^>]+>', '', answer)
                                # Clean up any remaining "Skip to content" artifacts
                                clean_answer = re.sub(r'(?i)skip\s+to\s+content.*?\n', '', clean_answer)
                                # Clean up excessive spacing
                                clean_answer = re.sub(r'\n{3,}', '\n\n', clean_answer)
                                clean_answer = re.sub(r'\s{2,}', ' ', clean_answer)
                                
                                st.markdown(clean_answer)
                                formatted_answer = clean_answer
                                output_format = OutputFormat.TEXT.value
                            else:
                                st.markdown(answer)
                                formatted_answer = answer
                                output_format = st.session_state.output_format
                        
                        # Add assistant message to chat with format info
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": formatted_answer,
                            "format": output_format
                        })
                    else:
                        no_docs_msg = "I don't have enough information to answer that. Try crawling more documentation first."
                        st.markdown(no_docs_msg)
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": no_docs_msg,
                            "format": OutputFormat.TEXT.value
                        })