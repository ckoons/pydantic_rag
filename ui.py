"""
Streamlit UI module for the Pydantic RAG application.
Handles user interaction and display components.
"""

import streamlit as st
import os
import tempfile
from typing import List, Dict, Optional, BinaryIO, Any

from db import get_all_documents, reset_database
from crawler import process_url
from search import search_similar, prepare_context
from embeddings import get_embedding, get_answer
from documents import process_document_with_chunking, detect_file_type, DocumentProcessingError
from caching import cache_stats
from vector_store import vector_store

def show_crawled_docs() -> List[Dict[str, str]]:
    """Display all crawled documents in the database"""
    docs = get_all_documents()
    
    if not docs:
        st.info("No documents have been crawled yet.")
        return []
    
    return docs

async def process_document_upload(uploaded_file) -> str:
    """
    Process an uploaded document file
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        Result message
    """
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp_file:
            # Write uploaded content to temp file
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            # Detect file type
            doc_type = detect_file_type(tmp_path)
            
            if doc_type == 'unknown':
                return f"Error: Unsupported file type for {uploaded_file.name}"
            
            # Process document with chunking
            chunks = process_document_with_chunking(
                tmp_path, 
                document_type=doc_type,
                chunk_size=1000, 
                chunk_overlap=200
            )
            
            if not chunks:
                return f"Error: Could not extract any content from {uploaded_file.name}"
            
            # Get embeddings and store chunks
            success_count = 0
            error_count = 0
            
            for i, chunk in enumerate(chunks):
                # Generate a unique URL-like identifier for each chunk
                chunk_id = f"file://{uploaded_file.name}#chunk{i+1}"
                chunk_title = f"{uploaded_file.name} (Chunk {i+1}/{len(chunks)})"
                
                # Get embedding
                embedding = await get_embedding(chunk)
                
                if embedding:
                    # Store in database
                    from db import store_document
                    doc_id = await store_document(chunk_id, chunk_title, chunk, embedding)
                    if doc_id:
                        success_count += 1
                    else:
                        error_count += 1
                else:
                    error_count += 1
            
            # Remove temp file
            os.unlink(tmp_path)
            
            if success_count > 0:
                return f"Successfully processed {uploaded_file.name} - {success_count} chunks stored, {error_count} errors"
            else:
                return f"Failed to process {uploaded_file.name} - no chunks were stored"
                
        except DocumentProcessingError as e:
            os.unlink(tmp_path)
            return f"Error processing document: {str(e)}"
        except Exception as e:
            os.unlink(tmp_path)
            return f"Unexpected error: {str(e)}"
            
    except Exception as e:
        return f"Error handling upload: {str(e)}"

async def show_crawler_ui():
    """Display the crawler UI tab"""
    st.header("Web Crawler")
    
    # Create tabs for different crawling methods
    crawl_tabs = st.tabs(["Web Crawler", "Document Upload", "Stats"])
    
    # Web crawler tab
    with crawl_tabs[0]:
        st.write("Enter a URL to crawl for documentation")
        
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
    
    # Document upload tab
    with crawl_tabs[1]:
        st.write("Upload PDF, DOCX, or text files to extract information")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload documents", 
            type=["pdf", "docx", "txt", "md", "html", "htm"],
            accept_multiple_files=True
        )
        
        # Process button
        if uploaded_files and st.button("Process Documents"):
            for uploaded_file in uploaded_files:
                with st.status(f"Processing {uploaded_file.name}...") as status:
                    result = await process_document_upload(uploaded_file)
                    if "Error" in result:
                        status.update(label=result, state="error")
                    else:
                        status.update(label=result, state="complete")
    
    # Stats tab
    with crawl_tabs[2]:
        st.write("System statistics")
        
        if st.button("Refresh Stats"):
            # Vector store stats
            vector_stats = vector_store.get_stats()
            st.subheader("Vector Database")
            st.write(f"Number of documents: {vector_stats['num_documents']}")
            st.write(f"Number of vectors: {vector_stats['num_vectors']}")
            st.write(f"Vector dimension: {vector_stats['dimension']}")
            
            # Cache stats
            cache_information = cache_stats()
            st.subheader("Cache")
            
            embedding_cache = cache_information["embedding_cache"]
            st.write(f"Embedding cache: {embedding_cache['size']} items, {embedding_cache['disk_size_bytes'] / (1024*1024):.2f} MB")
            
            document_cache = cache_information["document_cache"]
            st.write(f"Document cache: {document_cache['size']} items, {document_cache['disk_size_bytes'] / (1024*1024):.2f} MB")
            
            query_cache = cache_information["query_cache"]
            st.write(f"Query cache: {query_cache['size']} items, {query_cache['disk_size_bytes'] / (1024*1024):.2f} MB")
    
    
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
                    exclude_patterns=exclude_list
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
                            exclude_patterns=None
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
                            exclude_patterns=None
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
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
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
                # Search for similar documents
                similar_docs = await search_similar(query, top_k=3)
                
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
                        answer = await get_answer(query, context)
                        
                    # Clean up any remaining HTML or navigation artifacts in the answer
                    import re
                    # Remove any HTML tags
                    clean_answer = re.sub(r'<[^>]+>', '', answer)
                    # Clean up any remaining "Skip to content" artifacts
                    clean_answer = re.sub(r'(?i)skip\s+to\s+content.*?\n', '', clean_answer)
                    # Clean up excessive spacing
                    clean_answer = re.sub(r'\n{3,}', '\n\n', clean_answer)
                    clean_answer = re.sub(r'\s{2,}', ' ', clean_answer)
                    
                    # Display the answer
                    st.markdown(clean_answer)
                    
                    # Add assistant message to chat
                    st.session_state.messages.append({"role": "assistant", "content": clean_answer})
                else:
                    no_docs_msg = "I don't have enough information to answer that. Try crawling more documentation first."
                    st.markdown(no_docs_msg)
                    st.session_state.messages.append({"role": "assistant", "content": no_docs_msg})