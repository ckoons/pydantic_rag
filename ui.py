"""
Streamlit UI module for the Pydantic RAG application.
Handles user interaction and display components.
"""

import streamlit as st
from typing import List, Dict

from db import get_all_documents
from crawler import process_url
from search import search_similar, prepare_context
from embeddings import get_answer

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
    
    if st.button("Crawl URL") and url_input:
        # Debug info
        st.write(f"Starting crawl with depth: {st.session_state.crawl_depth}, timeout: {st.session_state.crawl_timeout}")
        
        with st.status("Crawling...") as status:
            result = await process_url(url_input, depth=st.session_state.crawl_depth, 
                                      timeout=st.session_state.crawl_timeout, progress_bar=status)
            if "Error" in result:
                st.error(result)
            else:
                st.success(result)
    
    # Show crawled documents
    st.subheader("Crawled Documents")
    docs = show_crawled_docs()
    if docs:
        for doc in docs:
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
                    # Show sources in an expander
                    with st.expander("View sources"):
                        for i, (similarity, doc_id, url, title, content) in enumerate(similar_docs):
                            st.markdown(f"**Source {i+1}**: [{title or url}]({url})")
                            st.markdown(f"Similarity: {similarity:.2f}")
                            st.markdown(f"**Content preview**: {content[:500]}...")
                    
                    # Create context from similar documents
                    context = prepare_context(similar_docs)
                    
                    # Generate answer
                    with st.spinner("Generating answer from context..."):
                        answer = await get_answer(query, context)
                        
                    # Show debug information
                    st.write(f"Response length: {len(answer)} characters")
                    
                    # Display the answer
                    st.markdown(answer)
                    
                    # Add assistant message to chat
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    no_docs_msg = "I don't have enough information to answer that. Try crawling more documentation first."
                    st.markdown(no_docs_msg)
                    st.session_state.messages.append({"role": "assistant", "content": no_docs_msg})