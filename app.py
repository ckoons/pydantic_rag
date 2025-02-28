"""
Main application module for the Pydantic RAG system.
Initializes the application and handles routing between components.
"""

import asyncio
import streamlit as st
import os
from dotenv import load_dotenv

from db import setup_database
from ui import show_chat_ui, show_crawler_ui

# Load environment variables
load_dotenv()

async def main():
    """Main application entry point"""
    st.set_page_config(
        page_title="Pydantic AI Documentation Assistant", 
        page_icon="🤖", 
        layout="wide"
    )
    
    # Setup database on first run
    setup_database()
    
    st.title("Pydantic AI Documentation Assistant")
    
    # Add footer information
    with st.sidebar:
        st.markdown("---")
        st.subheader("About")
        st.markdown("""
        This application uses RAG (Retrieval Augmented Generation) to provide
        accurate answers about Pydantic and other documentation.
        
        **Features:**
        - Multiple LLM providers (OpenAI, Anthropic)
        - Multiple embedding models
        - Structured output formats
        - Web crawling with URL filtering
        - Document search with vector similarity
        """)
        
        # Environment info
        st.markdown("---")
        st.markdown("**Current Configuration:**")
        st.markdown(f"- Default LLM: `{os.getenv('LLM_PROVIDER', 'openai')}`")
        st.markdown(f"- Default Embeddings: `{os.getenv('EMBEDDING_PROVIDER', 'openai')}`")
        
        # Version info
        st.markdown("---")
        st.markdown("**Version:** 1.0.0")
    
    # Create tabs for different functions
    tabs = st.tabs(["Chat", "Crawler"])
    
    # Chat tab (default view)
    with tabs[0]:
        await show_chat_ui()
    
    # Crawler tab
    with tabs[1]:
        await show_crawler_ui()

if __name__ == "__main__":
    asyncio.run(main())