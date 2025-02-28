"""
Main application module for the Pydantic RAG system.
Initializes the application and handles routing between components.
"""

import asyncio
import streamlit as st

from db import setup_database
from ui import show_chat_ui, show_crawler_ui

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
    
    # Create tabs for different functions
    tabs = st.tabs(["Chat", "Crawler"])
    
    # Crawler tab
    with tabs[1]:
        await show_crawler_ui()
    
    # Chat tab
    with tabs[0]:
        await show_chat_ui()

if __name__ == "__main__":
    asyncio.run(main())