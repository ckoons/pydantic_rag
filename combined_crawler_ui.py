import os
import asyncio
import sqlite3
import requests
import json
import numpy as np
import streamlit as st
from bs4 import BeautifulSoup
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup database
import os
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pydantic_docs_simple.db")

def setup_database():
    """Create a simple SQLite database for storing documentation"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create docs table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS docs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        url TEXT NOT NULL,
        title TEXT,
        content TEXT NOT NULL,
        embedding BLOB
    )
    ''')
    
    conn.commit()
    conn.close()
    print(f"Database setup complete at {DB_PATH}")
    return f"Database setup completed at {DB_PATH}"

# Initialize OpenAI client
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def get_embedding(text):
    """Get embedding from OpenAI"""
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text[:8000]  # Limit text length to avoid token limits
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None

def extract_links(soup, base_url):
    """Extract links from a BeautifulSoup parsed page"""
    links = []
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        # Handle relative URLs
        if href.startswith('/'):
            from urllib.parse import urlparse
            parsed_base = urlparse(base_url)
            href = f"{parsed_base.scheme}://{parsed_base.netloc}{href}"
        # Only include http/https URLs
        if href.startswith(('http://', 'https://')):
            links.append(href)
    return links

async def process_url(url, depth=1, timeout=30, progress_bar=None, processed_urls=None):
    """Process a single URL: fetch content, extract title, get embedding, store in db"""
    if processed_urls is None:
        processed_urls = set()
    
    # Skip if already processed
    if url in processed_urls:
        return f"Already processed: {url}"
    
    # Add to processed set
    processed_urls.add(url)
    
    status_msg = f"Processing: {url} (depth {depth}, timeout {timeout}s)"
    if progress_bar:
        progress_bar.progress(0.1, text=status_msg)
    else:
        print(status_msg)
    
    # Fetch content
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        html = response.text
        if progress_bar:
            progress_bar.progress(0.3, text="Content fetched")
    except Exception as e:
        error_msg = f"Error fetching {url}: {e}"
        if progress_bar:
            progress_bar.error(error_msg)
        else:
            print(error_msg)
        return error_msg
    
    # Parse with BeautifulSoup
    soup = BeautifulSoup(html, 'html.parser')
    if progress_bar:
        progress_bar.progress(0.4, text="Parsing content")
    
    # Extract title and content
    title = soup.title.string if soup.title else url
    
    # Get main content
    content = soup.get_text(separator="\n\n")
    if progress_bar:
        progress_bar.progress(0.5, text="Getting embedding")
    
    # Generate embedding
    embedding = await get_embedding(content)
    
    if embedding:
        if progress_bar:
            progress_bar.progress(0.8, text="Storing in database")
        
        # Convert embedding to bytes for storage
        embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()
        
        # Store in database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO docs (url, title, content, embedding) VALUES (?, ?, ?, ?)",
            (url, title, content, embedding_bytes)
        )
        conn.commit()
        conn.close()
        success_msg = f"Stored {url} in database"
        
        # Process linked pages if depth > 1
        if depth > 1:
            if progress_bar:
                progress_bar.progress(0.9, text=f"Crawling linked pages (depth {depth-1})")
            
            # Extract links
            links = extract_links(soup, url)
            
            # Process each link with reduced depth
            for link in links[:5]:  # Limit to first 5 links to avoid excessive crawling
                await process_url(link, depth=depth-1, timeout=timeout, 
                                 progress_bar=None, processed_urls=processed_urls)
        
        if progress_bar:
            progress_bar.progress(1.0, text="Complete!")
        else:
            print(success_msg)
        return success_msg
    else:
        error_msg = f"Skipping {url} - no embedding generated"
        if progress_bar:
            progress_bar.error(error_msg)
        else:
            print(error_msg)
        return error_msg

async def search_similar(query, top_k=3):
    """Search for similar documents using vector similarity"""
    # Get query embedding
    query_embedding = await get_embedding(query)
    if not query_embedding:
        return []
    
    # Convert to numpy array
    query_vector = np.array(query_embedding, dtype=np.float32)
    
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get all documents
    cursor.execute("SELECT id, url, title, content, embedding FROM docs")
    results = cursor.fetchall()
    
    # Calculate similarities
    similarities = []
    for doc_id, url, title, content, embedding_bytes in results:
        if embedding_bytes:
            # Convert stored bytes back to numpy array
            doc_vector = np.frombuffer(embedding_bytes, dtype=np.float32)
            
            # Calculate cosine similarity
            similarity = np.dot(query_vector, doc_vector) / (np.linalg.norm(query_vector) * np.linalg.norm(doc_vector))
            similarities.append((similarity, doc_id, url, title, content))
    
    # Sort by similarity (highest first)
    similarities.sort(reverse=True)
    
    # Return top k results
    return similarities[:top_k]

async def get_answer(query, context):
    """Generate an answer using GPT-4 with RAG"""
    try:
        # Determine which model to use based on environment variables
        model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        
        response = await openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an AI assistant specialized in Pydantic AI documentation. Answer questions based on the provided context."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating answer: {e}"

def show_crawled_docs():
    """Display all crawled documents in the database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT url, title FROM docs")
    results = cursor.fetchall()
    conn.close()
    
    if not results:
        st.info("No documents have been crawled yet.")
        return []
    
    docs = []
    for url, title in results:
        docs.append({"url": url, "title": title or url})
    
    return docs

async def main():
    st.set_page_config(page_title="Pydantic AI Documentation Assistant", page_icon="🤖", layout="wide")
    
    # Setup database on first run
    setup_database()
    
    st.title("Pydantic AI Documentation Assistant")
    
    # Create tabs for different functions
    tabs = st.tabs(["Chat", "Crawler"])
    
    # Crawler tab
    with tabs[1]:
        st.header("Web Crawler")
        st.write("Enter a URL to crawl for documentation")
        
        url_input = st.text_input("URL to crawl")
        
        # Add crawling options
        col1, col2 = st.columns(2)
        with col1:
            depth = st.slider("Crawl Depth", 1, 5, 1, 
                             help="How many links deep to follow from the starting URL")
        with col2:
            timeout = st.slider("Timeout (seconds)", 10, 120, 30, 
                               help="Maximum time to wait for a page to load")
        
        if st.button("Crawl URL") and url_input:
            with st.status("Crawling...") as status:
                result = await process_url(url_input, depth=depth, timeout=timeout, progress_bar=status)
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
    
    # Chat tab
    with tabs[0]:
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
                                with st.expander("Content preview"):
                                    st.markdown(content[:500] + "...")
                        
                        # Create context from similar documents
                        context = "\n\n".join([content for _, _, _, _, content in similar_docs])
                        
                        # Generate answer
                        answer = await get_answer(query, context)
                        st.markdown(answer)
                        
                        # Add assistant message to chat
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    else:
                        no_docs_msg = "I don't have enough information to answer that. Try crawling more documentation first."
                        st.markdown(no_docs_msg)
                        st.session_state.messages.append({"role": "assistant", "content": no_docs_msg})

if __name__ == "__main__":
    asyncio.run(main())