# Pydantic AI Documentation Assistant

A simplified RAG (Retrieval-Augmented Generation) system for crawling, storing, and querying Pydantic AI documentation.

## Features

- Web crawler for fetching documentation
- Vector search for finding relevant information 
- Chat interface for asking questions
- Simple SQLite database storage

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your API keys (or copy from example):
```bash
cp .env.example .env
# Then edit .env to add your actual OpenAI API key
```

## Usage

Run the Streamlit application:
```bash
streamlit run app.py
```

The application has two main tabs:
1. **Chat** - Ask questions about Pydantic AI documentation
2. **Crawler** - Crawl websites to build the knowledge base

### Crawling Websites

1. Navigate to the "Crawler" tab
2. Enter a URL to crawl (e.g., "https://ai.pydantic.dev/")
3. Adjust crawl depth and timeout settings
4. Click "Crawl URL"

### Asking Questions

1. Navigate to the "Chat" tab
2. Type your question in the input box
3. View sources used to answer your question in the expandable section

## Project Structure

- `app.py` - Main application entry point
- `db.py` - Database operations
- `embeddings.py` - OpenAI embeddings and completions
- `crawler.py` - Web crawling functionality
- `search.py` - Vector similarity search
- `ui.py` - Streamlit UI components

## How It Works

1. The crawler fetches web pages and extracts their text content
2. Content is stored in a SQLite database along with vector embeddings
3. When you ask a question, the system:
   - Converts your question to an embedding
   - Finds similar content in the database
   - Uses the similar content as context for generating an answer