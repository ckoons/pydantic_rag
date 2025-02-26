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
pip install streamlit openai python-dotenv beautifulsoup4 requests numpy
```

3. Create a `.env` file with your API keys:
```
OPENAI_API_KEY=your-openai-key
LLM_MODEL=gpt-4o-mini  # or another OpenAI model
```

## Usage

Run the Streamlit application:
```bash
streamlit run combined_crawler_ui.py
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

## How It Works

1. The crawler fetches web pages and extracts their text content
2. Content is stored in a SQLite database along with vector embeddings
3. When you ask a question, the system:
   - Converts your question to an embedding
   - Finds similar content in the database
   - Uses the similar content as context for generating an answer