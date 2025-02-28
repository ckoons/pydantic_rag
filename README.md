# Pydantic AI Documentation Assistant

A RAG (Retrieval-Augmented Generation) system for crawling, storing, and querying documentation with support for multiple LLM providers and output formats.

## Features

- Web crawler for fetching documentation with URL filtering
- Vector search for finding relevant information 
- Chat interface for asking questions
- Support for multiple LLM providers (OpenAI, Anthropic)
- Support for multiple embedding providers (OpenAI, Sentence Transformers)
- Structured output formats (Text, Markdown, HTML, JSON)
- Simple SQLite database storage
- Docker support for easy deployment

## Setup

### Option 1: Local Installation

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
# Then edit .env to add your API keys
```

4. Run the Streamlit application:
```bash
streamlit run app.py
```

### Option 2: Docker (Recommended)

1. Make sure you have Docker and Docker Compose installed

2. Run the container setup script:
```bash
./run_docker.sh
```

3. Or build and run manually:
```bash
# Copy the environment file and edit it
cp .env.example .env

# Build and run with Docker Compose
docker-compose up --build
```

4. Access the application at http://localhost:8501

For more details on Docker deployment, see [DOCKER.md](DOCKER.md).

The application has two main tabs:
1. **Chat** - Ask questions about Pydantic AI documentation
2. **Crawler** - Crawl websites to build the knowledge base

### Crawling Websites

1. Navigate to the "Crawler" tab
2. Enter a URL to crawl (e.g., "https://ai.pydantic.dev/")
3. Adjust crawl depth and timeout settings
4. Optionally configure URL filtering:
   - **Include patterns**: Only URLs matching at least one pattern will be crawled
   - **Exclude patterns**: URLs matching any pattern will be skipped
   - Patterns use Python regular expressions (e.g., `docs/.*` to match all docs)
5. Click "Crawl URL"

#### Example URL Filtering

To only crawl documentation pages and exclude blog posts:

**Include patterns:**
```
/docs/
/reference/
```

**Exclude patterns:**
```
/blog/
/news/
```

### Asking Questions

1. Navigate to the "Chat" tab
2. Type your question in the input box
3. View sources used to answer your question in the expandable section

## Project Structure

- `app.py` - Main application entry point
- `db.py` - Database operations
- `embeddings.py` - Embedding and LLM client handling 
- `llm_providers.py` - Provider abstractions for LLMs and embeddings
- `output_formats.py` - Output format handling and Pydantic models
- `crawler.py` - Web crawling functionality
- `search.py` - Vector similarity search 
- `ui.py` - Streamlit UI components
- `Dockerfile` - Docker container definition
- `docker-compose.yml` - Docker Compose configuration

## How It Works

1. The crawler fetches web pages and extracts their text content
2. Content is stored in a SQLite database along with vector embeddings (from OpenAI or Sentence Transformers)
3. When you ask a question, the system:
   - Converts your question to an embedding
   - Finds similar content in the database
   - Uses the similar content as context for generating an answer
   - Formats the output in your preferred format (Text, Markdown, HTML, JSON)
   - Can use different LLM providers (OpenAI, Anthropic) to generate answers