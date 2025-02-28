# Docker Setup for Pydantic RAG

This document explains how to run the Pydantic RAG application using Docker.

## Prerequisites

- Docker installed on your system
- Docker Compose installed on your system
- API keys for the LLM providers you intend to use (OpenAI, Anthropic)

## Getting Started

### 1. Create an environment file

Copy the example environment file and add your API keys:

```bash
cp .env.example .env
```

Edit the `.env` file and add your API keys and configuration:

```
# Required API keys
OPENAI_API_KEY=your-openai-key-here
ANTHROPIC_API_KEY=your-anthropic-key-here

# Optional: Configure which provider and models to use
LLM_PROVIDER=openai
OPENAI_MODEL=gpt-4o-mini
```

### 2. Build and run the container

Use Docker Compose to build and run the application:

```bash
docker-compose up --build
```

This will:
- Build the Docker image for the application
- Start the container
- Mount the `./data` directory for database persistence
- Expose the application on port 8501

### 3. Access the application

Once running, access the application in your browser at:

```
http://localhost:8501
```

## Configuration Options

The application can be configured using environment variables in the `.env` file:

### API Keys
- `OPENAI_API_KEY`: Your OpenAI API key
- `ANTHROPIC_API_KEY`: Your Anthropic API key

### LLM Configuration
- `LLM_PROVIDER`: Which LLM provider to use (`openai` or `anthropic`)
- `OPENAI_MODEL`: Which OpenAI model to use (default: `gpt-4o-mini`)
- `ANTHROPIC_MODEL`: Which Anthropic model to use (default: `claude-3-sonnet-20240229`)

### Embedding Configuration
- `EMBEDDING_PROVIDER`: Which embedding provider to use (`openai` or `sentence_transformers`)
- `OPENAI_EMBEDDING_MODEL`: Which OpenAI embedding model to use (default: `text-embedding-3-small`)
- `HF_EMBEDDING_MODEL`: Which Sentence Transformers model to use (default: `sentence-transformers/all-MiniLM-L6-v2`)

### Database Configuration
- `DATABASE_PATH`: Path to the SQLite database file (default: `/app/data/pydantic_rag.db`)

### Cache Configuration
- `ENABLE_CACHE`: Whether to enable caching (default: `true`)
- `CACHE_TTL`: Cache time-to-live in seconds (default: `3600`)

## Stopping the Application

To stop the application, press `Ctrl+C` in the terminal where Docker Compose is running.

To stop and remove the containers:

```bash
docker-compose down
```

## Persistent Data

The application stores data in a SQLite database located in the `./data` directory. This directory is mounted as a volume in the container, ensuring your crawled pages and embeddings persist between container restarts.

## Resource Usage

The container includes support for sentence-transformers models that can use GPU acceleration if available. If you're running on a system with NVIDIA GPUs, you may want to modify the Docker Compose file to include GPU support.