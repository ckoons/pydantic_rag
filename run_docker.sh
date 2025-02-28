#!/bin/bash

# Create data directory if it doesn't exist
mkdir -p ./data

# Check if .env file exists, create from example if not
if [ ! -f .env ]; then
    echo "Creating .env file from .env.example..."
    cp .env.example .env
    echo "Please edit .env file to add your API keys."
    echo "You can do this by running: nano .env"
    exit 1
fi

# Run docker-compose
echo "Starting Pydantic RAG container..."
docker-compose up --build

# Note: This script will build and run the container
# To stop the container, press Ctrl+C
# To run in the background, use: docker-compose up -d --build