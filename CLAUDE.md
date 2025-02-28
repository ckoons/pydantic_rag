# Pydantic RAG Project Guidelines

## Commands
- Run application: `streamlit run app.py`
- Install dependencies: `pip install -r requirements.txt`
- Create virtual environment: `python -m venv venv && source venv/bin/activate`
- Run tests: `pytest tests/`
- Run specific test: `pytest tests/test_file.py::test_function`
- Run tests with verbosity: `pytest -xvs tests/`

## Code Style Guidelines
- **Imports**: Group standard library imports first, then third-party imports, then local imports
- **Typing**: Use type hints for function parameters and return values
- **Naming**: 
  - Functions/variables: snake_case
  - Classes: PascalCase
  - Constants: UPPER_SNAKE_CASE
- **Error handling**: Use try/except blocks with specific exceptions
- **Documentation**: Include docstrings for functions and classes using triple quotes
- **Async**: Use asyncio for I/O-bound operations
- **Database**: Use SQLite for storage, with proper connection handling
- **Environment**: Store configuration in .env files, load with python-dotenv
- **File Length**: Keep files under 120 lines of code (comments excluded)
- **Aesthetics**: Use consistent spacing and formatting for readability
- **Tests**: Write comprehensive tests for all functionality

## Project Structure
- **app.py** - Main application entry point
- **db.py** - Database operations and storage
- **embeddings.py** - OpenAI embedding and LLM operations
- **crawler.py** - Web crawling and content extraction
- **search.py** - Vector similarity search
- **ui.py** - Streamlit UI components
- **tests/** - Test files (pytest)

## Project Status
- **Completed**:
  - Refactored monolithic application into modular components
  - Added comprehensive test suite with pytest
  - Implemented URL filtering for the crawler (include/exclude patterns)
  - Fixed UI issues with nested expanders
  - Added proper error handling

- **Enhancements Roadmap**:
  1. ✅ Add filtering options to control which links are followed during crawling
  2. Implement user authentication for the Streamlit interface 
  3. Add support for PDF and other document formats
  4. Integrate with Pydantic models for structured data extraction
  5. Implement caching to reduce API calls and improve performance
  6. Create a Docker container for easy deployment
  7. Add support for other embedding models (e.g., local models via LlamaIndex)