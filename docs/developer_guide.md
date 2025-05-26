# Developer Guide

This guide is for developers who want to contribute to the Crawl4AI RAG MCP Server or build upon it.

## Development Setup

### Prerequisites

- Python 3.12+
- Git
- Docker (optional, for testing)
- VS Code or your preferred IDE

### Setting Up the Development Environment

1. **Fork and clone the repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/mcp-crawl4ai-rag.git
   cd mcp-crawl4ai-rag
   ```

2. **Create a virtual environment**:
   ```bash
   # Using uv (recommended)
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   
   # Or using venv
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install in development mode**:
   ```bash
   # Using uv
   uv pip install -e ".[dev]"
   
   # Or using pip
   pip install -e ".[dev]"
   
   # Run Crawl4AI setup
   crawl4ai-setup
   ```

4. **Set up pre-commit hooks** (optional but recommended):
   ```bash
   pre-commit install
   ```

5. **Configure environment variables**:
   ```bash
   cp .env.template .env
   # Edit .env with your API keys
   ```

## Project Structure

```
mcp-crawl4ai-rag/
├── src/                      # Source code
│   ├── crawl4ai_mcp.py      # Main MCP server
│   ├── auth_mcp.py          # Authentication middleware
│   └── utils.py             # Utility functions
├── tests/                    # Test suite
│   ├── test_crawling.py     # Crawling tests
│   ├── test_search.py       # Search tests
│   └── test_integration.py  # End-to-end tests
├── docs/                     # Documentation
├── scripts/                  # Utility scripts
├── crawled_pages.sql        # Database schema
├── Dockerfile               # Container definition
├── pyproject.toml          # Project metadata
├── .env.template           # Environment template
└── README.md               # Project readme
```

## Code Style and Standards

### Python Style Guide

We follow PEP 8 with some modifications:

- Line length: 120 characters
- Use type hints for all functions
- Docstrings for all public functions
- Black for code formatting
- isort for import sorting

### Docstring Format

```python
def function_name(param1: str, param2: int = 5) -> dict:
    """
    Brief description of what the function does.
    
    Args:
        param1: Description of param1
        param2: Description of param2 with default
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When param1 is empty
        
    Example:
        >>> function_name("test", 10)
        {"result": "success"}
    """
```

### Commit Message Convention

Follow conventional commits:

```
type(scope): subject

body (optional)

footer (optional)
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test changes
- `chore`: Build process or auxiliary tool changes

Examples:
```
feat(crawler): add support for PDF extraction
fix(search): handle empty query strings
docs(api): update smart_crawl_url examples
```

## Architecture and Design Patterns

### Adding New Tools

To add a new MCP tool:

1. **Define the tool function** in `crawl4ai_mcp.py`:
   ```python
   @mcp.tool()
   async def your_new_tool(ctx: Context, param1: str, param2: int = 10) -> str:
       """
       Tool description for MCP clients.
       
       Args:
           ctx: MCP context (injected automatically)
           param1: Description
           param2: Description with default
           
       Returns:
           JSON string with results
       """
       try:
           # Get resources from context
           crawler = ctx.request_context.lifespan_context.crawler
           supabase = ctx.request_context.lifespan_context.supabase_client
           
           # Tool implementation
           result = await your_logic(param1, param2)
           
           return json.dumps({
               "success": True,
               "data": result
           }, indent=2)
       except Exception as e:
           return json.dumps({
               "success": False,
               "error": str(e)
           }, indent=2)
   ```

2. **Add tests** in `tests/test_your_tool.py`

3. **Update documentation** in `docs/api.md`

### Extending the Crawler

To add custom crawling strategies:

1. **Create a new crawler function**:
   ```python
   async def crawl_special_content(
       crawler: AsyncWebCrawler, 
       url: str,
       **kwargs
   ) -> List[Dict[str, Any]]:
       """
       Custom crawler for special content types.
       """
       config = CrawlerRunConfig(
           cache_mode=CacheMode.BYPASS,
           # Your custom configuration
       )
       
       result = await crawler.arun(url=url, config=config)
       
       # Custom processing logic
       processed_content = process_special_content(result)
       
       return [{
           'url': url,
           'markdown': processed_content
       }]
   ```

2. **Integrate with smart_crawl_url**:
   ```python
   # In smart_crawl_url
   if is_special_content(url):
       crawl_results = await crawl_special_content(crawler, url)
   ```

### Custom Embedding Strategies

To implement custom embedding generation:

1. **Create embedding provider interface**:
   ```python
   class EmbeddingProvider(ABC):
       @abstractmethod
       async def create_embeddings(self, texts: List[str]) -> List[List[float]]:
           pass
   ```

2. **Implement provider**:
   ```python
   class LocalEmbeddingProvider(EmbeddingProvider):
       def __init__(self, model_name: str):
           self.model = load_model(model_name)
           
       async def create_embeddings(self, texts: List[str]) -> List[List[float]]:
           # Your implementation
           return self.model.encode(texts)
   ```

3. **Update utils.py** to use provider:
   ```python
   # Configure based on environment
   if os.getenv("USE_LOCAL_EMBEDDINGS"):
       provider = LocalEmbeddingProvider("all-MiniLM-L6-v2")
   else:
       provider = OpenAIEmbeddingProvider()
   ```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_crawling.py

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test
pytest tests/test_crawling.py::test_single_page_crawl
```

### Writing Tests

Example test structure:

```python
import pytest
from unittest.mock import Mock, patch
import json

@pytest.mark.asyncio
async def test_crawl_single_page():
    """Test single page crawling functionality."""
    # Arrange
    mock_ctx = Mock()
    mock_crawler = Mock()
    mock_ctx.request_context.lifespan_context.crawler = mock_crawler
    
    # Mock crawler response
    mock_result = Mock()
    mock_result.success = True
    mock_result.markdown = "# Test Content"
    mock_result.links = {"internal": [], "external": []}
    
    with patch.object(mock_crawler, 'arun', return_value=mock_result):
        # Act
        result = await crawl_single_page(mock_ctx, "https://example.com")
        
    # Assert
    result_data = json.loads(result)
    assert result_data["success"] is True
    assert result_data["chunks_stored"] > 0
```

### Integration Tests

For integration tests with real services:

```python
@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="No API key")
async def test_real_embedding_generation():
    """Test real embedding generation with OpenAI."""
    # Test with actual API
    embeddings = create_embeddings_batch(["test text"])
    assert len(embeddings) == 1
    assert len(embeddings[0]) == 1536
```

## Debugging

### Local Debugging

1. **Enable verbose logging**:
   ```python
   # In your code
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **Use VS Code debugger**:
   ```json
   // .vscode/launch.json
   {
       "version": "0.2.0",
       "configurations": [
           {
               "name": "Debug MCP Server",
               "type": "python",
               "request": "launch",
               "program": "${workspaceFolder}/src/crawl4ai_mcp.py",
               "env": {
                   "TRANSPORT": "stdio",
                   "OPENAI_API_KEY": "${env:OPENAI_API_KEY}"
               }
           }
       ]
   }
   ```

3. **Interactive debugging**:
   ```python
   # Add breakpoints
   import pdb; pdb.set_trace()
   
   # Or use IPython
   from IPython import embed; embed()
   ```

### Testing with MCP Inspector

```bash
# Install MCP Inspector
npm install -g @modelcontextprotocol/inspector

# Run server with inspector
mcp-inspector python src/crawl4ai_mcp.py
```

## Performance Optimization

### Profiling

```python
# Add profiling decorator
import cProfile
import pstats
from functools import wraps

def profile(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()
        
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        stats.print_stats(10)
        
        return result
    return wrapper

# Use on slow functions
@profile
async def slow_function():
    pass
```

### Memory Optimization

```python
# Monitor memory usage
import psutil
import os

def log_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")
```

## Contributing

### Submitting Pull Requests

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**:
   - Write tests for new functionality
   - Update documentation
   - Follow code style guidelines

3. **Test your changes**:
   ```bash
   # Run tests
   pytest
   
   # Check code style
   black src/
   isort src/
   
   # Run type checking (if configured)
   mypy src/
   ```

4. **Commit with conventional commits**:
   ```bash
   git add .
   git commit -m "feat(tool): add new amazing feature"
   ```

5. **Push and create PR**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **PR description template**:
   ```markdown
   ## Description
   Brief description of changes
   
   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Breaking change
   - [ ] Documentation update
   
   ## Testing
   - [ ] Tests pass locally
   - [ ] Added new tests
   - [ ] Updated documentation
   
   ## Screenshots (if applicable)
   ```

### Code Review Process

1. All PRs require at least one review
2. CI must pass (tests, linting)
3. Documentation must be updated
4. Breaking changes need migration guide

## Release Process

### Version Numbering

We follow semantic versioning (MAJOR.MINOR.PATCH):

- MAJOR: Breaking changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes

### Release Steps

1. **Update version** in `pyproject.toml`
2. **Update CHANGELOG.md**
3. **Create release commit**:
   ```bash
   git commit -m "chore: release v1.2.3"
   ```
4. **Tag the release**:
   ```bash
   git tag -a v1.2.3 -m "Release version 1.2.3"
   ```
5. **Push to GitHub**:
   ```bash
   git push origin main --tags
   ```

## Advanced Topics

### Building Custom MCP Servers

Use this project as a template:

```python
# your_custom_mcp.py
from auth_mcp import AuthenticatedFastMCP
from contextlib import asynccontextmanager

@asynccontextmanager
async def custom_lifespan(server):
    # Initialize your resources
    yield YourContext()

mcp = AuthenticatedFastMCP(
    "your-custom-mcp",
    description="Your custom MCP server",
    lifespan=custom_lifespan
)

@mcp.tool()
async def your_tool(ctx: Context, param: str) -> str:
    # Your implementation
    pass
```

### Deployment Strategies

See [Deployment Guide](deployment.md) for production deployment options:
- Docker Compose
- Kubernetes
- Coolify
- Cloud platforms

## Resources

### Documentation
- [MCP Specification](https://modelcontextprotocol.io/)
- [Crawl4AI Documentation](https://crawl4ai.com/)
- [Supabase Documentation](https://supabase.com/docs)
- [pgvector Documentation](https://github.com/pgvector/pgvector)

### Community
- [GitHub Discussions](https://github.com/coleam00/mcp-crawl4ai-rag/discussions)
- [Discord Server](#) (if applicable)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/mcp)

### Tools
- [MCP Inspector](https://github.com/modelcontextprotocol/inspector)
- [Supabase CLI](https://supabase.com/docs/guides/cli)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)

## Getting Help

If you're stuck:

1. Check existing [issues](https://github.com/coleam00/mcp-crawl4ai-rag/issues)
2. Search [discussions](https://github.com/coleam00/mcp-crawl4ai-rag/discussions)
3. Ask in the community channels
4. Create a detailed issue with reproduction steps