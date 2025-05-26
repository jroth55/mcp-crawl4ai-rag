<h1 align="center">Crawl4AI RAG MCP Server</h1>

<p align="center">
  <em>Web Crawling and RAG Capabilities for AI Agents and AI Coding Assistants</em>
</p>

A powerful implementation of the [Model Context Protocol (MCP)](https://modelcontextprotocol.io) integrated with [Crawl4AI](https://crawl4ai.com) and [Supabase](https://supabase.com/) for providing AI agents and AI coding assistants with advanced web crawling and RAG capabilities.

With this MCP server, you can <b>scrape anything</b> and then <b>use that knowledge anywhere</b> for RAG.

This MCP server was originally developed by [coleam00](https://github.com/coleam00) as a knowledge engine for AI coding assistants.

This fork enhances the original with:
- **Intelligent Crawl Boundaries**: Automatic subdirectory restriction with customizable `prefix` parameter to prevent uncontrolled crawling
- **Authentication Support**: `MCP_SERVER_API_KEY` for secure production deployments  
- **Enhanced Claude Code Integration**: Detailed setup instructions for Coolify and Claude Code
- **Improved Safety**: Default behavior now restricts crawling to current subdirectory instead of entire domains

## 🆕 New Features & Improvements

### Intelligent Crawl Boundaries
The latest version introduces **automatic subdirectory restriction** to prevent uncontrolled crawling:

- **Default Behavior**: When you crawl `https://docs.example.com/api/v2/`, it now automatically restricts to the `/api/v2/` subdirectory
- **Custom Boundaries**: Use the `prefix` parameter to set custom crawl boundaries
- **Single Page Mode**: Set `prefix` equal to the URL to index only that specific page
- **Override**: To crawl an entire domain, explicitly set the prefix to the root domain

This prevents accidentally crawling thousands of pages when you only wanted documentation from a specific section!

### Enhanced Reliability & Performance
- **Sitemap Support**: Now handles gzipped sitemaps (`.xml.gz`) and sitemap index files that reference multiple sitemaps
- **Timeout Protection**: All crawl operations now have configurable timeouts (30s default) to prevent hanging on slow sites
- **Smart Chunking**: Improved code block handling ensures markdown code blocks are never split mid-block
- **Concurrent Safety**: Uses database upserts instead of delete+insert to prevent race conditions during parallel crawling
- **Better Error Handling**: OpenAI API key validation at startup with clear error messages

## Overview

This MCP server provides tools that enable AI agents to crawl websites, store content in a vector database (Supabase), and perform RAG over the crawled content. It follows the best practices for building MCP servers based on the [Mem0 MCP server template](https://github.com/coleam00/mcp-mem0/) I provided on my channel previously.

## Vision

The Crawl4AI RAG MCP server is just the beginning. Here's where we're headed:

1. **Multiple Embedding Models**: Expanding beyond OpenAI to support a variety of embedding models, including the ability to run everything locally with Ollama for complete control and privacy.

2. **Advanced RAG Strategies**: Implementing sophisticated retrieval techniques like contextual retrieval, late chunking, and others to move beyond basic "naive lookups" and significantly enhance the power and precision of the RAG system.

3. **Enhanced Chunking Strategy**: Implementing a Context 7-inspired chunking approach that focuses on examples and creates distinct, semantically meaningful sections for each chunk, improving retrieval precision.

4. **Performance Optimization**: Increasing crawling and indexing speed to make it more realistic to "quickly" index new documentation to then leverage it within the same prompt in an AI coding assistant.

5. **Local-First Architecture**: Supporting fully local deployments with local embedding models and vector databases for privacy-conscious users.

## Features

- **Smart URL Detection**: Automatically detects and handles different URL types (regular webpages, sitemaps, text files)
- **Intelligent Boundary Control**: Automatically restricts crawling to the current subdirectory by default, preventing scope creep
- **Recursive Crawling**: Follows internal links to discover content up to 3 levels deep
- **Parallel Processing**: Efficiently crawls multiple pages simultaneously
- **Content Chunking**: Intelligently splits content by headers and size for better processing
- **Vector Search**: Performs RAG over crawled content, optionally filtering by data source for precision
- **Source Retrieval**: Retrieve sources available for filtering to guide the RAG process

## Tools

The server provides four essential web crawling and search tools:

### 1. `crawl_single_page`
Quickly crawl a single web page without following links.
- **Use cases**: Single articles, specific documentation pages, quick content grabs
- **Features**: Automatic chunking, embedding generation, immediate storage
- **Parameters**: `url` (required)
- **Returns**: Success status, chunks stored, content length, link counts

### 2. `smart_crawl_url`
Intelligently crawl websites with automatic content detection and boundary control.
- **Use cases**: Full documentation sites, knowledge bases, multi-page content
- **Auto-detection**: Handles sitemaps (.xml, .xml.gz), text files, and regular webpages
- **Boundary control**: 
  - Default: Auto-restricts to current subdirectory
  - Custom: Use `prefix` parameter for precise control
  - Protocol-agnostic: Treats HTTP and HTTPS as equivalent
- **Parameters**:
  - `url`: Starting URL (required)
  - `max_depth`: Recursion depth (default: 3, max: 10)
  - `max_concurrent`: Parallel sessions (default: 10, max: 50)
  - `chunk_size`: Characters per chunk (default: 5000)
  - `prefix`: Custom crawl boundary (optional)
- **Returns**: Success status, crawl type, pages crawled, chunks stored, URLs processed

### 3. `get_available_sources`
List all domains that have been crawled and indexed.
- **Use cases**: Pre-search discovery, source filtering, content inventory
- **Features**: Efficient distinct query, sorted output
- **Returns**: List of unique source domains with count

### 4. `perform_rag_query`
Search indexed content using semantic similarity.
- **Use cases**: Q&A, documentation search, code examples, knowledge retrieval
- **Features**: Vector similarity search, optional source filtering, relevance scoring
- **Parameters**:
  - `query`: Natural language search query (required)
  - `source`: Filter by domain (optional)
  - `match_count`: Results to return (default: 5, max: 10)
- **Returns**: Matched chunks with content, metadata, URLs, and similarity scores

## Prerequisites

- [Docker/Docker Desktop](https://www.docker.com/products/docker-desktop/) if running the MCP server as a container (recommended)
- [Python 3.12+](https://www.python.org/downloads/) if running the MCP server directly through uv
- [Supabase](https://supabase.com/) (database for RAG)
- [OpenAI API key](https://platform.openai.com/api-keys) (for generating embeddings)

## Installation

### Using Docker (Recommended)

1. Clone this repository:
   ```bash
   git clone https://github.com/coleam00/mcp-crawl4ai-rag.git
   cd mcp-crawl4ai-rag
   ```

2. Build the Docker image:
   ```bash
   docker build -t mcp/crawl4ai-rag --build-arg PORT=8051 .
   ```

3. Create a `.env` file based on the configuration section below

### Using uv directly (no Docker)

1. Clone this repository:
   ```bash
   git clone https://github.com/coleam00/mcp-crawl4ai-rag.git
   cd mcp-crawl4ai-rag
   ```

2. Install uv if you don't have it:
   ```bash
   pip install uv
   ```

3. Create and activate a virtual environment:
   ```bash
   uv venv
   .venv\Scripts\activate
   # on Mac/Linux: source .venv/bin/activate
   ```

4. Install dependencies:
   ```bash
   uv pip install -e .
   crawl4ai-setup
   ```

5. Create a `.env` file based on the configuration section below

## Database Setup

Before running the server, you need to set up the database with the pgvector extension:

1. Go to the SQL Editor in your Supabase dashboard (create a new project first if necessary)

2. Create a new query and paste the contents of `crawled_pages.sql`

3. Run the query to create:
   - The `crawled_pages` table with vector search capabilities
   - The `match_crawled_pages` function for semantic search
   - The `get_unique_sources` function for efficient source listing
   - Necessary indexes for performance
   - Row-level security policies

**Note**: The SQL file includes optimizations for concurrent operations and efficient querying.

## Configuration

### Environment Variables

Create a `.env` file in the project root (or set in Coolify) with the following variables:

```
# MCP Server Configuration
HOST=0.0.0.0
PORT=8051
TRANSPORT=sse

# Authentication (optional but recommended for production)
MCP_SERVER_API_KEY=your_secure_api_key

# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key

# Supabase Configuration
SUPABASE_URL=your_supabase_project_url
SUPABASE_SERVICE_KEY=your_supabase_service_key

# Optional Configuration (with defaults)
# CRAWLER_MEMORY_THRESHOLD=70.0      # Memory threshold for adaptive dispatcher (%)
# CRAWLER_CHECK_INTERVAL=1.0         # Check interval for memory monitoring (seconds)
# CRAWLER_TIMEOUT=30000              # Timeout per page in milliseconds
# MAX_DOCUMENT_LENGTH=25000          # Maximum document length for contextual embeddings
# EMBEDDING_MODEL=text-embedding-3-small  # OpenAI embedding model to use
# MODEL_CHOICE=gpt-3.5-turbo         # Model for contextual embeddings (if using)
```

**Note**: When `MCP_SERVER_API_KEY` is set, all SSE connections must include the `X-API-Key` header with this value.

## Running the Server

### Using Docker

```bash
docker run --env-file .env -p 8051:8051 mcp/crawl4ai-rag
```

### Using Python

```bash
uv run src/crawl4ai_mcp.py
```

The server will start and listen on the configured host and port.

## Integration with MCP Clients

### Claude Code Setup (with Coolify deployment)

When running the MCP server on Coolify with environment variables configured there, you need to add the SSE endpoint with authentication:

```bash
# Add the MCP server with authentication
claude mcp add --transport sse crawl4ai-rag \
  -e X-API-Key=your_mcp_server_api_key \
  https://your-coolify-domain.com/sse
```

**Important**: The API key you pass here should match the `MCP_SERVER_API_KEY` environment variable set in your Coolify deployment.

#### Managing the MCP server:

```bash
# List all configured servers
claude mcp list

# Get details for this server
claude mcp get crawl4ai-rag

# Remove the server if needed
claude mcp remove crawl4ai-rag

# Check MCP server status in Claude Code
# Type this in the chat:
/mcp
```

### SSE Configuration (for other MCP clients)

Once you have the server running with SSE transport, you can connect to it using this configuration:

```json
{
  "mcpServers": {
    "crawl4ai-rag": {
      "transport": "sse",
      "url": "http://localhost:8051/sse",
      "headers": {
        "X-API-Key": "your_mcp_server_api_key"
      }
    }
  }
}
```

> **Note for Windsurf users**: Use `serverUrl` instead of `url` in your configuration:
> ```json
> {
>   "mcpServers": {
>     "crawl4ai-rag": {
>       "transport": "sse",
>       "serverUrl": "http://localhost:8051/sse",
>       "headers": {
>         "X-API-Key": "your_mcp_server_api_key"
>       }
>     }
>   }
> }
> ```
>
> **Note for Docker users**: Use `host.docker.internal` instead of `localhost` if your client is running in a different container. This will apply if you are using this MCP server within n8n!

### Stdio Configuration

Add this server to your MCP configuration for Claude Desktop, Windsurf, or any other MCP client:

```json
{
  "mcpServers": {
    "crawl4ai-rag": {
      "command": "python",
      "args": ["path/to/crawl4ai-mcp/src/crawl4ai_mcp.py"],
      "env": {
        "TRANSPORT": "stdio",
        "OPENAI_API_KEY": "your_openai_api_key",
        "SUPABASE_URL": "your_supabase_url",
        "SUPABASE_SERVICE_KEY": "your_supabase_service_key"
      }
    }
  }
}
```

### Docker with Stdio Configuration

```json
{
  "mcpServers": {
    "crawl4ai-rag": {
      "command": "docker",
      "args": ["run", "--rm", "-i", 
               "-e", "TRANSPORT", 
               "-e", "OPENAI_API_KEY", 
               "-e", "SUPABASE_URL", 
               "-e", "SUPABASE_SERVICE_KEY", 
               "mcp/crawl4ai"],
      "env": {
        "TRANSPORT": "stdio",
        "OPENAI_API_KEY": "your_openai_api_key",
        "SUPABASE_URL": "your_supabase_url",
        "SUPABASE_SERVICE_KEY": "your_supabase_service_key"
      }
    }
  }
}
```

## Recent Updates

### Major Bug Fixes & Improvements (Latest Release)

#### 🛡️ Security & Reliability
- **Domain Boundary Protection**: Crawler now strictly follows only internal links from the same domain
- **Thread-Safe Operations**: Added proper locking mechanisms for concurrent URL tracking
- **Race Condition Prevention**: Database operations use upsert with conflict resolution
- **Resource Cleanup**: Improved error handling in crawler lifecycle management

#### 🚀 Performance Enhancements
- **Optimized Source Extraction**: Added SQL function for efficient distinct source queries
- **Concurrent Processing Fix**: Fixed critical bug causing mismatched embeddings in parallel processing
- **Smart Chunking**: Improved markdown chunking that respects code block boundaries
- **Configurable Timeouts**: All operations now have configurable timeouts (default: 30s)

#### ✅ Data Integrity
- **Error Propagation**: No more silent failures - errors are properly raised and logged
- **Content Storage**: Fixed to store original content while using contextual embeddings
- **URL Normalization**: Consistent protocol-agnostic URL handling across all operations
- **Type Safety**: Handles both dictionary and string link formats from crawlers

#### 🔧 Configuration & Validation
- **Comprehensive Input Validation**: All inputs validated with sensible limits
- **URL Validation**: Proper format and scheme validation for all URLs
- **Model Validation**: Validates OpenAI models before use
- **Environment Variables**: Made all hardcoded values configurable (see Configuration section)

#### 📚 Enhanced Features
- **Sitemap Support**: Now handles gzipped sitemaps (.xml.gz) and sitemap index files
- **Document Truncation**: Indicates when documents are truncated for context
- **Better Error Messages**: Improved error messages throughout for easier debugging
- **File Detection**: Enhanced logic for distinguishing files from directories in URLs

## Usage Examples

### Crawling with Automatic Subdirectory Restriction (NEW)

```python
# Crawl only the /api/v2/ subdirectory (automatic restriction)
smart_crawl_url(url="https://docs.example.com/api/v2/")

# Result: Only crawls pages under /api/v2/, not the entire docs.example.com
```

### Crawling with Custom Boundaries

```python
# Crawl entire domain (override subdirectory restriction)
smart_crawl_url(
    url="https://docs.example.com/api/v2/",
    prefix="https://docs.example.com/"
)

# Index only a single page
smart_crawl_url(
    url="https://docs.example.com/api/v2/auth.html",
    prefix="https://docs.example.com/api/v2/auth.html"
)

# Crawl different subdirectory than the entrypoint
smart_crawl_url(
    url="https://docs.example.com/",
    prefix="https://docs.example.com/tutorials/"
)
```

### Performing RAG Queries

```python
# Search across all crawled content
perform_rag_query(query="how to authenticate")

# Search only within a specific domain
perform_rag_query(
    query="rate limiting",
    source="docs.example.com"
)
```

## Troubleshooting

### Common Issues

1. **"OPENAI_API_KEY not set" error**
   - Ensure your OpenAI API key is set in the `.env` file or environment
   - The server will warn at startup if the key is missing

2. **Crawler hangs on slow websites**
   - Adjust the `CRAWLER_TIMEOUT` environment variable (default: 30000ms)
   - Consider reducing `max_concurrent` for rate-limited sites

3. **"Invalid model" errors**
   - Ensure `MODEL_CHOICE` is set to a valid OpenAI model
   - Valid models: gpt-3.5-turbo, gpt-4, gpt-4-turbo, gpt-4o, gpt-4o-mini

4. **Database connection issues**
   - Verify `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` are correct
   - Ensure the database setup SQL has been run successfully

5. **Memory issues with large crawls**
   - Adjust `CRAWLER_MEMORY_THRESHOLD` (default: 70%)
   - Reduce `max_concurrent` to limit parallel operations

### Performance Tips

- For large documentation sites, use the sitemap URL if available
- Set appropriate `max_depth` to avoid crawling too deep
- Use `prefix` parameter to limit crawl scope
- Monitor memory usage with `CRAWLER_MEMORY_THRESHOLD`

## Building Your Own Server

This implementation provides a foundation for building more complex MCP servers with web crawling capabilities. To build your own:

1. Add your own tools by creating methods with the `@mcp.tool()` decorator
2. Create your own lifespan function to add your own dependencies
3. Modify the `utils.py` file for any helper functions you need
4. Extend the crawling capabilities by adding more specialized crawlers

### Contributing

Contributions are welcome! Please:
- Report bugs via GitHub issues
- Submit pull requests for improvements
- Follow the existing code style and patterns
- Add tests for new functionality
- Update documentation as needed