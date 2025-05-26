# Usage Guide

This guide covers how to use the Crawl4AI RAG MCP Server effectively, including running the server, integrating with MCP clients, and practical examples.

## Running the Server

### Using Docker

```bash
# Run with environment file
docker run --env-file .env -p 8051:8051 mcp/crawl4ai-rag

# Run with individual environment variables
docker run -e OPENAI_API_KEY=your_key \
           -e SUPABASE_URL=your_url \
           -e SUPABASE_SERVICE_KEY=your_key \
           -p 8051:8051 mcp/crawl4ai-rag
```

### Using Python

```bash
# With virtual environment activated
python src/crawl4ai_mcp.py

# Or using uv directly
uv run src/crawl4ai_mcp.py
```

The server will start on `http://localhost:8051` by default.

## Integration with MCP Clients

### Claude Code (Claude Desktop)

#### With Coolify Deployment

If you've deployed the server on Coolify:

```bash
# Add the MCP server with authentication
claude mcp add --transport sse crawl4ai-rag \
  -e X-API-Key=your_mcp_server_api_key \
  https://your-coolify-domain.com/sse
```

#### Local Development

For local development without authentication:

```bash
# Add local SSE server
claude mcp add --transport sse crawl4ai-rag \
  http://localhost:8051/sse
```

#### Managing Claude Code Integration

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

### Windsurf Configuration

Add to your Windsurf settings:

```json
{
  "mcpServers": {
    "crawl4ai-rag": {
      "transport": "sse",
      "serverUrl": "http://localhost:8051/sse",
      "headers": {
        "X-API-Key": "your_mcp_server_api_key"
      }
    }
  }
}
```

**Note**: Windsurf uses `serverUrl` instead of `url`.

### Stdio Configuration

For MCP clients that support stdio transport:

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

### Docker with Stdio

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
               "mcp/crawl4ai-rag"],
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

## Available Tools

The server provides four main tools:

### 1. crawl_single_page
Crawl a single web page without following links.

### 2. smart_crawl_url
Intelligently crawl websites with automatic content detection.

### 3. get_available_sources
List all domains that have been crawled.

### 4. perform_rag_query
Search indexed content using semantic similarity.

For detailed API documentation, see the [API Reference](api.md).

## Usage Examples

### Basic Web Crawling

#### Single Page Crawling

```python
# Crawl a single documentation page
crawl_single_page(url="https://docs.example.com/api/authentication")

# Result:
{
  "success": true,
  "url": "https://docs.example.com/api/authentication",
  "chunks_stored": 5,
  "content_length": 12543,
  "links_count": {
    "internal": 23,
    "external": 4
  }
}
```

#### Smart Crawling with Auto-Detection

```python
# Crawl a documentation section (auto-restricts to /api/v2/)
smart_crawl_url(url="https://docs.example.com/api/v2/")

# Crawl from a sitemap
smart_crawl_url(url="https://example.com/sitemap.xml")

# Crawl a text file
smart_crawl_url(url="https://example.com/documentation.txt")
```

### Advanced Crawling with Boundary Control

#### Subdirectory Restriction (Default Behavior)

```python
# Automatically restricts to /tutorials/ subdirectory
smart_crawl_url(url="https://docs.example.com/tutorials/getting-started")

# Only pages under /tutorials/ will be crawled
```

#### Custom Crawl Boundaries

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

# Start from homepage but only crawl /blog/ section
smart_crawl_url(
    url="https://example.com",
    prefix="https://example.com/blog"
)
```

### Controlling Crawl Depth and Performance

```python
# Shallow crawl (only follow links 1 level deep)
smart_crawl_url(
    url="https://docs.example.com",
    max_depth=1
)

# Reduce concurrent sessions for rate-limited sites
smart_crawl_url(
    url="https://api.example.com/docs",
    max_concurrent=5
)

# Larger chunks for technical documentation
smart_crawl_url(
    url="https://docs.example.com",
    chunk_size=8000
)
```

### Discovering Available Sources

```python
# List all crawled domains
get_available_sources()

# Result:
{
  "success": true,
  "sources": [
    "docs.example.com",
    "api.example.com",
    "blog.example.com"
  ],
  "count": 3
}
```

### Performing RAG Queries

#### Basic Search

```python
# Search across all sources
perform_rag_query(query="how to authenticate with the API")

# Result includes matched chunks with similarity scores
{
  "success": true,
  "query": "how to authenticate with the API",
  "results": [
    {
      "url": "https://docs.example.com/api/auth",
      "content": "Authentication is handled via API keys...",
      "metadata": {
        "source": "docs.example.com",
        "headers": "## Authentication",
        "chunk_index": 2
      },
      "similarity": 0.89
    }
  ],
  "count": 5
}
```

#### Filtered Search

```python
# Search only within specific domain
perform_rag_query(
    query="rate limiting",
    source="api.example.com"
)

# Get more results
perform_rag_query(
    query="error handling",
    match_count=10
)
```

## Common Workflows

### 1. Building a Documentation Knowledge Base

```python
# Step 1: Crawl the documentation site
smart_crawl_url(url="https://docs.yourproject.com")

# Step 2: Verify it was indexed
sources = get_available_sources()
# Check that "docs.yourproject.com" is in the list

# Step 3: Query the documentation
perform_rag_query(query="how to install the package")
```

### 2. Indexing Multiple Documentation Sites

```python
# Crawl multiple documentation sites
smart_crawl_url(url="https://docs.framework1.com")
smart_crawl_url(url="https://api.framework2.com/docs")
smart_crawl_url(url="https://guides.framework3.com")

# Search across all of them
perform_rag_query(query="authentication best practices")

# Or search specific framework
perform_rag_query(
    query="authentication best practices",
    source="docs.framework1.com"
)
```

### 3. Keeping Documentation Updated

```python
# Re-crawl to get latest updates
# The system uses upsert, so it will update existing content
smart_crawl_url(
    url="https://docs.example.com/api/",
    max_depth=2  # Shallow crawl for faster updates
)
```

### 4. Research and Information Gathering

```python
# Crawl a blog or news site
smart_crawl_url(
    url="https://blog.example.com",
    prefix="https://blog.example.com/2024/"  # Only 2024 posts
)

# Search for specific topics
perform_rag_query(query="machine learning trends")
```

## Best Practices

### 1. Crawling Strategy

- **Start Small**: Test with `crawl_single_page` before doing recursive crawls
- **Use Sitemaps**: When available, use sitemap URLs for efficient crawling
- **Set Appropriate Depth**: Use `max_depth=1` or `2` for faster, focused crawls
- **Respect Rate Limits**: Reduce `max_concurrent` if you encounter rate limiting

### 2. Boundary Control

- **Default is Good**: The automatic subdirectory restriction prevents accidental over-crawling
- **Be Specific**: Use custom `prefix` values to precisely control scope
- **Single Page Mode**: Set `prefix` equal to URL for indexing individual pages

### 3. Search Optimization

- **Use Natural Language**: The semantic search understands context
- **Filter by Source**: Use source filtering for more relevant results
- **Adjust Match Count**: Start with default 5, increase if needed

### 4. Performance Tips

- **Chunk Size**: Larger chunks (8000-10000) work well for technical docs
- **Concurrent Sessions**: 10 is usually safe, reduce for slower sites
- **Memory Management**: The crawler automatically manages memory usage

## Integration Tips

### For AI Coding Assistants

1. **Index Project Documentation**: Crawl your project's docs for context
2. **Include API References**: Ensure API documentation is well-indexed
3. **Regular Updates**: Re-crawl periodically to keep knowledge current

### For Research Applications

1. **Domain-Specific Crawling**: Use prefix to focus on relevant sections
2. **Multiple Sources**: Index various authoritative sources
3. **Query Refinement**: Use specific, detailed queries for best results

### For Q&A Systems

1. **Comprehensive Indexing**: Crawl entire knowledge bases
2. **Source Attribution**: Use metadata to cite sources
3. **Similarity Thresholds**: Filter results by similarity score

## Next Steps

- Review the [API Reference](api.md) for detailed parameter documentation
- Check [Architecture Overview](architecture.md) to understand how it works
- See [Troubleshooting](troubleshooting.md) for common issues
- Explore [Developer Guide](developer_guide.md) to extend functionality