# API Reference

This document provides detailed API documentation for all tools available in the Crawl4AI RAG MCP Server.

## Table of Contents

- [crawl_single_page](#crawl_single_page)
- [smart_crawl_url](#smart_crawl_url)
- [get_available_sources](#get_available_sources)
- [perform_rag_query](#perform_rag_query)

## crawl_single_page

Crawl a single web page without following any links and store its content for RAG queries.

### Function Signature

```python
async def crawl_single_page(ctx: Context, url: str) -> str
```

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `ctx` | Context | Yes | The MCP server provided context (automatically injected) |
| `url` | string | Yes | The exact URL of the web page to crawl |

### Returns

JSON string containing:

```json
{
  "success": true,
  "url": "https://example.com/page",
  "chunks_stored": 5,
  "content_length": 12543,
  "links_count": {
    "internal": 23,
    "external": 4
  }
}
```

### Error Response

```json
{
  "success": false,
  "url": "https://example.com/page",
  "error": "Error message describing what went wrong"
}
```

### Examples

```python
# Crawl a single article
crawl_single_page(url="https://blog.example.com/post/ai-trends-2024")

# Crawl a documentation page
crawl_single_page(url="https://docs.example.com/api/authentication")

# Crawl a product page
crawl_single_page(url="https://shop.example.com/products/item-123")
```

### Use Cases

- Quickly grabbing content from one specific page
- Storing a single article or documentation page
- Getting content without recursive crawling
- Testing before doing larger crawls

### Notes

- Content is automatically chunked based on headers and size
- Each chunk is stored with embeddings for semantic search
- The tool respects robots.txt and crawl delays
- Uses a 30-second timeout by default

## smart_crawl_url

Intelligently crawl websites and store all content for RAG queries. Auto-detects sitemaps and follows links.

### Function Signature

```python
async def smart_crawl_url(
    ctx: Context, 
    url: str, 
    max_depth: int = 3, 
    max_concurrent: int = 10, 
    chunk_size: int = 5000, 
    prefix: str = None,
    sitemap_max_depth: int = None
) -> str
```

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `ctx` | Context | Yes | - | The MCP server provided context (automatically injected) |
| `url` | string | Yes | - | Starting URL (can be webpage, sitemap.xml, or .txt file) |
| `max_depth` | integer | No | 3 | How many link levels deep to crawl (0-10) |
| `max_concurrent` | integer | No | 10 | Number of parallel browser sessions (1-50) |
| `chunk_size` | integer | No | 5000 | Characters per chunk for storage |
| `prefix` | string | No | Auto-detected | URL prefix to restrict crawling scope |
| `sitemap_max_depth` | integer | No | From env | Maximum recursion depth for nested sitemap indexes |

### Auto-Detection Behavior

The tool automatically detects the URL type:

1. **Sitemaps** (`.xml`, `.xml.gz`): Extracts all URLs and crawls in parallel
2. **Text files** (`.txt`): Directly retrieves and stores content
3. **Webpages**: Recursively follows internal links up to max_depth

### Boundary Control

The `prefix` parameter controls crawling boundaries:

- **Default (None)**: Auto-restricts to current subdirectory
- **Custom prefix**: Only crawls URLs starting with specified prefix
- **Single page**: Set `prefix = url` to index only that page
- **Protocol-agnostic**: HTTP and HTTPS are treated as equivalent

### Returns

JSON string containing:

```json
{
  "success": true,
  "url": "https://docs.example.com",
  "crawl_type": "webpage",
  "pages_crawled": 42,
  "pages_processed": 40,
  "chunks_prepared": 180,
  "chunks_stored": 156,
  "urls_crawled": [
    "https://docs.example.com/",
    "https://docs.example.com/getting-started",
    "https://docs.example.com/api",
    "https://docs.example.com/tutorials",
    "https://docs.example.com/reference",
    "..."
  ]
}
```

When partial failures occur, an additional `partial_failures` object is included:

```json
{
  "success": true,
  "url": "https://docs.example.com",
  "crawl_type": "webpage",
  "pages_crawled": 42,
  "pages_processed": 40,
  "chunks_prepared": 180,
  "chunks_stored": 156,
  "partial_failures": {
    "storage_errors": 24,
    "failed_batches": 2,
    "total_batches": 5,
    "success_rate": "86.7%"
  },
  "urls_crawled": ["..."]
}
```

### Error Response

```json
{
  "success": false,
  "url": "https://example.com",
  "error": "No URLs found in sitemap"
}
```

### Examples

#### Basic Crawling

```python
# Crawl with auto-detected boundaries
smart_crawl_url(url="https://docs.example.com/api/v2/")
# Auto-restricts to /api/v2/ subdirectory

# Crawl from a sitemap
smart_crawl_url(url="https://example.com/sitemap.xml")

# Crawl a text file
smart_crawl_url(url="https://raw.githubusercontent.com/example/repo/main/README.txt")
```

#### Advanced Boundary Control

```python
# Crawl entire domain (override subdirectory restriction)
smart_crawl_url(
    url="https://docs.example.com/api/",
    prefix="https://docs.example.com/"
)

# Index only a single page
smart_crawl_url(
    url="https://example.com/guide.html",
    prefix="https://example.com/guide.html"
)

# Start from homepage but only crawl blog section
smart_crawl_url(
    url="https://example.com",
    prefix="https://example.com/blog/"
)
```

#### Performance Tuning

```python
# Shallow crawl for faster results
smart_crawl_url(
    url="https://docs.example.com",
    max_depth=1
)

# Reduce concurrency for rate-limited sites
smart_crawl_url(
    url="https://api.example.com/docs",
    max_concurrent=5
)

# Larger chunks for code documentation
smart_crawl_url(
    url="https://docs.example.com",
    chunk_size=10000
)
```

### Use Cases

- Building knowledge bases from documentation sites
- Indexing entire websites or specific sections
- Processing sitemaps for comprehensive coverage
- Creating searchable archives of web content

### Notes

- Automatically handles gzipped sitemaps and sitemap index files
- Respects the same-domain policy for security
- Uses memory-adaptive dispatching to prevent OOM errors
- Implements retry logic for failed pages

## get_available_sources

List all domains/sources that have been crawled and are available for RAG queries.

### Function Signature

```python
async def get_available_sources(ctx: Context) -> str
```

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `ctx` | Context | Yes | The MCP server provided context (automatically injected) |

### Returns

JSON string containing:

```json
{
  "success": true,
  "sources": [
    "docs.example.com",
    "api.example.com",
    "blog.example.com",
    "help.example.com"
  ],
  "count": 4
}
```

### Error Response

```json
{
  "success": false,
  "error": "Database connection error"
}
```

### Examples

```python
# Get all available sources
sources = get_available_sources()

# Use the result to inform RAG queries
if "docs.example.com" in sources["sources"]:
    perform_rag_query(
        query="installation guide",
        source="docs.example.com"
    )
```

### Use Cases

- Pre-flight check before performing searches
- Building source selection interfaces
- Verifying successful crawls
- Discovering what content is available

### Notes

- Returns unique domain names only
- Sorted alphabetically for consistency
- Efficient query using database indexes
- Limited to 1000 sources for performance

## perform_rag_query

Search crawled content using semantic similarity (RAG). Returns relevant chunks matching your query.

### Function Signature

```python
async def perform_rag_query(
    ctx: Context, 
    query: str, 
    source: str = None, 
    match_count: int = 5
) -> str
```

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `ctx` | Context | Yes | - | The MCP server provided context (automatically injected) |
| `query` | string | Yes | - | Natural language search query |
| `source` | string | No | None | Filter results to specific domain |
| `match_count` | integer | No | 5 | Number of relevant chunks to return (1-10 recommended) |

### Returns

JSON string containing:

```json
{
  "success": true,
  "query": "how to authenticate with API",
  "source_filter": "docs.example.com",
  "results": [
    {
      "url": "https://docs.example.com/api/auth",
      "content": "Authentication is handled via API keys. To authenticate, include your API key in the X-API-Key header...",
      "metadata": {
        "chunk_index": 2,
        "url": "https://docs.example.com/api/auth",
        "source": "docs.example.com",
        "headers": "## Authentication; ### API Keys",
        "char_count": 1543,
        "word_count": 234,
        "crawl_time": "2024-01-15T10:30:00Z",
        "contextual_embedding": true
      },
      "similarity": 0.89
    }
  ],
  "count": 5
}
```

### Error Response

```json
{
  "success": false,
  "query": "search query",
  "error": "Failed to create embedding: API key invalid"
}
```

### Examples

#### Basic Search

```python
# Search across all sources
perform_rag_query(query="how to handle rate limits")

# Search for code examples
perform_rag_query(query="Python code example for authentication")

# Search for specific errors
perform_rag_query(query="ConnectionRefusedError troubleshooting")
```

#### Filtered Search

```python
# Search only within specific domain
perform_rag_query(
    query="rate limiting best practices",
    source="api.example.com"
)

# Search docs for installation instructions
perform_rag_query(
    query="pip install instructions",
    source="docs.python-project.com"
)
```

#### Adjusting Results

```python
# Get more results for comprehensive coverage
perform_rag_query(
    query="error handling strategies",
    match_count=10
)

# Get fewer, more relevant results
perform_rag_query(
    query="specific configuration option",
    match_count=3
)
```

### Understanding Results

#### Metadata Fields

- `chunk_index`: Position of chunk within the original page
- `source`: Domain where content was found
- `headers`: Markdown headers present in the chunk
- `char_count`: Number of characters in the chunk
- `word_count`: Number of words in the chunk
- `crawl_time`: When the content was indexed
- `contextual_embedding`: Whether contextual enhancement was used

#### Similarity Score

- `1.0`: Perfect match
- `0.8-1.0`: Highly relevant
- `0.6-0.8`: Relevant
- `0.4-0.6`: Somewhat relevant
- `< 0.4`: Low relevance

### Use Cases

- Question answering over documentation
- Finding code examples and API usage
- Troubleshooting error messages
- Research and information retrieval
- Building chatbots with web knowledge

### Advanced Usage

#### Combining with Source Discovery

```python
# First, discover available sources
sources = get_available_sources()

# Then search specific sources
for source in sources["sources"]:
    if "docs" in source:
        results = perform_rag_query(
            query="authentication",
            source=source
        )
```

#### Building Q&A Systems

```python
def answer_question(question):
    # Search for answer
    results = perform_rag_query(
        query=question,
        match_count=3
    )
    
    # Check if we found relevant content
    if results["success"] and results["count"] > 0:
        top_result = results["results"][0]
        if top_result["similarity"] > 0.7:
            return {
                "answer": top_result["content"],
                "source": top_result["url"],
                "confidence": top_result["similarity"]
            }
    
    return {"answer": "No relevant information found"}
```

### Notes

- Uses OpenAI's text-embedding-3-small model by default
- Semantic search understands context and synonyms
- Results are ordered by similarity score
- Supports filtering by metadata fields
- Implements efficient vector similarity search using pgvector

## Environment Variables

The following environment variables affect API behavior:

### Crawler Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `CRAWLER_TIMEOUT` | 30000 | Timeout per page in milliseconds |
| `CRAWLER_MEMORY_THRESHOLD` | 70.0 | Memory threshold for adaptive dispatcher (%) |
| `CRAWLER_CHECK_INTERVAL` | 1.0 | Memory check interval in seconds |
| `MAX_DOCUMENT_LENGTH` | 25000 | Max document length for contextual embeddings |
| `SITEMAP_MAX_DEPTH` | 2 | Maximum recursion depth for nested sitemap indexes |

### OpenAI Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_MODEL` | text-embedding-3-small | OpenAI embedding model |
| `MODEL_CHOICE` | None | Model for contextual embeddings (optional) |
| `OPENAI_MAX_RETRIES` | 3 | Maximum retries for OpenAI API calls |
| `OPENAI_RETRY_DELAY` | 1.0 | Initial delay between retries (seconds) |
| `OPENAI_TIMEOUT` | 30 | Timeout for OpenAI API calls (seconds) |

### Batch Processing Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_BATCH_SIZE` | 20 | Number of text chunks per OpenAI embeddings API call |
| `DOCUMENT_BATCH_SIZE` | 10 | Number of documents to process in memory at once |
| `THREAD_POOL_MAX_WORKERS` | 10 | Max concurrent threads for contextual embeddings |
| `BATCH_FAILURE_THRESHOLD` | 0.5 | Fail if more than this percentage of batches fail |

## Error Handling

All tools return consistent error responses:

```json
{
  "success": false,
  "error": "Descriptive error message"
}
```

Common error types:

- **Validation Errors**: Invalid URLs, parameters out of range
- **Network Errors**: Timeouts, connection failures
- **API Errors**: Invalid API keys, rate limits
- **Database Errors**: Connection issues, query failures

## Rate Limiting

The server implements several rate limiting strategies:

1. **Concurrent Crawling**: Limited by `max_concurrent` parameter
2. **Memory Management**: Automatic throttling based on system memory
3. **API Rate Limits**: Respects OpenAI and Supabase rate limits
4. **Crawl Delays**: Respects robots.txt crawl-delay directives

## Best Practices

1. **Start Small**: Test with single pages before large crawls
2. **Use Appropriate Depths**: max_depth=1-2 for most use cases
3. **Filter Searches**: Use source parameter for better relevance
4. **Monitor Usage**: Check crawled page counts and API usage
5. **Handle Errors**: Always check the success field in responses