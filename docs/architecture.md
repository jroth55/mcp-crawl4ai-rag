# Architecture Overview

This document provides a comprehensive overview of the Crawl4AI RAG MCP Server architecture, including system components, data flow, and design decisions.

## System Architecture

```
┌─────────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│                     │     │                  │     │                  │
│   MCP Clients       │────▶│  MCP Server      │────▶│  External APIs   │
│  (Claude, etc.)     │◀────│  (FastMCP)       │◀────│  (OpenAI)        │
│                     │     │                  │     │                  │
└─────────────────────┘     └──────────────────┘     └──────────────────┘
                                     │                          
                                     │                          
                            ┌────────▼─────────┐                
                            │                  │                
                            │   Crawl4AI       │                
                            │   (Async)        │                
                            │                  │                
                            └────────┬─────────┘                
                                     │                          
                            ┌────────▼─────────┐                
                            │                  │                
                            │   Supabase       │                
                            │   (pgvector)     │                
                            │                  │                
                            └──────────────────┘                
```

## Core Components

### 1. MCP Server (FastMCP)

The server is built on FastMCP, providing:

- **Transport Support**: Both SSE (Server-Sent Events) and stdio
- **Authentication**: Optional API key authentication via middleware
- **Tool Registration**: Decorator-based tool definition
- **Async Support**: Full async/await implementation

**Key Files**:
- `crawl4ai_mcp.py`: Main server implementation
- `auth_mcp.py`: Authentication middleware extension

### 2. Web Crawler (Crawl4AI)

Crawl4AI provides the web crawling capabilities:

- **Browser Automation**: Headless Chrome/Chromium via Playwright
- **Async Crawling**: Concurrent page processing
- **Content Extraction**: Markdown conversion and link extraction
- **Memory Management**: Adaptive memory-based throttling

**Key Features**:
- Automatic JavaScript rendering
- Cookie and session handling
- Robots.txt compliance
- Configurable timeouts

### 3. Vector Database (Supabase + pgvector)

Supabase with pgvector extension handles storage and search:

- **Vector Storage**: 1536-dimensional embeddings
- **Semantic Search**: Cosine similarity matching
- **Metadata Filtering**: JSONB-based filtering
- **Efficient Indexing**: IVFFlat indexes for performance

**Database Schema**:
```sql
crawled_pages (
    id: bigserial primary key
    url: varchar (page URL)
    chunk_number: integer (chunk index)
    content: text (original content)
    metadata: jsonb (structured metadata)
    embedding: vector(1536) (OpenAI embeddings)
    created_at: timestamp
)
```

### 4. Embedding Service (OpenAI)

OpenAI's API provides embedding generation:

- **Model**: text-embedding-3-small (1536 dimensions)
- **Batch Processing**: Multiple texts in single API call
- **Contextual Enhancement**: Optional context-aware embeddings
- **Error Handling**: Graceful fallback on failures

## Data Flow

### Crawling Flow

1. **URL Input** → MCP tool receives crawl request
2. **URL Analysis** → Detect type (webpage, sitemap, text)
3. **Content Fetching** → Crawl4AI retrieves content
4. **Link Extraction** → Parse internal/external links
5. **Recursive Crawling** → Follow links up to max_depth
6. **Content Processing** → Convert to markdown
7. **Chunking** → Split into semantic chunks
8. **Embedding Generation** → Create vector embeddings
9. **Storage** → Upsert to Supabase

### Search Flow

1. **Query Input** → Natural language search query
2. **Query Embedding** → Convert to vector
3. **Vector Search** → Find similar chunks
4. **Metadata Filtering** → Apply source filters
5. **Result Ranking** → Sort by similarity
6. **Response Formation** → Return formatted results

## Key Design Decisions

### 1. Chunking Strategy

The system uses intelligent markdown-aware chunking:

```python
def smart_chunk_markdown(text: str, chunk_size: int = 5000):
    # Respects code blocks (```)
    # Prefers paragraph boundaries
    # Falls back to sentence breaks
    # Maintains minimum chunk size (30% of target)
```

**Benefits**:
- Preserves code examples intact
- Maintains semantic coherence
- Optimal for embedding generation

### 2. Boundary Control

Automatic subdirectory restriction prevents over-crawling:

```python
# Default: Restrict to current subdirectory
url = "https://docs.example.com/api/v2/"
# Only crawls: /api/v2/*

# Override: Crawl entire domain
prefix = "https://docs.example.com/"
```

**Benefits**:
- Prevents accidental site-wide crawls
- Gives users precise control
- Protocol-agnostic matching

### 3. Concurrent Processing

Multi-level concurrency for performance:

1. **Page Level**: Multiple browsers via MemoryAdaptiveDispatcher
2. **Chunk Level**: Parallel embedding generation
3. **Batch Level**: Bulk database operations

```python
# Adaptive memory management
dispatcher = MemoryAdaptiveDispatcher(
    memory_threshold_percent=70.0,
    max_session_permit=10
)
```

### 4. Contextual Embeddings

Optional context enhancement for better retrieval:

```python
# When MODEL_CHOICE is set:
1. Generate context for chunk within document
2. Create embedding from contextualized text
3. Store original content with enhanced embedding
```

**Trade-offs**:
- Better retrieval accuracy
- Higher API costs
- Longer processing time

## Security Architecture

### 1. Authentication Layer

Optional API key authentication:

```python
# SSE requests require X-API-Key header
if MCP_SERVER_API_KEY is set:
    validate_api_key(request.headers["X-API-Key"])
```

### 2. Database Security

- **Service Role Key**: Full database access
- **Row Level Security**: Read-only public access
- **Prepared Statements**: SQL injection prevention

### 3. Input Validation

Comprehensive validation at all levels:

- URL format and scheme validation
- Parameter range checking
- Content size limits
- Timeout enforcement

## Performance Optimizations

### 1. Database Optimizations

```sql
-- Efficient vector search
CREATE INDEX ON crawled_pages 
USING ivfflat (embedding vector_cosine_ops);

-- Fast metadata filtering
CREATE INDEX idx_crawled_pages_metadata 
USING gin (metadata);

-- Source-specific queries
CREATE INDEX idx_crawled_pages_source 
ON crawled_pages ((metadata->>'source'));
```

### 2. Caching Strategy

- **Crawl4AI Cache**: Bypass for fresh content
- **Connection Pooling**: Reuse database connections
- **Batch Operations**: Reduce API calls

### 3. Memory Management

```python
# Automatic throttling based on system memory
if memory_usage > threshold:
    reduce_concurrent_sessions()
```

## Scalability Considerations

### Horizontal Scaling

The architecture supports horizontal scaling:

1. **Stateless Server**: Multiple instances behind load balancer
2. **Shared Database**: Supabase handles concurrent access
3. **Independent Crawlers**: Each instance manages its own crawlers

### Vertical Scaling

Performance scales with resources:

1. **More Memory**: Higher concurrent crawl sessions
2. **More CPU**: Faster content processing
3. **Better Network**: Increased crawl throughput

## Extension Points

### 1. Custom Embedders

Replace OpenAI with other providers:

```python
# Current
embeddings = openai.embeddings.create(...)

# Extension point for:
# - Local models (Ollama)
# - Other providers (Cohere, Anthropic)
# - Custom models
```

### 2. Alternative Storage

Swap Supabase for other vector databases:

```python
# Current
supabase.table("crawled_pages").upsert(...)

# Extension point for:
# - Pinecone
# - Weaviate
# - Qdrant
# - ChromaDB
```

### 3. Enhanced Chunking

Implement advanced chunking strategies:

```python
# Current: Header/paragraph-based

# Extension points:
# - Semantic chunking
# - Sliding window
# - Overlap strategies
# - Document-specific rules
```

### 4. Custom Crawl Strategies

Add specialized crawlers:

```python
# Current: Generic web crawler

# Extension points:
# - API documentation parsers
# - GitHub repository crawlers
# - PDF/document extractors
# - Dynamic content handlers
```

## Monitoring and Observability

### Logging

Comprehensive logging throughout:

```python
# Info level
print(f"INFO: Starting crawl of {url}")

# Warning level
print(f"WARNING: Rate limit approaching")

# Error level
print(f"ERROR: Failed to create embedding: {e}")
```

### Metrics

Key metrics to monitor:

1. **Crawl Performance**
   - Pages per second
   - Success/failure rates
   - Average page size

2. **Embedding Performance**
   - Embeddings per second
   - API latency
   - Error rates

3. **Search Performance**
   - Query latency
   - Result relevance
   - Cache hit rates

## Future Architecture Enhancements

### Planned Improvements

1. **Streaming Responses**: Real-time crawl progress
2. **Webhook Support**: Notify on crawl completion
3. **Scheduled Crawls**: Periodic content updates
4. **Incremental Updates**: Only crawl changed content

### Research Areas

1. **Advanced RAG**: Contextual retrieval, late chunking
2. **Multi-Modal**: Image and video content
3. **Cross-Language**: Multilingual support
4. **Federated Search**: Query multiple instances

## Conclusion

The Crawl4AI RAG MCP Server architecture is designed for:

- **Flexibility**: Easy to extend and customize
- **Performance**: Efficient concurrent processing
- **Reliability**: Robust error handling
- **Scalability**: Grows with your needs

The modular design allows for future enhancements while maintaining backward compatibility and system stability.