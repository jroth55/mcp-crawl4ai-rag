# Changelog

All notable changes to the Crawl4AI RAG MCP Server will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive documentation suite in `/docs` folder
- API reference with detailed parameter documentation
- Architecture overview explaining system design
- Developer guide for contributors
- Deployment guide covering multiple platforms
- Troubleshooting guide for common issues
- Health check endpoint at `/health` for monitoring
- Configurable sitemap recursion depth via `SITEMAP_MAX_DEPTH` environment variable
- Partial failure reporting in `smart_crawl_url` response with detailed statistics
- Thread pool configuration via `THREAD_POOL_MAX_WORKERS` environment variable
- Batch failure threshold configuration via `BATCH_FAILURE_THRESHOLD`
- Critical API error detection to skip retries for auth/invalid model errors
- Contextual embeddings support for improved RAG quality
- Token length validation and truncation for OpenAI API limits

### Changed
- Logging levels optimized - routine success operations now use DEBUG level
- RLS policies consolidated in `crawled_pages.sql` with idempotent DROP IF EXISTS statements
- Improved error handling with specific error types in responses
- Enhanced `smart_crawl_url` to accept `sitemap_max_depth` parameter
- Memory-efficient batch processing for large crawls

### Fixed
- **Critical: 17.5-hour hang issue** - Replaced blocking `requests.get()` with async `httpx` client
- **Binary file handling** - Added comprehensive filtering to prevent crawling non-HTML files
- **Database vector dimension mismatch** - Enforced 1536-dimensional embeddings
- **Contextual embedding asymmetry** - Added contextual prefix to query embeddings
- **Storage vs retrieval content mismatch** - Now stores both original and contextual content
- **Race conditions** - Added thread-safe URL tracking with locks
- **Silent data loss** - Added failure thresholds and proper error propagation
- **Code duplication** - Removed duplicate `is_binary_file` function definitions

### Removed
- Redundant `fix_rls_policies.sql` file (consolidated into `crawled_pages.sql`)

## [0.1.0] - 2024-01-XX

### Added
- Initial release of Crawl4AI RAG MCP Server
- Four core MCP tools:
  - `crawl_single_page` - Single page crawling
  - `smart_crawl_url` - Intelligent multi-page crawling
  - `get_available_sources` - List indexed domains
  - `perform_rag_query` - Semantic search over content
- Intelligent crawl boundaries with automatic subdirectory restriction
- Authentication support with `MCP_SERVER_API_KEY`
- Support for both SSE and stdio transports
- Docker containerization support
- Comprehensive README with setup instructions

### Features
- **Smart URL Detection**: Automatically handles webpages, sitemaps, and text files
- **Parallel Processing**: Efficient concurrent crawling
- **Content Chunking**: Intelligent markdown-aware splitting
- **Vector Search**: Semantic similarity using OpenAI embeddings
- **Memory Management**: Adaptive throttling based on system resources
- **Sitemap Support**: Handles regular and gzipped sitemaps
- **Error Handling**: Robust error handling and recovery

### Security
- Optional API key authentication
- Input validation on all parameters
- Row-level security in database
- No hardcoded credentials

### Performance
- Batch embedding generation
- Efficient database indexing
- Connection pooling
- Configurable timeouts

### Changed
- Enhanced from original fork with:
  - Better crawl boundary control
  - Production-ready authentication
  - Improved error messages
  - Extended documentation

### Fixed
- Race conditions in concurrent crawling
- Memory leaks in long-running crawls
- Proper handling of gzipped sitemaps
- Database upsert conflicts

## [Future Releases]

### Planned Features
- Multiple embedding model support
- Local embedding options (Ollama)
- Advanced RAG strategies (contextual retrieval, late chunking)
- Streaming responses for real-time progress
- Webhook support for crawl notifications
- Scheduled crawling capabilities
- Incremental content updates

### Under Consideration
- GraphQL API support
- Multi-language content support
- PDF and document extraction
- Integration with more MCP clients
- Distributed crawling architecture

---

For more details on each release, see the [GitHub Releases](https://github.com/coleam00/mcp-crawl4ai-rag/releases) page.