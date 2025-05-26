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

### Changed
- [TODO: Add changes made in this version]

### Fixed
- [TODO: Add bug fixes in this version]

### Removed
- [TODO: Add removed features in this version]

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