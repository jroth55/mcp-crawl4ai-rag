# Crawl4AI RAG MCP Server Documentation

Welcome to the documentation for the Crawl4AI RAG MCP Server. This server provides powerful web crawling and RAG (Retrieval-Augmented Generation) capabilities for AI agents and AI coding assistants.

## Quick Links

- [Installation Guide](installation.md) - Get started quickly
- [Usage Guide](usage.md) - Learn how to use the server
- [API Reference](api.md) - Detailed API documentation
- [Architecture Overview](architecture.md) - Understanding the system design
- [Developer Guide](developer_guide.md) - Contribute to the project
- [Deployment Guide](deployment.md) - Deploy to production
- [Troubleshooting](troubleshooting.md) - Common issues and solutions

## What is Crawl4AI RAG MCP Server?

The Crawl4AI RAG MCP Server is an implementation of the [Model Context Protocol (MCP)](https://modelcontextprotocol.io) that integrates [Crawl4AI](https://crawl4ai.com) for web crawling and [Supabase](https://supabase.com/) for vector storage and semantic search capabilities.

### Key Features

- **Smart URL Detection**: Automatically detects and handles different URL types (webpages, sitemaps, text files)
- **Intelligent Boundary Control**: Auto-restricts crawling to subdirectories by default, preventing scope creep
- **Recursive Crawling**: Follows internal links to discover content up to configurable depth
- **Parallel Processing**: Efficiently crawls multiple pages simultaneously
- **Content Chunking**: Intelligently splits content by headers and size for better processing
- **Vector Search**: Performs RAG over crawled content with semantic similarity
- **Authentication**: Optional API key authentication for secure deployments

### Use Cases

- Building knowledge bases from documentation sites
- Enabling AI agents to access and understand web content
- Creating Q&A systems over crawled content
- Indexing technical documentation for AI coding assistants
- Research and information retrieval from web sources

## Quick Start

1. **Install the server** (see [Installation Guide](installation.md))
2. **Configure environment variables** (see [Configuration](installation.md#configuration))
3. **Run the server** (see [Running the Server](usage.md#running-the-server))
4. **Connect your MCP client** (see [Integration](usage.md#integration-with-mcp-clients))
5. **Start crawling!** (see [Usage Examples](usage.md#usage-examples))

## Documentation Structure

This documentation is organized to help different audiences:

- **End Users**: Start with the [Installation](installation.md) and [Usage](usage.md) guides
- **Developers**: Check the [API Reference](api.md) and [Developer Guide](developer_guide.md)
- **System Administrators**: See [Deployment](deployment.md) and [Troubleshooting](troubleshooting.md)

## Getting Help

- **Issues**: Report bugs on [GitHub Issues](https://github.com/coleam00/mcp-crawl4ai-rag/issues)
- **Discussions**: Ask questions in [GitHub Discussions](https://github.com/coleam00/mcp-crawl4ai-rag/discussions)
- **Contributing**: See our [Developer Guide](developer_guide.md) for contribution guidelines

## Next Steps

Ready to get started? Head to the [Installation Guide](installation.md) to set up your Crawl4AI RAG MCP Server!