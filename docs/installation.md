# Installation Guide

This guide will walk you through installing and configuring the Crawl4AI RAG MCP Server.

## Prerequisites

Before installing, ensure you have the following:

- **Python 3.12+** or **Docker/Docker Desktop** (for containerized deployment)
- **[Supabase](https://supabase.com/)** account (free tier works)
- **[OpenAI API key](https://platform.openai.com/api-keys)** for generating embeddings
- **Git** for cloning the repository

## Installation Methods

### Method 1: Using Docker (Recommended)

Docker provides the most consistent experience across different platforms.

1. **Clone the repository**:
   ```bash
   git clone https://github.com/coleam00/mcp-crawl4ai-rag.git
   cd mcp-crawl4ai-rag
   ```

2. **Build the Docker image**:
   ```bash
   docker build -t mcp/crawl4ai-rag --build-arg PORT=8051 .
   ```

3. **Create environment file** (see [Configuration](#configuration) section)

4. **Run the container**:
   ```bash
   docker run --env-file .env -p 8051:8051 mcp/crawl4ai-rag
   ```

### Method 2: Using Python with uv

For development or non-containerized deployments.

1. **Clone the repository**:
   ```bash
   git clone https://github.com/coleam00/mcp-crawl4ai-rag.git
   cd mcp-crawl4ai-rag
   ```

2. **Install uv** (if not already installed):
   ```bash
   pip install uv
   ```

3. **Create and activate virtual environment**:
   ```bash
   uv venv
   
   # On Windows:
   .venv\Scripts\activate
   
   # On macOS/Linux:
   source .venv/bin/activate
   ```

4. **Install dependencies**:
   ```bash
   uv pip install -e .
   crawl4ai-setup
   ```

5. **Create environment file** (see [Configuration](#configuration) section)

### Method 3: Using pip (Alternative)

If you prefer using pip directly:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/coleam00/mcp-crawl4ai-rag.git
   cd mcp-crawl4ai-rag
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install package**:
   ```bash
   pip install -e .
   crawl4ai-setup
   ```

## Database Setup

The server requires a Supabase database with pgvector extension for storing and searching crawled content.

1. **Create a Supabase project**:
   - Go to [Supabase Dashboard](https://app.supabase.com/)
   - Click "New Project"
   - Fill in project details

2. **Set up the database**:
   - Navigate to SQL Editor in your project
   - Create a new query
   - Copy and paste the contents of `crawled_pages.sql`
   - Run the query

This creates:
- The `crawled_pages` table with vector search capabilities
- The `match_crawled_pages` function for semantic search
- The `get_unique_sources` function for efficient source listing
- Necessary indexes for performance
- Row-level security policies

## Configuration

### Environment Variables

Create a `.env` file in the project root with the following variables:

```bash
# MCP Server Configuration
HOST=0.0.0.0
PORT=8051
TRANSPORT=sse

# Authentication (optional but recommended for production)
MCP_SERVER_API_KEY=your_secure_api_key_here

# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Supabase Configuration
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_SERVICE_KEY=your_supabase_service_key_here

# Optional Configuration (with defaults)
# CRAWLER_MEMORY_THRESHOLD=70.0      # Memory threshold for adaptive dispatcher (%)
# CRAWLER_CHECK_INTERVAL=1.0         # Check interval for memory monitoring (seconds)
# CRAWLER_TIMEOUT=30000              # Timeout per page in milliseconds
# MAX_DOCUMENT_LENGTH=25000          # Maximum document length for contextual embeddings
# EMBEDDING_MODEL=text-embedding-3-small  # OpenAI embedding model to use
# MODEL_CHOICE=gpt-3.5-turbo         # Model for contextual embeddings (if using)
```

### Getting Your API Keys

#### OpenAI API Key
1. Visit [OpenAI Platform](https://platform.openai.com/)
2. Sign in or create an account
3. Navigate to API keys section
4. Create a new secret key
5. Copy and save it securely

#### Supabase Credentials
1. Go to your Supabase project dashboard
2. Navigate to Settings > API
3. Copy:
   - **Project URL** (looks like `https://xxxxx.supabase.co`)
   - **Service Role Key** (under "Service role" - has full access)

### Security Considerations

- **Never commit** your `.env` file to version control
- Use strong, unique values for `MCP_SERVER_API_KEY`
- Keep your API keys secure and rotate them regularly
- Use environment-specific configurations for different deployments

## Verifying Installation

### For Docker Installation

Check if the container is running:
```bash
docker ps | grep mcp/crawl4ai-rag
```

Test the health endpoint:
```bash
curl http://localhost:8051/health
```

### For Python Installation

Run the server directly:
```bash
python src/crawl4ai_mcp.py
```

You should see:
```
INFO: MCP_SERVER_API_KEY is set. API Key authentication enabled.
INFO: Server starting on http://0.0.0.0:8051
```

## Platform-Specific Notes

### Windows
- Use PowerShell or Command Prompt as Administrator for Docker commands
- Ensure Windows Subsystem for Linux (WSL2) is enabled for Docker Desktop
- Use backslashes in file paths or quote paths with spaces

### macOS
- Install Xcode Command Line Tools if prompted: `xcode-select --install`
- For Apple Silicon (M1/M2), Docker runs natively without emulation
- May need to allow Docker through Security & Privacy settings

### Linux
- Add your user to the docker group: `sudo usermod -aG docker $USER`
- Log out and back in for group changes to take effect
- SELinux users may need to add `:Z` flag for volume mounts

## Troubleshooting Installation

### Common Issues

1. **"OPENAI_API_KEY not set" error**:
   - Ensure the `.env` file is in the project root
   - Check that the key is properly quoted if it contains special characters
   - Verify the key is valid at [OpenAI Platform](https://platform.openai.com/)

2. **Docker build fails**:
   - Ensure Docker daemon is running
   - Check available disk space
   - Try `docker system prune` to clean up old images

3. **"ModuleNotFoundError" when running Python**:
   - Ensure virtual environment is activated
   - Re-run `uv pip install -e .` or `pip install -e .`
   - Check Python version is 3.12 or higher

4. **Supabase connection errors**:
   - Verify the URL format (should start with `https://`)
   - Ensure you're using the service role key, not the anon key
   - Check if the database setup SQL was run successfully

See the [Troubleshooting Guide](troubleshooting.md) for more detailed solutions.

## Next Steps

Once installation is complete:
1. Review the [Usage Guide](usage.md) to learn how to use the server
2. Check [API Reference](api.md) for detailed tool documentation
3. Set up [MCP client integration](usage.md#integration-with-mcp-clients)