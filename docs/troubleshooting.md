# Troubleshooting Guide

This guide helps you diagnose and resolve common issues with the Crawl4AI RAG MCP Server.

## Quick Diagnostics

Before diving into specific issues, run these checks:

```bash
# Check if server is running
curl http://localhost:8051/health

# Check environment variables
python -c "import os; print('OPENAI_API_KEY set:', bool(os.getenv('OPENAI_API_KEY')))"
python -c "import os; print('SUPABASE_URL set:', bool(os.getenv('SUPABASE_URL')))"

# Test database connection
python -c "from src.utils import get_supabase_client; client = get_supabase_client(); print('Connected to Supabase')"

# Check MCP connection (if using Claude)
claude mcp list
```

## Common Issues and Solutions

### Installation Issues

#### "ModuleNotFoundError: No module named 'crawl4ai'"

**Symptoms:**
```
ModuleNotFoundError: No module named 'crawl4ai'
```

**Solutions:**
1. Ensure virtual environment is activated:
   ```bash
   # Check current environment
   which python
   
   # Activate if needed
   source .venv/bin/activate  # or venv/bin/activate
   ```

2. Reinstall dependencies:
   ```bash
   uv pip install -e .
   # or
   pip install -e .
   ```

3. Run the crawl4ai setup:
   ```bash
   crawl4ai-setup
   ```

#### "crawl4ai-setup: command not found"

**Solutions:**
1. Install crawl4ai properly:
   ```bash
   pip install crawl4ai==0.6.2
   ```

2. Check if it's in PATH:
   ```bash
   pip show -f crawl4ai | grep crawl4ai-setup
   ```

### API Key Issues

#### "WARNING: OPENAI_API_KEY not set"

**Symptoms:**
- Embeddings fail to generate
- Search returns no results

**Solutions:**
1. Check .env file exists and contains key:
   ```bash
   cat .env | grep OPENAI_API_KEY
   ```

2. Ensure .env is in project root:
   ```bash
   ls -la | grep .env
   ```

3. For Docker, verify env file is passed:
   ```bash
   docker run --env-file .env ...
   ```

4. Test API key directly:
   ```python
   import openai
   openai.api_key = "your-key"
   response = openai.models.list()
   print("API key is valid")
   ```

#### "Invalid API Key" (MCP Authentication)

**Symptoms:**
```
event: error
data: {"error": "Invalid API Key"}
```

**Solutions:**
1. Verify MCP_SERVER_API_KEY matches:
   ```bash
   # In .env or environment
   echo $MCP_SERVER_API_KEY
   
   # In Claude Code
   claude mcp get crawl4ai-rag
   ```

2. Re-add with correct key:
   ```bash
   claude mcp remove crawl4ai-rag
   claude mcp add --transport sse crawl4ai-rag \
     -e X-API-Key=your_correct_key \
     http://localhost:8051/sse
   ```

### Database Connection Issues

#### "SUPABASE_URL and SUPABASE_SERVICE_KEY must be set"

**Solutions:**
1. Verify credentials in .env:
   ```bash
   # Should look like:
   SUPABASE_URL=https://xxxxxxxxxxxx.supabase.co
   SUPABASE_SERVICE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
   ```

2. Test connection:
   ```python
   from supabase import create_client
   url = "your-url"
   key = "your-key"
   client = create_client(url, key)
   # Should not raise error
   ```

#### "relation 'crawled_pages' does not exist"

**Solutions:**
1. Run the database setup SQL:
   - Go to Supabase SQL Editor
   - Paste contents of `crawled_pages.sql`
   - Execute the query

2. Verify table exists:
   ```sql
   SELECT table_name 
   FROM information_schema.tables 
   WHERE table_schema = 'public' 
   AND table_name = 'crawled_pages';
   ```

### Crawling Issues

#### Crawler Hangs or Times Out

**Symptoms:**
- No response after starting crawl
- Process seems stuck

**Solutions:**
1. Reduce timeout and concurrent sessions:
   ```python
   smart_crawl_url(
       url="https://slow-site.com",
       max_concurrent=3,  # Reduce from 10
   )
   ```

2. Set shorter timeout in environment:
   ```bash
   export CRAWLER_TIMEOUT=15000  # 15 seconds instead of 30
   ```

3. Check site accessibility:
   ```bash
   curl -I https://site-to-crawl.com
   ```

#### "No URLs found in sitemap"

**Solutions:**
1. Verify sitemap is accessible:
   ```bash
   curl https://example.com/sitemap.xml
   ```

2. Check if sitemap is gzipped:
   ```bash
   curl -H "Accept-Encoding: gzip" https://example.com/sitemap.xml.gz | gunzip
   ```

3. Try direct URL instead:
   ```python
   # Instead of sitemap
   smart_crawl_url(url="https://example.com/docs/")
   ```

#### Memory Issues During Crawling

**Symptoms:**
```
MemoryError: Unable to allocate memory
Process killed
```

**Solutions:**
1. Reduce concurrent crawls:
   ```bash
   export CRAWLER_MEMORY_THRESHOLD=50.0  # Lower threshold
   ```

2. Use smaller batches:
   ```python
   smart_crawl_url(
       url="https://large-site.com",
       max_concurrent=5,
       max_depth=1  # Shallow crawl
   )
   ```

3. Increase Docker memory:
   ```bash
   docker run -m 4g ...  # 4GB memory limit
   ```

### Search/RAG Issues

#### "Failed to create embedding"

**Solutions:**
1. Check OpenAI API status:
   ```bash
   curl https://status.openai.com/api/v2/status.json
   ```

2. Verify API limits:
   ```python
   # Check your usage at https://platform.openai.com/usage
   ```

3. Test embedding creation:
   ```python
   import openai
   openai.api_key = "your-key"
   response = openai.embeddings.create(
       model="text-embedding-3-small",
       input="test"
   )
   print("Embedding created successfully")
   ```

#### No Search Results

**Solutions:**
1. Verify content was indexed:
   ```python
   sources = get_available_sources()
   print(sources)  # Should show crawled domains
   ```

2. Check if content exists:
   ```sql
   -- In Supabase SQL Editor
   SELECT COUNT(*) FROM crawled_pages;
   SELECT DISTINCT metadata->>'source' FROM crawled_pages;
   ```

3. Try broader search:
   ```python
   # Remove source filter
   perform_rag_query(query="any content")
   
   # Increase match count
   perform_rag_query(query="test", match_count=10)
   ```

### Docker Issues

#### "Cannot connect to Docker daemon"

**Solutions:**
1. Start Docker:
   ```bash
   # macOS
   open -a Docker
   
   # Linux
   sudo systemctl start docker
   
   # Windows
   # Start Docker Desktop
   ```

2. Check Docker is running:
   ```bash
   docker ps
   ```

#### "Port already in use"

**Solutions:**
1. Find process using port:
   ```bash
   # macOS/Linux
   lsof -i :8051
   
   # Windows
   netstat -ano | findstr :8051
   ```

2. Use different port:
   ```bash
   export PORT=8052
   docker run -p 8052:8052 ...
   ```

### MCP Client Issues

#### "MCP server not responding"

**Solutions:**
1. Check server is running:
   ```bash
   ps aux | grep crawl4ai_mcp
   ```

2. Test SSE endpoint:
   ```bash
   curl -N http://localhost:8051/sse
   ```

3. Check Claude Code connection:
   ```
   # In Claude Code, type:
   /mcp
   ```

#### "Transport error" in Claude Code

**Solutions:**
1. Verify transport type:
   ```bash
   echo $TRANSPORT  # Should be "sse" or "stdio"
   ```

2. Re-add with correct transport:
   ```bash
   claude mcp remove crawl4ai-rag
   claude mcp add --transport sse crawl4ai-rag http://localhost:8051/sse
   ```

## Performance Issues

### Slow Crawling

**Diagnosis:**
```python
import time
start = time.time()
result = smart_crawl_url(url="https://example.com")
print(f"Crawl took {time.time() - start} seconds")
```

**Solutions:**
1. Use sitemap if available
2. Reduce max_depth
3. Increase max_concurrent (if not rate limited)
4. Check network speed

### High Memory Usage

**Monitor memory:**
```python
import psutil
print(f"Memory usage: {psutil.virtual_memory().percent}%")
```

**Solutions:**
1. Reduce chunk_size
2. Process in smaller batches
3. Restart server periodically

## Debugging Techniques

### Enable Verbose Logging

```python
# Add to crawl4ai_mcp.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Trace Requests

```python
# Add request tracing
@mcp.tool()
async def crawl_single_page(ctx: Context, url: str) -> str:
    print(f"DEBUG: Crawling {url}")
    # ... rest of function
```

### Test Individual Components

```python
# Test crawler independently
from crawl4ai import AsyncWebCrawler

async def test_crawler():
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun("https://example.com")
        print(f"Success: {result.success}")
        print(f"Content length: {len(result.markdown)}")

# Run test
import asyncio
asyncio.run(test_crawler())
```

## Getting Help

If these solutions don't resolve your issue:

1. **Collect diagnostic information**:
   ```bash
   # System info
   python --version
   pip list | grep -E "crawl4ai|mcp|supabase|openai"
   
   # Error logs
   # Copy full error message with stack trace
   ```

2. **Create detailed issue**:
   - Go to [GitHub Issues](https://github.com/coleam00/mcp-crawl4ai-rag/issues)
   - Use issue template
   - Include:
     - Steps to reproduce
     - Expected vs actual behavior
     - Environment details
     - Error messages

3. **Community support**:
   - Search existing issues first
   - Check discussions for similar problems
   - Provide minimal reproduction example

## Preventive Measures

### Regular Maintenance

1. **Update dependencies**:
   ```bash
   pip install --upgrade crawl4ai mcp supabase openai
   ```

2. **Monitor resources**:
   - Set up alerts for high memory/CPU
   - Track API usage and costs
   - Monitor crawl success rates

3. **Backup data**:
   - Regular Supabase backups
   - Export crawled content periodically

### Best Practices

1. **Test before production**:
   - Use test URLs first
   - Verify with small crawls
   - Check search results

2. **Set reasonable limits**:
   - max_depth: 1-3 for most cases
   - max_concurrent: 5-10
   - chunk_size: 3000-8000

3. **Handle errors gracefully**:
   - Check success field in responses
   - Implement retry logic
   - Log errors for debugging