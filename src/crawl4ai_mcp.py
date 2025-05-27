"""
MCP server for web crawling with Crawl4AI.

This server provides tools to crawl websites using Crawl4AI, automatically detecting
the appropriate crawl method based on URL type (sitemap, txt file, or regular webpage).
"""
from mcp.server.fastmcp import Context
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import List, Dict, Any
from urllib.parse import urlparse, urldefrag, urljoin
from xml.etree import ElementTree
from dotenv import load_dotenv
from supabase import Client
from pathlib import Path
import httpx
import asyncio
import json
import os
import re
import gzip
import threading
import logging
from datetime import datetime, timezone

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode, MemoryAdaptiveDispatcher
from utils import get_supabase_client, add_documents_to_supabase, search_documents
from auth_mcp import AuthenticatedFastMCP

# Configuration constants
DEFAULT_MEMORY_THRESHOLD = float(os.getenv("CRAWLER_MEMORY_THRESHOLD", "70.0"))
DEFAULT_CHECK_INTERVAL = float(os.getenv("CRAWLER_CHECK_INTERVAL", "1.0"))
MAX_DOCUMENT_LENGTH = int(os.getenv("MAX_DOCUMENT_LENGTH", "25000"))
DOCUMENT_BATCH_SIZE = int(os.getenv("DOCUMENT_BATCH_SIZE", "10"))  # Process N documents at a time
SITEMAP_MAX_DEPTH = int(os.getenv("SITEMAP_MAX_DEPTH", "2"))  # Max recursion depth for sitemap indexes

# Load environment variables from the project root .env file
project_root = Path(__file__).resolve().parent.parent
dotenv_path = project_root / '.env'

# Load environment variables without overriding system-set ones
load_dotenv(dotenv_path, override=False)

# Create a dataclass for our application context
@dataclass
class Crawl4AIContext:
    """Context for the Crawl4AI MCP server."""
    crawler: AsyncWebCrawler
    supabase_client: Client
    
@asynccontextmanager
async def crawl4ai_lifespan(server: AuthenticatedFastMCP) -> AsyncIterator[Crawl4AIContext]:
    """
    Manages the Crawl4AI client lifecycle.
    
    Args:
        server: The FastMCP server instance
        
    Yields:
        Crawl4AIContext: The context containing the Crawl4AI crawler and Supabase client
    """
    # Create browser configuration
    browser_config = BrowserConfig(
        headless=True,
        verbose=False
    )
    
    # Initialize the crawler
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.__aenter__()
    
    # Initialize Supabase client
    supabase_client = get_supabase_client()
    
    try:
        yield Crawl4AIContext(
            crawler=crawler,
            supabase_client=supabase_client
        )
    finally:
        # Clean up the crawler with proper error handling
        try:
            await crawler.__aexit__(None, None, None)
        except Exception as e:
            logger.error(f"Error during crawler cleanup: {e}")
            # Continue cleanup even if crawler cleanup fails

# Initialize AuthenticatedFastMCP server
mcp = AuthenticatedFastMCP(
    "mcp-crawl4ai-rag",
    description="Web crawler and RAG system. Crawl websites intelligently (auto-restricts to subdirectories), store content with embeddings, and search with semantic queries. Perfect for indexing documentation sites, building knowledge bases, and Q&A over web content.",
    lifespan=crawl4ai_lifespan,
    host=os.getenv("HOST", "0.0.0.0"),
    port=int(os.getenv("PORT", "8051"))
)

def normalize_url_for_comparison(url: str) -> str:
    """
    Normalize URL for consistent comparison.
    Removes trailing slashes and protocol for comparison.
    
    Args:
        url: URL to normalize
        
    Returns:
        Normalized URL without protocol and trailing slash
    """
    # Remove trailing slash
    url = url.rstrip('/')
    # Remove protocol for comparison
    return url.replace('https://', '').replace('http://', '')

def is_binary_file(url: str) -> bool:
    """
    Check if URL points to a binary file that shouldn't be crawled.
    
    Args:
        url: URL to check
        
    Returns:
        True if the URL points to a binary file, False otherwise
    """
    binary_extensions = [
        '.zip', '.gz', '.tar', '.rar', '.7z',  # Archives
        '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',  # Documents
        '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.ico',  # Images
        '.mp3', '.mp4', '.avi', '.mov', '.wmv', '.flv',  # Media
        '.exe', '.dmg', '.pkg', '.deb', '.rpm',  # Executables
        '.jar', '.war', '.ear',  # Java archives
        '.woff', '.woff2', '.ttf', '.eot',  # Fonts
    ]
    url_lower = url.lower()
    # Special handling for .xml.gz files (keep them as they're often sitemaps)
    if url_lower.endswith('.xml.gz'):
        return False
    return any(url_lower.endswith(ext) for ext in binary_extensions)

def is_sitemap(url: str) -> bool:
    """
    Check if a URL is a sitemap.
    
    Args:
        url: URL to check
        
    Returns:
        True if the URL is a sitemap, False otherwise
    """
    url_lower = url.lower()
    # Check for common sitemap patterns
    return (url_lower.endswith('sitemap.xml') or 
            url_lower.endswith('sitemap.xml.gz') or
            url_lower.endswith('sitemap_index.xml') or
            'sitemap' in urlparse(url_lower).path.lower())

def is_txt(url: str) -> bool:
    """
    Check if a URL is a text file.
    
    Args:
        url: URL to check
        
    Returns:
        True if the URL is a text file, False otherwise
    """
    return url.endswith('.txt')

async def parse_sitemap(sitemap_url: str, max_depth: int = SITEMAP_MAX_DEPTH, current_depth: int = 0) -> List[str]:
    """
    Parse a sitemap and extract URLs. Handles gzipped sitemaps and sitemap index files.
    
    Args:
        sitemap_url: URL of the sitemap
        max_depth: Maximum depth for following sitemap index files
        current_depth: Current recursion depth
        
    Returns:
        List of URLs found in the sitemap
    """
    if current_depth >= max_depth:
        return []
        
    try:
        headers = {'Accept-Encoding': 'gzip'}
        async with httpx.AsyncClient() as client:
            resp = await client.get(sitemap_url, timeout=30, headers=headers)
        urls = []

        if resp.status_code == 200:
            try:
                # Check if content is gzipped
                content = resp.content
                if sitemap_url.endswith('.gz') or resp.headers.get('Content-Encoding') == 'gzip':
                    try:
                        content = gzip.decompress(content)
                    except gzip.BadGzipFile:
                        # Not actually gzipped, use original content
                        pass
                
                tree = ElementTree.fromstring(content)
                
                # Check if this is a sitemap index file
                sitemap_nodes = tree.findall('.//{*}sitemap')
                if sitemap_nodes:
                    # This is a sitemap index - recursively parse each sitemap
                    for sitemap in sitemap_nodes:
                        loc = sitemap.find('./{*}loc')
                        if loc is not None and loc.text:
                            sub_urls = await parse_sitemap(loc.text, max_depth, current_depth + 1)
                            urls.extend(sub_urls)
                else:
                    # Regular sitemap - extract URLs
                    urls = [loc.text for loc in tree.findall('.//{*}loc') if loc.text]
            except Exception as e:
                logger.error(f"Error parsing sitemap XML from {sitemap_url}: {e}")
        else:
            logger.warning(f"Failed to fetch sitemap {sitemap_url}: HTTP {resp.status_code}")
    except httpx.RequestError as e:
        logger.error(f"Error fetching sitemap {sitemap_url}: {e}")
        urls = []

    return urls

def smart_chunk_markdown(text: str, chunk_size: int = 5000) -> List[str]:
    """
    Split text into chunks, respecting code blocks and paragraphs.
    
    Args:
        text: Markdown text to chunk
        chunk_size: Target size for each chunk
        
    Returns:
        List of text chunks
        
    Raises:
        ValueError: If chunk_size is invalid
    """
    if chunk_size <= 0:
        raise ValueError(f"Invalid chunk_size: {chunk_size}")
        
    if not text:
        return []
        
    chunks = []
    start = 0
    text_length = len(text)
    max_iterations = text_length // 100 + 1000  # Prevent infinite loops

    iteration = 0
    while start < text_length:
        # Calculate end position
        end = start + chunk_size

        # If we're at the end of the text, just take what's left
        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        chunk = text[start:end]
        
        # Check if we're inside a code block
        # Count opening and closing ``` to see if we're in a code block
        code_blocks_before = text[:end].count('```')
        if code_blocks_before % 2 == 1:
            # We're inside a code block, find the closing ```
            closing_block = text.find('```', end)
            if closing_block != -1:
                # Include the closing ``` in this chunk
                end = closing_block + 3
            # If no closing block found, fall through to other break points
        else:
            # Try to find a code block boundary first (```)
            code_block = chunk.rfind('```')
            if code_block != -1 and code_block > chunk_size * 0.3:
                # Check if this is an opening or closing block
                blocks_in_chunk = chunk[:code_block].count('```')
                if blocks_in_chunk % 2 == 0:
                    # This is an opening block, don't break here
                    # Look for the previous code block or paragraph break
                    prev_block = chunk[:code_block].rfind('```')
                    if prev_block != -1 and prev_block > chunk_size * 0.3:
                        end = start + prev_block + 3  # Include the ```
                    elif '\n\n' in chunk[:code_block]:
                        last_break = chunk[:code_block].rfind('\n\n')
                        if last_break > chunk_size * 0.3:
                            end = start + last_break
                else:
                    # This is a closing block, break after it
                    end = start + code_block + 3  # Include the ```

            # If no code block, try to break at a paragraph
            elif '\n\n' in chunk:
                # Find the last paragraph break
                last_break = chunk.rfind('\n\n')
                if last_break > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                    end = start + last_break

            # If no paragraph break, try to break at a sentence
            elif '. ' in chunk:
                # Find the last sentence break
                last_period = chunk.rfind('. ')
                if last_period > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                    end = start + last_period + 1

        # Extract chunk and clean it up
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start position for next chunk
        start = end
        
        # Safety check to prevent infinite loops
        iteration += 1
        if iteration > max_iterations:
            logger.error(f"Maximum iterations ({max_iterations}) exceeded in smart_chunk_markdown. Breaking loop.")
            # Add remaining text as final chunk if any
            if start < text_length:
                chunks.append(text[start:].strip())
            break

    return chunks

def extract_section_info(chunk: str) -> Dict[str, Any]:
    """
    Extracts headers and stats from a chunk.
    
    Args:
        chunk: Markdown chunk
        
    Returns:
        Dictionary with headers and stats
    """
    headers = re.findall(r'^(#+)\s+(.+)$', chunk, re.MULTILINE)
    header_str = '; '.join([f'{h[0]} {h[1]}' for h in headers]) if headers else ''

    return {
        "headers": header_str,
        "char_count": len(chunk),
        "word_count": len(chunk.split())
    }

async def process_crawl_results_batch(
    supabase_client: Client,
    crawl_results_batch: List[Dict[str, Any]],
    chunk_size: int,
    crawl_type: str
) -> Dict[str, int]:
    """
    Process a batch of crawl results and store in Supabase.
    Memory-efficient by processing documents in smaller batches.
    
    Args:
        supabase_client: Supabase client
        crawl_results_batch: Batch of crawl results to process
        chunk_size: Size for text chunking
        crawl_type: Type of crawl (webpage, sitemap, text_file)
        
    Returns:
        Dictionary with detailed statistics:
        - chunks_stored: Number of chunks successfully stored
        - chunks_prepared: Total chunks prepared for storage
        - pages_processed: Number of pages in this batch
        - storage_errors: Number of storage errors
    """
    urls = []
    chunk_numbers = []
    contents = []
    metadatas = []
    url_to_full_document = {}
    
    # Process each document in the batch
    for doc in crawl_results_batch:
        source_url = doc['url']
        md = doc['markdown']
        
        # Store full document for contextual embeddings
        url_to_full_document[source_url] = md
        
        # Chunk the document
        chunks = smart_chunk_markdown(md, chunk_size=chunk_size)
        
        for i, chunk in enumerate(chunks):
            urls.append(source_url)
            chunk_numbers.append(i)
            contents.append(chunk)
            
            # Extract metadata
            meta = extract_section_info(chunk)
            meta["chunk_index"] = i
            meta["url"] = source_url
            meta["source"] = urlparse(source_url).netloc
            meta["crawl_type"] = crawl_type
            meta["crawl_time"] = datetime.now(timezone.utc).isoformat()
            metadatas.append(meta)
    
    # Store in Supabase
    stats = {
        "chunks_stored": 0,
        "chunks_prepared": len(urls),
        "pages_processed": len(crawl_results_batch),
        "storage_errors": 0
    }
    
    if urls:
        try:
            chunks_stored = add_documents_to_supabase(
                supabase_client, urls, chunk_numbers, contents, 
                metadatas, url_to_full_document
            )
            stats["chunks_stored"] = chunks_stored
            if chunks_stored < stats["chunks_prepared"]:
                stats["storage_errors"] = stats["chunks_prepared"] - chunks_stored
        except Exception as e:
            logger.error(f"Error storing batch: {e}")
            stats["storage_errors"] = stats["chunks_prepared"]
            # Don't re-raise, let the caller handle partial failures
    
    return stats

@mcp.tool()
async def crawl_single_page(ctx: Context, url: str) -> str:
    """
    Crawl a single web page without following any links and store its content for RAG queries.
    
    Use this tool when you need to:
    - Quickly grab content from one specific page
    - Store a single article or documentation page
    - Get content without recursive crawling
    
    The content is automatically chunked and stored with embeddings for later semantic search.
    
    Args:
        ctx: The MCP server provided context
        url: The exact URL of the web page to crawl (e.g., "https://example.com/article.html")
    
    Returns:
        JSON with success status, chunks stored, content length, and link counts
    
    Example:
        crawl_single_page(url="https://docs.example.com/api/authentication")
    """
    try:
        # Validate URL
        if not url or not url.strip():
            return json.dumps({
                "success": False,
                "error": "URL is required"
            }, indent=2)
            
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return json.dumps({
                    "success": False,
                    "error": f"Invalid URL format: {url}"
                }, indent=2)
            if parsed.scheme not in ['http', 'https']:
                return json.dumps({
                    "success": False,
                    "error": f"Unsupported URL scheme: {parsed.scheme}"
                }, indent=2)
        except Exception as e:
            return json.dumps({
                "success": False,
                "error": f"Invalid URL: {str(e)}"
            }, indent=2)
        
        # Get the crawler from the context
        crawler = ctx.request_context.lifespan_context.crawler
        supabase_client = ctx.request_context.lifespan_context.supabase_client
        
        # Configure the crawl
        run_config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS, 
            stream=False
        )
        
        # Crawl the page
        result = await crawler.arun(url=url, config=run_config)
        
        if result.success and result.markdown:
            # Chunk the content
            chunks = smart_chunk_markdown(result.markdown)
            
            # Prepare data for Supabase
            urls = []
            chunk_numbers = []
            contents = []
            metadatas = []
            
            for i, chunk in enumerate(chunks):
                urls.append(url)
                chunk_numbers.append(i)
                contents.append(chunk)
                
                # Extract metadata
                meta = extract_section_info(chunk)
                meta["chunk_index"] = i
                meta["url"] = url
                meta["source"] = urlparse(url).netloc
                meta["crawl_time"] = datetime.now(timezone.utc).isoformat()
                metadatas.append(meta)
            
            # Create url_to_full_document mapping
            url_to_full_document = {url: result.markdown}
            
            # Add to Supabase
            add_documents_to_supabase(supabase_client, urls, chunk_numbers, contents, metadatas, url_to_full_document)
            
            return json.dumps({
                "success": True,
                "url": url,
                "chunks_stored": len(chunks),
                "content_length": len(result.markdown),
                "links_count": {
                    "internal": len(result.links.get("internal", [])),
                    "external": len(result.links.get("external", []))
                }
            }, indent=2)
        else:
            return json.dumps({
                "success": False,
                "url": url,
                "error": result.error_message
            }, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "url": url,
            "error": str(e)
        }, indent=2)

@mcp.tool()
async def smart_crawl_url(ctx: Context, url: str, max_depth: int = 3, max_concurrent: int = 10, chunk_size: int = 5000, prefix: str = None, sitemap_max_depth: int = None) -> str:
    """
    Intelligently crawl websites and store all content for RAG queries. Auto-detects sitemaps and follows links.
    
    USE THIS TOOL WHEN:
    - User asks to "crawl a website" or "index documentation"
    - You need to get ALL content from a site or subdirectory
    - Building a knowledge base from online documentation
    - The URL might be a sitemap.xml or regular webpage
    
    CRAWLING BEHAVIOR:
    - Sitemap URLs: Extracts and crawls all URLs in parallel
    - Text files (.txt): Directly retrieves the content
    - Webpages: Recursively follows internal links up to max_depth
    
    BOUNDARY CONTROL WITH PREFIX:
    - Default behavior: Auto-restricts to current subdirectory (e.g., https://dox.example.com/ stays within dox.example.com)
    - No subdirectory in URL: Crawls entire domain
    - Custom prefix: Only crawls URLs starting with that prefix
    - Single page: Set prefix = url to index only that page
    - Protocol-agnostic: HTTP and HTTPS are treated as equivalent (http://example.com matches https://example.com)
    - Note: The crawler follows ALL links (internal and external) but filters them by prefix
    
    Args:
        ctx: The MCP server provided context
        url: Starting URL (entrypoint) - can be webpage, sitemap.xml, or .txt file
        max_depth: How many link levels deep to crawl (default: 3, use 1 for shallow crawl)
        max_concurrent: Parallel browser sessions (default: 10, reduce if rate limited)
        chunk_size: Characters per chunk for storage (default: 5000)
        prefix: Optional - Only crawl URLs starting with this prefix. Defaults to current subdirectory.
        sitemap_max_depth: Maximum recursion depth for nested sitemap indexes (default: from SITEMAP_MAX_DEPTH env var)
    
    Returns:
        JSON with pages crawled, chunks stored, and list of URLs processed
    
    Examples:
        # Crawl /api/ subdirectory (auto-detected prefix)
        smart_crawl_url(url="https://docs.example.com/api/")
        
        # Crawl entire domain (override default subdirectory restriction)
        smart_crawl_url(url="https://docs.example.com/api/", prefix="https://docs.example.com/")
        
        # Index single page only
        smart_crawl_url(url="https://example.com/guide.html", prefix="https://example.com/guide.html")
        
        # Start from homepage but only follow links in /blog/
        smart_crawl_url(url="https://example.com", prefix="https://example.com/blog")
    """
    try:
        # Validate inputs
        if not url or not url.strip():
            return json.dumps({
                "success": False,
                "error": "URL is required"
            }, indent=2)
            
        # Validate URL format
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return json.dumps({
                    "success": False,
                    "error": f"Invalid URL format: {url}"
                }, indent=2)
            if parsed.scheme not in ['http', 'https']:
                return json.dumps({
                    "success": False,
                    "error": f"Unsupported URL scheme: {parsed.scheme}"
                }, indent=2)
        except Exception as e:
            return json.dumps({
                "success": False,
                "error": f"Invalid URL: {str(e)}"
            }, indent=2)
            
        if max_depth < 0:
            max_depth = 0
        elif max_depth > 10:
            logger.warning(f"max_depth {max_depth} is very high, capping at 10")
            max_depth = 10
            
        if max_concurrent <= 0:
            logger.warning(f"Invalid max_concurrent {max_concurrent}, using default 10")
            max_concurrent = 10
        elif max_concurrent > 50:
            logger.warning(f"max_concurrent {max_concurrent} is very high, capping at 50")
            max_concurrent = 50
            
        if chunk_size <= 0:
            logger.warning(f"Invalid chunk_size {chunk_size}, using default 5000")
            chunk_size = 5000
        
        # Get the crawler and Supabase client from the context
        crawler = ctx.request_context.lifespan_context.crawler
        supabase_client = ctx.request_context.lifespan_context.supabase_client
        
        # Set default prefix to subdirectory if not provided or empty
        if not prefix:
            # Extract the subdirectory path from the URL
            parsed = urlparse(url)
            path = parsed.path.rstrip('/')
            
            # If there's a path (not just domain root), use it as prefix
            if path and path != '/':
                # Find the last directory in the path (remove filename if present)
                # Check if the last segment looks like a file (has extension after last /)
                last_segment = path.split('/')[-1]
                if last_segment and '.' in last_segment and not last_segment.startswith('.'):
                    # Likely a file - use parent directory
                    path = '/'.join(path.split('/')[:-1])
                prefix = f"{parsed.scheme}://{parsed.netloc}{path}"
            else:
                # No subdirectory, use the full domain
                prefix = f"{parsed.scheme}://{parsed.netloc}"
        
        # No validation needed - prefix is a filter for discovered links, not a constraint on the starting URL
        # This allows crawling from any entry point but restricting discovered links to a specific path
        
        crawl_results = []
        crawl_type = "webpage"
        
        # Detect URL type and use appropriate crawl method
        if is_txt(url):
            # For text files, use simple crawl
            crawl_results = await crawl_markdown_file(crawler, url)
            crawl_type = "text_file"
        elif is_sitemap(url):
            # For sitemaps, extract URLs and crawl in parallel
            sitemap_urls = await parse_sitemap(url, max_depth=sitemap_max_depth if sitemap_max_depth is not None else SITEMAP_MAX_DEPTH)
            if not sitemap_urls:
                return json.dumps({
                    "success": False,
                    "url": url,
                    "error": "No URLs found in sitemap"
                }, indent=2)
            
            # Apply prefix filter to sitemap URLs if provided
            if prefix:
                prefix_normalized = normalize_url_for_comparison(prefix)
                filtered_urls = []
                for sitemap_url in sitemap_urls:
                    if normalize_url_for_comparison(sitemap_url).startswith(prefix_normalized):
                        filtered_urls.append(sitemap_url)
                sitemap_urls = filtered_urls
                
                if not sitemap_urls:
                    return json.dumps({
                        "success": False,
                        "url": url,
                        "error": f"No URLs in sitemap match the prefix '{prefix}'"
                    }, indent=2)
            
            crawl_results = await crawl_batch(crawler, sitemap_urls, max_concurrent=max_concurrent)
            crawl_type = "sitemap"
        else:
            # For regular URLs, use recursive crawl
            crawl_results = await crawl_recursive_internal_links(crawler, [url], max_depth=max_depth, max_concurrent=max_concurrent, prefix=prefix)
            crawl_type = "webpage"
        
        if not crawl_results:
            return json.dumps({
                "success": False,
                "url": url,
                "error": "No content found"
            }, indent=2)
        
        # Process results in batches for memory efficiency
        total_chunks_stored = 0
        total_chunks_prepared = 0
        pages_processed = 0
        total_storage_errors = 0
        batch_failures = 0
        
        # Process documents in batches
        for batch_start in range(0, len(crawl_results), DOCUMENT_BATCH_SIZE):
            batch_end = min(batch_start + DOCUMENT_BATCH_SIZE, len(crawl_results))
            batch = crawl_results[batch_start:batch_end]
            
            logger.debug(f"Processing document batch {batch_start//DOCUMENT_BATCH_SIZE + 1}: documents {batch_start+1}-{batch_end} of {len(crawl_results)}")
            
            try:
                # Process and store the batch
                batch_stats = await process_crawl_results_batch(
                    supabase_client, batch, chunk_size, crawl_type
                )
                
                # Aggregate statistics
                total_chunks_stored += batch_stats["chunks_stored"]
                total_chunks_prepared += batch_stats["chunks_prepared"]
                pages_processed += batch_stats["pages_processed"]
                total_storage_errors += batch_stats["storage_errors"]
                
                logger.debug(f"Batch {batch_start//DOCUMENT_BATCH_SIZE + 1} complete: {batch_stats['chunks_stored']}/{batch_stats['chunks_prepared']} chunks stored")
                
                if batch_stats["storage_errors"] > 0:
                    batch_failures += 1
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_start//DOCUMENT_BATCH_SIZE + 1}: {e}")
                batch_failures += 1
                # Continue with next batch even if one fails
                continue
        
        # Check if any chunks were stored
        if total_chunks_stored == 0 and total_chunks_prepared > 0:
            logger.error(f"Failed to store any chunks for {url}")
            return json.dumps({
                "success": False,
                "url": url,
                "error": "Failed to store any content in database"
            }, indent=2)
        
        # Prepare response with detailed statistics
        response = {
            "success": True,
            "url": url,
            "crawl_type": crawl_type,
            "pages_crawled": len(crawl_results),
            "pages_processed": pages_processed,
            "chunks_prepared": total_chunks_prepared,
            "chunks_stored": total_chunks_stored,
            "urls_crawled": [doc['url'] for doc in crawl_results][:5] + (["..."] if len(crawl_results) > 5 else [])
        }
        
        # Add partial failure details if any
        if total_storage_errors > 0 or batch_failures > 0:
            response["partial_failures"] = {
                "storage_errors": total_storage_errors,
                "failed_batches": batch_failures,
                "total_batches": (len(crawl_results) + DOCUMENT_BATCH_SIZE - 1) // DOCUMENT_BATCH_SIZE,
                "success_rate": f"{(total_chunks_stored / total_chunks_prepared * 100):.1f}%" if total_chunks_prepared > 0 else "0%"
            }
            
        return json.dumps(response, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "url": url,
            "error": str(e)
        }, indent=2)

async def crawl_markdown_file(crawler: AsyncWebCrawler, url: str) -> List[Dict[str, Any]]:
    """
    Crawl a .txt or markdown file.
    
    Args:
        crawler: AsyncWebCrawler instance
        url: URL of the file
        
    Returns:
        List of dictionaries with URL and markdown content
    """
    crawl_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS, 
        stream=False
    )

    result = await crawler.arun(url=url, config=crawl_config)
    if result.success and result.markdown:
        return [{'url': url, 'markdown': result.markdown}]
    else:
        logger.error(f"Failed to crawl {url}: {result.error_message}")
        return []

async def crawl_batch(crawler: AsyncWebCrawler, urls: List[str], max_concurrent: int = 10) -> List[Dict[str, Any]]:
    """
    Batch crawl multiple URLs in parallel.
    
    Args:
        crawler: AsyncWebCrawler instance
        urls: List of URLs to crawl
        max_concurrent: Maximum number of concurrent browser sessions
        
    Returns:
        List of dictionaries with URL and markdown content
    """
    # Filter out binary files
    filtered_urls = [url for url in urls if not is_binary_file(url)]
    skipped_count = len(urls) - len(filtered_urls)
    if skipped_count > 0:
        logger.debug(f"Skipping {skipped_count} binary files from batch crawl")
    
    crawl_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS, 
        stream=False,
        wait_until="domcontentloaded"  # Don't wait for all resources
    )
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=DEFAULT_MEMORY_THRESHOLD,
        check_interval=DEFAULT_CHECK_INTERVAL,
        max_session_permit=max_concurrent
    )

    results = await crawler.arun_many(urls=filtered_urls, config=crawl_config, dispatcher=dispatcher)
    successful_results = []
    for r in results:
        if r.success and r.markdown:
            successful_results.append({'url': r.url, 'markdown': r.markdown})
            logger.debug(f"Successfully crawled {r.url}")
        else:
            logger.warning(f"Failed to crawl {r.url}: {getattr(r, 'error_message', 'Unknown error')}")
    
    return successful_results

async def crawl_recursive_internal_links(crawler: AsyncWebCrawler, start_urls: List[str], max_depth: int = 3, max_concurrent: int = 10, prefix: str = None) -> List[Dict[str, Any]]:
    """
    Recursively crawl internal links from start URLs up to a maximum depth.
    
    Args:
        crawler: AsyncWebCrawler instance
        start_urls: List of starting URLs
        max_depth: Maximum recursion depth
        max_concurrent: Maximum number of concurrent browser sessions
        prefix: Optional URL prefix to restrict crawling scope
        
    Returns:
        List of dictionaries with URL and markdown content
    """
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS, 
        stream=False,
        wait_until="domcontentloaded"  # Don't wait for all resources
    )
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=DEFAULT_MEMORY_THRESHOLD,
        check_interval=DEFAULT_CHECK_INTERVAL,
        max_session_permit=max_concurrent
    )

    # Thread-safe set for visited URLs
    visited = set()
    visited_lock = threading.Lock()

    def normalize_url(url):
        return urldefrag(url)[0]
    

    # Filter out binary files from start URLs
    start_urls = [url for url in start_urls if not is_binary_file(url)]
    current_urls = set([normalize_url(u) for u in start_urls])
    results_all = []

    for depth in range(max_depth):
        logger.debug(f"Crawling depth {depth}, {len(current_urls)} URLs to process")
        
        # Thread-safe check for unvisited URLs
        with visited_lock:
            urls_to_crawl = [url for url in current_urls if url not in visited and not is_binary_file(url)]
            # Mark URLs as visited immediately to prevent duplicates
            for url in urls_to_crawl:
                visited.add(url)
                
        if not urls_to_crawl:
            logger.debug(f"No more URLs to crawl at depth {depth}")
            break

        logger.debug(f"Crawling {len(urls_to_crawl)} URLs at depth {depth}")
        results = await crawler.arun_many(urls=urls_to_crawl, config=run_config, dispatcher=dispatcher)
        next_level_urls = set()

        for result in results:
            if result.success and result.markdown:
                results_all.append({'url': result.url, 'markdown': result.markdown})
                logger.debug(f"Successfully crawled {result.url}")
                
                # Only check internal links from the same domain
                internal_links = result.links.get("internal", [])
                
                # Extract target domain from prefix for additional filtering
                prefix_parsed = urlparse(prefix)
                target_domain = prefix_parsed.netloc
                
                for link in internal_links:
                    # Handle both dict and string formats
                    if isinstance(link, dict):
                        link_href = link.get("href", "")
                    elif isinstance(link, str):
                        link_href = link
                    else:
                        continue
                        
                    if not link_href:
                        continue
                        
                    # Handle relative URLs by making them absolute
                    next_url = normalize_url(urljoin(result.url, link_href))
                    
                    # Skip binary files
                    if is_binary_file(next_url):
                        logger.debug(f"Skipping binary file: {next_url}")
                        continue
                    
                    # First check if the link is from the same domain
                    next_parsed = urlparse(next_url)
                    if next_parsed.netloc != target_domain:
                        continue
                    
                    # Always apply prefix filter (it's always set in smart_crawl_url)
                    # Use consistent normalization
                    if not normalize_url_for_comparison(next_url).startswith(normalize_url_for_comparison(prefix)):
                        continue
                    
                    # Thread-safe check before adding to next level
                    with visited_lock:
                        if next_url not in visited:
                            next_level_urls.add(next_url)
            else:
                logger.warning(f"Failed to crawl {result.url}: {getattr(result, 'error_message', 'Unknown error')}")

        current_urls = next_level_urls
        logger.debug(f"Found {len(next_level_urls)} new URLs for next depth")

    logger.info(f"Crawl complete. Total pages crawled: {len(results_all)}")
    return results_all

@mcp.tool()
async def get_available_sources(ctx: Context) -> str:
    """
    List all domains/sources that have been crawled and are available for RAG queries.
    
    USE THIS TOOL WHEN:
    - User asks "what sources are available?" or "what have you crawled?"
    - Before performing RAG queries to see available domains
    - You need to filter RAG results by source
    - Checking if a specific domain has been indexed
    
    Returns:
        JSON with list of source domains and count
    
    Example response:
        {
            "success": true,
            "sources": ["docs.example.com", "api.example.com", "blog.example.com"],
            "count": 3
        }
    """
    try:
        # Get the Supabase client from the context
        supabase_client = ctx.request_context.lifespan_context.supabase_client
        
        # Use a more efficient query - select distinct sources only
        # This is much faster than fetching all records
        result = supabase_client.from_('crawled_pages')\
            .select('metadata->>source')\
            .not_.is_('metadata->>source', 'null')\
            .limit(1000)\
            .execute()
            
        # Use a set to efficiently track unique sources
        unique_sources = set()
        
        # Extract the source values from the result
        if result.data:
            for item in result.data:
                if item and item.get('source'):
                    unique_sources.add(item['source'])
        
        # Convert set to sorted list for consistent output
        sources = sorted(list(unique_sources))
        
        return json.dumps({
            "success": True,
            "sources": sources,
            "count": len(sources)
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, indent=2)

@mcp.tool()
async def perform_rag_query(ctx: Context, query: str, source: str = None, match_count: int = 5) -> str:
    """
    Search crawled content using semantic similarity (RAG). Returns relevant chunks matching your query.
    
    USE THIS TOOL WHEN:
    - User asks questions about crawled content
    - Looking for specific information in documentation
    - Need code examples or API references
    - Searching across multiple crawled pages
    
    The tool uses embeddings to find semantically similar content, so it understands context and meaning.
    
    Args:
        ctx: The MCP server provided context
        query: Natural language search query (e.g., "how to authenticate with API" or "error handling examples")
        source: Optional - Filter results to specific domain (e.g., "docs.example.com")
        match_count: Number of relevant chunks to return (default: 5, max recommended: 10)
    
    Returns:
        JSON with matched content chunks, URLs, metadata, and similarity scores
    
    Examples:
        # Search all sources
        perform_rag_query(query="how to handle rate limits")
        
        # Search only docs.example.com
        perform_rag_query(query="authentication methods", source="docs.example.com")
        
        # Get more results
        perform_rag_query(query="error codes", match_count=10)
    """
    try:
        # Get the Supabase client from the context
        supabase_client = ctx.request_context.lifespan_context.supabase_client
        
        # Prepare filter if source is provided and not empty
        filter_metadata = None
        if source and source.strip():
            filter_metadata = {"source": source}
        
        # Perform the search
        results = search_documents(
            client=supabase_client,
            query=query,
            match_count=match_count,
            filter_metadata=filter_metadata
        )
        
        # Format the results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "url": result.get("url"),
                "content": result.get("content"),
                "metadata": result.get("metadata"),
                "similarity": result.get("similarity")
            })
        
        return json.dumps({
            "success": True,
            "query": query,
            "source_filter": source,
            "results": formatted_results,
            "count": len(formatted_results)
        }, indent=2)
    except ValueError as e:
        # Query validation errors
        return json.dumps({
            "success": False,
            "query": query,
            "error": f"Invalid query: {str(e)}",
            "error_type": "validation_error"
        }, indent=2)
    except RuntimeError as e:
        # API or database errors
        return json.dumps({
            "success": False,
            "query": query,
            "error": str(e),
            "error_type": "runtime_error"
        }, indent=2)
    except Exception as e:
        # Unexpected errors
        return json.dumps({
            "success": False,
            "query": query,
            "error": f"Unexpected error: {str(e)}",
            "error_type": "unknown_error"
        }, indent=2)

async def main():
    transport = os.getenv("TRANSPORT", "sse")
    if transport == 'sse':
        # Run the MCP server with sse transport
        await mcp.run_sse_async()
    else:
        # Run the MCP server with stdio transport
        await mcp.run_stdio_async()

if __name__ == "__main__":
    asyncio.run(main())