"""
Utility functions for the Crawl4AI MCP server.
"""
import os
import time
import logging
import concurrent.futures
from typing import List, Dict, Any, Optional, Tuple
from supabase import create_client, Client
import openai

# Configure logging
logger = logging.getLogger(__name__)

# Retry configuration
OPENAI_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "3"))
OPENAI_RETRY_DELAY = float(os.getenv("OPENAI_RETRY_DELAY", "1.0"))
OPENAI_TIMEOUT = int(os.getenv("OPENAI_TIMEOUT", "30"))

# Batch processing configuration
BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "20"))
THREAD_POOL_MAX_WORKERS = int(os.getenv("THREAD_POOL_MAX_WORKERS", "10"))
BATCH_FAILURE_THRESHOLD = float(os.getenv("BATCH_FAILURE_THRESHOLD", "0.5"))  # Fail if more than 50% of batches fail

# Load OpenAI API key for embeddings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

# Configuration constants
EMBEDDING_MODEL_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

# Token limits for OpenAI models
EMBEDDING_MODEL_TOKEN_LIMITS = {
    "text-embedding-3-small": 8191,
    "text-embedding-3-large": 8191,
    "text-embedding-ada-002": 8191,
}

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
# Force 1536 dimensions to match database schema
EMBEDDING_DIMENSION = 1536  # Database is hardcoded to vector(1536)
if EMBEDDING_MODEL not in EMBEDDING_MODEL_DIMENSIONS:
    logger.warning(f"Unknown embedding model '{EMBEDDING_MODEL}'. Using text-embedding-3-small.")
    EMBEDDING_MODEL = "text-embedding-3-small"
elif EMBEDDING_MODEL_DIMENSIONS[EMBEDDING_MODEL] != 1536:
    logger.warning(f"{EMBEDDING_MODEL} has {EMBEDDING_MODEL_DIMENSIONS[EMBEDDING_MODEL]} dimensions but database expects 1536. Using text-embedding-3-small.")
    EMBEDDING_MODEL = "text-embedding-3-small"

MAX_DOCUMENT_LENGTH = int(os.getenv("MAX_DOCUMENT_LENGTH", "25000"))

def is_critical_api_error(exception: Exception) -> bool:
    """
    Determine if an exception represents a critical API error that shouldn't be retried.
    
    Args:
        exception: The exception to check
        
    Returns:
        True if this is a critical error that shouldn't be retried
    """
    error_str = str(exception).lower()
    
    # Authentication/authorization errors
    if any(term in error_str for term in ['unauthorized', 'forbidden', 'invalid api key', 'authentication', '401', '403']):
        return True
    
    # Invalid request errors that won't be fixed by retrying
    if any(term in error_str for term in ['invalid model', 'model not found', 'invalid request']):
        return True
    
    return False

def retry_with_exponential_backoff(
    func,
    max_retries: int = OPENAI_MAX_RETRIES,
    initial_delay: float = OPENAI_RETRY_DELAY,
    exponential_base: float = 2,
    jitter: bool = True
):
    """
    Retry a function with exponential backoff.
    
    Args:
        func: Function to retry
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
        exponential_base: Base for exponential backoff
        jitter: Whether to add random jitter to delays
        
    Returns:
        Result of the function call
    """
    import random
    
    def wrapper(*args, **kwargs):
        num_retries = 0
        delay = initial_delay
        
        while True:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Check if this is a critical error that shouldn't be retried
                if is_critical_api_error(e):
                    logger.error(f"Critical API error in {func.__name__}: {str(e)}")
                    raise
                
                num_retries += 1
                
                if num_retries > max_retries:
                    logger.error(f"Maximum retries ({max_retries}) exceeded for {func.__name__}")
                    raise
                
                # Log the retry attempt
                logger.warning(f"Retry {num_retries}/{max_retries} for {func.__name__} after error: {str(e)}")
                
                # Calculate delay with optional jitter
                if jitter:
                    delay = delay * (1 + random.random() * 0.1)
                
                time.sleep(delay)
                
                # Exponential backoff
                delay *= exponential_base
                
    return wrapper

def get_supabase_client() -> Client:
    """
    Get a Supabase client with the URL and key from environment variables.
    
    Returns:
        Supabase client instance
    """
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")
    
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in environment variables")
    
    return create_client(url, key)

def create_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """
    Create embeddings for multiple texts in a single API call.
    
    Args:
        texts: List of texts to create embeddings for
        
    Returns:
        List of embeddings (each embedding is a list of floats)
    """
    if not texts:
        return []
    
    if not OPENAI_API_KEY:
        error_msg = "Cannot create embeddings - OPENAI_API_KEY not set"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    # Validate token lengths
    import tiktoken
    try:
        encoding = tiktoken.encoding_for_model(EMBEDDING_MODEL)
    except KeyError:
        # Fallback for unknown models
        encoding = tiktoken.get_encoding("cl100k_base")
    
    max_tokens = EMBEDDING_MODEL_TOKEN_LIMITS.get(EMBEDDING_MODEL, 8191)
    validated_texts = []
    
    for text in texts:
        tokens = encoding.encode(text)
        if len(tokens) > max_tokens:
            logger.warning(f"Text exceeds {max_tokens} token limit ({len(tokens)} tokens). Truncating...")
            # Truncate to fit within token limit
            truncated_tokens = tokens[:max_tokens]
            validated_texts.append(encoding.decode(truncated_tokens))
        else:
            validated_texts.append(text)
        
    # Create a retry-wrapped version of the API call
    @retry_with_exponential_backoff
    def create_embeddings_with_retry():
        return openai.embeddings.create(
            model=EMBEDDING_MODEL,
            input=validated_texts,
            timeout=OPENAI_TIMEOUT
        )
    
    try:
        response = create_embeddings_with_retry()
        return [item.embedding for item in response.data]
    except Exception as e:
        error_msg = f"Error creating batch embeddings after {OPENAI_MAX_RETRIES} retries: {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e

def create_embedding(text: str, contextual_prefix: Optional[str] = None) -> List[float]:
    """
    Create an embedding for a single text using OpenAI's API.
    
    Args:
        text: Text to create an embedding for
        contextual_prefix: Optional contextual prefix to add for retrieval
        
    Returns:
        List of floats representing the embedding
    """
    # Add contextual prefix if provided (for query embeddings)
    if contextual_prefix:
        text = f"{contextual_prefix}\n---\n{text}"
        
    try:
        embeddings = create_embeddings_batch([text])
        return embeddings[0] if embeddings else []
    except Exception as e:
        error_msg = f"Error creating embedding: {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e

def generate_contextual_embedding(full_document: str, chunk: str) -> Tuple[str, bool]:
    """
    Generate contextual information for a chunk within a document to improve retrieval.
    
    Args:
        full_document: The complete document text
        chunk: The specific chunk of text to generate context for
        
    Returns:
        Tuple containing:
        - The contextual text that situates the chunk within the document
        - Boolean indicating if contextual embedding was performed
    """
    model_choice = os.getenv("MODEL_CHOICE")
    
    # Check if model_choice is set and valid
    if not model_choice:
        return chunk, False
    
    # Validate model choice
    valid_models = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-4o-mini", 
                    "gpt-4.1-mini", "gpt-4.1-nano", "o1-preview", "o1-mini", "o3-mini", "o4-mini",
                    "deepseek/deepseek-r1-distill-qwen-32b", "deepseek/deepseek-r1-distill-llama-70b", 
                    "deepseek/deepseek-v3", "google/gemini-2.5-flash", "google/gemini-2.0-flash-thinking"]
    if model_choice not in valid_models:
        logger.warning(f"Invalid model '{model_choice}'. Using original chunk.")
        return chunk, False
    
    try:
        # Create the prompt for generating contextual information
        # Indicate if document was truncated
        doc_text = full_document[:MAX_DOCUMENT_LENGTH]
        truncation_note = " [Document truncated]" if len(full_document) > MAX_DOCUMENT_LENGTH else ""
        
        prompt = f"""<document{truncation_note}> 
{doc_text} 
</document>
Here is the chunk we want to situate within the whole document 
<chunk> 
{chunk}
</chunk> 
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""

        # Create a retry-wrapped version of the API call
        @retry_with_exponential_backoff
        def create_chat_completion_with_retry():
            # Check if using OpenRouter models
            if model_choice.startswith(("deepseek/", "google/", "meta/", "anthropic/")):
                # OpenRouter configuration
                import openai as openrouter_client
                openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
                if not openrouter_api_key:
                    logger.error("OPENROUTER_API_KEY not set for OpenRouter models")
                    return chunk, False
                
                # Create a new client for OpenRouter
                from openai import OpenAI
                client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=openrouter_api_key
                )
                
                return client.chat.completions.create(
                    model=model_choice,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that provides concise contextual information."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.6 if "deepseek" in model_choice else 0.3,  # DeepSeek recommends 0.6
                    max_tokens=200,
                    timeout=OPENAI_TIMEOUT
                )
            else:
                # Standard OpenAI API
                return openai.chat.completions.create(
                    model=model_choice,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that provides concise contextual information."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=200,
                    timeout=OPENAI_TIMEOUT
                )
        
        # Call the OpenAI API to generate contextual information
        response = create_chat_completion_with_retry()
        
        # Extract the generated context
        context = response.choices[0].message.content.strip()
        
        # Combine the context with the original chunk
        contextual_text = f"{context}\n---\n{chunk}"
        
        return contextual_text, True
    
    except Exception as e:
        logger.error(f"Error generating contextual embedding: {e}. Using original chunk instead.")
        return chunk, False

def process_chunk_with_context(args):
    """
    Process a single chunk with contextual embedding.
    This function is designed to be used with concurrent.futures.
    
    Args:
        args: Tuple containing (url, content, full_document)
        
    Returns:
        Tuple containing:
        - The contextual text that situates the chunk within the document
        - Boolean indicating if contextual embedding was performed
    """
    _, content, full_document = args  # url is not used in this function
    return generate_contextual_embedding(full_document, content)

def add_documents_to_supabase(
    client: Client, 
    urls: List[str], 
    chunk_numbers: List[int],
    contents: List[str], 
    metadatas: List[Dict[str, Any]],
    url_to_full_document: Dict[str, str],
    batch_size: int = None
) -> int:
    """
    Add documents to the Supabase crawled_pages table in batches.
    Uses upsert to handle concurrent operations safely.
    
    Args:
        client: Supabase client
        urls: List of URLs
        chunk_numbers: List of chunk numbers
        contents: List of document contents
        metadatas: List of document metadata
        url_to_full_document: Dictionary mapping URLs to their full document content
        batch_size: Size of each batch for insertion
        
    Returns:
        Number of successfully stored chunks
        
    Raises:
        ValueError: If input validation fails
        RuntimeError: If critical errors occur during processing
    """
    # Use environment variable if batch_size not provided
    if batch_size is None:
        batch_size = BATCH_SIZE
        
    # Validate inputs
    if not urls or not contents:
        logger.warning("No documents to add to Supabase")
        return 0
        
    if len(urls) != len(contents) or len(urls) != len(chunk_numbers) or len(urls) != len(metadatas):
        raise ValueError("Mismatched lengths for urls, contents, chunk_numbers, and metadatas")
    
    if batch_size <= 0:
        logger.warning(f"Invalid batch_size {batch_size}, using default {BATCH_SIZE}")
        batch_size = BATCH_SIZE
    
    # Note: Using upsert instead of delete+insert to avoid race conditions
    # The unique constraint on (url, chunk_number) will ensure no duplicates
    
    # Check if MODEL_CHOICE is set for contextual embeddings
    model_choice = os.getenv("MODEL_CHOICE")
    use_contextual_embeddings = bool(model_choice)
    
    # Track successful chunks
    successful_chunks = 0
    failed_batches = 0
    total_batches = (len(contents) + batch_size - 1) // batch_size
    
    # Log start of embedding generation if we have many documents
    if total_batches > 5:
        logger.info(f"[EMBEDDING PROGRESS] Starting embedding generation for {len(contents)} chunks in {total_batches} batches...")
    
    # Process in batches to avoid memory issues
    for i in range(0, len(contents), batch_size):
        batch_end = min(i + batch_size, len(contents))
        batch_num = i//batch_size + 1
        
        # Progress update every 10 batches or on first/last batch
        if total_batches > 5 and (batch_num == 1 or batch_num == total_batches or batch_num % 10 == 0):
            logger.info(f"[EMBEDDING PROGRESS] Processing batch {batch_num}/{total_batches} (chunks {i+1}-{batch_end} of {len(contents)})")
        
        # Get batch slices
        batch_urls = urls[i:batch_end]
        batch_chunk_numbers = chunk_numbers[i:batch_end]
        batch_contents = contents[i:batch_end]
        batch_metadatas = metadatas[i:batch_end]
        
        # Apply contextual embedding to each chunk if MODEL_CHOICE is set
        if use_contextual_embeddings:
            # Prepare arguments for parallel processing
            process_args = []
            for j, content in enumerate(batch_contents):
                url = batch_urls[j]
                full_document = url_to_full_document.get(url, "")
                process_args.append((url, content, full_document))
            
            # Process in parallel using ThreadPoolExecutor
            contextual_contents = [None] * len(batch_contents)  # Pre-allocate list with correct size
            with concurrent.futures.ThreadPoolExecutor(max_workers=THREAD_POOL_MAX_WORKERS) as executor:
                # Submit all tasks and collect results
                future_to_idx = {executor.submit(process_chunk_with_context, arg): idx 
                                for idx, arg in enumerate(process_args)}
                
                # Process results as they complete
                for future in concurrent.futures.as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        result, success = future.result()
                        contextual_contents[idx] = result  # Place in correct index
                        if success:
                            batch_metadatas[idx]["contextual_embedding"] = True
                    except Exception as e:
                        logger.error(f"Error processing chunk {idx}: {e}")
                        # Use original content as fallback
                        contextual_contents[idx] = batch_contents[idx]
            
            # Check if any items are still None (shouldn't happen but be safe)
            for i, content in enumerate(contextual_contents):
                if content is None:
                    logger.warning(f"No result for chunk {i}, using original content")
                    contextual_contents[i] = batch_contents[i]
        else:
            # If not using contextual embeddings, use original contents
            contextual_contents = batch_contents
        
        # Create embeddings for the entire batch at once
        try:
            batch_embeddings = create_embeddings_batch(contextual_contents)
        except Exception as e:
            # Check if this is a critical error that should stop all processing
            if is_critical_api_error(e):
                logger.error(f"Critical API error encountered: {e}")
                raise RuntimeError(f"Critical API error - cannot continue: {str(e)}")
            
            logger.error(f"Failed to create embeddings for batch {i//batch_size + 1}. Skipping batch: {e}")
            failed_batches += 1
            continue
        
        batch_data = []
        for j in range(len(contextual_contents)):
            # Extract metadata fields
            chunk_size = len(batch_contents[j])
            
            # Prepare data for insertion
            data = {
                "url": batch_urls[j],
                "chunk_number": batch_chunk_numbers[j],
                "content": batch_contents[j],  # Store original content
                "metadata": {
                    "chunk_size": chunk_size,
                    **batch_metadatas[j]
                },
                "embedding": batch_embeddings[j]  # Use embedding from contextual content
            }
            
            # If contextual embedding was used, store the contextual content
            if use_contextual_embeddings and batch_metadatas[j].get("contextual_embedding"):
                data["contextual_content"] = contextual_contents[j]
            
            batch_data.append(data)
        
        # Upsert batch into Supabase (insert or update based on unique constraint)
        try:
            client.table("crawled_pages").upsert(batch_data, on_conflict="url,chunk_number").execute()
            successful_chunks += len(batch_data)
            logger.debug(f"Successfully upserted batch {i//batch_size + 1} with {len(batch_data)} chunks")
        except Exception as e:
            logger.error(f"Error upserting batch {i//batch_size + 1} into Supabase: {e}")
            failed_batches += 1
    
    # Log summary
    total_batches = len(range(0, len(contents), batch_size))
    logger.info(f"Document storage complete: {successful_chunks}/{len(contents)} chunks stored, {failed_batches}/{total_batches} batches failed")
    
    # Check if failure threshold exceeded
    if total_batches > 0:
        failure_rate = failed_batches / total_batches
        if failure_rate > BATCH_FAILURE_THRESHOLD:
            error_msg = f"Batch failure rate ({failure_rate:.1%}) exceeded threshold ({BATCH_FAILURE_THRESHOLD:.1%}). {failed_batches}/{total_batches} batches failed."
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    # Raise error if no documents stored
    if failed_batches > 0 and successful_chunks == 0:
        raise RuntimeError(f"Failed to store any documents. All {failed_batches} batches failed.")
    
    return successful_chunks

def search_documents(
    client: Client, 
    query: str, 
    match_count: int = 10, 
    filter_metadata: Optional[Dict[str, Any]] = None,
    contextual_search: bool = True
) -> List[Dict[str, Any]]:
    """
    Search for documents in Supabase using vector similarity.
    
    Args:
        client: Supabase client
        query: Query text
        match_count: Maximum number of results to return
        filter_metadata: Optional metadata filter
        contextual_search: Whether to use contextual search (should match storage)
        
    Returns:
        List of matching documents
    """
    # Validate query
    if not query or not query.strip():
        error_msg = "Empty query provided for search"
        logger.error(error_msg)
        raise ValueError(error_msg)
        
    # Validate match_count
    if match_count <= 0:
        logger.warning(f"Invalid match_count {match_count}, using default 10")
        match_count = 10
    
    # Check if contextual embeddings are being used in storage
    use_contextual = os.getenv("MODEL_CHOICE") and contextual_search
    
    # Create embedding for the query
    try:
        if use_contextual:
            # Generate contextual prefix for the query
            contextual_prefix = "This is a search query looking for relevant content about:"
            query_embedding = create_embedding(query, contextual_prefix=contextual_prefix)
        else:
            query_embedding = create_embedding(query)
    except Exception as e:
        error_msg = f"Failed to create embedding for query: {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e
    
    # Execute the search using the match_crawled_pages function
    try:
        # Only include filter parameter if filter_metadata is provided and not empty
        params = {
            'query_embedding': query_embedding,
            'match_count': match_count
        }
        
        # Only add the filter if it's actually provided and not empty
        if filter_metadata:
            params['filter'] = filter_metadata  # Pass the dictionary directly, not JSON-encoded
        
        result = client.rpc('match_crawled_pages', params).execute()
        
        results = result.data
        
        # If we have contextual content stored, return that instead of original content
        for result in results:
            if result.get('contextual_content'):
                result['content'] = result['contextual_content']
                
        return results
    except Exception as e:
        error_msg = f"Database error searching documents: {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e