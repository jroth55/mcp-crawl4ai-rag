"""
Utility functions for the Crawl4AI MCP server.
"""
import os
import concurrent.futures
from typing import List, Dict, Any, Optional, Tuple
from supabase import create_client, Client
import openai

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
    print(f"WARNING: Unknown embedding model '{EMBEDDING_MODEL}'. Using text-embedding-3-small.")
    EMBEDDING_MODEL = "text-embedding-3-small"
elif EMBEDDING_MODEL_DIMENSIONS[EMBEDDING_MODEL] != 1536:
    print(f"WARNING: {EMBEDDING_MODEL} has {EMBEDDING_MODEL_DIMENSIONS[EMBEDDING_MODEL]} dimensions but database expects 1536. Using text-embedding-3-small.")
    EMBEDDING_MODEL = "text-embedding-3-small"

MAX_DOCUMENT_LENGTH = int(os.getenv("MAX_DOCUMENT_LENGTH", "25000"))

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
        print(f"ERROR: {error_msg}")
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
            print(f"WARNING: Text exceeds {max_tokens} token limit ({len(tokens)} tokens). Truncating...")
            # Truncate to fit within token limit
            truncated_tokens = tokens[:max_tokens]
            validated_texts.append(encoding.decode(truncated_tokens))
        else:
            validated_texts.append(text)
        
    try:
        response = openai.embeddings.create(
            model=EMBEDDING_MODEL,
            input=validated_texts
        )
        return [item.embedding for item in response.data]
    except Exception as e:
        error_msg = f"Error creating batch embeddings: {e}"
        print(error_msg)
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
        print(error_msg)
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
    valid_models = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-4o-mini", "o1-preview", "o1-mini"]
    if model_choice not in valid_models:
        print(f"WARNING: Invalid model '{model_choice}'. Using original chunk.")
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

        # Call the OpenAI API to generate contextual information
        response = openai.chat.completions.create(
            model=model_choice,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides concise contextual information."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=200
        )
        
        # Extract the generated context
        context = response.choices[0].message.content.strip()
        
        # Combine the context with the original chunk
        contextual_text = f"{context}\n---\n{chunk}"
        
        return contextual_text, True
    
    except Exception as e:
        print(f"Error generating contextual embedding: {e}. Using original chunk instead.")
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
    batch_size: int = 20
) -> None:
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
    """
    # Validate inputs
    if not urls or not contents:
        print("WARNING: No documents to add to Supabase")
        return
        
    if len(urls) != len(contents) or len(urls) != len(chunk_numbers) or len(urls) != len(metadatas):
        raise ValueError("Mismatched lengths for urls, contents, chunk_numbers, and metadatas")
    
    if batch_size <= 0:
        print(f"WARNING: Invalid batch_size {batch_size}, using default 20")
        batch_size = 20
    
    # Note: Using upsert instead of delete+insert to avoid race conditions
    # The unique constraint on (url, chunk_number) will ensure no duplicates
    
    # Check if MODEL_CHOICE is set for contextual embeddings
    model_choice = os.getenv("MODEL_CHOICE")
    use_contextual_embeddings = bool(model_choice)
    
    # Process in batches to avoid memory issues
    for i in range(0, len(contents), batch_size):
        batch_end = min(i + batch_size, len(contents))
        
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
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
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
                        print(f"Error processing chunk {idx}: {e}")
                        # Use original content as fallback
                        contextual_contents[idx] = batch_contents[idx]
            
            # Check if any items are still None (shouldn't happen but be safe)
            for i, content in enumerate(contextual_contents):
                if content is None:
                    print(f"Warning: No result for chunk {i}, using original content")
                    contextual_contents[i] = batch_contents[i]
        else:
            # If not using contextual embeddings, use original contents
            contextual_contents = batch_contents
        
        # Create embeddings for the entire batch at once
        try:
            batch_embeddings = create_embeddings_batch(contextual_contents)
        except Exception as e:
            print(f"Failed to create embeddings for batch. Skipping batch: {e}")
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
        except Exception as e:
            print(f"Error upserting batch into Supabase: {e}")

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
        print(f"ERROR: {error_msg}")
        raise ValueError(error_msg)
        
    # Validate match_count
    if match_count <= 0:
        print(f"WARNING: Invalid match_count {match_count}, using default 10")
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
        print(f"ERROR: {error_msg}")
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
        print(f"ERROR: {error_msg}")
        raise RuntimeError(error_msg) from e