# Fixes Applied to crawl4ai-mcp

## Summary
Fixed critical issues in the embedding and retrieval logic that were causing search failures and mismatches between stored and queried content.

## Changes Made

### 1. **Fixed Database Vector Dimension Mismatch**
- Enforced 1536-dimensional embeddings to match the database schema
- Added validation to prevent using models with different dimensions
- Added proper model configuration with dimension mapping

### 2. **Fixed Contextual Embedding Asymmetry**
- Added contextual prefix support to query embeddings when contextual storage is enabled
- Ensured queries and stored documents use the same embedding approach
- Added `contextual_content` storage to preserve what was actually embedded

### 3. **Fixed Storage vs Retrieval Content Mismatch**
- Now stores both original content and contextual content when using contextual embeddings
- Returns contextual content in search results when available
- Database migration script provided to add `contextual_content` column

### 4. **Added Token Length Validation**
- Added tiktoken dependency for accurate token counting
- Validates and truncates texts exceeding OpenAI's 8191 token limit
- Prevents API errors from oversized inputs

### 5. **Improved Error Handling**
- Consistent error handling - always raises exceptions instead of returning empty results
- Better error differentiation (validation vs runtime vs unknown errors)
- Clear error messages with proper error types in responses

### 6. **Fixed Configuration Issues**
- Proper embedding model validation
- Support for multiple OpenAI embedding models with correct dimensions
- Warning messages for unsupported configurations

## Migration Required

Run the following SQL migration on your Supabase database:

```sql
-- Add contextual_content column
ALTER TABLE crawled_pages 
ADD COLUMN IF NOT EXISTS contextual_content text;

-- Update the match function to return contextual_content
-- (see migration_add_contextual_content.sql for full script)
```

## Dependencies Added
- `tiktoken==0.7.0` - For accurate token counting and validation

## Environment Variables
No changes required, but ensure:
- `OPENAI_API_KEY` is set (now required, will error if missing)
- `EMBEDDING_MODEL` is one of: text-embedding-3-small (default), text-embedding-ada-002
- `MODEL_CHOICE` is set if you want contextual embeddings

## Testing Recommendations
1. Clear existing embeddings if switching between contextual and non-contextual modes
2. Test search functionality with both modes
3. Verify token truncation works for long documents
4. Check error handling with invalid API keys