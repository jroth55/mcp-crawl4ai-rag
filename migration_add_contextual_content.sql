-- Migration to add contextual_content column for better search accuracy

-- Add contextual_content column to store the text that was actually embedded
ALTER TABLE crawled_pages 
ADD COLUMN IF NOT EXISTS contextual_content text;

-- Update the match_crawled_pages function to return contextual_content
CREATE OR REPLACE FUNCTION match_crawled_pages (
  query_embedding vector(1536),
  match_count int default 10,
  filter jsonb DEFAULT '{}'::jsonb
) RETURNS TABLE (
  id bigint,
  url varchar,
  chunk_number integer,
  content text,
  contextual_content text,
  metadata jsonb,
  similarity float
)
LANGUAGE plpgsql
AS $$
#variable_conflict use_column
BEGIN
  RETURN QUERY
  SELECT
    id,
    url,
    chunk_number,
    content,
    contextual_content,
    metadata,
    1 - (crawled_pages.embedding <=> query_embedding) as similarity
  FROM crawled_pages
  WHERE metadata @> filter
  ORDER BY crawled_pages.embedding <=> query_embedding
  LIMIT match_count;
END;
$$;

-- Add an index on contextual_content for text search if needed
CREATE INDEX IF NOT EXISTS idx_crawled_pages_contextual_content 
ON crawled_pages USING gin(to_tsvector('english', contextual_content));