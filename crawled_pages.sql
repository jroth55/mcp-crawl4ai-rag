-- Enable the pgvector extension
create extension if not exists vector;

-- Create the documentation chunks table
create table crawled_pages (
    id bigserial primary key,
    url varchar not null,
    chunk_number integer not null,
    content text not null,  -- Original content
    contextual_content text,  -- Content with context (used when contextual embeddings are enabled)
    metadata jsonb not null default '{}'::jsonb,  -- Metadata including source, chunk_size, etc.
    embedding vector(1536),  -- OpenAI embeddings are 1536 dimensions
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    
    -- Add a unique constraint to prevent duplicate chunks for the same URL
    unique(url, chunk_number)
);

-- Create an index for better vector similarity search performance
create index on crawled_pages using ivfflat (embedding vector_cosine_ops);

-- Create an index on metadata for faster filtering
create index idx_crawled_pages_metadata on crawled_pages using gin (metadata);

-- Create an index on source for faster filtering by source
create index idx_crawled_pages_source on crawled_pages ((metadata->>'source'));

-- Create an index on contextual_content for text search (optional, but helpful for full-text search)
create index idx_crawled_pages_contextual_content on crawled_pages using gin(to_tsvector('english', contextual_content));

-- Create a function to search for documentation chunks
create or replace function match_crawled_pages (
  query_embedding vector(1536),
  match_count int default 10,
  filter jsonb DEFAULT '{}'::jsonb
) returns table (
  id bigint,
  url varchar,
  chunk_number integer,
  content text,
  contextual_content text,
  metadata jsonb,
  similarity float
)
language plpgsql
as $$
#variable_conflict use_column
begin
  return query
  select
    id,
    url,
    chunk_number,
    content,
    contextual_content,
    metadata,
    1 - (crawled_pages.embedding <=> query_embedding) as similarity
  from crawled_pages
  where metadata @> filter
  order by crawled_pages.embedding <=> query_embedding
  limit match_count;
end;
$$;

-- Create a function to get unique sources efficiently
create or replace function get_unique_sources()
returns table (source text)
language sql
stable
as $$
  select distinct metadata->>'source' as source
  from crawled_pages
  where metadata->>'source' is not null
  order by source;
$$;

-- Enable RLS on the table
alter table crawled_pages enable row level security;

-- Create a policy that allows anyone to read
create policy "Allow public read access"
  on crawled_pages
  for select
  to public
  using (true);

-- Create a policy that allows service role to insert
create policy "Allow service role insert"
  on crawled_pages
  for insert
  to service_role
  with check (true);

-- Create a policy that allows service role to update
create policy "Allow service role update"
  on crawled_pages
  for update
  to service_role
  using (true)
  with check (true);

-- Create a policy that allows service role to delete
create policy "Allow service role delete"
  on crawled_pages
  for delete
  to service_role
  using (true);