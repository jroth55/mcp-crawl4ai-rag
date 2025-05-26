-- Drop existing RLS policies if they exist
drop policy if exists "Allow public read access" on crawled_pages;
drop policy if exists "Allow service role insert" on crawled_pages;
drop policy if exists "Allow service role update" on crawled_pages;
drop policy if exists "Allow service role delete" on crawled_pages;

-- Recreate RLS policies with proper permissions
-- Allow anyone to read (for search functionality)
create policy "Allow public read access"
  on crawled_pages
  for select
  to public
  using (true);

-- Allow service role to insert (for crawling)
create policy "Allow service role insert"
  on crawled_pages
  for insert
  to service_role
  with check (true);

-- Allow service role to update (for upsert operations)
create policy "Allow service role update"
  on crawled_pages
  for update
  to service_role
  using (true)
  with check (true);

-- Allow service role to delete (for cleanup if needed)
create policy "Allow service role delete"
  on crawled_pages
  for delete
  to service_role
  using (true);

-- Verify policies
select * from pg_policies where tablename = 'crawled_pages';