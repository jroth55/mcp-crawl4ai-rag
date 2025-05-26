-- Drop existing RLS policies if they exist
drop policy if exists "Allow public read access" on crawled_pages;
drop policy if exists "Allow service role insert" on crawled_pages;
drop policy if exists "Allow service role update" on crawled_pages;
drop policy if exists "Allow service role delete" on crawled_pages;

-- Recreate RLS policies with anon access
-- Allow anyone to read (for search functionality)
create policy "Allow public read access"
  on crawled_pages
  for select
  to public
  using (true);

-- Allow anon to insert (for crawling)
create policy "Allow anon insert"
  on crawled_pages
  for insert
  to anon
  with check (true);

-- Allow anon to update (for upsert operations)
create policy "Allow anon update"
  on crawled_pages
  for update
  to anon
  using (true)
  with check (true);

-- Allow anon to delete (for cleanup if needed)
create policy "Allow anon delete"
  on crawled_pages
  for delete
  to anon
  using (true);

-- Verify policies
select * from pg_policies where tablename = 'crawled_pages';