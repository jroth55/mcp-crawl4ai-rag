# Fixing Row-Level Security (RLS) Errors

## Problem
When crawling websites, you may encounter this error:
```
Error upserting batch into Supabase: {'message': 'new row violates row-level security policy for table "crawled_pages"', 'code': '42501', 'hint': None, 'details': None}
```

This occurs because the `crawled_pages` table has Row-Level Security (RLS) enabled but lacks the necessary policies to allow the service role to insert, update, or delete data.

## Root Cause
If you're using an older version of `crawled_pages.sql` or if RLS policies were manually modified, you may be missing the necessary INSERT, UPDATE, or DELETE policies for the service role. When RLS is enabled on a table without appropriate policies, PostgreSQL denies all operations by default.

## Solution

### Option 1: Re-run the RLS Section from crawled_pages.sql (Recommended)
1. Connect to your Supabase database using the SQL editor in the Supabase dashboard
2. Run the RLS section from the `crawled_pages.sql` file (lines 77-111):
   ```sql
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
   ```

### Option 2: Manual Fix via Supabase Dashboard
1. Go to your Supabase project dashboard
2. Navigate to Authentication > Policies
3. Find the `crawled_pages` table
4. Add the following policies:
   - **Insert Policy**: 
     - Name: "Allow service role insert"
     - Target roles: service_role
     - WITH CHECK expression: `true`
   - **Update Policy**:
     - Name: "Allow service role update"
     - Target roles: service_role
     - USING expression: `true`
     - WITH CHECK expression: `true`
   - **Delete Policy**:
     - Name: "Allow service role delete"
     - Target roles: service_role
     - USING expression: `true`

### Option 3: Recreate the Table
If you're starting fresh, you can drop and recreate the table with the current `crawled_pages.sql` which includes all necessary policies.

**Note**: The latest version of `crawled_pages.sql` includes DROP POLICY IF EXISTS statements, making it idempotent. You can safely re-run the entire script or just the RLS section to reset policies to their default state.

## Verification
After applying the fix, verify the policies are correctly set:
```sql
select * from pg_policies where tablename = 'crawled_pages';
```

You should see 4 policies:
1. "Allow public read access" (SELECT)
2. "Allow service role insert" (INSERT)
3. "Allow service role update" (UPDATE)
4. "Allow service role delete" (DELETE)

## Prevention
Always ensure that when enabling RLS on a table, you create appropriate policies for all operations your application needs to perform. The service role (used when connecting with `SUPABASE_SERVICE_KEY`) needs explicit policies even though it has elevated privileges.

## Additional Notes
- The MCP server uses the `SUPABASE_SERVICE_KEY` which authenticates as the `service_role`
- The `service_role` bypasses RLS by default, but when RLS is enabled, explicit policies are still required
- The public read policy allows anyone to search the crawled content
- The service role policies allow the MCP server to manage the data