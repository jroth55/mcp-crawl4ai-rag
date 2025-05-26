-- Drop everything related to crawled_pages to start fresh
-- This script handles cases where objects may or may not exist

-- Drop the table first (this will cascade drop policies and indexes)
DROP TABLE IF EXISTS crawled_pages CASCADE;

-- Drop functions (they might exist even if table doesn't)
DROP FUNCTION IF EXISTS match_crawled_pages(vector(1536), integer, jsonb);
DROP FUNCTION IF EXISTS get_unique_sources();

-- Note: We don't drop the vector extension as other tables might use it
-- If you want to drop it too, uncomment the next line:
-- DROP EXTENSION IF EXISTS vector CASCADE;