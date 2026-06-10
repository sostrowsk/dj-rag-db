"""German tsvector trigger for DocumentChunk.search_vector.

Ported from arznei-muster-mello ai_vectorstore/migrations/0002_tsvector_trigger.py.
Keeps search_vector in sync with content at the database level, so bulk_create
and raw SQL inserts are covered without application-side hooks.
"""

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("scribe", "0001_initial"),
    ]

    operations = [
        migrations.RunSQL(
            sql="""
                CREATE OR REPLACE FUNCTION scribe_chunk_search_vector_trigger() RETURNS trigger AS $$
                BEGIN
                    NEW.search_vector := to_tsvector('german', NEW.content);
                    RETURN NEW;
                END;
                $$ LANGUAGE plpgsql;

                CREATE TRIGGER scribe_chunk_search_vector_update
                    BEFORE INSERT OR UPDATE OF content
                    ON scribe_documentchunk
                    FOR EACH ROW
                    EXECUTE FUNCTION scribe_chunk_search_vector_trigger();
            """,
            reverse_sql="""
                DROP TRIGGER IF EXISTS scribe_chunk_search_vector_update ON scribe_documentchunk;
                DROP FUNCTION IF EXISTS scribe_chunk_search_vector_trigger();
            """,
        ),
    ]
