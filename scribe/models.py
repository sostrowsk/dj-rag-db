from django.conf import settings
from django.contrib.postgres.indexes import GinIndex
from django.contrib.postgres.search import SearchVectorField
from django.db import models
from pgvector.django import HalfVectorField, HnswIndex


class DocumentChunk(models.Model):
    """Postgres source of truth for chunked document content + embeddings.

    Uses HalfVectorField (float16) instead of VectorField (float32) because
    pgvector's HNSW index supports up to 4000 dimensions with halfvec but
    only 2000 with regular vector. Azure text-embedding-3-large produces
    3072-dim vectors. Halving precision has negligible retrieval impact.

    ``search_vector`` is maintained by a database trigger (migration 0002)
    that runs ``to_tsvector('german', content)`` on INSERT/UPDATE of content.
    """

    # Namespace, identical to the legacy Milvus collection name:
    # "project_{id}", "client_{id}" or "general_chat".
    collection_name = models.CharField(max_length=128, db_index=True)

    project_document = models.ForeignKey(
        getattr(settings, "SCRIBE_PROJECT_DOCUMENT_MODEL", "data_room.ProtectedProjectDocument"),
        null=True,
        blank=True,
        on_delete=models.CASCADE,
        related_name="chunks",
    )
    client_document = models.ForeignKey(
        getattr(settings, "SCRIBE_CLIENT_DOCUMENT_MODEL", "data_room.ProtectedClientDocument"),
        null=True,
        blank=True,
        on_delete=models.CASCADE,
        related_name="chunks",
    )

    # Denormalized ids (kept even when the FK target is unknown/orphaned).
    document_id = models.BigIntegerField(db_index=True)
    project_id = models.BigIntegerField(null=True, blank=True, db_index=True)
    chunk_id = models.IntegerField(help_text="Sequential chunk ID within a document")

    # Contextualized content — this is what gets embedded and FTS-indexed.
    content = models.TextField()
    original_content = models.TextField(blank=True, default="")
    raw_section = models.TextField(blank=True, default="")
    document_path = models.TextField(blank=True, default="")
    image_path = models.TextField(blank=True, default="")
    page_number = models.IntegerField(null=True, blank=True)
    has_context = models.BooleanField(default=False)

    embedding = HalfVectorField(dimensions=3072)
    search_vector = SearchVectorField(null=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "Document Chunk"
        verbose_name_plural = "Document Chunks"
        indexes = [
            HnswIndex(
                name="scribe_chunk_emb_hnsw",
                fields=["embedding"],
                m=16,
                ef_construction=64,
                opclasses=["halfvec_cosine_ops"],
            ),
            GinIndex(
                name="scribe_chunk_search_gin",
                fields=["search_vector"],
            ),
            models.Index(fields=["collection_name", "document_id", "chunk_id"]),
        ]
        constraints = [
            models.UniqueConstraint(
                fields=["collection_name", "document_id", "chunk_id"],
                name="scribe_unique_chunk_per_collection_doc",
            ),
        ]

    def __str__(self):
        return f"Chunk {self.chunk_id} of doc {self.document_id} ({self.collection_name})"
