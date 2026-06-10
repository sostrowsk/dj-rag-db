"""Tests for the DocumentChunk ORM model (Postgres SSOT for vector chunks).

Phase A1 of the pgvector plan: DocumentChunk stores chunked document content
with halfvec embeddings (pgvector) and a German tsvector maintained by a
database trigger.
"""

import pytest
from django.contrib.postgres.search import SearchQuery
from django.db import IntegrityError

from data_room.tests.factories import ProtectedClientDocumentFactory, ProtectedDocumentFactory
from scribe.tests.factories import DocumentChunkFactory, deterministic_embedding


class TestDeterministicEmbedding:
    """Seedable fake embeddings for cosine-ordering tests."""

    def test_same_seed_yields_identical_vector(self):
        assert deterministic_embedding(seed=42) == deterministic_embedding(seed=42)

    def test_different_seeds_yield_different_vectors(self):
        assert deterministic_embedding(seed=1) != deterministic_embedding(seed=2)

    def test_vector_has_3072_dimensions(self):
        assert len(deterministic_embedding(seed=0)) == 3072

    def test_vector_is_unit_normalized(self):
        vec = deterministic_embedding(seed=7)
        norm = sum(v * v for v in vec) ** 0.5
        assert norm == pytest.approx(1.0, abs=1e-6)


@pytest.mark.django_db
class TestDocumentChunkModel:
    def test_factory_creates_chunk_with_persisted_embedding(self):
        chunk = DocumentChunkFactory()
        chunk.refresh_from_db()
        assert chunk.pk is not None
        # refresh_from_db returns a pgvector HalfVector wrapper
        assert len(chunk.embedding.to_list()) == 3072
        assert chunk.collection_name
        assert chunk.content

    def test_unique_constraint_rejects_duplicate_collection_document_chunk(self):
        DocumentChunkFactory(collection_name="project_99", document_id=1, chunk_id=0)
        with pytest.raises(IntegrityError):
            DocumentChunkFactory(collection_name="project_99", document_id=1, chunk_id=0)

    def test_same_chunk_id_allowed_in_different_collections(self):
        DocumentChunkFactory(collection_name="project_1", document_id=1, chunk_id=0)
        chunk = DocumentChunkFactory(collection_name="client_1", document_id=1, chunk_id=0)
        assert chunk.pk is not None

    def test_trigger_populates_search_vector_on_insert(self):
        chunk = DocumentChunkFactory(content="Die Finanzierung der Maschinen wurde genehmigt.")
        chunk.refresh_from_db()
        assert chunk.search_vector is not None

    def test_search_vector_matches_german_stemmed_query(self):
        from scribe.models import DocumentChunk

        chunk = DocumentChunkFactory(content="Die Maschinen wurden im Werk gebaut.")
        matches = DocumentChunk.objects.filter(
            pk=chunk.pk,
            search_vector=SearchQuery("Maschine", config="german"),
        )
        assert matches.exists()

    def test_trigger_updates_search_vector_on_content_update(self):
        from scribe.models import DocumentChunk

        chunk = DocumentChunkFactory(content="Photovoltaikanlage auf dem Dach.")
        chunk.content = "Leasingvertrag für Gabelstapler unterschrieben."
        chunk.save(update_fields=["content"])
        matches = DocumentChunk.objects.filter(
            pk=chunk.pk,
            search_vector=SearchQuery("Gabelstapler", config="german"),
        )
        assert matches.exists()

    def test_cascade_delete_via_project_document_fk(self):
        from scribe.models import DocumentChunk

        doc = ProtectedDocumentFactory()
        chunk = DocumentChunkFactory(
            project_document=doc,
            document_id=doc.id,
            project_id=doc.project_id,
            collection_name=f"project_{doc.project_id}",
        )
        doc.delete()
        assert not DocumentChunk.objects.filter(pk=chunk.pk).exists()

    def test_cascade_delete_via_client_document_fk(self):
        from scribe.models import DocumentChunk

        doc = ProtectedClientDocumentFactory()
        chunk = DocumentChunkFactory(
            client_document=doc,
            document_id=doc.id,
            project_id=None,
            collection_name=f"client_{doc.client_id}",
        )
        doc.delete()
        assert not DocumentChunk.objects.filter(pk=chunk.pk).exists()

    def test_both_document_fks_are_optional(self):
        chunk = DocumentChunkFactory(collection_name="general_chat", project_id=None)
        assert chunk.project_document is None
        assert chunk.client_document is None

    def test_reverse_accessor_is_chunks(self):
        doc = ProtectedDocumentFactory()
        DocumentChunkFactory(project_document=doc, document_id=doc.id)
        assert doc.chunks.count() == 1
