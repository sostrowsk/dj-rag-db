"""Tests for the SearchBackend abstraction and the settings-based backend factory."""

from ai_router.types import Document
from django.core.exceptions import ImproperlyConfigured
from django.test import SimpleTestCase, override_settings

from scribe.backends import get_search_backend
from scribe.backends.base import ChunkRecord, SearchBackend, SearchFilter, SearchResult


class TestBackendDataclasses(SimpleTestCase):
    """SearchFilter / SearchResult / ChunkRecord carry the backend-agnostic contract."""

    def test_search_filter_requires_only_collection_name(self):
        f = SearchFilter(collection_name="project_42")
        self.assertEqual(f.collection_name, "project_42")
        self.assertIsNone(f.project_id)
        self.assertIsNone(f.document_id)

    def test_search_filter_accepts_optional_ids(self):
        f = SearchFilter(collection_name="project_42", project_id=42, document_id=7)
        self.assertEqual(f.project_id, 42)
        self.assertEqual(f.document_id, 7)

    def test_search_result_holds_document_and_score(self):
        doc = Document(page_content="Inhalt", metadata={"page_number": 3})
        result = SearchResult(document=doc, score=0.0123)
        self.assertIs(result.document, doc)
        self.assertEqual(result.score, 0.0123)

    def test_chunk_record_defaults_to_empty_metadata(self):
        record = ChunkRecord(content="Text", embedding=[0.1, 0.2])
        self.assertEqual(record.metadata, {})

    def test_chunk_record_metadata_is_not_shared_between_instances(self):
        a = ChunkRecord(content="a", embedding=[0.0])
        b = ChunkRecord(content="b", embedding=[0.0])
        a.metadata["key"] = "value"
        self.assertEqual(b.metadata, {})


class TestSearchBackendInterface(SimpleTestCase):
    """SearchBackend is abstract and pins the method contract."""

    def test_cannot_be_instantiated_directly(self):
        with self.assertRaises(TypeError):
            SearchBackend()

    def test_declares_full_method_contract(self):
        expected = {
            "search",
            "insert_chunks",
            "delete",
            "drop_namespace",
            "count",
            "is_ready",
        }
        self.assertEqual(set(SearchBackend.__abstractmethods__), expected)


class TestGetSearchBackend(SimpleTestCase):
    """get_search_backend() resolves the backend from settings.VECTORSTORE_BACKEND."""

    @override_settings(VECTORSTORE_BACKEND="pgvector")
    def test_pgvector_setting_returns_pgvector_backend(self):
        from scribe.backends.pgvector_backend import PgvectorBackend

        backend = get_search_backend()
        self.assertIsInstance(backend, PgvectorBackend)
        self.assertIsInstance(backend, SearchBackend)

    @override_settings(VECTORSTORE_BACKEND="milvus")
    def test_milvus_setting_returns_milvus_backend(self):
        from scribe.backends.milvus_backend import MilvusBackend

        backend = get_search_backend()
        self.assertIsInstance(backend, MilvusBackend)
        self.assertIsInstance(backend, SearchBackend)

    @override_settings(VECTORSTORE_BACKEND="qdrant")
    def test_unknown_backend_raises_improperly_configured(self):
        with self.assertRaises(ImproperlyConfigured) as ctx:
            get_search_backend()
        self.assertIn("qdrant", str(ctx.exception))

    @override_settings(VECTORSTORE_BACKEND="pgvector")
    def test_returns_fresh_instance_per_call(self):
        self.assertIsNot(get_search_backend(), get_search_backend())
