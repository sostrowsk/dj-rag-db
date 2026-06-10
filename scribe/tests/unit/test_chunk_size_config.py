"""Configurable chunk-size support of the SCRIBE facade."""

from django.conf import settings
from django.test import TestCase, override_settings

from scribe.tests.mocks import mock_scribe_service


class TestChunkSizeConfiguration(TestCase):
    """Test configurable chunk size support."""

    collection_name = "test_collection"

    def _make_scribe(self, **kwargs):
        with mock_scribe_service():
            from scribe.scribe_milvus import SCRIBE

            return SCRIBE(self.collection_name, **kwargs)

    def test_default_chunk_sizes_from_settings(self):
        scribe = self._make_scribe()

        expected_min = getattr(settings, "SCRIBE_MIN_CHUNK_TOKENS", 500)
        expected_max = getattr(settings, "SCRIBE_MAX_CHUNK_TOKENS", int(scribe.embedding_size * 0.95))

        self.assertEqual(scribe.chunker.min_split_tokens, expected_min)
        self.assertEqual(scribe.chunker.max_split_tokens, expected_max)

    def test_custom_chunk_sizes_via_constructor(self):
        scribe = self._make_scribe(min_chunk_tokens=100, max_chunk_tokens=1000)

        self.assertEqual(scribe.chunker.min_split_tokens, 100)
        self.assertEqual(scribe.chunker.max_split_tokens, 1000)

    @override_settings(SCRIBE_MIN_CHUNK_TOKENS=250, SCRIBE_MAX_CHUNK_TOKENS=2000)
    def test_settings_override(self):
        scribe = self._make_scribe()

        self.assertEqual(scribe.chunker.min_split_tokens, 250)
        self.assertEqual(scribe.chunker.max_split_tokens, 2000)

    @override_settings(SCRIBE_MIN_CHUNK_TOKENS=250, SCRIBE_MAX_CHUNK_TOKENS=2000)
    def test_constructor_overrides_settings(self):
        scribe = self._make_scribe(min_chunk_tokens=150, max_chunk_tokens=1500)

        self.assertEqual(scribe.chunker.min_split_tokens, 150)
        self.assertEqual(scribe.chunker.max_split_tokens, 1500)
