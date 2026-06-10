"""SCRIBE facade construction: embedding-size detection.

Regression coverage: SCRIBE() construction must not require a live embedding
endpoint for known models — otherwise network hiccups break unrelated
operations like index removal (which generates no new embeddings).
"""

from django.test import TestCase

from scribe.tests.mocks import mock_scribe_service


class TestSCRIBEEmbeddingSize(TestCase):
    collection_name = "test_collection"

    def test_get_embedding_size_known_model_skips_api_call(self):
        with mock_scribe_service(embedding_model="text-embedding-3-large") as mocks:
            from scribe.scribe_milvus import SCRIBE

            scribe = SCRIBE(self.collection_name)
            self.assertEqual(scribe.embedding_size, 3072)
            mocks["embeddings"].embed_query.assert_not_called()

    def test_get_embedding_size_unknown_model_falls_back_to_probe(self):
        with mock_scribe_service(embedding_model="some-future-experimental-model") as mocks:
            mocks["embeddings"].embed_query.return_value = [0.1] * 999
            from scribe.scribe_milvus import SCRIBE

            scribe = SCRIBE(self.collection_name)
            self.assertEqual(scribe.embedding_size, 999)
            mocks["embeddings"].embed_query.assert_called_once()

    def test_embedding_size_exceeds_limit(self):
        with mock_scribe_service(embedding_model="some-future-experimental-model") as mocks:
            mocks["embeddings"].embed_query.return_value = [0.1] * 40000
            from scribe.scribe_milvus import SCRIBE

            with self.assertRaises(AssertionError):
                SCRIBE(self.collection_name)
