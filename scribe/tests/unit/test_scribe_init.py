"""SCRIBE facade initialization (Phase A6: backend-based, no eager Milvus connect)."""

import asyncio

from django.test import TestCase

from scribe.tests.mocks import mock_scribe_service


class TestSCRIBEInitialization(TestCase):
    collection_name = "test_collection"

    def test_init_sets_up_chunking_and_backends(self):
        with mock_scribe_service() as mocks:
            from scribe.scribe_milvus import SCRIBE

            scribe = SCRIBE(self.collection_name)

        self.assertEqual(scribe.collection_name, self.collection_name)
        self.assertIsNotNone(scribe.embeddings)
        self.assertIsNotNone(scribe.encoder)
        self.assertIsNotNone(scribe.chunker)
        self.assertEqual(scribe.chunker.name, "statistical_chunker")
        self.assertIsNotNone(scribe.contextualizer)
        self.assertIsNotNone(scribe.pdf_processor)
        self.assertIs(scribe.search_backend, mocks["backend"])
        self.assertIs(scribe.pg_backend, mocks["pg_backend"])

    def test_init_with_custom_chunk_sizes(self):
        with mock_scribe_service():
            from scribe.scribe_milvus import SCRIBE

            scribe = SCRIBE(self.collection_name, min_chunk_tokens=200, max_chunk_tokens=1000)

        self.assertEqual(scribe.min_chunk_tokens, 200)
        self.assertEqual(scribe.max_chunk_tokens, 1000)

    def test_async_create_method(self):
        async def run_test():
            with mock_scribe_service():
                from scribe.scribe_milvus import SCRIBE

                scribe = await SCRIBE.create(self.collection_name)
                self.assertEqual(scribe.collection_name, self.collection_name)
                self.assertIsNotNone(scribe.embeddings)

        asyncio.run(run_test())
