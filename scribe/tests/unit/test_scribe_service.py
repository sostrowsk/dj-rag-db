"""Phase A6: SCRIBE facade — Postgres-SSOT indexing + backend-agnostic search.

The facade keeps its public API names (process_pdf, add_documents_to_collection,
search_similar_chunks, delete_documents, drop_collection,
check_milvus_health[_static], close) but delegates to the Phase-A2 backends:

- search: embed query once, hybrid search via the configured backend, then
  adaptive cutoff over the fused RRF scores (settings-driven).
- indexing: embed chunk contents once (batched), always write DocumentChunk
  rows via PgvectorBackend (delete-then-insert per document = idempotent
  re-index); mirror into Milvus only when VECTORSTORE_BACKEND == "milvus".
- delete/drop: always Postgres; Milvus best-effort when configured.
"""

import asyncio

from ai_router.types import Document
from django.test import TestCase, override_settings

from scribe.backends import SearchFilter, SearchResult
from scribe.tests.factories import DocumentFactory
from scribe.tests.mocks import make_search_backend_mock, mock_scribe_service


def _search_results(scores):
    return [
        SearchResult(
            document=Document(page_content=f"chunk {i}", metadata={"document_id": 1, "page_number": i}),
            score=score,
        )
        for i, score in enumerate(scores)
    ]


def _make_scribe(collection_name="project_1"):
    from scribe.scribe_milvus import SCRIBE

    return SCRIBE(collection_name)


@override_settings(VECTORSTORE_BACKEND="pgvector")
class TestSearchSimilarChunks(TestCase):
    def test_search_embeds_query_once_and_passes_settings_to_backend(self):
        backend = make_search_backend_mock(results=_search_results([0.5, 0.4, 0.39]))
        with mock_scribe_service(backend=backend) as mocks:
            scribe = _make_scribe()
            asyncio.run(scribe.search_similar_chunks("Maschinen Leasing", project_id=1))

        mocks["embeddings"].embed_query.assert_called_once_with("Maschinen Leasing")
        kwargs = backend.search.await_args.kwargs
        self.assertEqual(kwargs["initial_fetch_k"], 150)
        self.assertEqual(kwargs["max_k"], 50)
        self.assertEqual(kwargs["rrf_k"], 60)
        self.assertEqual(
            kwargs["filters"],
            SearchFilter(collection_name="project_1", project_id=1, document_id=None),
        )

    def test_search_returns_document_score_tuples(self):
        backend = make_search_backend_mock(results=_search_results([0.5, 0.4, 0.39]))
        with mock_scribe_service(backend=backend):
            scribe = _make_scribe()
            results = asyncio.run(scribe.search_similar_chunks("query"))

        self.assertEqual(len(results), 3)
        for document, score in results:
            self.assertIsInstance(document, Document)
            self.assertIsInstance(score, float)
        self.assertEqual([score for _, score in results], [0.5, 0.4, 0.39])

    def test_search_applies_adaptive_cutoff_to_backend_results(self):
        # Sharp relative-floor break after 4 strong scores.
        scores = [1.0, 0.9, 0.8, 0.7, 0.005, 0.004, 0.003, 0.002]
        backend = make_search_backend_mock(results=_search_results(scores))
        with mock_scribe_service(backend=backend):
            scribe = _make_scribe()
            results = asyncio.run(scribe.search_similar_chunks("query"))

        self.assertEqual(len(results), 4)

    def test_search_with_return_diagnostics_yields_retrieval_log_contract(self):
        scores = [1.0, 0.9, 0.8, 0.7, 0.005, 0.004]
        backend = make_search_backend_mock(results=_search_results(scores))
        with mock_scribe_service(backend=backend):
            scribe = _make_scribe()
            results, diagnostics = asyncio.run(scribe.search_similar_chunks("query", return_diagnostics=True))

        self.assertEqual(len(results), 4)
        self.assertEqual(diagnostics["candidate_scores"], scores)
        self.assertEqual(diagnostics["final_k"], 4)
        self.assertEqual(
            diagnostics["cutoff_config"],
            {
                "rel_floor": 0.35,
                "elbow_drop": 0.45,
                "min_k": 3,
                "max_k": 50,
                "backend": "pgvector",
            },
        )

    def test_search_with_no_hits_returns_empty_list(self):
        backend = make_search_backend_mock(results=[])
        with mock_scribe_service(backend=backend):
            scribe = _make_scribe()
            results, diagnostics = asyncio.run(scribe.search_similar_chunks("query", return_diagnostics=True))

        self.assertEqual(results, [])
        self.assertEqual(diagnostics["final_k"], 0)
        self.assertEqual(diagnostics["candidate_scores"], [])

    def test_search_honors_explicit_max_k(self):
        backend = make_search_backend_mock(results=_search_results([0.5, 0.49, 0.48, 0.47, 0.46]))
        with mock_scribe_service(backend=backend):
            scribe = _make_scribe()
            results = asyncio.run(scribe.search_similar_chunks("query", max_k=2))

        self.assertEqual(backend.search.await_args.kwargs["max_k"], 2)
        self.assertEqual(len(results), 2)


@override_settings(VECTORSTORE_BACKEND="pgvector", SCRIBE_USE_CONTEXTUAL_RETRIEVAL=False)
class TestAddDocumentsToCollection(TestCase):
    def _documents(self, document_id=7, count=3):
        return [
            DocumentFactory(document_id=document_id, chunk_id=i, project_id=1, page_number=i + 1) for i in range(count)
        ]

    def test_add_documents_embeds_once_and_inserts_postgres_rows(self):
        documents = self._documents()
        with mock_scribe_service() as mocks:
            scribe = _make_scribe()
            asyncio.run(scribe.add_documents_to_collection(documents, document_text="Volltext"))

        mocks["embeddings"].embed_documents.assert_called_once()
        pg = mocks["pg_backend"]
        pg.insert_chunks.assert_called_once()
        records, filters = pg.insert_chunks.call_args.args
        self.assertEqual(len(records), 3)
        self.assertEqual(filters.collection_name, "project_1")
        self.assertEqual([r.content for r in records], [d.page_content for d in documents])
        self.assertTrue(all(r.embedding == [0.1] * 8 for r in records))
        # Milvus mirror must stay silent on the pgvector backend.
        mocks["backend"].insert_chunks.assert_not_called()
        mocks["milvus_backend"].insert_chunks.assert_not_called()

    def test_add_documents_predeletes_existing_chunks_for_idempotent_reindex(self):
        documents = self._documents(document_id=42)
        with mock_scribe_service() as mocks:
            scribe = _make_scribe()
            asyncio.run(scribe.add_documents_to_collection(documents, document_text="Volltext"))

        pg = mocks["pg_backend"]
        pg.delete.assert_called_once_with(SearchFilter(collection_name="project_1", document_id=42))

    def test_add_documents_maps_project_document_fk_from_collection_namespace(self):
        documents = self._documents(document_id=7)
        with mock_scribe_service() as mocks:
            scribe = _make_scribe()
            asyncio.run(scribe.add_documents_to_collection(documents, document_text="Volltext"))

        records, _ = mocks["pg_backend"].insert_chunks.call_args.args
        self.assertTrue(all(r.metadata["project_document_id"] == 7 for r in records))

    def test_add_documents_maps_client_document_fk_for_client_collections(self):
        documents = self._documents(document_id=9)
        with mock_scribe_service() as mocks:
            scribe = _make_scribe("client_5")
            asyncio.run(scribe.add_documents_to_collection(documents, document_text="Volltext"))

        records, _ = mocks["pg_backend"].insert_chunks.call_args.args
        self.assertTrue(all(r.metadata["client_document_id"] == 9 for r in records))
        self.assertTrue(all("project_document_id" not in r.metadata for r in records))

    @override_settings(VECTORSTORE_BACKEND="milvus")
    def test_add_documents_mirrors_same_embeddings_into_milvus_when_active(self):
        documents = self._documents(document_id=7)
        backend = make_search_backend_mock()
        with mock_scribe_service(backend=backend) as mocks:
            scribe = _make_scribe()
            asyncio.run(scribe.add_documents_to_collection(documents, document_text="Volltext"))

        pg_records, _ = mocks["pg_backend"].insert_chunks.call_args.args
        milvus_records, milvus_filters = backend.insert_chunks.call_args.args
        self.assertEqual(milvus_filters.collection_name, "project_1")
        self.assertEqual([r.embedding for r in milvus_records], [r.embedding for r in pg_records])
        backend.delete.assert_called_once_with(SearchFilter(collection_name="project_1", document_id=7))

    def test_add_documents_with_empty_list_is_a_noop(self):
        with mock_scribe_service() as mocks:
            scribe = _make_scribe()
            asyncio.run(scribe.add_documents_to_collection([]))

        mocks["pg_backend"].insert_chunks.assert_not_called()
        mocks["embeddings"].embed_documents.assert_not_called()


@override_settings(VECTORSTORE_BACKEND="pgvector", MILVUS_HOST="localhost", MILVUS_PORT="19530")
class TestDeleteAndDrop(TestCase):
    def test_delete_documents_always_deletes_postgres(self):
        with mock_scribe_service() as mocks:
            scribe = _make_scribe()
            asyncio.run(scribe.delete_documents(document_id=11))

        mocks["pg_backend"].delete.assert_called_once_with(
            SearchFilter(collection_name="project_1", project_id=None, document_id=11)
        )

    def test_delete_documents_milvus_failure_is_swallowed(self):
        milvus = make_search_backend_mock()
        milvus.delete.side_effect = ConnectionError("milvus down")
        with mock_scribe_service(milvus_backend=milvus) as mocks:
            scribe = _make_scribe()
            asyncio.run(scribe.delete_documents(document_id=11))

        milvus.delete.assert_called_once()
        mocks["pg_backend"].delete.assert_called_once()

    @override_settings(MILVUS_HOST=None)
    def test_delete_documents_skips_milvus_when_not_configured(self):
        with mock_scribe_service() as mocks:
            scribe = _make_scribe()
            asyncio.run(scribe.delete_documents(document_id=11))

        mocks["milvus_backend"].delete.assert_not_called()
        mocks["pg_backend"].delete.assert_called_once()

    def test_delete_documents_without_filters_is_a_noop(self):
        with mock_scribe_service() as mocks:
            scribe = _make_scribe()
            asyncio.run(scribe.delete_documents())

        mocks["pg_backend"].delete.assert_not_called()
        mocks["milvus_backend"].delete.assert_not_called()

    def test_drop_collection_drops_postgres_namespace_and_milvus_best_effort(self):
        milvus = make_search_backend_mock()
        milvus.drop_namespace.side_effect = ConnectionError("milvus down")
        with mock_scribe_service(milvus_backend=milvus) as mocks:
            scribe = _make_scribe()
            result = asyncio.run(scribe.drop_collection())

        self.assertTrue(result)
        mocks["pg_backend"].drop_namespace.assert_called_once_with("project_1")
        milvus.drop_namespace.assert_called_once_with("project_1")


@override_settings(VECTORSTORE_BACKEND="pgvector")
class TestHealthAndInitialization(TestCase):
    def test_check_milvus_health_delegates_to_search_backend_readiness(self):
        backend = make_search_backend_mock(ready=False)
        with mock_scribe_service(backend=backend):
            scribe = _make_scribe()
            self.assertFalse(scribe.check_milvus_health())

        backend = make_search_backend_mock(ready=True)
        with mock_scribe_service(backend=backend):
            scribe = _make_scribe()
            self.assertTrue(scribe.check_milvus_health())

    @override_settings(MILVUS_HOST=None)
    def test_check_milvus_health_static_skips_when_milvus_not_configured(self):
        from scribe.scribe_milvus import SCRIBE

        result = SCRIBE.check_milvus_health_static()
        self.assertEqual(result["status"], "skipped")

    def test_initialize_collection_reports_backend_readiness_and_count(self):
        backend = make_search_backend_mock(ready=True)
        backend.count.return_value = 12
        with mock_scribe_service(backend=backend):
            scribe = _make_scribe()
            status = scribe.initialize_collection()

        self.assertEqual(status, {"success": True, "existed": True, "entity_count": 12})

    def test_initialize_collection_fails_when_backend_not_ready(self):
        backend = make_search_backend_mock(ready=False)
        with mock_scribe_service(backend=backend):
            scribe = _make_scribe()
            status = scribe.initialize_collection()

        self.assertFalse(status["success"])

    @override_settings(VECTORSTORE_BACKEND="milvus")
    def test_initialize_collection_ensures_milvus_collection_when_active(self):
        backend = make_search_backend_mock(ready=True)
        backend.ensure_collection.return_value = True  # collection was created
        with mock_scribe_service(backend=backend):
            scribe = _make_scribe()
            status = scribe.initialize_collection()

        backend.ensure_collection.assert_called_once_with("project_1")
        self.assertTrue(status["success"])
        self.assertFalse(status["existed"])
