"""Integration tests for PgvectorBackend against a real Postgres + pgvector.

Phase A3 of the pgvector plan: FTS branch (German OR-of-terms), dense branch
(CosineDistance + SET LOCAL hnsw.ef_search), async search() orchestration and
the CRUD surface (insert/delete/drop_namespace/count/is_ready).
"""

import asyncio

import pytest
from asgiref.sync import sync_to_async
from django.db import connection, connections
from django.test.utils import CaptureQueriesContext

from scribe.backends.base import ChunkRecord, SearchFilter
from scribe.backends.pgvector_backend import PgvectorBackend
from scribe.models import DocumentChunk
from scribe.tests.factories import DocumentChunkFactory, deterministic_embedding

COLLECTION = "project_1"


def _filters(**kwargs):
    return SearchFilter(collection_name=COLLECTION, **kwargs)


@pytest.mark.django_db
class TestFtsSearch:
    def test_matches_any_term_not_all(self):
        # Neither chunk contains BOTH terms; AND semantics would return nothing.
        DocumentChunkFactory(collection_name=COLLECTION, content="Die Finanzierung der Anlage ist gesichert.")
        DocumentChunkFactory(collection_name=COLLECTION, content="Der Leasingvertrag wurde unterschrieben.")

        results = PgvectorBackend._fts_search_sync(
            "Finanzierung Leasingvertrag", limit=50, filter_kwargs={"collection_name": COLLECTION}
        )

        assert len(results) == 2, "each chunk matches one term — OR semantics must return both"

    def test_german_stemming_matches_inflected_form(self):
        DocumentChunkFactory(collection_name=COLLECTION, content="Mehrere Maschinen wurden geleast.")

        results = PgvectorBackend._fts_search_sync("Maschine", limit=50, filter_kwargs={"collection_name": COLLECTION})

        assert len(results) == 1

    def test_no_match_returns_empty(self):
        DocumentChunkFactory(collection_name=COLLECTION, content="Die Finanzierung der Anlage ist gesichert.")

        results = PgvectorBackend._fts_search_sync(
            "Quantencomputer", limit=50, filter_kwargs={"collection_name": COLLECTION}
        )

        assert results == []

    def test_filters_by_collection_name(self):
        DocumentChunkFactory(collection_name=COLLECTION, content="Finanzierung der Anlage.")
        DocumentChunkFactory(collection_name="client_9", content="Finanzierung der Anlage.")

        results = PgvectorBackend._fts_search_sync(
            "Finanzierung", limit=50, filter_kwargs={"collection_name": COLLECTION}
        )

        assert len(results) == 1

    def test_filters_by_document_id(self):
        match = DocumentChunkFactory(collection_name=COLLECTION, document_id=11, content="Finanzierung der Anlage.")
        DocumentChunkFactory(collection_name=COLLECTION, document_id=22, content="Finanzierung der Anlage.")

        results = PgvectorBackend._fts_search_sync(
            "Finanzierung", limit=50, filter_kwargs={"collection_name": COLLECTION, "document_id": 11}
        )

        assert [r["id"] for r in results] == [match.id]

    def test_filters_by_project_id(self):
        match = DocumentChunkFactory(collection_name=COLLECTION, project_id=5, content="Finanzierung der Anlage.")
        DocumentChunkFactory(collection_name=COLLECTION, project_id=6, content="Finanzierung der Anlage.")

        results = PgvectorBackend._fts_search_sync(
            "Finanzierung", limit=50, filter_kwargs={"collection_name": COLLECTION, "project_id": 5}
        )

        assert [r["id"] for r in results] == [match.id]

    def test_results_carry_rank_for_fusion(self):
        DocumentChunkFactory(collection_name=COLLECTION, content="Finanzierung Finanzierung Finanzierung.")

        results = PgvectorBackend._fts_search_sync(
            "Finanzierung", limit=50, filter_kwargs={"collection_name": COLLECTION}
        )

        assert results[0]["rank"] > 0


@pytest.mark.django_db
class TestDenseSearch:
    def test_closest_embedding_ranks_first(self):
        DocumentChunkFactory(collection_name=COLLECTION, embedding=deterministic_embedding(seed=1))
        target = DocumentChunkFactory(collection_name=COLLECTION, embedding=deterministic_embedding(seed=2))
        DocumentChunkFactory(collection_name=COLLECTION, embedding=deterministic_embedding(seed=3))

        results = PgvectorBackend._dense_search_sync(
            deterministic_embedding(seed=2), limit=10, filter_kwargs={"collection_name": COLLECTION}
        )

        assert results[0]["id"] == target.id
        assert results[0]["distance"] == pytest.approx(0.0, abs=1e-3)

    def test_sets_ef_search_to_limit(self):
        with CaptureQueriesContext(connection) as ctx:
            PgvectorBackend._dense_search_sync(
                deterministic_embedding(seed=1), limit=150, filter_kwargs={"collection_name": COLLECTION}
            )

        set_stmts = [q["sql"] for q in ctx.captured_queries if "hnsw.ef_search" in q["sql"].lower()]
        assert set_stmts, "expected a SET LOCAL hnsw.ef_search statement before the dense query"
        assert "150" in set_stmts[0]

    def test_ef_search_floor_is_40(self):
        with CaptureQueriesContext(connection) as ctx:
            PgvectorBackend._dense_search_sync(
                deterministic_embedding(seed=1), limit=5, filter_kwargs={"collection_name": COLLECTION}
            )

        set_stmts = [q["sql"] for q in ctx.captured_queries if "hnsw.ef_search" in q["sql"].lower()]
        assert set_stmts, "expected a SET LOCAL hnsw.ef_search statement"
        assert "40" in set_stmts[0], "should not lower ef_search below the pgvector default of 40"

    def test_filters_by_collection_name(self):
        match = DocumentChunkFactory(collection_name=COLLECTION, embedding=deterministic_embedding(seed=1))
        DocumentChunkFactory(collection_name="client_9", embedding=deterministic_embedding(seed=1))

        results = PgvectorBackend._dense_search_sync(
            deterministic_embedding(seed=1), limit=10, filter_kwargs={"collection_name": COLLECTION}
        )

        assert [r["id"] for r in results] == [match.id]


@pytest.mark.django_db(transaction=True)
class TestHybridSearch:
    """search() runs both branches via sync_to_async (thread = own connection),
    hence transaction=True so the worker thread sees committed test data."""

    @pytest.fixture(autouse=True)
    def _close_executor_thread_connections(self):
        """Close the sync_to_async executor thread's DB connection after each
        test, otherwise the test-DB teardown warns about a lingering session."""
        yield
        asyncio.run(sync_to_async(connections.close_all)())

    def test_fused_results_hydrate_documents_with_metadata(self):
        chunk = DocumentChunkFactory(
            collection_name=COLLECTION,
            document_id=42,
            content="Die Finanzierung der Maschinen ist gesichert.",
            original_content="Finanzierung der Maschinen.",
            document_path="docs/vertrag.pdf",
            page_number=3,
            has_context=True,
            embedding=deterministic_embedding(seed=2),
        )
        DocumentChunkFactory(
            collection_name=COLLECTION,
            content="Unrelated Inhalt ohne Treffer.",
            embedding=deterministic_embedding(seed=9),
        )

        backend = PgvectorBackend()
        results = asyncio.run(
            backend.search(
                query="Finanzierung",
                query_embedding=deterministic_embedding(seed=2),
                filters=_filters(),
            )
        )

        assert results
        top = results[0]
        assert top.document.page_content == chunk.content
        assert top.document.metadata["document_id"] == 42
        assert top.document.metadata["document_path"] == "docs/vertrag.pdf"
        assert top.document.metadata["page_number"] == 3
        assert top.document.metadata["has_context"] is True
        assert top.document.metadata["original_content"] == "Finanzierung der Maschinen."
        assert top.score > 0
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_chunk_in_both_branches_outranks_single_branch_hits(self):
        both = DocumentChunkFactory(
            collection_name=COLLECTION,
            content="Finanzierung der Anlage.",
            embedding=deterministic_embedding(seed=2),
        )
        DocumentChunkFactory(
            collection_name=COLLECTION,
            content="Leasingvertrag unterschrieben.",  # FTS miss for the query
            embedding=deterministic_embedding(seed=3),
        )

        backend = PgvectorBackend()
        results = asyncio.run(
            backend.search(
                query="Finanzierung",
                query_embedding=deterministic_embedding(seed=2),
                filters=_filters(),
            )
        )

        assert results[0].document.page_content == both.content
        assert len(results) == 2

    def test_empty_query_returns_empty(self):
        DocumentChunkFactory(collection_name=COLLECTION)

        backend = PgvectorBackend()
        results = asyncio.run(
            backend.search(query="", query_embedding=deterministic_embedding(seed=1), filters=_filters())
        )

        assert results == []

    def test_max_k_caps_result_count(self):
        for i in range(5):
            DocumentChunkFactory(
                collection_name=COLLECTION,
                content="Finanzierung der Anlage.",
                document_id=100 + i,
                embedding=deterministic_embedding(seed=i),
            )

        backend = PgvectorBackend()
        results = asyncio.run(
            backend.search(
                query="Finanzierung",
                query_embedding=deterministic_embedding(seed=0),
                filters=_filters(),
                max_k=2,
            )
        )

        assert len(results) == 2


@pytest.mark.django_db
class TestInsertChunks:
    def _record(self, document_id=1, chunk_id=0, **meta):
        metadata = {
            "document_id": document_id,
            "chunk_id": chunk_id,
            "project_id": 1,
            "document_path": "docs/test.pdf",
            "page_number": 2,
            "original_content": "Original.",
            "raw_section": "## Abschnitt",
            "image_path": "",
            "has_context": True,
        }
        metadata.update(meta)
        return ChunkRecord(
            content="Kontextualisierter Inhalt.",
            embedding=deterministic_embedding(seed=chunk_id),
            metadata=metadata,
        )

    def test_inserts_rows_with_all_fields(self):
        backend = PgvectorBackend()
        inserted = backend.insert_chunks([self._record(chunk_id=0), self._record(chunk_id=1)], _filters())

        assert inserted == 2
        chunk = DocumentChunk.objects.get(collection_name=COLLECTION, document_id=1, chunk_id=0)
        assert chunk.content == "Kontextualisierter Inhalt."
        assert chunk.project_id == 1
        assert chunk.document_path == "docs/test.pdf"
        assert chunk.page_number == 2
        assert chunk.original_content == "Original."
        assert chunk.raw_section == "## Abschnitt"
        assert chunk.has_context is True

    def test_maps_project_document_fk_from_metadata(self):
        from data_room.tests.factories import ProtectedDocumentFactory

        doc = ProtectedDocumentFactory()
        backend = PgvectorBackend()
        backend.insert_chunks(
            [self._record(document_id=doc.id, project_document_id=doc.id)],
            SearchFilter(collection_name=f"project_{doc.project_id}"),
        )

        chunk = DocumentChunk.objects.get(document_id=doc.id)
        assert chunk.project_document_id == doc.id

    def test_maps_client_document_fk_from_metadata(self):
        from data_room.tests.factories import ProtectedClientDocumentFactory

        doc = ProtectedClientDocumentFactory()
        backend = PgvectorBackend()
        backend.insert_chunks(
            [self._record(document_id=doc.id, client_document_id=doc.id)],
            SearchFilter(collection_name=f"client_{doc.client_id}"),
        )

        chunk = DocumentChunk.objects.get(document_id=doc.id)
        assert chunk.client_document_id == doc.id

    def test_reinsert_is_idempotent_via_ignore_conflicts(self):
        backend = PgvectorBackend()
        backend.insert_chunks([self._record()], _filters())
        backend.insert_chunks([self._record()], _filters())  # must not raise IntegrityError

        assert DocumentChunk.objects.filter(collection_name=COLLECTION).count() == 1

    def test_empty_list_inserts_nothing(self):
        backend = PgvectorBackend()
        assert backend.insert_chunks([], _filters()) == 0


@pytest.mark.django_db
class TestDeleteDropCount:
    def test_delete_by_document_id_within_collection(self):
        DocumentChunkFactory(collection_name=COLLECTION, document_id=1, chunk_id=0)
        DocumentChunkFactory(collection_name=COLLECTION, document_id=1, chunk_id=1)
        keep = DocumentChunkFactory(collection_name=COLLECTION, document_id=2, chunk_id=0)

        backend = PgvectorBackend()
        deleted = backend.delete(_filters(document_id=1))

        assert deleted == 2
        assert list(DocumentChunk.objects.values_list("id", flat=True)) == [keep.id]

    def test_drop_namespace_removes_only_that_collection(self):
        DocumentChunkFactory(collection_name=COLLECTION)
        other = DocumentChunkFactory(collection_name="client_9")

        backend = PgvectorBackend()
        assert backend.drop_namespace(COLLECTION) is True
        assert list(DocumentChunk.objects.values_list("id", flat=True)) == [other.id]

    def test_count_respects_filters(self):
        DocumentChunkFactory(collection_name=COLLECTION, document_id=1, chunk_id=0)
        DocumentChunkFactory(collection_name=COLLECTION, document_id=1, chunk_id=1)
        DocumentChunkFactory(collection_name=COLLECTION, document_id=2, chunk_id=0)
        DocumentChunkFactory(collection_name="client_9", document_id=1, chunk_id=0)

        backend = PgvectorBackend()
        assert backend.count(_filters()) == 3
        assert backend.count(_filters(document_id=1)) == 2


@pytest.mark.django_db
class TestIsReady:
    def test_ready_when_db_reachable_and_vector_extension_installed(self):
        assert PgvectorBackend().is_ready() is True
