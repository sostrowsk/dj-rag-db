"""Tests for the Phase A7 rollout management commands.

- backfill_chunks_from_milvus: Milvus -> DocumentChunk rows (no re-embedding)
- reindex_documents: full rebuild via the normal SCRIBE pipeline
- rebuild_milvus_from_postgres: Milvus as derived index from Postgres SSOT

Milvus is always mocked (scribe/tests/mocks.py); Postgres is real.
"""

from io import StringIO
from unittest.mock import Mock, patch

import pytest
from django.core.management import call_command

from data_room.tests.factories import ProtectedClientDocumentFactory, ProtectedDocumentFactory
from scribe.models import DocumentChunk
from scribe.tests.factories import DocumentChunkFactory, deterministic_embedding
from scribe.tests.mocks import mock_milvus_client


def _milvus_row(document_id, chunk_id=0, content="Maschinen und Anlagen", **overrides):
    row = {
        "content": content,
        "embedding": deterministic_embedding(seed=chunk_id),
        "document_id": document_id,
        "project_id": overrides.pop("project_id", 0),
        "chunk_id": chunk_id,
        "document_path": "docs/test.pdf",
        "raw_section": "",
        "image_path": "",
        "original_content": "",
        "page_number": 3,
        "has_context": True,
    }
    row.update(overrides)
    return row


def _make_query_iterator(batches):
    """Mock pymilvus QueryIterator: next() yields batches, then []."""
    iterator = Mock()
    iterator.next = Mock(side_effect=list(batches) + [[]])
    iterator.close = Mock()
    return iterator


def _wire_iterators(client, batches_by_collection):
    client.query_iterator.side_effect = lambda collection_name, **kwargs: _make_query_iterator(
        batches_by_collection.get(collection_name, [])
    )


# ---------------------------------------------------------------------------
# backfill_chunks_from_milvus
# ---------------------------------------------------------------------------


@pytest.mark.django_db
class TestBackfillChunksFromMilvus:
    def test_backfill_creates_rows_and_maps_project_document_fk(self):
        doc = ProtectedDocumentFactory()
        collection = f"project_{doc.project_id}"
        with mock_milvus_client(collections=[collection]) as mocks:
            _wire_iterators(
                mocks["client"], {collection: [[_milvus_row(doc.id, chunk_id=0), _milvus_row(doc.id, chunk_id=1)]]}
            )
            call_command("backfill_chunks_from_milvus", stdout=StringIO())

        chunks = DocumentChunk.objects.filter(collection_name=collection).order_by("chunk_id")
        assert chunks.count() == 2
        first = chunks.first()
        assert first.project_document_id == doc.id
        assert first.client_document_id is None
        assert first.document_id == doc.id
        assert first.content == "Maschinen und Anlagen"
        assert first.page_number == 3
        assert first.has_context is True
        assert len(first.embedding.to_list()) == 3072

    def test_backfill_maps_client_document_fk(self):
        doc = ProtectedClientDocumentFactory()
        collection = f"client_{doc.client_id}"
        with mock_milvus_client(collections=[collection]) as mocks:
            _wire_iterators(mocks["client"], {collection: [[_milvus_row(doc.id)]]})
            call_command("backfill_chunks_from_milvus", stdout=StringIO())

        chunk = DocumentChunk.objects.get(collection_name=collection)
        assert chunk.client_document_id == doc.id
        assert chunk.project_document_id is None

    def test_backfill_inserts_orphans_with_null_fk_and_reports_them(self):
        collection = "project_424242"
        out = StringIO()
        with mock_milvus_client(collections=[collection]) as mocks:
            _wire_iterators(mocks["client"], {collection: [[_milvus_row(999999991)]]})
            call_command("backfill_chunks_from_milvus", stdout=out)

        chunk = DocumentChunk.objects.get(collection_name=collection)
        assert chunk.project_document_id is None
        assert chunk.document_id == 999999991
        assert "1" in out.getvalue() and "Orphan" in out.getvalue()

    def test_backfill_reassigns_duplicate_chunk_ids_instead_of_dropping(self):
        """Regression (Codex P2): Milvus-Bestandsdaten enthalten doppelte
        (document_id, chunk_id)-Paare (Text- vs. Image-Chunks mit eigenem
        idx ab 0) — die Duplikate duerfen nicht via ignore_conflicts still
        verworfen werden, sondern bekommen die naechste freie chunk_id."""
        doc = ProtectedDocumentFactory()
        collection = f"project_{doc.project_id}"
        batch1 = [
            _milvus_row(doc.id, chunk_id=0, content="Text 0"),
            _milvus_row(doc.id, chunk_id=1, content="Text 1"),
        ]
        # Image-Chunk aus Folge-Batch kollidiert mit Text-Chunk 0
        batch2 = [_milvus_row(doc.id, chunk_id=0, content="Bild 0", image_path="/tmp/img.png")]
        with mock_milvus_client(collections=[collection]) as mocks:
            _wire_iterators(mocks["client"], {collection: [batch1, batch2]})
            call_command("backfill_chunks_from_milvus", stdout=StringIO())

        chunks = DocumentChunk.objects.filter(collection_name=collection)
        assert chunks.count() == 3
        assert set(chunks.values_list("content", flat=True)) == {"Text 0", "Text 1", "Bild 0"}
        chunk_ids = list(chunks.values_list("chunk_id", flat=True))
        assert len(chunk_ids) == len(set(chunk_ids)), f"chunk_ids not unique: {chunk_ids}"

    def test_backfill_skips_non_scribe_collections(self):
        with mock_milvus_client(collections=["some_other_collection"]) as mocks:
            call_command("backfill_chunks_from_milvus", stdout=StringIO())
            mocks["client"].query_iterator.assert_not_called()
        assert DocumentChunk.objects.count() == 0

    def test_backfill_collection_flag_limits_to_one_collection(self):
        doc = ProtectedDocumentFactory()
        target = f"project_{doc.project_id}"
        with mock_milvus_client(collections=[target, "general_chat"]) as mocks:
            _wire_iterators(mocks["client"], {target: [[_milvus_row(doc.id)]], "general_chat": [[_milvus_row(1)]]})
            call_command("backfill_chunks_from_milvus", "--collection", target, stdout=StringIO())

        assert DocumentChunk.objects.filter(collection_name=target).count() == 1
        assert DocumentChunk.objects.filter(collection_name="general_chat").count() == 0

    def test_backfill_dry_run_writes_nothing(self):
        doc = ProtectedDocumentFactory()
        collection = f"project_{doc.project_id}"
        out = StringIO()
        with mock_milvus_client(collections=[collection]) as mocks:
            _wire_iterators(mocks["client"], {collection: [[_milvus_row(doc.id)]]})
            call_command("backfill_chunks_from_milvus", "--dry-run", stdout=out)

        assert DocumentChunk.objects.count() == 0
        assert "Dry-Run" in out.getvalue() or "dry-run" in out.getvalue().lower()

    def test_backfill_rerun_does_not_duplicate_chunks(self):
        doc = ProtectedDocumentFactory()
        collection = f"project_{doc.project_id}"
        for _ in range(2):
            with mock_milvus_client(collections=[collection]) as mocks:
                _wire_iterators(mocks["client"], {collection: [[_milvus_row(doc.id, chunk_id=7)]]})
                call_command("backfill_chunks_from_milvus", stdout=StringIO())

        assert DocumentChunk.objects.filter(collection_name=collection, chunk_id=7).count() == 1

    def test_backfill_rerun_recovers_missing_chunk_without_duplicating_content(self):
        """Regression (Codex P2, Folge-Finding): Re-Run gegen eine teilweise
        backfillte Collection (Bestand aus der Zeit VOR dem Duplikat-Fix, dem
        also der Image-Chunk fehlt). Auch wenn der Milvus-Stream den
        Image-Chunk ZUERST liefert, darf weder bestehender Content dupliziert
        noch der fehlende Chunk erneut verworfen werden."""
        doc = ProtectedDocumentFactory()
        collection = f"project_{doc.project_id}"
        # Bestand aus altem (kaputtem) Backfill: nur die Text-Chunks 0 und 1
        DocumentChunkFactory(
            collection_name=collection, project_document=doc, document_id=doc.id, chunk_id=0, content="Text 0"
        )
        DocumentChunkFactory(
            collection_name=collection, project_document=doc, document_id=doc.id, chunk_id=1, content="Text 1"
        )
        # Milvus liefert den kollidierenden Image-Chunk vor den Text-Chunks
        batch = [
            _milvus_row(doc.id, chunk_id=0, content="Bild 0", image_path="/tmp/img.png"),
            _milvus_row(doc.id, chunk_id=0, content="Text 0"),
            _milvus_row(doc.id, chunk_id=1, content="Text 1"),
        ]
        with mock_milvus_client(collections=[collection]) as mocks:
            _wire_iterators(mocks["client"], {collection: [batch]})
            call_command("backfill_chunks_from_milvus", stdout=StringIO())

        chunks = DocumentChunk.objects.filter(collection_name=collection)
        assert chunks.count() == 3
        contents = list(chunks.values_list("content", flat=True))
        assert sorted(contents) == ["Bild 0", "Text 0", "Text 1"], f"unexpected contents: {contents}"
        chunk_ids = list(chunks.values_list("chunk_id", flat=True))
        assert len(chunk_ids) == len(set(chunk_ids)), f"chunk_ids not unique: {chunk_ids}"

    def test_backfill_general_chat_has_no_fk(self):
        with mock_milvus_client(collections=["general_chat"]) as mocks:
            _wire_iterators(mocks["client"], {"general_chat": [[_milvus_row(55)]]})
            call_command("backfill_chunks_from_milvus", stdout=StringIO())

        chunk = DocumentChunk.objects.get(collection_name="general_chat")
        assert chunk.project_document_id is None
        assert chunk.client_document_id is None


# ---------------------------------------------------------------------------
# reindex_documents
# ---------------------------------------------------------------------------


@pytest.mark.django_db
class TestReindexDocuments:
    def _call(self, *args, **kwargs):
        out = StringIO()
        # Lazily resolved via scribe.conf (import_string), so patch the source.
        with patch("data_room.tasks.index_document.index_document_task") as task:
            call_command("reindex_documents", *args, "--user-id", "1", stdout=out, **kwargs)
        return task, out.getvalue()

    def test_reindex_dispatches_pipeline_for_project_and_client_docs(self):
        project_doc = ProtectedDocumentFactory(markdown="# Bilanz", tokens=100)
        client_doc = ProtectedClientDocumentFactory(markdown="# GuV", tokens=200)
        task, _ = self._call()
        task.delay.assert_any_call(project_doc.id, 1, "ProtectedProjectDocument")
        task.delay.assert_any_call(client_doc.id, 1, "ProtectedClientDocument")
        assert task.delay.call_count == 2

    def test_reindex_skips_documents_without_markdown(self):
        ProtectedDocumentFactory(markdown="")
        task, _ = self._call()
        task.delay.assert_not_called()

    def test_reindex_only_missing_skips_docs_with_existing_chunks(self):
        indexed = ProtectedDocumentFactory(markdown="# A", tokens=10)
        missing = ProtectedDocumentFactory(markdown="# B", tokens=10)
        DocumentChunkFactory(
            collection_name=f"project_{indexed.project_id}",
            document_id=indexed.id,
            project_id=indexed.project_id,
        )
        task, _ = self._call("--only-missing")
        task.delay.assert_called_once_with(missing.id, 1, "ProtectedProjectDocument")

    def test_reindex_project_flag_filters_to_project_docs(self):
        target = ProtectedDocumentFactory(markdown="# A", tokens=10)
        ProtectedDocumentFactory(markdown="# B", tokens=10)  # other project
        ProtectedClientDocumentFactory(markdown="# C", tokens=10)
        task, _ = self._call("--project", str(target.project_id))
        task.delay.assert_called_once_with(target.id, 1, "ProtectedProjectDocument")

    def test_reindex_document_flag_selects_single_document(self):
        target = ProtectedDocumentFactory(markdown="# A", tokens=10)
        ProtectedDocumentFactory(markdown="# B", tokens=10)
        task, _ = self._call("--document", str(target.id))
        task.delay.assert_called_once_with(target.id, 1, "ProtectedProjectDocument")

    def test_reindex_limit_caps_dispatch_count(self):
        ProtectedDocumentFactory(markdown="# A", tokens=10)
        ProtectedDocumentFactory(markdown="# B", tokens=10)
        task, _ = self._call("--limit", "1")
        assert task.delay.call_count == 1

    def test_reindex_logs_doc_count_and_token_sum_before_start(self):
        # ProtectedBaseDocument.save() recomputes tokens from markdown, so
        # assert against the persisted values rather than factory kwargs.
        doc_a = ProtectedDocumentFactory(markdown="# A")
        doc_b = ProtectedDocumentFactory(markdown="# B")
        doc_a.refresh_from_db()
        doc_b.refresh_from_db()
        _, output = self._call()
        assert "2 Dokumente" in output
        assert f"~{doc_a.tokens + doc_b.tokens} Tokens" in output


# ---------------------------------------------------------------------------
# rebuild_milvus_from_postgres
# ---------------------------------------------------------------------------


@pytest.mark.django_db
class TestRebuildMilvusFromPostgres:
    def test_rebuild_drops_recreates_and_inserts_from_postgres(self):
        chunk = DocumentChunkFactory(collection_name="project_9", project_id=9, document_id=4, chunk_id=0)
        with mock_milvus_client(insert_count=1) as mocks:
            client = mocks["client"]
            # drop_namespace sees the old collection, ensure_collection then re-creates it
            client.has_collection.side_effect = [True, False]
            call_command("rebuild_milvus_from_postgres", stdout=StringIO())

        client.drop_collection.assert_called_once_with("project_9")
        client.create_collection.assert_called_once()
        client.insert.assert_called_once()
        _, kwargs = client.insert.call_args
        rows = kwargs["data"]
        assert len(rows) == 1
        assert rows[0]["content"] == chunk.content
        assert rows[0]["document_id"] == 4
        # embeddings come straight from Postgres (halfvec -> float16 precision)
        assert rows[0]["embedding"] == pytest.approx(list(chunk.embedding), abs=1e-2)

    def test_rebuild_truncates_overlong_content_with_warning(self):
        DocumentChunkFactory(collection_name="project_9", content="x" * 70000)
        out = StringIO()
        with mock_milvus_client(insert_count=1) as mocks:
            client = mocks["client"]
            client.has_collection.side_effect = [True, False]
            call_command("rebuild_milvus_from_postgres", stdout=out)

        _, kwargs = client.insert.call_args
        assert len(kwargs["data"][0]["content"].encode("utf-8")) <= 65000
        assert "truncat" in out.getvalue().lower() or "gek" in out.getvalue().lower()

    def test_rebuild_collection_flag_limits_to_one_collection(self):
        DocumentChunkFactory(collection_name="project_1", project_id=1)
        DocumentChunkFactory(collection_name="client_2", project_id=None)
        with mock_milvus_client(insert_count=1) as mocks:
            client = mocks["client"]
            client.has_collection.side_effect = [True, False]
            call_command("rebuild_milvus_from_postgres", "--collection", "client_2", stdout=StringIO())

        client.drop_collection.assert_called_once_with("client_2")
        _, kwargs = client.insert.call_args
        assert client.insert.call_args[1]["collection_name"] == "client_2"
        assert len(kwargs["data"]) == 1

    def test_rebuild_dry_run_makes_no_milvus_writes(self):
        DocumentChunkFactory(collection_name="project_1")
        out = StringIO()
        with mock_milvus_client() as mocks:
            call_command("rebuild_milvus_from_postgres", "--dry-run", stdout=out)
            mocks["client"].drop_collection.assert_not_called()
            mocks["client"].insert.assert_not_called()
        assert "Dry-Run" in out.getvalue() or "dry-run" in out.getvalue().lower()

    def test_rebuild_makes_no_embedding_api_calls(self):
        DocumentChunkFactory(collection_name="project_1")
        with mock_milvus_client(insert_count=1) as mocks:
            mocks["client"].has_collection.side_effect = [True, False]
            with patch("ai_router.azure_client.openai_embeddings") as embeddings:
                call_command("rebuild_milvus_from_postgres", stdout=StringIO())
                embeddings.assert_not_called()
