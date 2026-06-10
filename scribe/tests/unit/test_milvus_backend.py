"""Tests for the MilvusBackend (direct pymilvus MilvusClient, no langchain).

All MilvusClient interaction is mocked via scribe.tests.mocks.mock_milvus_client;
schema/index/request construction is asserted on the recorded call arguments.
"""

import asyncio

from django.test import SimpleTestCase, override_settings
from pymilvus import DataType

from scribe.backends.base import ChunkRecord, SearchFilter
from scribe.backends.milvus_backend import MilvusBackend
from scribe.tests.mocks import mock_milvus_client


def _field_calls(schema_mock):
    """Map field_name -> kwargs for every schema.add_field(...) call."""
    fields = {}
    for call in schema_mock.add_field.call_args_list:
        args, kwargs = call
        name = kwargs.get("field_name", args[0] if args else None)
        datatype = kwargs.get("datatype", args[1] if len(args) > 1 else None)
        merged = dict(kwargs)
        merged["datatype"] = datatype
        fields[name] = merged
    return fields


class TestEnsureCollection(SimpleTestCase):
    """ensure_collection builds the SCRIBE schema with BM25 function and modern indexes."""

    def test_creates_schema_with_expected_fields(self):
        with mock_milvus_client() as mocks:
            backend = MilvusBackend(embedding_dim=3072)
            created = backend.ensure_collection("project_42")

        self.assertTrue(created)
        fields = _field_calls(mocks["schema"])
        self.assertEqual(
            set(fields),
            {
                "id",
                "content",
                "embedding",
                "bm25_sparse",
                "document_id",
                "project_id",
                "chunk_id",
                "document_path",
                "raw_section",
                "image_path",
                "original_content",
                "page_number",
                "has_context",
            },
        )
        self.assertTrue(fields["id"]["is_primary"])
        self.assertTrue(fields["id"]["auto_id"])
        self.assertEqual(fields["id"]["datatype"], DataType.INT64)
        self.assertEqual(fields["content"]["datatype"], DataType.VARCHAR)
        self.assertEqual(fields["content"]["max_length"], 65535)
        self.assertTrue(fields["content"]["enable_analyzer"])
        self.assertEqual(fields["embedding"]["datatype"], DataType.FLOAT_VECTOR)
        self.assertEqual(fields["embedding"]["dim"], 3072)
        self.assertEqual(fields["bm25_sparse"]["datatype"], DataType.SPARSE_FLOAT_VECTOR)
        self.assertEqual(fields["document_id"]["datatype"], DataType.INT64)
        self.assertEqual(fields["project_id"]["datatype"], DataType.INT64)
        self.assertEqual(fields["chunk_id"]["datatype"], DataType.INT64)
        self.assertEqual(fields["page_number"]["datatype"], DataType.INT64)
        self.assertTrue(fields["page_number"]["nullable"])
        self.assertEqual(fields["has_context"]["datatype"], DataType.BOOL)
        for varchar_field in ("document_path", "raw_section", "image_path", "original_content"):
            self.assertEqual(fields[varchar_field]["datatype"], DataType.VARCHAR)
            self.assertEqual(fields[varchar_field]["max_length"], 65535)

    def test_respects_custom_embedding_dim(self):
        with mock_milvus_client() as mocks:
            MilvusBackend(embedding_dim=1536).ensure_collection("project_42")

        fields = _field_calls(mocks["schema"])
        self.assertEqual(fields["embedding"]["dim"], 1536)

    def test_adds_bm25_function_from_content_to_sparse(self):
        with mock_milvus_client() as mocks:
            MilvusBackend().ensure_collection("project_42")

        mocks["schema"].add_function.assert_called_once()
        function = mocks["schema"].add_function.call_args[0][0]
        self.assertEqual(function.input_field_names, ["content"])
        self.assertEqual(function.output_field_names, ["bm25_sparse"])

    def test_creates_hnsw_cosine_and_sparse_bm25_indexes(self):
        with mock_milvus_client() as mocks:
            MilvusBackend().ensure_collection("project_42")

        index_calls = {
            call.kwargs["field_name"]: call.kwargs for call in mocks["index_params"].add_index.call_args_list
        }
        self.assertEqual(index_calls["embedding"]["index_type"], "HNSW")
        self.assertEqual(index_calls["embedding"]["metric_type"], "COSINE")
        self.assertEqual(index_calls["embedding"]["params"], {"M": 16, "efConstruction": 64})
        self.assertEqual(index_calls["bm25_sparse"]["index_type"], "SPARSE_INVERTED_INDEX")
        self.assertEqual(index_calls["bm25_sparse"]["metric_type"], "BM25")

        create_kwargs = mocks["client"].create_collection.call_args.kwargs
        self.assertEqual(create_kwargs["collection_name"], "project_42")
        self.assertIs(create_kwargs["schema"], mocks["schema"])
        self.assertIs(create_kwargs["index_params"], mocks["index_params"])

    def test_is_idempotent_when_collection_exists(self):
        with mock_milvus_client(has_collection=True) as mocks:
            created = MilvusBackend().ensure_collection("project_42")

        self.assertFalse(created)
        mocks["client"].create_collection.assert_not_called()
        mocks["schema"].add_field.assert_not_called()


class TestHybridSearch(SimpleTestCase):
    """search() builds dense + BM25 AnnSearchRequests fused via RRFRanker."""

    @staticmethod
    def _hit(hit_id=1, distance=0.032, **entity):
        defaults = {
            "content": "Chunk-Inhalt",
            "document_id": 7,
            "document_path": "doc.pdf",
            "page_number": 3,
            "original_content": "",
            "has_context": True,
        }
        defaults.update(entity)
        return {"id": hit_id, "distance": distance, "entity": defaults}

    def test_builds_dense_and_sparse_requests_with_rrf_ranker(self):
        with mock_milvus_client() as mocks:
            backend = MilvusBackend()
            asyncio.run(
                backend.search(
                    query="Leasingvertrag Maschinen",
                    query_embedding=[0.1] * 4,
                    filters=SearchFilter(collection_name="project_42"),
                    initial_fetch_k=99,
                    max_k=11,
                    rrf_k=17,
                )
            )

        call = mocks["client"].hybrid_search.call_args
        self.assertEqual(call.kwargs["collection_name"], "project_42")
        self.assertEqual(call.kwargs["limit"], 11)
        self.assertEqual(
            call.kwargs["output_fields"],
            ["content", "document_id", "document_path", "page_number", "original_content", "has_context"],
        )

        dense_req, sparse_req = call.kwargs["reqs"]
        self.assertEqual(dense_req.anns_field, "embedding")
        self.assertEqual(dense_req.param, {"metric_type": "COSINE"})
        self.assertEqual(dense_req.limit, 99)
        self.assertEqual(sparse_req.anns_field, "bm25_sparse")
        self.assertEqual(sparse_req.param, {"metric_type": "BM25"})
        self.assertEqual(sparse_req.limit, 99)

        ranker = call.kwargs["ranker"]
        self.assertEqual(ranker.dict()["params"]["k"], 17)

    def test_applies_document_and_project_filter_expr(self):
        with mock_milvus_client() as mocks:
            asyncio.run(
                MilvusBackend().search(
                    query="Bilanz",
                    query_embedding=[0.1] * 4,
                    filters=SearchFilter(collection_name="project_42", project_id=42, document_id=7),
                )
            )

        dense_req, sparse_req = mocks["client"].hybrid_search.call_args.kwargs["reqs"]
        self.assertEqual(dense_req.expr, "project_id == 42 and document_id == 7")
        self.assertEqual(sparse_req.expr, "project_id == 42 and document_id == 7")

    def test_maps_hits_to_search_results_with_fused_score(self):
        hits = [
            self._hit(hit_id=1, distance=0.032, original_content="Roher Inhalt"),
            self._hit(hit_id=2, distance=0.016, content="Zweiter Chunk", original_content=""),
        ]
        with mock_milvus_client(hybrid_search_results=hits):
            results = asyncio.run(
                MilvusBackend().search(
                    query="Bilanz",
                    query_embedding=[0.1] * 4,
                    filters=SearchFilter(collection_name="project_42"),
                )
            )

        self.assertEqual(len(results), 2)
        first, second = results
        self.assertEqual(first.score, 0.032)
        self.assertEqual(first.document.page_content, "Chunk-Inhalt")
        self.assertEqual(first.document.metadata["document_id"], 7)
        self.assertEqual(first.document.metadata["document_path"], "doc.pdf")
        self.assertEqual(first.document.metadata["page_number"], 3)
        self.assertTrue(first.document.metadata["has_context"])
        self.assertEqual(first.document.metadata["original_content"], "Roher Inhalt")
        # empty original_content is omitted, mirroring the pgvector backend
        self.assertNotIn("original_content", second.document.metadata)
        self.assertEqual(second.score, 0.016)

    def test_empty_query_returns_empty_without_calling_milvus(self):
        with mock_milvus_client() as mocks:
            results = asyncio.run(
                MilvusBackend().search(
                    query="",
                    query_embedding=[0.1] * 4,
                    filters=SearchFilter(collection_name="project_42"),
                )
            )

        self.assertEqual(results, [])
        mocks["client"].hybrid_search.assert_not_called()


class TestInsertChunks(SimpleTestCase):
    """insert_chunks sends pre-computed embeddings, ensuring the collection first."""

    def test_inserts_rows_with_precomputed_embeddings(self):
        chunks = [
            ChunkRecord(
                content="Erster Chunk",
                embedding=[0.1, 0.2],
                metadata={
                    "document_id": 7,
                    "project_id": 42,
                    "chunk_id": 0,
                    "document_path": "doc.pdf",
                    "page_number": 3,
                    "has_context": True,
                    "original_content": "Roh",
                },
            ),
            ChunkRecord(content="Zweiter Chunk", embedding=[0.3, 0.4], metadata={"chunk_id": 1}),
        ]
        filters = SearchFilter(collection_name="project_42", project_id=42, document_id=7)

        with mock_milvus_client(insert_count=2) as mocks:
            inserted = MilvusBackend().insert_chunks(chunks, filters)

        self.assertEqual(inserted, 2)
        call = mocks["client"].insert.call_args
        self.assertEqual(call.kwargs["collection_name"], "project_42")
        rows = call.kwargs["data"]
        self.assertEqual(rows[0]["content"], "Erster Chunk")
        self.assertEqual(rows[0]["embedding"], [0.1, 0.2])
        self.assertEqual(rows[0]["document_id"], 7)
        self.assertEqual(rows[0]["project_id"], 42)
        self.assertEqual(rows[0]["chunk_id"], 0)
        self.assertEqual(rows[0]["document_path"], "doc.pdf")
        self.assertEqual(rows[0]["page_number"], 3)
        self.assertTrue(rows[0]["has_context"])
        self.assertEqual(rows[0]["original_content"], "Roh")
        # missing metadata falls back to the filter scope / empty defaults
        self.assertEqual(rows[1]["document_id"], 7)
        self.assertEqual(rows[1]["project_id"], 42)
        self.assertEqual(rows[1]["chunk_id"], 1)
        self.assertEqual(rows[1]["raw_section"], "")
        self.assertFalse(rows[1]["has_context"])

    def test_ensures_collection_before_insert(self):
        with mock_milvus_client(insert_count=1) as mocks:
            MilvusBackend().insert_chunks(
                [ChunkRecord(content="Chunk", embedding=[0.1])],
                SearchFilter(collection_name="project_42"),
            )

        mocks["client"].create_collection.assert_called_once()

    def test_empty_chunk_list_returns_zero_without_insert(self):
        with mock_milvus_client() as mocks:
            inserted = MilvusBackend().insert_chunks([], SearchFilter(collection_name="project_42"))

        self.assertEqual(inserted, 0)
        mocks["client"].insert.assert_not_called()


class TestDeleteDropCount(SimpleTestCase):
    """delete / drop_namespace / count map onto MilvusClient expressions."""

    def test_delete_uses_filter_expression(self):
        with mock_milvus_client(has_collection=True, delete_count=5) as mocks:
            deleted = MilvusBackend().delete(SearchFilter(collection_name="project_42", document_id=7))

        self.assertEqual(deleted, 5)
        call = mocks["client"].delete.call_args
        self.assertEqual(call.kwargs["collection_name"], "project_42")
        self.assertEqual(call.kwargs["filter"], "document_id == 7")

    def test_delete_whole_namespace_uses_match_all_expression(self):
        with mock_milvus_client(has_collection=True, delete_count=12) as mocks:
            deleted = MilvusBackend().delete(SearchFilter(collection_name="project_42"))

        self.assertEqual(deleted, 12)
        self.assertEqual(mocks["client"].delete.call_args.kwargs["filter"], "id >= 0")

    def test_delete_returns_zero_when_collection_missing(self):
        with mock_milvus_client(has_collection=False) as mocks:
            deleted = MilvusBackend().delete(SearchFilter(collection_name="project_42", document_id=7))

        self.assertEqual(deleted, 0)
        mocks["client"].delete.assert_not_called()

    def test_drop_namespace_drops_existing_collection(self):
        with mock_milvus_client(has_collection=True) as mocks:
            self.assertTrue(MilvusBackend().drop_namespace("project_42"))

        mocks["client"].drop_collection.assert_called_once_with("project_42")

    def test_drop_namespace_is_noop_for_missing_collection(self):
        with mock_milvus_client(has_collection=False) as mocks:
            self.assertTrue(MilvusBackend().drop_namespace("project_42"))

        mocks["client"].drop_collection.assert_not_called()

    def test_drop_namespace_returns_false_on_error(self):
        with mock_milvus_client(has_collection=True) as mocks:
            mocks["client"].drop_collection.side_effect = RuntimeError("kaputt")
            self.assertFalse(MilvusBackend().drop_namespace("project_42"))

    def test_count_queries_count_star_with_filter(self):
        with mock_milvus_client(has_collection=True, count_result=23) as mocks:
            total = MilvusBackend().count(SearchFilter(collection_name="project_42", document_id=7))

        self.assertEqual(total, 23)
        call = mocks["client"].query.call_args
        self.assertEqual(call.kwargs["collection_name"], "project_42")
        self.assertEqual(call.kwargs["filter"], "document_id == 7")
        self.assertEqual(call.kwargs["output_fields"], ["count(*)"])

    def test_count_returns_zero_when_collection_missing(self):
        with mock_milvus_client(has_collection=False) as mocks:
            self.assertEqual(MilvusBackend().count(SearchFilter(collection_name="project_42")), 0)

        mocks["client"].query.assert_not_called()


class TestReadiness(SimpleTestCase):
    """is_ready / health_check replace the legacy connector-level health check."""

    def test_is_ready_true_when_server_responds(self):
        with mock_milvus_client(collections=["project_1"]):
            self.assertTrue(MilvusBackend().is_ready())

    def test_is_ready_false_on_connection_error(self):
        with mock_milvus_client() as mocks:
            mocks["client"].list_collections.side_effect = RuntimeError("nicht erreichbar")
            self.assertFalse(MilvusBackend().is_ready())

    def test_health_check_reports_ok_with_collection_count(self):
        with mock_milvus_client(collections=["project_1", "client_2"]):
            result = MilvusBackend.health_check()

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["collections_count"], 2)
        self.assertIn("response_time_ms", result)

    def test_health_check_reports_error_on_failure(self):
        with mock_milvus_client() as mocks:
            mocks["client"].list_collections.side_effect = RuntimeError("nicht erreichbar")
            result = MilvusBackend.health_check()

        self.assertEqual(result["status"], "error")
        self.assertIn("nicht erreichbar", result["error"])

    @override_settings(MILVUS_HOST="")
    def test_health_check_skips_when_milvus_not_configured(self):
        with mock_milvus_client() as mocks:
            result = MilvusBackend.health_check()

        self.assertEqual(result["status"], "skipped")
        mocks["cls"].assert_not_called()

    def test_client_connects_to_configured_host_and_port(self):
        with override_settings(MILVUS_HOST="milvus.example", MILVUS_PORT=19531):
            with mock_milvus_client() as mocks:
                MilvusBackend().is_ready()

        mocks["cls"].assert_called_once_with(uri="http://milvus.example:19531")
