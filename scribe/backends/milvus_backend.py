"""Milvus search backend via direct pymilvus MilvusClient (no langchain).

Modernized replacement for the legacy langchain_milvus-based modules
(``scribe/milvus/``, removed in Phase A8): dense index is HNSW/COSINE
(was IVF_FLAT/L2), sparse retrieval is Milvus-native BM25, fusion happens
server-side via RRFRanker.

Note: Milvus's RRFRanker has no per-branch weights, so ``dense_weight`` /
``sparse_weight`` are ignored here (both branches contribute 1/(k+rank)).
``hit.distance`` of a hybrid search IS the fused RRF score (higher = better),
i.e. the same scale family as the pgvector backend's weighted RRF.
"""

from __future__ import annotations

import asyncio
import logging
import time

from ai_router.types import Document
from django.conf import settings
from pymilvus import AnnSearchRequest, DataType, Function, FunctionType, MilvusClient, RRFRanker

from .base import ChunkRecord, SearchBackend, SearchFilter, SearchResult

logger = logging.getLogger(__name__)

#: Mirrors HybridRetriever.output_fields so ai_chat result handling stays compatible.
OUTPUT_FIELDS = [
    "content",
    "document_id",
    "document_path",
    "page_number",
    "original_content",
    "has_context",
]

#: Milvus VARCHAR fields are capped at 65535 bytes.
VARCHAR_MAX_LENGTH = 65535


def _build_uri() -> str:
    return f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}"


class MilvusBackend(SearchBackend):
    """Hybrid search via Milvus AnnSearchRequest (HNSW/COSINE + BM25) + RRFRanker."""

    def __init__(self, embedding_dim: int = 3072):
        self.embedding_dim = embedding_dim
        self._client: MilvusClient | None = None

    @property
    def client(self) -> MilvusClient:
        """Lazily connected MilvusClient (no connection at instantiation time)."""
        if self._client is None:
            self._client = MilvusClient(uri=_build_uri())
        return self._client

    # -- schema / collection management -----------------------------------

    def ensure_collection(self, collection_name: str) -> bool:
        """Create the SCRIBE collection if missing. Returns True if created."""
        if self.client.has_collection(collection_name):
            return False

        schema = self.client.create_schema(auto_id=True, enable_dynamic_field=False)
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(
            field_name="content",
            datatype=DataType.VARCHAR,
            max_length=VARCHAR_MAX_LENGTH,
            enable_analyzer=True,
        )
        schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=self.embedding_dim)
        schema.add_field(field_name="bm25_sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field(field_name="document_id", datatype=DataType.INT64)
        schema.add_field(field_name="project_id", datatype=DataType.INT64)
        schema.add_field(field_name="chunk_id", datatype=DataType.INT64)
        schema.add_field(field_name="document_path", datatype=DataType.VARCHAR, max_length=VARCHAR_MAX_LENGTH)
        schema.add_field(field_name="raw_section", datatype=DataType.VARCHAR, max_length=VARCHAR_MAX_LENGTH)
        schema.add_field(field_name="image_path", datatype=DataType.VARCHAR, max_length=VARCHAR_MAX_LENGTH)
        schema.add_field(field_name="original_content", datatype=DataType.VARCHAR, max_length=VARCHAR_MAX_LENGTH)
        schema.add_field(field_name="page_number", datatype=DataType.INT64, nullable=True)
        schema.add_field(field_name="has_context", datatype=DataType.BOOL, nullable=True)
        schema.add_function(
            Function(
                name="bm25",
                function_type=FunctionType.BM25,
                input_field_names=["content"],
                output_field_names=["bm25_sparse"],
            )
        )

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 64},
        )
        index_params.add_index(
            field_name="bm25_sparse",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
        )

        self.client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
            consistency_level="Strong",
        )
        logger.info(f"Created Milvus collection '{collection_name}' (dim={self.embedding_dim})")
        return True

    # -- search ------------------------------------------------------------

    async def search(
        self,
        query: str,
        query_embedding: list[float],
        filters: SearchFilter,
        initial_fetch_k: int = 150,
        max_k: int = 50,
        rrf_k: int = 60,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
    ) -> list[SearchResult]:
        if not query:
            return []

        expr = self._build_filter_expr(filters)
        reqs = [
            AnnSearchRequest(
                data=[query_embedding],
                anns_field="embedding",
                param={"metric_type": "COSINE"},
                limit=initial_fetch_k,
                expr=expr,
            ),
            AnnSearchRequest(
                data=[query],
                anns_field="bm25_sparse",
                param={"metric_type": "BM25"},
                limit=initial_fetch_k,
                expr=expr,
            ),
        ]

        raw = await asyncio.to_thread(
            self.client.hybrid_search,
            collection_name=filters.collection_name,
            reqs=reqs,
            ranker=RRFRanker(rrf_k),
            limit=max_k,
            output_fields=OUTPUT_FIELDS,
        )

        hits = raw[0] if raw else []
        results = [SearchResult(document=self._hit_to_document(hit), score=float(hit["distance"])) for hit in hits]
        logger.info(f"Milvus hybrid search in {filters.collection_name}: {len(results)} results (rrf_k={rrf_k})")
        return results

    # -- write path ----------------------------------------------------------

    def insert_chunks(self, chunks: list[ChunkRecord], filters: SearchFilter) -> int:
        if not chunks:
            return 0

        self.ensure_collection(filters.collection_name)

        rows = []
        for record in chunks:
            meta = record.metadata
            rows.append(
                {
                    "content": record.content,
                    "embedding": record.embedding,
                    # INT64 fields are non-nullable in the schema -> default to 0.
                    "document_id": self._first_int(meta.get("document_id"), filters.document_id),
                    "project_id": self._first_int(meta.get("project_id"), filters.project_id),
                    "chunk_id": self._first_int(meta.get("chunk_id")),
                    "document_path": meta.get("document_path") or "",
                    "raw_section": meta.get("raw_section") or "",
                    "image_path": meta.get("image_path") or "",
                    "original_content": meta.get("original_content") or "",
                    "page_number": meta.get("page_number"),
                    "has_context": bool(meta.get("has_context", False)),
                }
            )

        result = self.client.insert(collection_name=filters.collection_name, data=rows)
        inserted = int(result.get("insert_count", len(rows)))
        logger.info(f"Inserted {inserted} chunks into Milvus collection {filters.collection_name}")
        return inserted

    def delete(self, filters: SearchFilter) -> int:
        if not self.client.has_collection(filters.collection_name):
            return 0
        # MilvusClient.delete requires a filter; "id >= 0" matches every row.
        expr = self._build_filter_expr(filters) or "id >= 0"
        result = self.client.delete(collection_name=filters.collection_name, filter=expr)
        deleted = int(result.get("delete_count", 0))
        logger.info(f"Deleted {deleted} chunks from Milvus matching {filters}")
        return deleted

    def drop_namespace(self, collection_name: str) -> bool:
        try:
            if self.client.has_collection(collection_name):
                self.client.drop_collection(collection_name)
                logger.info(f"Dropped Milvus collection {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to drop Milvus collection {collection_name}: {e}")
            return False

    def count(self, filters: SearchFilter) -> int:
        if not self.client.has_collection(filters.collection_name):
            return 0
        result = self.client.query(
            collection_name=filters.collection_name,
            filter=self._build_filter_expr(filters) or "",
            output_fields=["count(*)"],
        )
        return int(result[0]["count(*)"]) if result else 0

    # -- health --------------------------------------------------------------

    def is_ready(self) -> bool:
        try:
            self.client.list_collections()
            return True
        except Exception as e:
            logger.error(f"Milvus backend not ready: {e}")
            return False

    @staticmethod
    def health_check() -> dict:
        """Static health check (replaces SCRIBE.check_milvus_health_static)."""
        start_time = time.time()
        if not getattr(settings, "MILVUS_HOST", None):
            return {"status": "skipped", "reason": "Milvus not configured", "response_time_ms": 0}
        try:
            client = MilvusClient(uri=_build_uri())
            collections = client.list_collections()
            return {
                "status": "ok",
                "collections_count": len(collections),
                "response_time_ms": int((time.time() - start_time) * 1000),
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "response_time_ms": int((time.time() - start_time) * 1000),
            }

    # -- internal helpers ------------------------------------------------------

    @staticmethod
    def _build_filter_expr(filters: SearchFilter) -> str | None:
        parts = []
        if filters.project_id is not None:
            parts.append(f"project_id == {int(filters.project_id)}")
        if filters.document_id is not None:
            parts.append(f"document_id == {int(filters.document_id)}")
        return " and ".join(parts) or None

    @staticmethod
    def _first_int(*values) -> int:
        for value in values:
            if value is not None:
                return int(value)
        return 0

    @staticmethod
    def _hit_to_document(hit: dict) -> Document:
        """Convert a hybrid_search hit to an ai_router Document.

        Metadata keys mirror HybridRetriever / PgvectorBackend so ai_chat
        result handling stays backend-agnostic (original_content only when
        present).
        """
        entity = hit.get("entity", {})
        metadata = {
            "document_id": entity.get("document_id"),
            "document_path": entity.get("document_path") or None,
            "page_number": entity.get("page_number"),
            "has_context": entity.get("has_context", False),
        }
        if entity.get("original_content"):
            metadata["original_content"] = entity["original_content"]
        return Document(page_content=entity.get("content", ""), metadata=metadata)
