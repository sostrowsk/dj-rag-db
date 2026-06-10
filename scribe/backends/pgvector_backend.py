"""Postgres/pgvector search backend: hybrid dense + FTS search with RRF fusion.

Port of arznei-muster-mello ``ai_vectorstore/backends/pgvector_backend.py``,
adapted to scribe's collection-name namespaces and metadata contract.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from ai_router.types import Document
from asgiref.sync import sync_to_async
from django.conf import settings
from django.contrib.postgres.search import SearchQuery, SearchRank
from django.db import connection, transaction
from django.db.models import F
from pgvector.django import CosineDistance

from scribe.models import DocumentChunk

from .base import ChunkRecord, SearchBackend, SearchFilter, SearchResult

logger = logging.getLogger(__name__)


class PgvectorBackend(SearchBackend):
    """Hybrid search over DocumentChunk (HNSW dense + German FTS, weighted RRF fusion)."""

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

        filter_kwargs = self._build_filter_kwargs(filters)

        dense_results = await self._dense_search(query_embedding, initial_fetch_k, filter_kwargs)
        fts_results = await self._fts_search(query, initial_fetch_k, filter_kwargs)

        fused = self._rrf_fusion(
            dense_results,
            fts_results,
            rrf_k,
            max_k,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
        )

        chunk_ids = [item["id"] for item in fused]
        if not chunk_ids:
            return []

        chunks = await sync_to_async(lambda: {c.id: c for c in DocumentChunk.objects.filter(id__in=chunk_ids)})()

        results = []
        for item in fused:
            chunk = chunks.get(item["id"])
            if not chunk:
                continue
            results.append(SearchResult(document=self._chunk_to_document(chunk), score=item["score"]))

        logger.info(
            f"Hybrid search in {filters.collection_name}: {len(results)} results "
            f"(dense={len(dense_results)}, fts={len(fts_results)}, rrf_k={rrf_k})"
        )
        return results

    def insert_chunks(self, chunks: list[ChunkRecord], filters: SearchFilter) -> int:
        if not chunks:
            return 0

        rows = []
        for record in chunks:
            meta = record.metadata
            rows.append(
                DocumentChunk(
                    collection_name=filters.collection_name,
                    project_document_id=meta.get("project_document_id"),
                    client_document_id=meta.get("client_document_id"),
                    document_id=meta.get("document_id", filters.document_id),
                    project_id=meta.get("project_id", filters.project_id),
                    chunk_id=meta.get("chunk_id", 0),
                    content=record.content,
                    original_content=meta.get("original_content") or "",
                    raw_section=meta.get("raw_section") or "",
                    document_path=meta.get("document_path") or "",
                    image_path=meta.get("image_path") or "",
                    page_number=meta.get("page_number"),
                    has_context=bool(meta.get("has_context", False)),
                    embedding=record.embedding,
                )
            )

        created = DocumentChunk.objects.bulk_create(rows, ignore_conflicts=True)
        logger.info(f"Inserted {len(created)} chunks into {filters.collection_name}")
        return len(created)

    def delete(self, filters: SearchFilter) -> int:
        count, _ = DocumentChunk.objects.filter(**self._build_filter_kwargs(filters)).delete()
        logger.info(f"Deleted {count} chunks matching {filters}")
        return count

    def drop_namespace(self, collection_name: str) -> bool:
        try:
            count, _ = DocumentChunk.objects.filter(collection_name=collection_name).delete()
            logger.info(f"Dropped namespace {collection_name} ({count} chunks)")
            return True
        except Exception as e:
            logger.error(f"Failed to drop namespace {collection_name}: {e}")
            return False

    def count(self, filters: SearchFilter) -> int:
        return DocumentChunk.objects.filter(**self._build_filter_kwargs(filters)).count()

    def is_ready(self) -> bool:
        """DB reachable and the pgvector extension installed."""
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
                return cursor.fetchone() is not None
        except Exception as e:
            logger.error(f"pgvector backend not ready: {e}")
            return False

    # -- internal helpers ------------------------------------------------

    @staticmethod
    def _build_filter_kwargs(filters: SearchFilter) -> dict:
        """Translate a SearchFilter into ORM filter kwargs."""
        kwargs = {"collection_name": filters.collection_name}
        if filters.project_id is not None:
            kwargs["project_id"] = filters.project_id
        if filters.document_id is not None:
            kwargs["document_id"] = filters.document_id
        return kwargs

    @staticmethod
    def _dense_search_sync(query_embedding: list[float], limit: int, filter_kwargs: dict) -> list[dict]:
        """Dense vector search using cosine distance (synchronous core).

        Raises hnsw.ef_search to at least ``limit`` for this transaction, so
        the HNSW index actually returns up to initial_fetch_k candidates
        instead of the pgvector default cap of 40. SET LOCAL is
        transaction-scoped (pgbouncer-safe), so the query must run inside the
        same atomic block.
        """
        ef_search = max(int(limit), 40)
        with transaction.atomic():
            with connection.cursor() as cursor:
                # ef_search is a validated int — safe to inline (SET rejects bind params).
                cursor.execute(f"SET LOCAL hnsw.ef_search = {ef_search}")
            return list(
                DocumentChunk.objects.filter(**filter_kwargs)
                .annotate(distance=CosineDistance("embedding", query_embedding))
                .order_by("distance")[:limit]
                .values("id", "distance")
            )

    @staticmethod
    async def _dense_search(query_embedding: list[float], limit: int, filter_kwargs: dict) -> list[dict]:
        return await sync_to_async(PgvectorBackend._dense_search_sync)(query_embedding, limit, filter_kwargs)

    @staticmethod
    def _build_fts_query(query: str, config: str) -> SearchQuery:
        """Build an OR-of-terms tsquery so a chunk matching ANY term ranks.

        Plain/websearch SearchQuery ANDs all lexemes, so a multi-term query
        only matches chunks containing every term — usually nothing. ORing
        the terms keeps recall; SearchRank still ranks chunks matching more
        terms higher.
        """
        terms = query.split()
        if not terms:
            return SearchQuery(query, config=config)
        combined = SearchQuery(terms[0], config=config)
        for term in terms[1:]:
            combined |= SearchQuery(term, config=config)
        return combined

    @staticmethod
    def _fts_search_sync(query: str, limit: int, filter_kwargs: dict) -> list[dict]:
        """Full-text search using the trigger-maintained German tsvector."""
        fts_config = getattr(settings, "VECTORSTORE_SEARCH_CONFIG", "german")
        search_query = PgvectorBackend._build_fts_query(query, fts_config)

        return list(
            DocumentChunk.objects.filter(search_vector=search_query, **filter_kwargs)
            .annotate(rank=SearchRank(F("search_vector"), search_query))
            .order_by("-rank")[:limit]
            .values("id", "rank")
        )

    @staticmethod
    async def _fts_search(query: str, limit: int, filter_kwargs: dict) -> list[dict]:
        return await sync_to_async(PgvectorBackend._fts_search_sync)(query, limit, filter_kwargs)

    @staticmethod
    def _rrf_fusion(
        dense_results: list[dict],
        fts_results: list[dict],
        rrf_k: int,
        max_results: int,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
    ) -> list[dict]:
        """Weighted Reciprocal Rank Fusion of dense + FTS results.

        Each list contributes ``weight / (rrf_k + rank + 1)`` per item, so the
        dense/sparse weights actually steer the fused ranking. Python's sort is
        stable and the score dict preserves insertion order (dense first), so
        ties resolve deterministically in favor of the dense branch.
        """
        scores: dict[int, float] = defaultdict(float)

        for rank_pos, item in enumerate(dense_results):
            scores[item["id"]] += dense_weight / (rrf_k + rank_pos + 1)

        for rank_pos, item in enumerate(fts_results):
            scores[item["id"]] += sparse_weight / (rrf_k + rank_pos + 1)

        sorted_ids = sorted(scores.keys(), key=lambda chunk_id: scores[chunk_id], reverse=True)[:max_results]
        return [{"id": chunk_id, "score": scores[chunk_id]} for chunk_id in sorted_ids]

    @staticmethod
    def _chunk_to_document(chunk: DocumentChunk) -> Document:
        """Convert a DocumentChunk row to an ai_router Document.

        Metadata keys mirror today's HybridRetriever result shape so ai_chat
        result handling stays compatible (original_content only when present).
        """
        metadata = {
            "document_id": chunk.document_id,
            "document_path": chunk.document_path or None,
            "page_number": chunk.page_number,
            "has_context": chunk.has_context,
        }
        if chunk.original_content:
            metadata["original_content"] = chunk.original_content
        return Document(page_content=chunk.content, metadata=metadata)
