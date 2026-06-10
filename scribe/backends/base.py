"""Abstract base for scribe search backends (pgvector | milvus).

Contract decisions (see plan Phase A2):
- Embeddings are always computed by the service layer (once per chunk),
  never by a backend.
- `SearchResult.score` is a fused weighted-RRF score (higher = better),
  comparable across backends.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Optional

from ai_router.types import Document


@dataclass
class SearchFilter:
    """Backend-agnostic filter: namespace plus optional scope narrowing."""

    collection_name: str
    project_id: Optional[int] = None
    document_id: Optional[int] = None


@dataclass
class SearchResult:
    """A single search hit: document plus fused RRF score (higher = better)."""

    document: Document
    score: float


@dataclass
class ChunkRecord:
    """A chunk ready for insertion, with pre-computed embedding."""

    content: str
    embedding: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)


class SearchBackend(abc.ABC):
    """Abstract interface every vector store backend must implement."""

    @abc.abstractmethod
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
        """Hybrid search (dense + sparse, RRF-fused), sorted by score desc."""

    @abc.abstractmethod
    def insert_chunks(self, chunks: list[ChunkRecord], filters: SearchFilter) -> int:
        """Insert chunks into the namespace given by filters. Returns count inserted."""

    @abc.abstractmethod
    def delete(self, filters: SearchFilter) -> int:
        """Delete all chunks matching filters. Returns count deleted."""

    @abc.abstractmethod
    def drop_namespace(self, collection_name: str) -> bool:
        """Drop an entire namespace/collection. Returns True on success."""

    @abc.abstractmethod
    def count(self, filters: SearchFilter) -> int:
        """Count chunks matching filters."""

    @abc.abstractmethod
    def is_ready(self) -> bool:
        """Check whether the backend is operational."""
