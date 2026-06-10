"""Unit tests for PgvectorBackend pure helpers (no DB).

Phase A3 of the pgvector plan. Weighted RRF fusion is the core ranking
primitive: each branch (dense, FTS) contributes
``weight / (rrf_k + rank + 1)`` per hit, so per-query weights actually
steer the fused ordering (port of arznei test_rrf_fusion.py).
"""

import pytest

from scribe.backends.base import SearchFilter
from scribe.backends.pgvector_backend import PgvectorBackend

# doc 1 is the dense-only top hit, doc 2 is the sparse-only (FTS) top hit.
DENSE = [{"id": 1, "distance": 0.1}]
FTS = [{"id": 2, "rank": 0.9}]


class TestWeightedRrfFusion:
    def test_dense_weight_dominates(self):
        fused = PgvectorBackend._rrf_fusion(DENSE, FTS, rrf_k=60, max_results=2, dense_weight=0.9, sparse_weight=0.1)
        assert fused[0]["id"] == 1, "dense-only hit must rank first when dense_weight is high"
        assert fused[0]["score"] > fused[1]["score"]

    def test_sparse_weight_dominates(self):
        fused = PgvectorBackend._rrf_fusion(DENSE, FTS, rrf_k=60, max_results=2, dense_weight=0.1, sparse_weight=0.9)
        assert fused[0]["id"] == 2, "FTS-only hit must rank first when sparse_weight is high"
        assert fused[0]["score"] > fused[1]["score"]

    def test_overlap_accumulates_both_weights(self):
        # doc 3 appears in both lists -> gets both weighted contributions.
        dense = [{"id": 3, "distance": 0.1}, {"id": 1, "distance": 0.2}]
        fts = [{"id": 3, "rank": 0.9}, {"id": 2, "rank": 0.5}]
        fused = PgvectorBackend._rrf_fusion(dense, fts, rrf_k=60, max_results=3, dense_weight=0.5, sparse_weight=0.5)
        assert fused[0]["id"] == 3, "doc present in both lists should rank first"

    def test_rank_zero_score_is_weight_over_rrf_k_plus_one(self):
        fused = PgvectorBackend._rrf_fusion(DENSE, [], rrf_k=60, max_results=1, dense_weight=0.5, sparse_weight=0.5)
        assert fused[0]["score"] == pytest.approx(0.5 / 61)

    def test_larger_rrf_k_lowers_scores(self):
        low_k = PgvectorBackend._rrf_fusion(DENSE, FTS, rrf_k=10, max_results=2)
        high_k = PgvectorBackend._rrf_fusion(DENSE, FTS, rrf_k=100, max_results=2)
        assert low_k[0]["score"] > high_k[0]["score"]

    def test_equal_scores_keep_stable_dense_first_order(self):
        # Equal weights, both docs at rank 0 of their branch -> identical
        # scores. Python's sort is stable, so the first-encountered branch
        # (dense) must keep its position deterministically.
        fused = PgvectorBackend._rrf_fusion(DENSE, FTS, rrf_k=60, max_results=2, dense_weight=0.5, sparse_weight=0.5)
        assert fused[0]["score"] == pytest.approx(fused[1]["score"])
        assert [item["id"] for item in fused] == [1, 2]

    def test_max_results_truncates_fused_list(self):
        dense = [{"id": i, "distance": i / 10} for i in range(1, 6)]
        fused = PgvectorBackend._rrf_fusion(dense, [], rrf_k=60, max_results=3)
        assert len(fused) == 3
        assert [item["id"] for item in fused] == [1, 2, 3]

    def test_empty_inputs_yield_empty_fusion(self):
        assert PgvectorBackend._rrf_fusion([], [], rrf_k=60, max_results=10) == []


class TestBuildFilterKwargs:
    def test_collection_only(self):
        kwargs = PgvectorBackend._build_filter_kwargs(SearchFilter(collection_name="project_1"))
        assert kwargs == {"collection_name": "project_1"}

    def test_optional_scope_narrowing(self):
        kwargs = PgvectorBackend._build_filter_kwargs(
            SearchFilter(collection_name="client_7", project_id=3, document_id=42)
        )
        assert kwargs == {"collection_name": "client_7", "project_id": 3, "document_id": 42}
