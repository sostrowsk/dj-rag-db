"""Tests for the adaptive per-query cutoff (Phase A5).

``adaptive_cutoff`` is a pure function over fused-RRF scores (descending,
higher = better). No Django settings, no DB.
"""

import pytest

from scribe.retrieval.adaptive_cutoff import adaptive_cutoff


class TestAdaptiveCutoffEdgeCases:
    def test_empty_scores_return_zero(self):
        assert adaptive_cutoff([]) == 0

    def test_single_score_returns_one(self):
        assert adaptive_cutoff([0.8]) == 1

    def test_single_zero_score_returns_one(self):
        # top <= 0: ratios are meaningless, fall back to min_k capped by len.
        assert adaptive_cutoff([0.0]) == 1

    def test_all_zero_scores_return_min_k_capped_by_length(self):
        assert adaptive_cutoff([0.0] * 10, min_k=3) == 3
        assert adaptive_cutoff([0.0, 0.0], min_k=3) == 2

    def test_all_negative_scores_return_min_k_capped_by_length(self):
        assert adaptive_cutoff([-0.1, -0.2, -0.3, -0.4], min_k=3) == 3

    def test_unsorted_scores_raise_value_error(self):
        with pytest.raises(ValueError, match="descending"):
            adaptive_cutoff([0.5, 0.9, 0.4])


class TestAdaptiveCutoffNoCut:
    def test_all_equal_scores_keep_everything_up_to_max_k(self):
        # No relative drop at all -> no cut -> clamped to max_k.
        assert adaptive_cutoff([0.7] * 60, max_k=50) == 50

    def test_all_equal_scores_shorter_than_max_k_keep_everything(self):
        assert adaptive_cutoff([0.7] * 8, max_k=50) == 8


class TestAdaptiveCutoffElbow:
    def test_sharp_elbow_after_four_cuts_at_four(self):
        # Drop (0.95 - 0.30) / 0.95 ~= 0.684 > elbow_drop=0.45 -> keep 4.
        scores = [1.0, 0.98, 0.96, 0.95, 0.30, 0.29, 0.28]
        assert adaptive_cutoff(scores, min_k=3, elbow_drop=0.45) == 4

    def test_elbow_exactly_at_threshold_does_not_cut(self):
        # Drop is exactly elbow_drop (strict ">" required to cut).
        scores = [1.0, 0.5]
        assert adaptive_cutoff(scores, min_k=1, rel_floor=0.35, elbow_drop=0.5) == 2


class TestAdaptiveCutoffRelativeFloor:
    def test_gradual_decay_cuts_when_relative_floor_is_crossed(self):
        # 10% decay per step: no single-step elbow (0.1 < 0.45), but
        # 0.9**10 ~= 0.3487 < rel_floor=0.35 -> keep the first 10.
        scores = [0.9**i for i in range(20)]
        assert adaptive_cutoff(scores, min_k=3, max_k=50, rel_floor=0.35, elbow_drop=0.45) == 10

    def test_negative_tail_score_is_cut_by_relative_floor(self):
        scores = [1.0, 0.9, 0.8, 0.7, -0.2]
        assert adaptive_cutoff(scores, min_k=1, elbow_drop=0.99) == 4


class TestAdaptiveCutoffClamping:
    def test_result_never_below_min_k(self):
        # Elbow already after the top hit, but min_k=3 wins.
        scores = [1.0, 0.1, 0.05, 0.04, 0.03]
        assert adaptive_cutoff(scores, min_k=3) == 3

    def test_result_never_above_max_k(self):
        scores = [1.0] * 30
        assert adaptive_cutoff(scores, max_k=10) == 10

    def test_min_k_capped_by_number_of_scores(self):
        assert adaptive_cutoff([1.0, 0.9], min_k=5) == 2

    def test_max_k_wins_when_below_min_k(self):
        # max_k is the hard upper bound, even against min_k.
        assert adaptive_cutoff([1.0, 0.9, 0.8, 0.7], min_k=3, max_k=1) == 1

    def test_max_k_wins_when_below_min_k_for_non_positive_top(self):
        assert adaptive_cutoff([0.0, 0.0, 0.0], min_k=3, max_k=2) == 2

    def test_defaults_match_phase_a5_contract(self):
        # Defaults: max_k=50, min_k=3, rel_floor=0.35, elbow_drop=0.45.
        scores = [1.0, 0.98, 0.96, 0.95, 0.30, 0.29, 0.28]
        assert adaptive_cutoff(scores) == 4
