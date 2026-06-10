"""Adaptive per-query cutoff over fused-RRF scores (Phase A5).

Pure function, backend-agnostic: both PgvectorBackend and MilvusBackend
return fused weighted-RRF scores (higher = better), so the cutoff works on
score *ratios* only and needs no normalisation. Settings wiring happens in
the service layer (Phase A6), never here.
"""

from typing import Sequence


def adaptive_cutoff(
    scores: Sequence[float],
    max_k: int = 50,
    min_k: int = 3,
    rel_floor: float = 0.35,
    elbow_drop: float = 0.45,
) -> int:
    """Return how many of the top ``scores`` to keep.

    Args:
        scores: Fused-RRF scores sorted descending, higher = better.
        max_k: Hard upper bound on the number of kept results.
        min_k: Lower bound (capped by ``len(scores)``).
        rel_floor: Cut once ``scores[i] / scores[0] < rel_floor``.
        elbow_drop: Cut once a single step drops by more than this
            fraction: ``(scores[i-1] - scores[i]) / scores[i-1] > elbow_drop``.

    Returns:
        Number of results to keep, clamped to
        ``[min(min_k, len(scores)), min(max_k, len(scores))]``. ``max_k`` is
        the hard upper bound and wins if ``max_k < min_k``.
        An empty list returns 0. A non-positive top score makes ratios
        meaningless and returns ``min(min_k, max_k, len(scores))``.

    Raises:
        ValueError: If ``scores`` is not sorted descending. Backends always
            return ranked results, so unsorted input indicates a caller bug —
            fail fast instead of silently re-sorting.
    """
    n = len(scores)
    if n == 0:
        return 0

    if any(scores[i] < scores[i + 1] for i in range(n - 1)):
        raise ValueError("scores must be sorted descending (higher = better)")

    ceil_k = min(max_k, n)
    floor_k = min(min_k, ceil_k)  # max_k is the hard upper bound

    top = scores[0]
    if top <= 0:
        return floor_k

    cut = n
    for i in range(1, n):
        if scores[i] / top < rel_floor:
            cut = i
            break
        prev = scores[i - 1]
        if prev > 0 and (prev - scores[i]) / prev > elbow_drop:
            cut = i
            break

    return max(floor_k, min(cut, ceil_k))
