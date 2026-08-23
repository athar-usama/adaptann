"""Evaluation metrics shared by the tests and the benchmark demo."""

from __future__ import annotations

from collections.abc import Sequence


def recall_at_k(approx_ids: Sequence[int], exact_ids: Sequence[int]) -> float:
    """Fraction of ``exact_ids`` (the true top-k) that appear anywhere in
    ``approx_ids`` (the index's returned top-k). 1.0 means perfect recall."""
    if not exact_ids:
        return 1.0
    exact_set = set(exact_ids)
    hits = sum(1 for i in approx_ids if i in exact_set)
    return hits / len(exact_ids)
