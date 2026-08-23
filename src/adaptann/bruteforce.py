"""Exact k-nearest-neighbors, used as ground truth for recall@k and as the
baseline every approximate result in this package is checked against."""

from __future__ import annotations

import numpy as np

from .distance import sq_euclidean


def brute_force_knn(vectors: np.ndarray, query: np.ndarray, k: int) -> np.ndarray:
    """Returns the ``k`` row-indices of ``vectors`` closest to ``query``,
    sorted nearest-first."""
    dists = sq_euclidean(query, vectors)
    k = min(k, len(vectors))
    idx = np.argpartition(dists, k - 1)[:k]
    return idx[np.argsort(dists[idx])]
