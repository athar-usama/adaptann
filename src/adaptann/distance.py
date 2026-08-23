"""Vectorized distance functions. Everything in this package works with
squared Euclidean distance internally (monotonic with Euclidean distance,
so it never changes a nearest-neighbor ranking, and skips a sqrt on every
comparison)."""

from __future__ import annotations

import numpy as np


def sq_euclidean(query: np.ndarray, points: np.ndarray) -> np.ndarray:
    """``query``: (dim,). ``points``: (n, dim). Returns (n,) squared distances."""
    diff = points - query
    return np.einsum("ij,ij->i", diff, diff)


def sq_euclidean_one(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return float(np.dot(diff, diff))
