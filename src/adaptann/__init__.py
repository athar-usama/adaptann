"""adaptann: a from-scratch HNSW vector index with an optional self-tuning
layer that densifies the local neighborhood of frequently-queried nodes
as query traffic drifts, without a full rebuild.

    from adaptann import HNSW
    from adaptann.bruteforce import brute_force_knn
    from adaptann.metrics import recall_at_k

See the package README for the drifting-workload benchmark this is built
to demonstrate.
"""

from .hnsw import HNSW

__all__ = ["HNSW"]
__version__ = "0.1.0"
