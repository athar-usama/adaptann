import copy

import numpy as np

from adaptann.bruteforce import brute_force_knn
from adaptann.hnsw import HNSW
from adaptann.metrics import recall_at_k


def _clustered_dataset(rng, cluster_sizes, dim=32, spread=0.4, gap=6.0):
    """One cluster per entry in ``cluster_sizes``, centers spread far apart
    (``gap``) so a cluster's members are only reachable from other clusters
    through the graph's higher layers, not by being nearby in space."""
    vectors = []
    centers = []
    for size in cluster_sizes:
        center = rng.normal(scale=gap, size=dim)
        centers.append(center)
        vectors.append(center + rng.normal(scale=spread, size=(size, dim)))
    return np.concatenate(vectors, axis=0), centers


def _build_index(vectors, **kwargs):
    index = HNSW(dim=vectors.shape[1], **kwargs)
    for v in vectors:
        index.insert(v)
    return index


def test_promotion_only_fires_above_the_hit_threshold():
    rng = np.random.default_rng(0)
    vectors, _ = _clustered_dataset(rng, [300], dim=16)
    index = _build_index(vectors, M=12, ef_construction=60, seed=1, self_tuning=True, promotion_threshold=1000)

    q = vectors[0]
    for _ in range(20):
        index.search(q, k=5, ef=40)

    assert index.promotions == 0  # threshold never reached


def test_repeated_queries_trigger_promotions():
    rng = np.random.default_rng(0)
    vectors, _ = _clustered_dataset(rng, [300], dim=16)
    index = _build_index(vectors, M=12, ef_construction=60, seed=1, self_tuning=True, promotion_threshold=5)

    q = vectors[0]
    for _ in range(60):
        index.search(q, k=5, ef=40)

    assert index.promotions > 0
    assert len(index.widened) > 0
    # a densified node's layer-0 degree should exceed the construction-time cap
    densified_degrees = [len(index.graph[0][nid]) for nid in index.widened]
    assert max(densified_degrees) > index.M_max0


def test_self_tuning_improves_recall_at_fixed_ef_for_a_cold_cluster():
    """The scenario this feature is for: a cluster that's spatially isolated
    (few bridges in from the rest of the graph) and, once reached, only
    sparsely wired internally, because construction used a small ``M`` and
    ``ef_construction`` and had no way to know this region would later get
    hit hard. At a deliberately tight search budget (small ``ef``), a
    static index misses some of that cluster's true neighbors; a
    self-tuning index that has seen enough repeat traffic there should
    recover them, at the same ef, by then having denser local edges to
    search over.

    An earlier version of this mechanism promoted hot nodes to higher
    graph layers instead of densifying layer 0, and a distance-computation
    comparison against a static baseline showed that made total search
    cost worse, not better: layer 0 (bounded by ``ef``) was already the
    dominant cost and was insensitive to entry-point quality once the
    descent already landed nearby, which it did even without promotion.
    Recall at a fixed, tight ef is the metric that actually reflects the
    real bottleneck.
    """
    rng = np.random.default_rng(0)
    # One big "typical" cluster and one smaller, sparser, far-away "cold"
    # cluster: enough members and internal spread that low-ef search inside
    # it is genuinely hard, not a straight brute-force check.
    vectors, centers = _clustered_dataset(rng, [600, 60], dim=32, spread=1.6, gap=10.0)

    base = _build_index(vectors, M=5, ef_construction=20, seed=2, self_tuning=False)

    static = copy.deepcopy(base)
    adaptive = copy.deepcopy(base)
    adaptive.self_tuning = True
    adaptive.promotion_threshold = 5
    adaptive.widen_factor = 4

    cold_center = centers[1]

    def cold_query():
        return cold_center + rng.normal(scale=1.6, size=32)

    # Warm up the adaptive index with a burst of queries into the cold region.
    for _ in range(200):
        adaptive.search(cold_query(), k=10, ef=15)

    assert adaptive.promotions > 0, "expected the cold cluster's nodes to get densified"

    tight_ef = 10

    def avg_recall(index, n=40):
        recalls = []
        for _ in range(n):
            q = cold_query()
            exact = brute_force_knn(vectors, q, 10)
            approx = [nid for nid, _ in index.search(q, k=10, ef=tight_ef)]
            recalls.append(recall_at_k(approx, exact.tolist()))
        return float(np.mean(recalls))

    static_recall = avg_recall(static)
    adaptive_recall = avg_recall(adaptive)

    assert adaptive_recall > static_recall, (
        f"expected higher recall@10 (ef={tight_ef}) after warm-up: "
        f"static={static_recall:.2f} adaptive={adaptive_recall:.2f}"
    )
