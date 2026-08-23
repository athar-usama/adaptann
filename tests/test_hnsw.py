import numpy as np

from adaptann.bruteforce import brute_force_knn
from adaptann.hnsw import HNSW
from adaptann.metrics import recall_at_k


def _random_dataset(n=500, dim=16, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, dim)), rng


def _build_index(vectors, **kwargs):
    index = HNSW(dim=vectors.shape[1], **kwargs)
    for v in vectors:
        index.insert(v)
    return index


def test_recall_is_high_against_brute_force_on_random_data():
    vectors, rng = _random_dataset(n=500, dim=16)
    index = _build_index(vectors, M=16, ef_construction=100, seed=1)

    queries = rng.normal(size=(30, 16))
    k = 10
    recalls = []
    for q in queries:
        exact = brute_force_knn(vectors, q, k)
        approx = [nid for nid, _ in index.search(q, k, ef=50)]
        recalls.append(recall_at_k(approx, exact.tolist()))

    assert np.mean(recalls) > 0.85, f"mean recall too low: {np.mean(recalls):.2f}"


def test_search_returns_distances_consistent_with_brute_force():
    vectors, rng = _random_dataset(n=200, dim=8)
    index = _build_index(vectors, M=12, ef_construction=80, seed=2)

    q = rng.normal(size=8)
    results = index.search(q, k=5, ef=50)
    for nid, dist in results:
        expected = float(np.linalg.norm(vectors[nid] - q))
        assert abs(dist - expected) < 1e-9


def test_layer_membership_is_monotonic():
    """A node present at layer l must also be present at every layer < l
    (the standard HNSW invariant): if levels[i] == L, graph[l] must contain
    node i for every l in 0..L."""
    vectors, _ = _random_dataset(n=300, dim=12)
    index = _build_index(vectors, M=10, ef_construction=60, seed=3)

    for node_id, level in enumerate(index.levels):
        for layer in range(level + 1):
            assert node_id in index.graph[layer], f"node {node_id} (level {level}) missing from layer {layer}"


def test_neighbor_degree_never_exceeds_the_cap():
    vectors, _ = _random_dataset(n=400, dim=10)
    M = 10
    index = _build_index(vectors, M=M, ef_construction=60, seed=4)

    for layer, layer_graph in enumerate(index.graph):
        cap = index.M_max0 if layer == 0 else M
        for node_id, neighbors in layer_graph.items():
            assert len(neighbors) <= cap, f"node {node_id} at layer {layer} has {len(neighbors)} > {cap} neighbors"


def test_neighbor_edges_are_bidirectional():
    vectors, _ = _random_dataset(n=250, dim=10)
    index = _build_index(vectors, M=8, ef_construction=50, seed=5)

    for layer_graph in index.graph:
        for node_id, neighbors in layer_graph.items():
            for other in neighbors:
                assert node_id in layer_graph[other], (
                    f"edge {node_id}->{other} is not mirrored back"
                )


def test_empty_index_search_returns_nothing():
    index = HNSW(dim=4)
    assert index.search(np.zeros(4), k=5) == []
