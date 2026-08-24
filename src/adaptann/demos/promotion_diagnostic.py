"""Reproduces the diagnosis behind the README's redesign story: a
per-layer breakdown of search cost for the *first* self-tuning mechanism
this project tried, promoting hot nodes to higher graph layers, against
the static baseline.

The upper-layer-promotion mechanism no longer exists in ``hnsw.py`` (it
was a genuine dead end, not merely renamed), so this script reconstructs
it here, using only the same private methods ``HNSW`` already exposes to
itself (``_search_layer``, ``_connect_at_layer``), and reruns the
comparison for real, averaged over several seeds, rather than quoting
old numbers from memory. What comes out is stable across seeds: layer 0,
the ef-bounded pass that dominates total cost, is statistically
unaffected by promotion, because promotion never touches a layer-0 edge.
This is supporting diagnostic material, not part of the main benchmark:
the claim it backs is entirely about *why* promotion was abandoned, which
the current codebase's own tests do not need to re-prove every run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..hnsw import HNSW
from ..viz import plot_promotion_vs_densify

ROOT = Path(__file__).resolve().parent.parent.parent.parent
ASSETS_DIR = ROOT / "assets"

DIM = 32
MAX_PROMOTION_LEVEL = 4


def _clustered_dataset(rng, cluster_sizes, dim=32, spread=0.4, gap=6.0):
    vectors, centers = [], []
    for size in cluster_sizes:
        center = rng.normal(scale=gap, size=dim)
        centers.append(center)
        vectors.append(center + rng.normal(scale=spread, size=(size, dim)))
    return np.concatenate(vectors, axis=0), centers


def _promote_old_mechanism(index: HNSW, node_id: int) -> None:
    """The abandoned mechanism: give a hot node one more graph layer, the
    same way a fresh insert climbs the hierarchy. Reconstructed exactly as
    it worked before it was replaced by ``HNSW._densify``."""
    current_level = index.levels[node_id]
    if current_level >= MAX_PROMOTION_LEVEL:
        return
    new_level = current_level + 1
    vector = index.vectors[node_id]
    curr_nearest = [index.entry_point]
    for layer in range(index.max_level, new_level, -1):
        curr_nearest = [nid for _, nid in index._search_layer(vector, curr_nearest, ef=1, layer=layer)]
    index._connect_at_layer(node_id, vector, curr_nearest, new_level)
    index.levels[node_id] = new_level
    if new_level > index.max_level:
        index.entry_point = node_id
        index.max_level = new_level


def _warm_up_with_promotion(index: HNSW, queries, k: int, ef: int, threshold: int) -> None:
    hit_counts: dict[int, int] = {}
    for q in queries:
        result_ids = [nid for nid, _ in index.search(q, k, ef=ef)]
        for nid in result_ids:
            hit_counts[nid] = hit_counts.get(nid, 0) + 1
            if hit_counts[nid] == threshold:
                _promote_old_mechanism(index, nid)


def _search_with_layer_breakdown(index: HNSW, query, k: int, ef: int) -> tuple[int, int]:
    """Mirrors ``HNSW.search``'s own descent exactly, but returns
    ``(upper_layers_cost, layer_0_cost)`` instead of results, so the two
    phases of one query's cost can be charted separately."""
    query = np.asarray(query, dtype=np.float64)
    before = index.distance_computations
    curr_nearest = [index.entry_point]
    for layer in range(index.max_level, 0, -1):
        curr_nearest = [nid for _, nid in index._search_layer(query, curr_nearest, ef=1, layer=layer)]
    upper_cost = index.distance_computations - before

    before_l0 = index.distance_computations
    index._search_layer(query, curr_nearest, ef, layer=0)
    layer0_cost = index.distance_computations - before_l0
    return upper_cost, layer0_cost


def _run_one_seed(seed: int) -> tuple[tuple[float, float], tuple[float, float]]:
    # These are the original scenario's parameters: a generously-connected
    # graph (M=14, ef_construction=80) and a generous search ef=50, the
    # same setup that first surfaced this mechanism's problem. The sparser,
    # harder-to-search parameters used elsewhere in this project's
    # benchmark are what let `_densify` show a *recall* win instead; this
    # script is only about the promotion mechanism's cost, so it stays
    # faithful to the scenario that first caught it.
    rng = np.random.default_rng(seed)
    vectors, centers = _clustered_dataset(rng, [600, 20], dim=DIM, spread=0.5, gap=8.0)
    cold_center = centers[1]

    def cold_query():
        return cold_center + rng.normal(scale=0.5, size=DIM)

    base = HNSW(dim=DIM, M=14, ef_construction=80, seed=seed + 100, self_tuning=False)
    for v in vectors:
        base.insert(v)

    import copy

    static = copy.deepcopy(base)
    promoted = copy.deepcopy(base)

    warm_up_queries = [cold_query() for _ in range(150)]
    _warm_up_with_promotion(promoted, warm_up_queries, k=10, ef=50, threshold=5)

    def avg_breakdown(index, n=60):
        uppers, layer0s = [], []
        for _ in range(n):
            index.distance_computations = 0
            u, l0 = _search_with_layer_breakdown(index, cold_query(), k=10, ef=50)
            uppers.append(u)
            layer0s.append(l0)
        return float(np.mean(uppers)), float(np.mean(layer0s))

    return avg_breakdown(static), avg_breakdown(promoted)


def main() -> None:
    # Averaged over several independent seeds: any one seed's query
    # sample is noisy enough (a few points either way) that a single run
    # could make promotion look marginally better or worse by chance.
    # What is stable across seeds, and what this chart actually backs, is
    # that layer 0 (the dominant, ef-bounded cost) barely moves between
    # the two, because promotion never touches layer-0 edges at all.
    seeds = range(5)
    statics, promoteds = zip(*(_run_one_seed(s) for s in seeds), strict=True)
    static_upper = float(np.mean([s[0] for s in statics]))
    static_l0 = float(np.mean([s[1] for s in statics]))
    promoted_upper = float(np.mean([p[0] for p in promoteds]))
    promoted_l0 = float(np.mean([p[1] for p in promoteds]))

    for label, upper, layer0 in (
        ("static", static_upper, static_l0),
        ("promoted", promoted_upper, promoted_l0),
    ):
        print(f"{label:10s} upper={upper:.1f}  layer0={layer0:.1f}  total={upper + layer0:.1f}")
    print(f"(averaged over {len(seeds)} seeds, 60 queries each)")

    ASSETS_DIR.mkdir(exist_ok=True)
    plot_promotion_vs_densify(
        static=(static_upper, static_l0),
        promoted=(promoted_upper, promoted_l0),
        path=ASSETS_DIR / "promotion_vs_densify.png",
    )
    print(f"wrote {ASSETS_DIR / 'promotion_vs_densify.png'}")


if __name__ == "__main__":
    main()
