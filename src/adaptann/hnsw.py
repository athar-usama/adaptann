"""A from-scratch HNSW (Hierarchical Navigable Small World) approximate
nearest-neighbor index (Malkov & Yashunin, 2016), with one addition
standard from-scratch clones of this algorithm don't have: an optional
self-tuning layer that densifies the local (layer 0) neighborhood of
frequently-returned nodes as query traffic drifts, without a full
rebuild. See the README for the mechanism, and why an earlier version
that promoted hot nodes to higher graph layers instead turned out not
to help.

Construction and search follow the paper's algorithm directly: greedy
descent through layers with a single-candidate search (``ef=1``) to find a
good entry point, then a best-first ``SEARCH-LAYER`` at the target layer(s)
with a beam of size ``ef``, connecting each new node to its ``M`` closest
candidates and pruning any neighbor that now exceeds its layer's degree cap.
"""

from __future__ import annotations

import heapq
import math
from collections import defaultdict

import numpy as np

from .distance import sq_euclidean_one


class HNSW:
    def __init__(
        self,
        dim: int,
        *,
        M: int = 16,
        ef_construction: int = 100,
        self_tuning: bool = False,
        promotion_threshold: int = 20,
        widen_factor: int = 3,
        seed: int = 0,
    ):
        self.dim = dim
        self.M = M
        self.M_max0 = 2 * M  # layer 0 keeps roughly double the degree, standard HNSW choice
        self.ef_construction = ef_construction
        self.ml = 1.0 / math.log(M)
        self.rng = np.random.default_rng(seed)

        self.vectors: list[np.ndarray] = []
        self.levels: list[int] = []
        self.graph: list[dict[int, set[int]]] = [{}]  # graph[layer][node_id] -> neighbor ids
        self.entry_point: int | None = None
        self.max_level = -1

        self.self_tuning = self_tuning
        self.promotion_threshold = promotion_threshold
        self.widen_factor = widen_factor
        self.hit_counts: dict[int, int] = defaultdict(int)
        self.widened: set[int] = set()
        self.promotions = 0

        # A scale-invariant cost metric: counts every squared-distance
        # evaluation done by _search_layer, regardless of how fast a given
        # machine happens to run at the moment. Reset it (`.distance_computations
        # = 0`) before a call whose cost you want to measure in isolation.
        self.distance_computations = 0

    # -- construction ---------------------------------------------------------
    def _random_level(self) -> int:
        return min(int(-math.log(self.rng.random()) * self.ml), 16)

    def _ensure_layer(self, layer: int) -> None:
        while len(self.graph) <= layer:
            self.graph.append({})

    def _search_layer(self, query: np.ndarray, entry_points: list[int], ef: int, layer: int) -> list[tuple[float, int]]:
        """Best-first search within a single layer. Returns up to ``ef``
        ``(squared_distance, node_id)`` pairs, sorted nearest-first."""
        visited = set(entry_points)
        candidates: list[tuple[float, int]] = []
        results: list[tuple[float, int]] = []  # max-heap via negated distance
        for ep in entry_points:
            d = sq_euclidean_one(query, self.vectors[ep])
            self.distance_computations += 1
            heapq.heappush(candidates, (d, ep))
            heapq.heappush(results, (-d, ep))

        while candidates:
            d, current = heapq.heappop(candidates)
            worst = -results[0][0]
            if len(results) >= ef and d > worst:
                break
            for neighbor in self.graph[layer].get(current, ()):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                dn = sq_euclidean_one(query, self.vectors[neighbor])
                self.distance_computations += 1
                worst = -results[0][0]
                if len(results) < ef or dn < worst:
                    heapq.heappush(candidates, (dn, neighbor))
                    heapq.heappush(results, (-dn, neighbor))
                    if len(results) > ef:
                        heapq.heappop(results)

        return sorted((-nd, nid) for nd, nid in results)

    @staticmethod
    def _select_neighbors_simple(candidates: list[tuple[float, int]], m: int) -> list[tuple[float, int]]:
        return candidates[:m]

    def _connect_at_layer(self, node_id: int, vector: np.ndarray, entry_points: list[int], layer: int) -> list[int]:
        """Search ``layer`` from ``entry_points``, connect ``node_id`` to its
        ``M`` closest candidates there, and prune any neighbor whose degree
        cap that connection pushed past. Returns the new candidates for the
        next layer down."""
        self._ensure_layer(layer)
        candidates = self._search_layer(vector, entry_points, self.ef_construction, layer)
        neighbors = self._select_neighbors_simple(candidates, self.M)
        self.graph[layer][node_id] = {nid for _, nid in neighbors}

        m_max = self.M_max0 if layer == 0 else self.M
        for _, nid in neighbors:
            self.graph[layer].setdefault(nid, set())
            self.graph[layer][nid].add(node_id)
            if len(self.graph[layer][nid]) > m_max:
                self._prune(nid, layer, m_max)

        return [nid for _, nid in neighbors] or entry_points

    def _prune(self, node_id: int, layer: int, m_max: int) -> None:
        neighbors = list(self.graph[layer][node_id])
        ranked = sorted((sq_euclidean_one(self.vectors[node_id], self.vectors[n]), n) for n in neighbors)
        keep = {n for _, n in ranked[:m_max]}
        for dropped in set(neighbors) - keep:
            self.graph[layer][dropped].discard(node_id)
        self.graph[layer][node_id] = keep

    def insert(self, vector: np.ndarray) -> int:
        vector = np.asarray(vector, dtype=np.float64)
        node_id = len(self.vectors)
        self.vectors.append(vector)
        level = self._random_level()
        self.levels.append(level)
        self._ensure_layer(level)

        if self.entry_point is None:
            for layer in range(level + 1):
                self.graph[layer][node_id] = set()
            self.entry_point = node_id
            self.max_level = level
            return node_id

        curr_nearest = [self.entry_point]
        for layer in range(self.max_level, level, -1):
            curr_nearest = [nid for _, nid in self._search_layer(vector, curr_nearest, ef=1, layer=layer)]

        for layer in range(min(level, self.max_level), -1, -1):
            curr_nearest = self._connect_at_layer(node_id, vector, curr_nearest, layer)

        if level > self.max_level:
            self.entry_point = node_id
            self.max_level = level
        return node_id

    # -- search -----------------------------------------------------------------
    def search(self, query: np.ndarray, k: int, ef: int | None = None) -> list[tuple[int, float]]:
        """Returns up to ``k`` ``(node_id, euclidean_distance)`` pairs,
        nearest-first. If ``self_tuning`` is enabled, this also records a
        "hit" for every returned node and may promote hot nodes afterward."""
        if self.entry_point is None:
            return []
        query = np.asarray(query, dtype=np.float64)
        ef = ef if ef is not None else max(self.ef_construction, k)

        curr_nearest = [self.entry_point]
        for layer in range(self.max_level, 0, -1):
            curr_nearest = [nid for _, nid in self._search_layer(query, curr_nearest, ef=1, layer=layer)]

        candidates = self._search_layer(query, curr_nearest, ef, layer=0)
        top = candidates[:k]
        result_ids = [nid for _, nid in top]

        if self.self_tuning:
            self._record_hits(result_ids)

        return [(nid, math.sqrt(d)) for d, nid in top]

    # -- self-tuning --------------------------------------------------------------
    #
    # The first version of this promoted hot nodes to higher graph layers,
    # the same way a fresh insert climbs the hierarchy. Measured against a
    # deep-copied static baseline on an identical, fixed query set, that
    # made total search cost *worse*, not better: a per-layer breakdown
    # showed the layer-0 pass (bounded by `ef`) already cost the same in
    # both variants, because the upper layers already routed a query into
    # roughly the right neighborhood regardless of promotion. Promotion
    # only added one more layer to descend through. The layer that
    # actually bounds recall and search cost is layer 0, so that's what
    # self-tuning densifies instead: a hot node's own local neighborhood
    # gets wider, not its position in the hierarchy. See
    # ``tests/test_self_tuning.py`` for the recall-at-fixed-ef comparison
    # that replaced the flawed distance-computation comparison.
    def _record_hits(self, result_ids: list[int]) -> None:
        for nid in result_ids:
            self.hit_counts[nid] += 1
            if self.hit_counts[nid] >= self.promotion_threshold:
                self._densify(nid)

    def _densify(self, node_id: int) -> None:
        """Widen a hot node's layer-0 neighbor set beyond the normal degree
        cap, one time, by searching further out from it than construction
        did and keeping whatever new, still-close neighbors that search
        turns up. Existing neighbors are never pruned back down: growing
        the hot region's mutual connectivity is the entire point."""
        if node_id in self.widened:
            self.hit_counts[node_id] = 0
            return

        vector = self.vectors[node_id]
        wide_ef = self.ef_construction * self.widen_factor
        candidates = self._search_layer(vector, [node_id], wide_ef, layer=0)
        wide_cap = self.M_max0 * self.widen_factor
        chosen = [nid for _, nid in candidates if nid != node_id][:wide_cap]

        for nid in chosen:
            self.graph[0][node_id].add(nid)
            self.graph[0].setdefault(nid, set()).add(node_id)

        self.widened.add(node_id)
        self.hit_counts[node_id] = 0
        self.promotions += 1

    def __len__(self) -> int:
        return len(self.vectors)
