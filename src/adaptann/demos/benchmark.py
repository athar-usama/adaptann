"""Simulates a drifting query workload against a from-scratch HNSW index,
built once, then queried two ways in parallel: as a static baseline, and
as a self-tuning index that densifies hot regions as it sees them, with
no rebuild. This script is what produced the numbers and charts in the
README, not an illustration of them.

The dataset is one large, densely-sampled "core" cluster plus several
smaller, more sparsely-connected "topic" clusters, each far from the
others. Construction (with a small ``M`` and ``ef_construction``, so it
doesn't have unlimited budget to spend) has no way to know which topic
will matter later. Query traffic then drifts: it hammers one topic for a
while, then moves on to the next, cycling back around more than once so
the run can show whether an index that adapted earlier stays adapted.
"""

from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np

from ..bruteforce import brute_force_knn
from ..hnsw import HNSW
from ..metrics import recall_at_k
from ..viz import plot_cold_start_recovery, plot_recall_over_time, plot_summary_bars, render_degree_diagram_svg

ROOT = Path(__file__).resolve().parent.parent.parent.parent
ASSETS_DIR = ROOT / "assets"

DIM = 32
CORE_N = 500
N_TOPICS = 5
TOPIC_N = 60
CORE_SPREAD = 1.2
TOPIC_SPREAD = 1.6
TOPIC_GAP = 12.0
M = 5
EF_CONSTRUCTION = 20
K = 10
SEARCH_EF = 10
PROMOTION_THRESHOLD = 5
WIDEN_FACTOR = 4
N_CYCLES = 2
BATCHES_PER_EPOCH = 8
BATCH_SIZE = 15
SEED = 7


def _build_dataset(rng: np.random.Generator) -> tuple[np.ndarray, list[np.ndarray]]:
    core = rng.normal(scale=CORE_SPREAD, size=(CORE_N, DIM))
    topic_centers = [rng.normal(scale=TOPIC_GAP, size=DIM) for _ in range(N_TOPICS)]
    topics = [c + rng.normal(scale=TOPIC_SPREAD, size=(TOPIC_N, DIM)) for c in topic_centers]
    vectors = np.concatenate([core, *topics], axis=0)
    return vectors, topic_centers


def main() -> None:
    rng = np.random.default_rng(SEED)
    vectors, topic_centers = _build_dataset(rng)

    base = HNSW(dim=DIM, M=M, ef_construction=EF_CONSTRUCTION, seed=SEED + 1, self_tuning=False)
    for v in vectors:
        base.insert(v)

    static = copy.deepcopy(base)
    adaptive = copy.deepcopy(base)
    adaptive.self_tuning = True
    adaptive.promotion_threshold = PROMOTION_THRESHOLD
    adaptive.widen_factor = WIDEN_FACTOR

    def query_for(topic_idx: int) -> np.ndarray:
        return topic_centers[topic_idx % N_TOPICS] + rng.normal(scale=TOPIC_SPREAD, size=DIM)

    epochs = N_TOPICS * N_CYCLES
    static_series: list[float] = []
    adaptive_series: list[float] = []
    topic_per_epoch: list[int] = []
    static_lat: list[float] = []
    adaptive_lat: list[float] = []
    per_topic_first_batch: dict[int, dict[str, float]] = {}

    for epoch in range(epochs):
        topic_idx = epoch % N_TOPICS
        topic_per_epoch.append(topic_idx)
        for batch in range(BATCHES_PER_EPOCH):
            s_recalls, a_recalls = [], []
            for _ in range(BATCH_SIZE):
                q = query_for(topic_idx)
                exact = brute_force_knn(vectors, q, K).tolist()

                t0 = time.perf_counter()
                s_approx = [nid for nid, _ in static.search(q, K, ef=SEARCH_EF)]
                static_lat.append(time.perf_counter() - t0)

                t0 = time.perf_counter()
                a_approx = [nid for nid, _ in adaptive.search(q, K, ef=SEARCH_EF)]
                adaptive_lat.append(time.perf_counter() - t0)

                s_recalls.append(recall_at_k(s_approx, exact))
                a_recalls.append(recall_at_k(a_approx, exact))

            static_series.append(float(np.mean(s_recalls)))
            adaptive_series.append(float(np.mean(a_recalls)))

            snapshot = {"static": static_series[-1], "adaptive": adaptive_series[-1]}
            if batch == 0 and topic_idx not in per_topic_first_batch:
                per_topic_first_batch[topic_idx] = snapshot
            second_key = f"{topic_idx}_second"
            if batch == 0 and second_key not in per_topic_first_batch and epoch >= N_TOPICS:
                per_topic_first_batch[second_key] = snapshot

    epoch_boundaries = [i * BATCHES_PER_EPOCH for i in range(epochs + 1)]

    mean_static = float(np.mean(static_series))
    mean_adaptive = float(np.mean(adaptive_series))
    p99_static_ms = float(np.percentile(static_lat, 99)) * 1000
    p99_adaptive_ms = float(np.percentile(adaptive_lat, 99)) * 1000

    batch_topics = np.repeat(topic_per_epoch, BATCHES_PER_EPOCH)
    per_topic_static_mean = {
        t: float(np.mean([s for s, top in zip(static_series, batch_topics, strict=True) if top == t]))
        for t in range(N_TOPICS)
    }
    worst_topic = min(per_topic_static_mean, key=per_topic_static_mean.get)

    print(f"dataset: {len(vectors)} vectors ({CORE_N} core + {N_TOPICS} x {TOPIC_N} topic), dim={DIM}")
    print(f"index: M={M} ef_construction={EF_CONSTRUCTION}, search ef={SEARCH_EF}, k={K}")
    print(f"simulated drift: {epochs} epochs x {BATCHES_PER_EPOCH} batches x {BATCH_SIZE} queries "
          f"= {epochs * BATCHES_PER_EPOCH * BATCH_SIZE} queries per index")
    print(f"self-tuning densifications: {adaptive.promotions}")
    print(f"mean recall@{K}: static {mean_static:.1%}  self-tuning {mean_adaptive:.1%}")
    print(f"p99 search latency: static {p99_static_ms:.2f}ms  self-tuning {p99_adaptive_ms:.2f}ms")
    print(
        "note: at this dataset scale, Python-level overhead per distance call dominates\n"
        "wall-clock latency, the same caveat as this author's speculative-decoding project;\n"
        "recall@k, not latency, is the metric that isolates what self-tuning actually changes."
    )
    worst_recall = per_topic_static_mean[worst_topic]
    print(f"hardest-to-reach cluster: topic {worst_topic} (lowest mean static recall, {worst_recall:.1%})")

    not_widened = set(range(len(adaptive.vectors))) - adaptive.widened
    normal_degree = float(np.mean([len(adaptive.graph[0].get(nid, ())) for nid in not_widened]))
    widened_degree = float(np.mean([len(adaptive.graph[0][nid]) for nid in adaptive.widened]))
    print(
        f"layer-0 degree: {normal_degree:.1f} typical (cap {adaptive.M_max0}) vs "
        f"{widened_degree:.1f} for the {len(adaptive.widened)} densified nodes"
    )

    if worst_topic in per_topic_first_batch and f"{worst_topic}_second" in per_topic_first_batch:
        first = per_topic_first_batch[worst_topic]
        second = per_topic_first_batch[f"{worst_topic}_second"]
        print(
            f"topic {worst_topic}, first burst of traffic (cold): "
            f"static {first['static']:.0%}  self-tuning {first['adaptive']:.0%}"
        )
        print(
            f"topic {worst_topic}, one full cycle later (self-tuning has since densified it, "
            f"and never forgets): static {second['static']:.0%}  self-tuning {second['adaptive']:.0%}"
        )

    ASSETS_DIR.mkdir(exist_ok=True)
    plot_recall_over_time(
        static_series, adaptive_series, epoch_boundaries, topic_per_epoch,
        ASSETS_DIR / "recall_over_time.png",
    )
    plot_summary_bars(
        mean_static, mean_adaptive, p99_static_ms, p99_adaptive_ms,
        ASSETS_DIR / "summary_bars.png",
    )
    if worst_topic in per_topic_first_batch and f"{worst_topic}_second" in per_topic_first_batch:
        first = per_topic_first_batch[worst_topic]
        second = per_topic_first_batch[f"{worst_topic}_second"]
        plot_cold_start_recovery(
            first["static"], first["adaptive"], second["static"], second["adaptive"],
            worst_topic, ASSETS_DIR / "cold_start_recovery.png",
        )
    render_degree_diagram_svg(normal_degree, widened_degree, adaptive.M_max0, ASSETS_DIR / "degree_widening.svg")
    print(f"wrote charts to {ASSETS_DIR}")


if __name__ == "__main__":
    main()
