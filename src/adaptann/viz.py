"""Charts for the drifting-workload benchmark.

Shared visual language with this author's other from-scratch projects:
slate gray for the static baseline, emerald green for the self-tuning
index, amber only for callout annotations. Bars and lines get their value
labeled directly so the numbers are legible without reading the axis.
"""

from __future__ import annotations

_STATIC = "#64748b"
_ADAPTIVE = "#059669"
_ADAPTIVE_DARK = "#065f46"
_ACCENT = "#b45309"
_GRID = "#e5e7eb"
_TEXT = "#1f2937"
_MUTED = "#6b7280"
_BAND_A = "#f8fafc"
_BAND_B = "#f1f5f9"


def _style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(_MUTED)
    ax.spines["bottom"].set_color(_MUTED)
    ax.tick_params(colors=_MUTED)
    ax.title.set_color(_TEXT)
    ax.xaxis.label.set_color(_TEXT)
    ax.yaxis.label.set_color(_TEXT)


def plot_recall_over_time(
    static_recalls: list[float],
    adaptive_recalls: list[float],
    epoch_boundaries: list[int],
    topic_labels: list[int],
    path,
    *,
    title: str = "Recall@k under a drifting query workload",
) -> None:
    """One line per index, batch index on the x-axis, recall@k on the y.
    Alternating shaded bands mark each epoch (a burst of traffic hitting
    one topic cluster), labeled with which topic is hot, so a reader can
    see the static line stay flat while the adaptive line recovers each
    time traffic drifts to an under-connected region."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("white")
    x = list(range(len(static_recalls)))

    for i in range(len(epoch_boundaries) - 1):
        band = _BAND_A if i % 2 == 0 else _BAND_B
        ax.axvspan(epoch_boundaries[i], epoch_boundaries[i + 1], color=band, zorder=0)
        mid = (epoch_boundaries[i] + epoch_boundaries[i + 1]) / 2
        ax.text(
            mid, 1.045, f"topic {topic_labels[i]}", ha="center", va="bottom",
            fontsize=8.5, color=_MUTED,
        )

    ax.axhline(1.0, color=_GRID, linewidth=1, zorder=1)
    ax.plot(x, static_recalls, color=_STATIC, linewidth=2, label="static", zorder=3)
    ax.plot(x, adaptive_recalls, color=_ADAPTIVE, linewidth=2, label="self-tuning", zorder=4)
    ahead = [a >= s for a, s in zip(adaptive_recalls, static_recalls, strict=True)]
    ax.fill_between(x, static_recalls, adaptive_recalls, where=ahead, color=_ADAPTIVE, alpha=0.08, zorder=2)

    ax.set_xlim(0, len(x) - 1)
    ax.set_ylim(0, 1.12)
    ax.set_xlabel("query batch (time)")
    ax.set_ylabel("recall@k")
    ax.set_title(title, fontsize=13, fontweight="bold", loc="left")
    ax.grid(axis="y", color=_GRID, linewidth=0.8, zorder=0)
    ax.legend(loc="lower left", frameon=False, labelcolor=_TEXT)
    _style_axis(ax)

    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def plot_summary_bars(
    mean_static_recall: float,
    mean_adaptive_recall: float,
    p99_static_ms: float,
    p99_adaptive_ms: float,
    path,
) -> None:
    """One figure, two panels: mean recall@k and p99 search latency,
    static vs. self-tuning, side by side."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    fig.patch.set_facecolor("white")
    labels = ["static", "self-tuning"]
    colors = [_STATIC, _ADAPTIVE]

    for ax, values, title, ylabel, fmt in (
        (ax1, [mean_static_recall, mean_adaptive_recall], "Mean recall@k (whole run)", "recall@k", "{:.1%}"),
        (ax2, [p99_static_ms, p99_adaptive_ms], "p99 search latency", "milliseconds", "{:.2f}"),
    ):
        bars = ax.bar(labels, values, color=colors, width=0.55, zorder=3)
        for bar, val in zip(bars, values, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(), fmt.format(val),
                ha="center", va="bottom", fontsize=11, color=_TEXT, fontweight="bold",
            )
        ax.set_title(title, fontsize=12, fontweight="bold", loc="left")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, max(values) * 1.25)
        ax.grid(axis="y", color=_GRID, linewidth=0.8, zorder=0)
        _style_axis(ax)

    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def plot_cold_start_recovery(
    static_first_visit: float,
    adaptive_first_visit: float,
    static_second_visit: float,
    adaptive_second_visit: float,
    topic: int,
    path,
) -> None:
    """The sharpest single number in this project: the worst-connected
    topic cluster's recall@k on its first burst of traffic (cold, nobody
    promoted yet) versus its second burst, a full cycle later (adaptive
    has already densified it and never forgets; static hasn't changed at
    all, because it can't)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("white")

    groups = ["first visit\n(cold)", "second visit\n(one cycle later)"]
    x = range(len(groups))
    width = 0.32

    static_vals = [static_first_visit, static_second_visit]
    adaptive_vals = [adaptive_first_visit, adaptive_second_visit]

    bars1 = ax.bar([i - width / 2 for i in x], static_vals, width, color=_STATIC, label="static", zorder=3)
    bars2 = ax.bar([i + width / 2 for i in x], adaptive_vals, width, color=_ADAPTIVE, label="self-tuning", zorder=3)
    for bars in (bars1, bars2):
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{bar.get_height():.0%}",
                ha="center", va="bottom", fontsize=11, color=_TEXT, fontweight="bold",
            )

    ax.set_xticks(list(x))
    ax.set_xticklabels(groups)
    ax.set_ylabel("recall@k")
    ax.set_ylim(0, 1.18)
    title = f"Topic {topic}, the hardest-to-reach cluster: cold vs. warmed up"
    ax.set_title(title, fontsize=12, fontweight="bold", loc="left")
    ax.grid(axis="y", color=_GRID, linewidth=0.8, zorder=0)
    ax.legend(loc="upper left", frameon=False, labelcolor=_TEXT)
    _style_axis(ax)

    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)
