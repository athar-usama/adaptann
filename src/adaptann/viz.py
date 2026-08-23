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
_SHADOW = "#0f172a"


def _style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(_MUTED)
    ax.tick_params(colors=_MUTED, left=False)
    ax.title.set_color(_TEXT)
    ax.xaxis.label.set_color(_TEXT)
    ax.yaxis.label.set_color(_TEXT)


def _lighten(hex_color: str, amount: float) -> tuple:
    from matplotlib.colors import to_rgb

    r, g, b = to_rgb(hex_color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


def _gradient_bar(ax, x_center: float, height: float, width: float, color: str, *, zorder: int = 3) -> None:
    """A single bar with a rounded top, a soft vertical color gradient, and
    a faint drop shadow, instead of a flat-filled rectangle. The rounded
    bottom corners a plain FancyBboxPatch would draw are pushed below
    ``y=0`` and cropped away by the axes' own clip box, so only the top
    stays visibly rounded."""
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import FancyBboxPatch

    if height <= 0:
        return
    radius = min(width, height) * 0.22

    shadow = FancyBboxPatch(
        (x_center - width / 2 + width * 0.05, -radius),
        width, height + radius,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=0, facecolor=_SHADOW, alpha=0.10, zorder=zorder - 1,
        transform=ax.transData,
    )
    ax.add_patch(shadow)

    outline = FancyBboxPatch(
        (x_center - width / 2, -radius), width, height + radius,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=0, facecolor="none", zorder=zorder,
        transform=ax.transData,
    )
    ax.add_patch(outline)

    cmap = LinearSegmentedColormap.from_list("bar", [_lighten(color, 0.55), color])
    gradient = np.linspace(0, 1, 256).reshape(-1, 1)
    im = ax.imshow(
        gradient, extent=(x_center - width / 2, x_center + width / 2, -radius, height),
        origin="lower", aspect="auto", cmap=cmap, zorder=zorder,
        transform=ax.transData,
    )
    im.set_clip_path(outline)


def _value_chip(ax, x_center: float, y: float, text: str, *, color: str = _TEXT) -> None:
    ax.text(
        x_center, y, text, ha="center", va="bottom", fontsize=12.5, color=color, fontweight="bold",
        bbox={"boxstyle": "round,pad=0.32", "facecolor": "white", "edgecolor": _GRID, "linewidth": 1},
        zorder=6,
    )


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
    static vs. self-tuning, side by side, as rounded gradient bars with a
    soft shadow and a labeled value chip above each one."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 5))
    fig.patch.set_facecolor("white")
    labels = ["static", "self-tuning"]
    colors = [_STATIC, _ADAPTIVE]
    bar_width = 0.5

    for ax, values, title, ylabel, fmt in (
        (ax1, [mean_static_recall, mean_adaptive_recall], "Mean recall@k, whole run", "recall@k", "{:.1%}"),
        (ax2, [p99_static_ms, p99_adaptive_ms], "p99 search latency", "milliseconds", "{:.2f} ms"),
    ):
        ax.set_facecolor(_BAND_A)
        for i, (val, color) in enumerate(zip(values, colors, strict=True)):
            _gradient_bar(ax, i, val, bar_width, color)
            _value_chip(ax, i, val, fmt.format(val), color=_ADAPTIVE_DARK if color == _ADAPTIVE else _TEXT)

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_xlim(-0.6, len(labels) - 0.4)
        ax.set_title(title, fontsize=13, fontweight="bold", loc="left", pad=14)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, max(values) * 1.32)
        ax.grid(axis="y", color="white", linewidth=1.4, zorder=0)
        ax.set_axisbelow(True)
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
    all, because it can't). Rounded gradient bars with a value chip on
    each, matching the rest of this project's charts."""
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor(_BAND_A)

    groups = ["first visit  (cold)", "second visit  (one cycle later)"]
    centers = [0.0, 1.35]
    offset = 0.28
    width = 0.42

    static_vals = [static_first_visit, static_second_visit]
    adaptive_vals = [adaptive_first_visit, adaptive_second_visit]

    for c, s_val, a_val in zip(centers, static_vals, adaptive_vals, strict=True):
        _gradient_bar(ax, c - offset, s_val, width, _STATIC)
        _value_chip(ax, c - offset, s_val, f"{s_val:.0%}")
        _gradient_bar(ax, c + offset, a_val, width, _ADAPTIVE)
        _value_chip(ax, c + offset, a_val, f"{a_val:.0%}", color=_ADAPTIVE_DARK)

    ax.set_xticks(centers)
    ax.set_xticklabels(groups, fontsize=11)
    ax.set_xlim(-0.75, centers[-1] + 0.75)
    ax.set_ylabel("recall@k")
    ax.set_ylim(0, 1.22)
    title = f"Topic {topic}, the hardest-to-reach cluster: cold vs. warmed up"
    ax.set_title(title, fontsize=13, fontweight="bold", loc="left", pad=14)
    ax.grid(axis="y", color="white", linewidth=1.4, zorder=0)
    ax.set_axisbelow(True)

    handles = [
        mpatches.Patch(facecolor=_STATIC, label="static"),
        mpatches.Patch(facecolor=_ADAPTIVE, label="self-tuning"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False, labelcolor=_TEXT, fontsize=10.5)
    _style_axis(ax)

    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)
