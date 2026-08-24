"""Charts for the drifting-workload benchmark.

Visual language: amber for the static baseline, teal for the self-tuning
index (validated as a colorblind-safe pair, not eyeballed), plus a
diagonal hatch on every static bar as a second, non-color-dependent way
to tell the two apart. Bars are flat, fully rounded "pill" shapes with a
value label set directly inside each one, a deliberately different
silhouette from a conventional flat-topped column chart.
"""

from __future__ import annotations

_STATIC = "#b45309"
_STATIC_DARK = "#7c2d12"
_ADAPTIVE = "#0d9488"
_ADAPTIVE_DARK = "#115e59"
_REJECTED = "#7c3aed"
_REJECTED_DARK = "#4c1d95"
_GRID = "#e5e7eb"
_TEXT = "#1f2937"
_MUTED = "#6b7280"
_BAND_A = "#f8fafc"
_BAND_B = "#f1f5f9"


def _style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(_MUTED)
    ax.tick_params(colors=_MUTED, left=False)
    ax.title.set_color(_TEXT)
    ax.xaxis.label.set_color(_TEXT)
    ax.yaxis.label.set_color(_TEXT)


def _pill_bar(
    ax, x_center: float, height: float, width: float, color: str, text: str, *,
    hatch: str | None = None, zorder: int = 3,
) -> None:
    """A flat, fully-rounded "pill" bar (not a gradient column): solid
    fill, a thin white separator stroke, an optional diagonal hatch (used
    for the static series, so the two are still distinguishable without
    relying on color at all), and its value set as white text inside the
    bar rather than floating above it."""
    from matplotlib.patches import FancyBboxPatch

    if height <= 0:
        return
    radius = min(width, height) * 0.4

    patch = FancyBboxPatch(
        (x_center - width / 2, -radius), width, height + radius,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=1.8, edgecolor="white", facecolor=color, hatch=hatch,
        zorder=zorder, transform=ax.transData,
    )
    ax.add_patch(patch)

    if height > radius * 2.4:
        ax.text(
            x_center, height - radius * 0.9, text, ha="center", va="top",
            fontsize=12.5, color="white", fontweight="bold", zorder=zorder + 2,
        )
    else:
        ax.text(
            x_center, height + radius * 0.5, text, ha="center", va="bottom",
            fontsize=12.5, color=_TEXT, fontweight="bold", zorder=zorder + 2,
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
    static vs. self-tuning, side by side, as flat hatch-vs-solid pill
    bars with the value set inside each one."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 5))
    fig.patch.set_facecolor("white")
    labels = ["static", "self-tuning"]
    colors = [_STATIC, _ADAPTIVE]
    hatches = ["////", None]
    bar_width = 0.5

    for ax, values, title, ylabel, fmt in (
        (ax1, [mean_static_recall, mean_adaptive_recall], "Mean recall@k, whole run", "recall@k", "{:.1%}"),
        (ax2, [p99_static_ms, p99_adaptive_ms], "p99 search latency", "milliseconds", "{:.2f} ms"),
    ):
        ax.set_facecolor(_BAND_A)
        for i, (val, color, hatch) in enumerate(zip(values, colors, hatches, strict=True)):
            _pill_bar(ax, i, val, bar_width, color, fmt.format(val), hatch=hatch)

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
    all, because it can't). Flat hatch-vs-solid pill bars, matching the
    rest of this project's charts."""
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
        _pill_bar(ax, c - offset, s_val, width, _STATIC, f"{s_val:.0%}", hatch="////")
        _pill_bar(ax, c + offset, a_val, width, _ADAPTIVE, f"{a_val:.0%}")

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


def plot_promotion_vs_densify(static: tuple[float, float], promoted: tuple[float, float], path) -> None:
    """The chart behind the redesign story: mean distance computations per
    query, split into upper-layer descent vs. the layer-0 pass, for the
    static baseline and for the abandoned upper-layer-promotion
    mechanism, averaged over several seeds. Layer 0 barely moves between
    the two, which is the actual diagnosis: promotion cannot help,
    because it never touches the layer that dominates cost. ``promoted``
    is marked with a cross-hatch and a violet fill, distinct in kind (not
    just color) from the amber static baseline, since it is not a real
    alternative, just a documented dead end; see the summary and
    cold-start charts elsewhere in this README for the mechanism that
    replaced it."""
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor(_BAND_A)

    groups = ["upper layers\n(descent to ef=1)", "layer 0\n(the ef-bounded pass)"]
    centers = [0.0, 1.3]
    offsets = [-0.24, 0.24]
    width = 0.4
    series = [(_STATIC, None, "static"), (_REJECTED, "xxxx", "promoted (abandoned)")]
    values_by_group = [[static[0], promoted[0]], [static[1], promoted[1]]]
    max_val = max(max(v) for v in values_by_group)

    # The corner radius has to be measured against ``width`` (x-data-units),
    # not against the y-scale: this axis spans ~2 x-units but ~350 y-units,
    # so a radius picked as a fraction of the y-range would be many times
    # wider than the bars themselves. `min(width, val)` keeps it sane for
    # both the short "upper layers" and tall "layer 0" bars alike, since
    # every bar here is far taller than it is wide.
    for group_center, values in zip(centers, values_by_group, strict=True):
        for offset, val, (color, hatch, _label) in zip(offsets, values, series, strict=True):
            x_center = group_center + offset
            radius = min(width, val) * 0.4
            patch = FancyBboxPatch(
                (x_center - width / 2, -radius), width, val + radius,
                boxstyle=f"round,pad=0,rounding_size={radius}",
                linewidth=1.8, edgecolor="white", facecolor=color, hatch=hatch, zorder=3,
                transform=ax.transData,
            )
            ax.add_patch(patch)
            ax.text(
                x_center, val + max_val * 0.02, f"{val:.0f}", ha="center", va="bottom",
                fontsize=12, fontweight="bold", color=_TEXT, zorder=6,
            )

    ax.set_xticks(centers)
    ax.set_xticklabels(groups, fontsize=11)
    ax.set_xlim(centers[0] - 0.7, centers[-1] + 0.7)
    ax.set_ylabel("mean distance computations")
    ax.set_ylim(0, max_val * 1.22)
    ax.set_title(
        "Layer 0, the dominant cost, barely moves: promotion never touches it",
        fontsize=13, fontweight="bold", loc="left", pad=14,
    )
    ax.grid(axis="y", color="white", linewidth=1.4, zorder=0)
    ax.set_axisbelow(True)

    handles = [mpatches.Patch(facecolor=c, hatch=h, label=label) for c, h, label in series]
    ax.legend(handles=handles, loc="upper right", frameon=False, labelcolor=_TEXT, fontsize=10)
    _style_axis(ax)

    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def plot_degree_widening(normal_degree: float, widened_degree: float, degree_cap: int, path) -> None:
    """The chart behind the honest-cost section: a densified node's actual
    layer-0 degree against a typical, never-hot node's, both measured on
    the same warmed-up index from ``demos/benchmark.py``, with the normal
    construction-time degree cap marked as a reference line. This is the
    concrete shape of "later searches through it examine more
    candidates": not a multiplier claimed in prose, a real average."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor(_BAND_A)

    labels = ["typical node", "densified node"]
    values = [normal_degree, widened_degree]
    colors = [_STATIC, _ADAPTIVE]
    hatches = ["////", None]
    width = 0.5

    for i, (val, color, hatch) in enumerate(zip(values, colors, hatches, strict=True)):
        _pill_bar(ax, i, val, width, color, f"{val:.1f}", hatch=hatch)

    ax.axhline(degree_cap, color=_MUTED, linewidth=1.5, linestyle="--", zorder=4)
    ax.text(
        1.62, degree_cap, f" cap: {degree_cap} ", color=_MUTED, fontsize=9.5,
        va="center", ha="left", bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none"},
        zorder=5,
    )

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_xlim(-0.6, 2.0)
    ax.set_ylabel("layer-0 neighbor count")
    ax.set_ylim(0, max(values) * 1.3)
    ax.set_title(
        "Densified nodes end up far more connected than typical ones",
        fontsize=13, fontweight="bold", loc="left", pad=14,
    )
    ax.grid(axis="y", color="white", linewidth=1.4, zorder=0)
    ax.set_axisbelow(True)
    _style_axis(ax)

    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)
