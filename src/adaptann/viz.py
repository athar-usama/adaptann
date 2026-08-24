"""Charts for the drifting-workload benchmark.

Visual language: amber for the static baseline, teal for the self-tuning
index (validated as a colorblind-safe pair, not eyeballed), plus a
diagonal hatch on every static bar as a second, non-color-dependent way
to tell the two apart. Bars are flat, fully rounded "pill" shapes with a
value label set directly inside each one, a deliberately different
silhouette from a conventional flat-topped column chart.
"""

from __future__ import annotations

from pathlib import Path

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
    """The chart behind the redesign story, as a dot-and-stem comparison
    rather than another bar chart: for each of the two search phases, one
    dot for the static baseline and one for the abandoned promotion
    mechanism, on a shared stem. Two dots landing on almost the same
    point *is* the finding, and a bar chart tends to visually exaggerate
    small differences through bar-height weight; a dot plot doesn't."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 4.6))
    fig.patch.set_facecolor("white")

    rows = [
        (ax1, "layer 0  (the ef-bounded pass)", static[1], promoted[1]),
        (ax2, "upper layers  (descent to ef=1)", static[0], promoted[0]),
    ]

    for ax, label, static_val, promoted_val in rows:
        ax.set_facecolor(_BAND_A)
        lo, hi = min(static_val, promoted_val), max(static_val, promoted_val)
        pad = max(hi * 0.35, 3)
        ax.plot([lo, hi], [0, 0], color=_MUTED, linewidth=2, zorder=2, solid_capstyle="round")
        ax.scatter([static_val], [0], s=260, color=_STATIC, zorder=3, edgecolor="white", linewidth=1.5)
        ax.scatter(
            [promoted_val], [0], s=260, color=_REJECTED, marker="X", zorder=4, edgecolor="white", linewidth=1.2
        )
        ax.text(
            static_val, 0.55, f"static\n{static_val:.0f}", ha="center", va="bottom",
            fontsize=10.5, color=_STATIC_DARK, fontweight="bold",
        )
        ax.text(
            promoted_val, -0.55, f"promoted\n{promoted_val:.0f}", ha="center", va="top",
            fontsize=10.5, color=_REJECTED_DARK, fontweight="bold",
        )

        ax.set_ylim(-1.3, 1.3)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_yticks([])
        ax.set_ylabel(label, fontsize=10.5, rotation=0, ha="right", va="center", labelpad=10)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(colors=_MUTED, labelsize=9)

    fig.suptitle(
        "Layer 0, the dominant cost, barely moves: promotion never touches it",
        fontsize=13, fontweight="bold", x=0.02, ha="left", y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def render_degree_diagram_svg(normal_degree: float, widened_degree: float, degree_cap: int, path) -> None:
    """The figure behind the honest-cost section, as an actual node-and-edges
    drawing rather than a bar chart: a hub with ``normal_degree`` spokes
    next to a hub with ``widened_degree`` spokes, one spoke per real
    layer-0 neighbor, both counts measured on the same warmed-up index
    from ``demos/benchmark.py``. The point ("densified nodes end up far
    more connected") is the actual density of lines on the page, not an
    abstraction of it."""
    import math

    bg, text, muted = "#f8fafc", "#1f2937", "#6b7280"
    panel_w = 340
    hub_r, spoke_len, dot_r = 15, 105, 3.4
    top_margin, label_block_h, bottom_margin = 40, 54, 16
    width = panel_w * 2 + 20
    height = top_margin + spoke_len + hub_r + spoke_len + label_block_h + bottom_margin

    def panel(cx: float, cy: float, degree: float, color: str, label: str) -> list[str]:
        n = max(int(round(degree)), 1)
        elems = []
        for i in range(n):
            angle = 2 * math.pi * i / n - math.pi / 2
            ex = cx + spoke_len * math.cos(angle)
            ey = cy + spoke_len * math.sin(angle)
            elems.append(
                f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
                f'stroke="{color}" stroke-width="1.1" opacity="0.55"/>'
            )
            elems.append(f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="{dot_r}" fill="{color}"/>')
        elems.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{hub_r}" fill="{bg}" stroke="{color}" stroke-width="3"/>')
        elems.append(
            f'<text x="{cx:.1f}" y="{cy + spoke_len + 34:.1f}" font-size="15" font-weight="bold" '
            f'fill="{text}" text-anchor="middle">{label}</text>'
        )
        elems.append(
            f'<text x="{cx:.1f}" y="{cy + spoke_len + 54:.1f}" font-size="12.5" fill="{muted}" '
            f'text-anchor="middle">{degree:.1f} layer-0 neighbors</text>'
        )
        return elems

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="Consolas, Menlo, monospace">',
        f'<rect width="{width}" height="{height}" fill="{bg}"/>',
        f'<text x="20" y="26" font-size="14" fill="{text}">Densified nodes end up far more '
        'connected than typical ones</text>',
    ]
    cy = top_margin + spoke_len
    svg += panel(panel_w / 2, cy, normal_degree, _STATIC, f"typical node (cap {degree_cap})")
    svg += panel(panel_w * 1.5 + 20, cy, widened_degree, _ADAPTIVE, "densified node")
    svg.append("</svg>")

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(svg), encoding="utf-8")
