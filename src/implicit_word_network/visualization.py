# visualization.py
"""Plotting helpers (require the ``viz`` extra)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from implicit_word_network._utils import require

if TYPE_CHECKING:
    import networkx as nx

    from implicit_word_network.network.graph import ImplicitNetwork

CATEGORICAL_PALETTE: tuple[str, ...] = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
)
"""Colour-vision-safe categorical hues, assigned to entity types in fixed order."""

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
EDGE_COLOR = "#8f8e88"


def plot_network(
    network: ImplicitNetwork | nx.Graph,
    *,
    min_weight: float = 0.0,
    top_k: int | None = 100,
    labels: Sequence[str] | None = None,
    label_order: Sequence[str] | None = None,
    min_component_size: int = 1,
    ax: Any | None = None,
    layout: str = "spring",
    seed: int = 42,
    node_size: tuple[float, float] = (90.0, 1100.0),
    edge_width: tuple[float, float] = (0.6, 5.5),
    with_labels: bool = True,
    max_labels: int | None = None,
    font_size: float = 8.5,
    legend: bool = True,
    palette: Sequence[str] = CATEGORICAL_PALETTE,
    title: str | None = None,
    output_path: str | None = None,
    show: bool = False,
) -> Any:
    """Draw the entity layer of a network with matplotlib.

    Node area scales with the number of mentions, edge width and opacity with
    the edge weight, and node colour encodes the entity type (fixed hue per
    type, listed in a legend). Labels are set in text ink with a surface halo
    and nudged apart so they do not overlap. Isolated entities are not drawn.

    Args:
        network: An ``ImplicitNetwork`` or a graph produced by ``to_networkx``
            (node attributes ``text``, ``label``, ``count``; edge ``weight``).
        min_weight: Drop edges lighter than this (networks only).
        top_k: Keep only the heaviest ``top_k`` edges (networks only; ``None``
            keeps all).
        labels: Restrict to these entity types (networks only).
        label_order: Entity types in the order they take palette slots.
            Defaults to types sorted by total mentions (most frequent first).
        min_component_size: Drop connected components with fewer nodes, so
            small satellites do not squeeze the main network into a corner.
        ax: Matplotlib axes to draw on (a new figure is created otherwise).
        layout: ``"spring"``, ``"kamada_kawai"`` or ``"circular"``.
        seed: Random seed of the spring layout.
        node_size: Minimum and maximum node area in points².
        edge_width: Minimum and maximum edge width in points.
        with_labels: Draw entity names.
        max_labels: Label only the ``max_labels`` most mentioned entities.
        font_size: Label font size in points.
        legend: Show the entity-type legend.
        palette: Categorical colours, assigned to types in ``label_order``.
        title: Optional figure title.
        output_path: Save the figure to this path when given.
        show: Call ``plt.show()``.

    Returns:
        The matplotlib axes.
    """
    require("matplotlib", extra="viz", purpose="plot_network")
    import matplotlib.pyplot as plt
    import networkx as nx
    from matplotlib import patheffects
    from matplotlib.lines import Line2D

    from implicit_word_network.network.graph import ImplicitNetwork

    graph = (
        network.to_networkx(
            min_weight=min_weight, top_k=top_k, labels=labels, include_isolated=False
        )
        if isinstance(network, ImplicitNetwork)
        else network.copy()
    )
    graph.remove_nodes_from([n for n, degree in dict(graph.degree()).items() if degree == 0])
    if min_component_size > 1:
        for component in list(nx.connected_components(graph)):
            if len(component) < min_component_size:
                graph.remove_nodes_from(component)

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6.5))
    figure = ax.figure
    figure.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_axis_off()
    if graph.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "no edges to draw", ha="center", va="center", color=TEXT_SECONDARY)
        return ax

    nodes = list(graph.nodes)
    data = [graph.nodes[n] for n in nodes]
    counts = np.array([float(d.get("count", 1)) for d in data])
    types = [str(d.get("label", "")) for d in data]

    # ---- colours: one hue per entity type, fixed order
    if label_order is None:
        totals: dict[str, float] = {}
        for label, count in zip(types, counts):
            totals[label] = totals.get(label, 0.0) + count
        label_order = sorted(totals, key=lambda lbl: (-totals[lbl], lbl))
    colour_of = {
        label: palette[i % len(palette)] for i, label in enumerate(label_order) if label in types
    }
    node_colours = [colour_of.get(t, TEXT_SECONDARY) for t in types]

    # ---- sizes: area proportional to mentions, clipped to a readable range
    lo, hi = node_size
    span = counts.max() - counts.min()
    sizes = (
        lo + (hi - lo) * (counts - counts.min()) / span if span > 0 else np.full_like(counts, lo)
    )

    # ---- layout
    if layout == "kamada_kawai":
        positions = nx.kamada_kawai_layout(graph, weight="weight")
    elif layout == "circular":
        positions = nx.circular_layout(graph)
    else:
        k = 1.6 / np.sqrt(max(graph.number_of_nodes(), 1))
        positions = nx.spring_layout(graph, k=k, iterations=300, seed=seed, weight="weight")

    # ---- edges: recessive grey, width and opacity by weight, heavy on top
    edges = sorted(graph.edges(data=True), key=lambda e: float(e[2].get("weight", 1.0)))
    weights = np.array([float(e[2].get("weight", 1.0)) for e in edges])
    w_lo, w_hi = edge_width
    w_span = weights.max() - weights.min() if len(weights) else 0.0
    widths = (
        w_lo + (w_hi - w_lo) * (weights - weights.min()) / w_span
        if w_span > 0
        else np.full_like(weights, w_lo)
    )
    alphas = (
        0.22 + 0.5 * (widths - w_lo) / (w_hi - w_lo) if w_hi > w_lo else np.full_like(widths, 0.5)
    )
    for (u, v, _), width, alpha in zip(edges, widths, alphas):
        (x0, y0), (x1, y1) = positions[u], positions[v]
        ax.plot(
            [x0, x1],
            [y0, y1],
            color=EDGE_COLOR,
            linewidth=width,
            alpha=float(alpha),
            zorder=1,
            solid_capstyle="round",
        )

    # ---- nodes with a surface ring
    xs = np.array([positions[n][0] for n in nodes])
    ys = np.array([positions[n][1] for n in nodes])
    ax.scatter(xs, ys, s=sizes, c=node_colours, edgecolors=SURFACE, linewidths=1.4, zorder=3)
    ax.margins(0.12)

    # ---- labels: text ink with a halo, nudged apart
    if with_labels:
        order = np.argsort(-counts, kind="stable")
        if max_labels is not None:
            order = order[: max(max_labels, 0)]
        halo = [patheffects.withStroke(linewidth=3.0, foreground=SURFACE)]
        radii_pt = np.sqrt(sizes / np.pi)  # scatter area (pt²) -> radius (pt)
        offsets = _spread_labels(
            ax,
            xs[order],
            ys[order],
            [str(data[i].get("text", nodes[i])) for i in order],
            radii_pt[order],
            font_size,
        )
        for i, (dx, dy) in zip(order, offsets):
            ax.annotate(
                str(data[i].get("text", nodes[i])),
                (xs[i], ys[i]),
                xytext=(dx, dy),
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=font_size,
                color=TEXT_PRIMARY,
                path_effects=halo,
                zorder=4,
            )

    # ---- legend and title
    if legend and len(colour_of) > 1:
        handles = [
            Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                markersize=8,
                markerfacecolor=colour_of[lbl],
                markeredgecolor=SURFACE,
                label=lbl,
            )
            for lbl in label_order
            if lbl in colour_of
        ]
        ax.legend(
            handles=handles,
            loc="lower left",
            frameon=False,
            fontsize=font_size,
            labelcolor=TEXT_SECONDARY,
            handletextpad=0.4,
            borderaxespad=0.2,
        )
    if title:
        ax.set_title(title, loc="left", fontsize=font_size + 2.5, color=TEXT_PRIMARY, pad=10)

    if output_path:
        figure.savefig(output_path, bbox_inches="tight", dpi=160, facecolor=SURFACE)
    if show:
        plt.show()
    return ax


def _spread_labels(
    ax: Any,
    xs: np.ndarray,
    ys: np.ndarray,
    texts: Sequence[str],
    radii_pt: np.ndarray,
    font_size: float,
    *,
    iterations: int = 200,
) -> list[tuple[float, float]]:
    """Compute label offsets (in points) that avoid label/label and label/node overlaps.

    Labels start centred above their node. Overlapping boxes are separated
    along the axis of minimum translation, labels are pushed out of every
    node disc, and a weak pull keeps each label near its own node. A greedy,
    deterministic approximation that is good enough for a few hundred labels.
    """
    n = len(texts)
    if n == 0:
        return []
    figure = ax.figure
    ax.autoscale_view()
    pt = float(figure.dpi) / 72.0
    px = ax.transData.transform(np.column_stack([xs, ys]))
    widths = np.array([max(len(t), 1) * font_size * 0.58 * pt + 4 for t in texts])
    heights = np.full(n, font_size * 1.3 * pt)
    radii = radii_pt * pt
    anchors = px.copy()
    anchors[:, 1] += radii + 2 * pt + heights / 2
    centres = anchors.copy()

    for _ in range(iterations):
        moved = False
        left, right = centres[:, 0] - widths / 2, centres[:, 0] + widths / 2
        bottom, top = centres[:, 1] - heights / 2, centres[:, 1] + heights / 2
        for i in range(n):
            for j in range(i + 1, n):
                ox = min(right[i], right[j]) - max(left[i], left[j])
                oy = min(top[i], top[j]) - max(bottom[i], bottom[j])
                if ox <= 0 or oy <= 0:
                    continue
                moved = True
                if ox < oy:
                    side = 1.0 if centres[i, 0] >= centres[j, 0] else -1.0
                    shift = (ox / 2 + 1.0) * side
                    centres[i, 0] += shift
                    centres[j, 0] -= shift
                else:
                    side = 1.0 if centres[i, 1] >= centres[j, 1] else -1.0
                    shift = (oy / 2 + 1.0) * side
                    centres[i, 1] += shift
                    centres[j, 1] -= shift
                left, right = centres[:, 0] - widths / 2, centres[:, 0] + widths / 2
                bottom, top = centres[:, 1] - heights / 2, centres[:, 1] + heights / 2
        # push labels out of node discs (their own and others')
        for i in range(n):
            dx = np.clip(px[:, 0], left[i], right[i]) - px[:, 0]
            dy = np.clip(px[:, 1], bottom[i], top[i]) - px[:, 1]
            dist = np.hypot(dx, dy)
            hits = np.flatnonzero(dist < radii + 1.5 * pt)
            for hit in hits.tolist():
                vec = centres[i] - px[hit]
                norm = float(np.hypot(vec[0], vec[1])) or 1.0
                push = float(radii[hit] + 1.5 * pt - dist[hit]) + 1.0
                centres[i] += vec / norm * push
                moved = True
        # weak pull back towards the anchor above the node
        centres += 0.06 * (anchors - centres)
        if not moved:
            break
    offsets = (centres - px) / pt
    return [(float(dx), float(dy)) for dx, dy in offsets]
