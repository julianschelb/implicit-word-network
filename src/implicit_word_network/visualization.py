# visualization.py
"""Plotting helpers (require the ``viz`` extra)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from implicit_word_network._utils import require

if TYPE_CHECKING:
    import networkx as nx

    from implicit_word_network.network.graph import ImplicitNetwork


def plot_network(
    network: ImplicitNetwork | nx.Graph,
    *,
    min_weight: float = 0.0,
    top_k: int | None = 100,
    ax: Any | None = None,
    layout: str = "spring",
    seed: int = 42,
    node_scale: float = 120.0,
    edge_scale: float = 2.0,
    with_labels: bool = True,
    font_size: int = 8,
    output_path: str | None = None,
    show: bool = False,
) -> Any:
    """Draw the entity layer of a network with matplotlib.

    Node sizes scale with mention counts, edge widths with edge weights and
    node colours encode entity types.

    Args:
        network: An ``ImplicitNetwork``
            or a graph produced by ``to_networkx``.
        min_weight: Drop edges lighter than this (networks only).
        top_k: Keep only the heaviest ``top_k`` edges (networks only).
        ax: Matplotlib axes to draw on (a new figure is created otherwise).
        layout: ``"spring"``, ``"kamada_kawai"`` or ``"circular"``.
        seed: Random seed of the spring layout.
        node_scale: Node size per mention.
        edge_scale: Edge width per unit of weight.
        with_labels: Draw entity names.
        font_size: Label font size.
        output_path: Save the figure to this path when given.
        show: Call ``plt.show()``.

    Returns:
        The matplotlib axes.
    """
    require("matplotlib", extra="viz", purpose="plot_network")
    import matplotlib.pyplot as plt
    import networkx as nx

    from implicit_word_network.network.graph import ImplicitNetwork

    graph = (
        network.to_networkx(min_weight=min_weight, top_k=top_k, include_isolated=False)
        if isinstance(network, ImplicitNetwork)
        else network
    )
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))

    if layout == "kamada_kawai":
        positions = nx.kamada_kawai_layout(graph)
    elif layout == "circular":
        positions = nx.circular_layout(graph)
    else:
        positions = nx.spring_layout(graph, seed=seed, weight="weight")

    labels = sorted({data.get("label", "") for _, data in graph.nodes(data=True)})
    palette = plt.get_cmap("tab10")
    colour_of = {label: palette(i % 10) for i, label in enumerate(labels)}
    node_colours = [colour_of[data.get("label", "")] for _, data in graph.nodes(data=True)]
    node_sizes = [
        node_scale * max(float(data.get("count", 1)), 1.0) for _, data in graph.nodes(data=True)
    ]
    widths = [edge_scale * float(data.get("weight", 1.0)) for _, _, data in graph.edges(data=True)]

    nx.draw_networkx_nodes(
        graph, positions, ax=ax, node_color=node_colours, node_size=node_sizes, alpha=0.85
    )
    nx.draw_networkx_edges(graph, positions, ax=ax, width=widths, alpha=0.4)
    if with_labels:
        nx.draw_networkx_labels(
            graph,
            positions,
            ax=ax,
            labels={n: d.get("text", n) for n, d in graph.nodes(data=True)},
            font_size=font_size,
        )
    ax.set_axis_off()
    if output_path:
        ax.figure.savefig(output_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    return ax
