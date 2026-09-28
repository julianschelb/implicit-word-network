# network/export.py
"""Conversion of implicit networks to NetworkX graphs and plain data structures."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from implicit_word_network.network._types import EntityNode, TermNode

if TYPE_CHECKING:
    import networkx as nx

    from implicit_word_network.network.graph import ImplicitNetwork


def entity_node_id(node: EntityNode) -> str:
    """Stable string identifier of an entity node (``"<norm>|<label>"``)."""
    return f"{node.norm}|{node.label}"


def term_node_id(node: TermNode) -> str:
    """Stable string identifier of a term node (``"<text>|<pos>|term"``)."""
    return f"{node.text}|{node.pos}|term"


def to_networkx(
    network: ImplicitNetwork,
    *,
    min_weight: float = 0.0,
    top_k: int | None = None,
    labels: Sequence[str] | None = None,
    include_isolated: bool = True,
    include_terms: bool = False,
    max_terms_per_entity: int | None = 10,
    min_term_count: int = 1,
) -> nx.Graph:
    """Convert the entity layer of a network into a ``networkx.Graph``.

    Entity nodes carry ``kind="entity"``, ``text``, ``label`` and ``count``
    attributes; edges carry ``weight`` and ``count``. Optionally, term nodes
    (``kind="term"``) are attached to entities with ``weight`` equal to the
    number of shared sentences.

    Args:
        network: Source network.
        min_weight: Drop entity edges lighter than this.
        top_k: Keep only the heaviest ``top_k`` entity edges.
        labels: Restrict to these entity types.
        include_isolated: Keep entities without edges.
        include_terms: Add term nodes and entity–term edges.
        max_terms_per_entity: Number of strongest terms per entity to add.
        min_term_count: Minimum shared-sentence count of an entity–term edge.

    Returns:
        An undirected graph.
    """
    import networkx as nx

    graph = nx.Graph(window=network.config.window, decay=network.config.decay)
    allowed = None if labels is None else set(labels)
    entities = [e for e in network.entities() if allowed is None or e.label in allowed]
    for node in entities:
        graph.add_node(
            entity_node_id(node), kind="entity", text=node.text, label=node.label, count=node.count
        )
    for edge in network.edges(min_weight=min_weight, top_k=top_k, labels=labels):
        graph.add_edge(
            entity_node_id(edge.source),
            entity_node_id(edge.target),
            weight=edge.weight,
            count=edge.count,
            kind="entity-entity",
        )
    if include_terms:
        for node in entities:
            for term, count in network.entity_terms(node, k=max_terms_per_entity):
                if count < min_term_count:
                    continue
                term_id = term_node_id(term)
                if term_id not in graph:
                    graph.add_node(
                        term_id, kind="term", text=term.text, pos=term.pos, count=term.count
                    )
                graph.add_edge(
                    entity_node_id(node),
                    term_id,
                    weight=float(count),
                    count=count,
                    kind="entity-term",
                )
    if not include_isolated:
        graph.remove_nodes_from([n for n, degree in dict(graph.degree()).items() if degree == 0])
    return graph


def to_dict(
    network: ImplicitNetwork,
    *,
    min_weight: float = 0.0,
    top_k: int | None = None,
    labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    """JSON-serialisable representation of the entity layer.

    Returns:
        A dictionary with ``config``, ``nodes`` (id, text, norm, label, count)
        and ``edges`` (source, target, weight, count) entries.
    """
    allowed = None if labels is None else set(labels)
    nodes = [
        {
            "id": entity_node_id(e),
            "text": e.text,
            "norm": e.norm,
            "label": e.label,
            "count": e.count,
        }
        for e in network.entities()
        if allowed is None or e.label in allowed
    ]
    edges = [
        {
            "source": entity_node_id(edge.source),
            "target": entity_node_id(edge.target),
            "weight": edge.weight,
            "count": edge.count,
        }
        for edge in network.edges(min_weight=min_weight, top_k=top_k, labels=labels)
    ]
    return {
        "config": network.config.to_dict(),
        "n_documents": network.n_documents,
        "n_sentences": network.n_sentences,
        "nodes": nodes,
        "edges": edges,
    }


def to_json(network: ImplicitNetwork, path: str | Path, **kwargs: Any) -> Path:
    """Write ``to_dict`` output to a JSON file."""
    path = Path(path)
    path.write_text(
        json.dumps(to_dict(network, **kwargs), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return path


def to_edgelist(
    network: ImplicitNetwork,
    *,
    min_weight: float = 0.0,
    top_k: int | None = None,
    labels: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Flat edge records (one dict per entity–entity edge), e.g. for CSV export."""
    return [
        {
            "source": edge.source.text,
            "source_label": edge.source.label,
            "target": edge.target.text,
            "target_label": edge.target.label,
            "weight": edge.weight,
            "count": edge.count,
        }
        for edge in network.edges(min_weight=min_weight, top_k=top_k, labels=labels)
    ]


def to_pandas(network: ImplicitNetwork, **kwargs: Any) -> tuple[Any, Any]:
    """Return ``(nodes, edges)`` DataFrames (requires the ``pandas`` extra)."""
    from implicit_word_network._utils import require

    require("pandas", extra="pandas", purpose="to_pandas")
    import pandas as pd

    data = to_dict(network, **kwargs)
    return pd.DataFrame(data["nodes"]), pd.DataFrame(data["edges"])
