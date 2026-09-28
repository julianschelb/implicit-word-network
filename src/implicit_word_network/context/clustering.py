# context/clustering.py
"""Context-aware clustering of parallel entity edges (CIEN).

In a plain implicit network all cooccurrences of two entities are aggregated
into one edge, which conflates relations that arise in different contexts
(Spitz & Gertz, 2018). Contextual implicit entity networks keep parallel
edges apart by embedding the text of every cooccurrence window and clustering
the embeddings with DBSCAN (cosine distance, ``eps = 0.25``, ``min_samples =
1`` in ECCE). This module implements that step on top of
``ImplicitNetwork``, either on demand for
single edges or batched for many edges at once.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from tqdm import tqdm

from implicit_word_network.context.embedders import BagOfWordsEmbedder, BaseContextEmbedder
from implicit_word_network.network._types import Cooccurrence
from implicit_word_network.network.graph import EntityLike, ImplicitNetwork

Metric = Literal["cosine", "euclidean"]

# =============================================================================
# Distances & DBSCAN
# =============================================================================


def cosine_distances(vectors: NDArray[np.floating]) -> NDArray[np.float64]:
    """Pairwise cosine distances ``1 - cos(x, y)`` (zero vectors get distance 1)."""
    x = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    unit = np.divide(x, norms, out=np.zeros_like(x), where=norms > 0)
    distances = 1.0 - unit @ unit.T
    np.fill_diagonal(distances, 0.0)
    return np.asarray(np.clip(distances, 0.0, 2.0), dtype=np.float64)


def euclidean_distances(vectors: NDArray[np.floating]) -> NDArray[np.float64]:
    """Pairwise Euclidean distances."""
    x = np.asarray(vectors, dtype=np.float64)
    squared = np.sum(x * x, axis=1)
    distances = squared[:, None] + squared[None, :] - 2.0 * (x @ x.T)
    np.fill_diagonal(distances, 0.0)
    return np.asarray(np.sqrt(np.clip(distances, 0.0, None)), dtype=np.float64)


def dbscan(distances: NDArray[np.floating], eps: float, min_samples: int = 1) -> NDArray[np.int64]:
    """DBSCAN on a precomputed distance matrix.

    A point is a *core* point when at least ``min_samples`` points (itself
    included) lie within ``eps``. Clusters are the connected components of
    core points; non-core points within ``eps`` of a core point join that
    core point's cluster (border points); all others are noise (``-1``).
    Cluster labels are numbered ``0, 1, ...`` in order of first appearance.

    Args:
        distances: Square, symmetric distance matrix.
        eps: Neighbourhood radius.
        min_samples: Minimum neighbourhood size of a core point.

    Returns:
        Integer cluster label per point (``-1`` = noise).
    """
    matrix = np.asarray(distances, dtype=np.float64)
    n = matrix.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.int64)
    if matrix.shape != (n, n):
        raise ValueError("distances must be a square matrix")
    if eps < 0:
        raise ValueError("eps must be >= 0")
    if min_samples < 1:
        raise ValueError("min_samples must be >= 1")

    neighbors = matrix <= eps
    np.fill_diagonal(neighbors, True)
    core = neighbors.sum(axis=1) >= min_samples
    labels = np.full(n, -1, dtype=np.int64)
    if not core.any():
        return labels

    core_adjacency = neighbors & core[:, None] & core[None, :]
    _, components = connected_components(
        sparse.csr_matrix(core_adjacency.astype(np.int8)), directed=False
    )
    labels[core] = components[core]

    border = ~core & (neighbors & core[None, :]).any(axis=1)
    if border.any():
        first_core = np.argmax(neighbors[border] & core[None, :], axis=1)
        labels[border] = labels[first_core]

    assigned = labels >= 0
    values = labels[assigned]
    unique, first_index = np.unique(values, return_index=True)
    rank = np.empty(unique.shape[0], dtype=np.int64)
    rank[np.argsort(first_index, kind="stable")] = np.arange(unique.shape[0])
    labels[assigned] = rank[np.searchsorted(unique, values)]
    return labels


# =============================================================================
# Edge clustering
# =============================================================================


@dataclass
class EdgeContextCluster:
    """A group of cooccurrences of two entities that share a similar context.

    Attributes:
        label: Cluster label (``-1`` for DBSCAN noise).
        cooccurrences: Member cooccurrences.
        contexts: Context texts, parallel to ``cooccurrences``.
        weight: Sum of the members' decayed weights (the cluster's edge weight).
        centroid: Mean embedding of the members (``None`` if not kept).
    """

    label: int
    cooccurrences: list[Cooccurrence]
    contexts: list[str]
    weight: float
    centroid: NDArray[np.float32] | None = field(default=None, repr=False)

    @property
    def size(self) -> int:
        """Number of cooccurrences in the cluster."""
        return len(self.cooccurrences)


class ContextualEdgeClusterer:
    """Cluster the cooccurrence contexts of entity pairs (CIEN edge splitting).

    Args:
        embedder: Context embedder. Defaults to the dependency-free
            ``BagOfWordsEmbedder``; use
            ``SentenceTransformerEmbedder``
            for neural contexts as in ECCE.
        eps: DBSCAN neighbourhood radius (``0.25`` cosine distance in ECCE).
        min_samples: DBSCAN core-point threshold (``1`` = no noise).
        metric: ``"cosine"`` or ``"euclidean"``.
        batch_size: Number of contexts embedded per model call. Contexts of
            many edges are pooled into batches of this size.
        keep_centroids: Store the mean embedding of every cluster.

    Example:
        ```python
        from implicit_word_network import ContextualEdgeClusterer

        clusterer = ContextualEdgeClusterer(eps=0.4)
        clusters = clusterer.cluster_edge(network, ("Feynman", "PERSON"), ("Caltech", "ORG"))
        for cluster in clusters:
            print(cluster.size, round(cluster.weight, 2), cluster.contexts[0][:60])
        ```
    """

    def __init__(
        self,
        embedder: BaseContextEmbedder | None = None,
        *,
        eps: float = 0.25,
        min_samples: int = 1,
        metric: Metric = "cosine",
        batch_size: int = 256,
        keep_centroids: bool = True,
    ) -> None:
        if metric not in ("cosine", "euclidean"):
            raise ValueError("metric must be 'cosine' or 'euclidean'")
        self.embedder: BaseContextEmbedder = (
            embedder if embedder is not None else BagOfWordsEmbedder()
        )
        self.eps = eps
        self.min_samples = min_samples
        self.metric: Metric = metric
        self.batch_size = batch_size
        self.keep_centroids = keep_centroids

    # ---------- Public API ----------

    def cluster_edge(
        self, network: ImplicitNetwork, a: EntityLike, b: EntityLike
    ) -> list[EdgeContextCluster]:
        """Cluster the cooccurrences of one entity pair on demand."""
        cooccurrences = network.cooccurrences(a, b)
        contexts = [network.context_of(c) for c in cooccurrences]
        embeddings = self._encode(contexts)
        return self._cluster(cooccurrences, contexts, embeddings)

    def cluster_edges(
        self,
        network: ImplicitNetwork,
        pairs: Iterable[tuple[EntityLike, EntityLike]] | None = None,
        *,
        min_weight: float = 0.0,
        top_k: int | None = None,
        show_progress: bool = False,
    ) -> dict[tuple[int, int], list[EdgeContextCluster]]:
        """Cluster many edges, embedding their contexts in pooled batches.

        Args:
            network: Source network (must store sentence texts).
            pairs: Entity pairs to cluster. Defaults to all edges passing
                ``min_weight`` / ``top_k``.
            min_weight: Edge weight threshold used when ``pairs`` is ``None``.
            top_k: Number of heaviest edges used when ``pairs`` is ``None``.
            show_progress: Display a progress bar.

        Returns:
            Mapping ``(entity_id_a, entity_id_b)`` (``a < b``) to clusters.
        """
        if pairs is None:
            pairs = [
                (edge.source.id, edge.target.id)
                for edge in network.edges(min_weight=min_weight, top_k=top_k)
            ]
        results: dict[tuple[int, int], list[EdgeContextCluster]] = {}
        pending: list[tuple[tuple[int, int], list[Cooccurrence], list[str]]] = []
        pending_contexts = 0

        def flush() -> None:
            nonlocal pending, pending_contexts
            if not pending:
                return
            all_contexts = [ctx for _, _, contexts in pending for ctx in contexts]
            embeddings = self._encode(all_contexts)
            offset = 0
            for key, cooccurrences, contexts in pending:
                block = embeddings[offset : offset + len(contexts)]
                offset += len(contexts)
                results[key] = self._cluster(cooccurrences, contexts, block)
            pending = []
            pending_contexts = 0

        for a, b in tqdm(
            list(pairs), disable=not show_progress, desc="Clustering edges", unit="edge"
        ):
            ids = (network._resolve_entity(a), network._resolve_entity(b))
            key = (min(ids), max(ids))
            cooccurrences = network.cooccurrences(*key)
            contexts = [network.context_of(c) for c in cooccurrences]
            pending.append((key, cooccurrences, contexts))
            pending_contexts += len(contexts)
            if pending_contexts >= self.batch_size:
                flush()
        flush()
        return results

    # ---------- Internals ----------

    def _encode(self, contexts: list[str]) -> NDArray[np.float32]:
        if not contexts:
            return np.empty((0, 0), dtype=np.float32)
        return self.embedder.encode(contexts, batch_size=self.batch_size)

    def _cluster(
        self,
        cooccurrences: list[Cooccurrence],
        contexts: list[str],
        embeddings: NDArray[np.float32],
    ) -> list[EdgeContextCluster]:
        n = len(cooccurrences)
        if n == 0:
            return []
        if n == 1:
            labels = np.zeros(1, dtype=np.int64)
        else:
            distances = (
                cosine_distances(embeddings)
                if self.metric == "cosine"
                else euclidean_distances(embeddings)
            )
            labels = dbscan(distances, self.eps, self.min_samples)

        order: list[int] = []
        members: dict[int, list[int]] = {}
        for index, label in enumerate(labels.tolist()):
            if label not in members:
                members[label] = []
                order.append(label)
            members[label].append(index)

        clusters: list[EdgeContextCluster] = []
        for label in order:
            rows = members[label]
            centroid = None
            if self.keep_centroids and embeddings.size:
                centroid = np.asarray(embeddings[rows].mean(axis=0), dtype=np.float32)
            clusters.append(
                EdgeContextCluster(
                    label=label,
                    cooccurrences=[cooccurrences[i] for i in rows],
                    contexts=[contexts[i] for i in rows],
                    weight=float(sum(cooccurrences[i].weight for i in rows)),
                    centroid=centroid,
                )
            )
        return clusters
