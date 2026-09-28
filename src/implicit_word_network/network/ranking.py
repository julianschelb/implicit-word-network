# network/ranking.py
"""LOAD edge weighting and entity-centric ranking queries.

Implements the directed importance weights and the query model of the LOAD
graph (Spitz & Gertz, 2016; Spitz, Almasian & Gertz, 2017):

- ``load_weight_matrix``: ω(x, y) = log(|Y| / |N(x) ∩ Y|) · Σ_i exp(−δ_i(x, y)),
  where Y is the set of entities of y's type and N(x) the neighbourhood of x.
- ``rank_entities``: single-entity queries rank by ω(q, x) / ω_max; multi-entity
  queries use r = c + s with cohesion c(x) = |N(x) ∩ Q| − 1 and the normalised
  weight sum s(x) = Σ_q ω(q, x) / s_max.
- ``rank_sentences``: r = c + s with c(x) the number of query entities in the
  sentence and s(x) = |N(x) ∩ T_Q| / |T_Q| for the union T_Q of the k most
  important terms of the query entities.
- ``rank_documents``: c(p) = max c(x), s(p) = Σ s(x) (normalised) over the
  sentences of a document.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from implicit_word_network.network._ops import incidence_matrix
from implicit_word_network.network._types import DocumentRef, EntityNode, SentenceRef

if TYPE_CHECKING:
    from implicit_word_network.network.graph import EntityLike, ImplicitNetwork

Weighting = Literal["raw", "load"]
"""``"raw"``: undirected ω = Σ exp(−δ); ``"load"``: directed, type-normalised LOAD weight."""


def load_weight_matrix(network: ImplicitNetwork) -> sparse.csr_matrix:
    """Directed LOAD weights ``ω(x, y) = log(|Y| / |N(x) ∩ Y|) · Σ exp(−δ)``.

    Row ``x`` holds the importance of every neighbour ``y`` *for* ``x``; the
    matrix is not symmetric. Pairs where ``x`` is connected to every entity of
    ``y``'s type get weight ``0`` (``log 1``).
    """
    raw = network.entity_entity_matrix.tocoo()
    n = network.n_entities
    if raw.nnz == 0:
        return sparse.csr_matrix((n, n), dtype=np.float64)
    labels = network.entity_label_ids()
    n_labels = int(labels.max()) + 1 if labels.size else 0
    type_size = np.bincount(labels, minlength=n_labels).astype(np.float64)
    rows = raw.row.astype(np.int64)
    cols = raw.col.astype(np.int64)
    # |N(x) ∩ Y| for every (x, Y): count distinct neighbours per type.
    degree = incidence_matrix(rows, labels[cols], (n, n_labels)).toarray().astype(np.float64)
    factor = np.log(type_size[labels[cols]] / degree[rows, labels[cols]])
    data = raw.data.astype(np.float64) * factor
    matrix = sparse.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    matrix.eliminate_zeros()  # log(1) = 0 when x touches every entity of y's type
    return matrix


def _weights(network: ImplicitNetwork, weighting: Weighting) -> sparse.csr_matrix:
    if weighting == "raw":
        return network.entity_entity_matrix
    if weighting == "load":
        return network.load_weight_matrix()
    raise ValueError("weighting must be 'raw' or 'load'")


def _query_ids(network: ImplicitNetwork, query: EntityLike | Sequence[EntityLike]) -> list[int]:
    items: Sequence[EntityLike]
    if isinstance(query, (EntityNode, int)) or (
        isinstance(query, tuple) and len(query) == 2 and isinstance(query[0], str)
    ):
        items = [query]  # type: ignore[list-item]
    else:
        items = list(query)  # type: ignore[arg-type]
    ids = [network._resolve_entity(item) for item in items]
    if not ids:
        raise ValueError("The query must contain at least one entity")
    return list(dict.fromkeys(ids))


def rank_entities(
    network: ImplicitNetwork,
    query: EntityLike | Sequence[EntityLike],
    *,
    label: str | None = None,
    k: int | None = 10,
    weighting: Weighting = "load",
) -> list[tuple[EntityNode, float]]:
    """Rank entities by their relation to one or more query entities (EVELIN).

    Args:
        network: Source network.
        query: One entity or a set of entities.
        label: Restrict results to this entity type.
        k: Number of results (``None`` for all).
        weighting: Edge weights used for the scores.

    Returns:
        ``(EntityNode, score)`` pairs sorted by decreasing score. Single-entity
        queries yield scores in ``[0, 1]``; multi-entity queries yield
        ``c + s ∈ [0, |Q|]``.
    """
    ids = _query_ids(network, query)
    weights = _weights(network, weighting)
    block = weights[ids, :].tocoo()
    n = network.n_entities
    total = np.bincount(block.col, weights=block.data, minlength=n).astype(np.float64)
    connected = np.bincount(block.col, minlength=n).astype(np.int64)
    candidates = connected > 0
    candidates[ids] = False
    if label is not None:
        candidates &= network.entity_label_ids() == network._label_id(label)
    index = np.flatnonzero(candidates)
    if index.size == 0:
        return []
    if len(ids) == 1:
        scores = total[index] / max(float(total[index].max()), np.finfo(float).tiny)
    else:
        s_max = max(float(total[index].max()), np.finfo(float).tiny)
        scores = (connected[index] - 1).astype(np.float64) + total[index] / s_max
    order = np.argsort(-scores, kind="stable")
    if k is not None:
        order = order[: max(k, 0)]
    counts = network.entity_counts()
    return [(network._entity_node(int(index[i]), counts), float(scores[i])) for i in order]


def _important_terms(
    network: ImplicitNetwork, ids: Sequence[int], n_terms: int
) -> NDArray[np.int64]:
    et = network.entity_term_matrix
    terms: list[int] = []
    for entity_id in ids:
        row = et.getrow(entity_id).tocoo()
        order = np.argsort(-row.data, kind="stable")[:n_terms]
        terms.extend(int(t) for t in row.col[order])
    return np.array(sorted(set(terms)), dtype=np.int64)


def _sentence_scores(
    network: ImplicitNetwork, ids: Sequence[int], n_terms: int
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    n_sentences = network.n_sentences
    entity_block = network.sentence_entity_matrix[:, ids].tocoo()
    cohesion = np.bincount(entity_block.row, minlength=n_sentences).astype(np.int64)
    terms = _important_terms(network, ids, n_terms)
    if terms.size:
        term_block = network.sentence_term_matrix[:, terms].tocoo()
        overlap = np.bincount(term_block.row, minlength=n_sentences).astype(np.float64)
        term_score = overlap / float(terms.size)
    else:
        term_score = np.zeros(network.n_sentences, dtype=np.float64)
    return cohesion, term_score


def rank_sentences(
    network: ImplicitNetwork,
    query: EntityLike | Sequence[EntityLike],
    *,
    k: int | None = 10,
    n_terms: int = 10,
) -> list[tuple[SentenceRef, float]]:
    """Rank sentences that describe the query entities (EVELIN sentence queries).

    Only sentences containing at least one query entity are returned.

    Args:
        network: Source network.
        query: One entity or a set of entities.
        k: Number of results (``None`` for all).
        n_terms: Number of most important terms per query entity used for the
            term-overlap component.
    """
    ids = _query_ids(network, query)
    cohesion, term_score = _sentence_scores(network, ids, n_terms)
    scores = cohesion.astype(np.float64) + term_score
    index = np.flatnonzero(cohesion > 0)
    order = index[np.argsort(-scores[index], kind="stable")]
    if k is not None:
        order = order[: max(k, 0)]
    return [(network.sentence(int(i)), float(scores[i])) for i in order]


def rank_documents(
    network: ImplicitNetwork,
    query: EntityLike | Sequence[EntityLike],
    *,
    k: int | None = 10,
    n_terms: int = 10,
) -> list[tuple[DocumentRef, float]]:
    """Rank documents (LOAD pages) for the query entities.

    ``c(p)`` is the maximum sentence cohesion in the document and ``s(p)`` the
    sum of the sentence term scores, normalised by the maximum over documents.
    Only documents containing at least one query entity are returned.
    """
    ids = _query_ids(network, query)
    cohesion, term_score = _sentence_scores(network, ids, n_terms)
    sent_doc = network.sentence_document()
    n_docs = network.n_documents
    doc_cohesion = np.zeros(n_docs, dtype=np.int64)
    np.maximum.at(doc_cohesion, sent_doc, cohesion)
    doc_terms = np.zeros(n_docs, dtype=np.float64)
    np.add.at(doc_terms, sent_doc, term_score)
    index = np.flatnonzero(doc_cohesion > 0)
    if index.size == 0:
        return []
    s_max = max(float(doc_terms[index].max()), np.finfo(float).tiny)
    scores = doc_cohesion.astype(np.float64) + doc_terms / s_max
    order = index[np.argsort(-scores[index], kind="stable")]
    if k is not None:
        order = order[: max(k, 0)]
    return [(network.document(int(i)), float(scores[i])) for i in order]
