# network/graph.py
"""The ``ImplicitNetwork`` container.

An implicit entity network (Spitz & Gertz, 2016; Spitz, 2019) is a
heterogeneous graph over a document collection with four node classes:

- **documents** and **sentences** (containment structure),
- **entities**, unique by normalised name and type, linked to the
  sentences they are mentioned in,
- **terms** (non-entity words), linked to the sentences they occur in and to
  the entities they share a sentence with.

Two entities are connected when they cooccur inside a context window of
``c`` sentences. Parallel edges are aggregated into a single weight

    ω(v, w) = Σ_{i ∈ I_{v,w}} exp(-δ_i(v, w)),

the sum of exponentially decayed sentence distances over all cooccurrences.

Internally the network is stored as append-only NumPy columns plus lazily
compacted sparse matrices. Entity–entity weights are computed with a sparse
triple product ``Ω = Sᵀ K S`` (``S``: sentence × entity mention counts,
``K``: banded sentence-distance kernel) per added batch, which vectorises the
cooccurrence aggregation and keeps updates additive.
"""

from __future__ import annotations

import gc
import json
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from tqdm import tqdm

from implicit_word_network.annotation import AnnotatedDocument
from implicit_word_network.document import ID
from implicit_word_network.network._ops import (
    DecayFunction,
    all_window_pairs,
    band_kernel,
    band_structure,
    incidence_matrix,
    resolve_decay,
    window_pairs,
)
from implicit_word_network.network._storage import GrowableArray, SparseAccumulator
from implicit_word_network.network._types import (
    Cooccurrence,
    CooccurrenceTable,
    DocumentRef,
    EntityEdge,
    EntityNode,
    Mention,
    NetworkConfig,
    SentenceRef,
    TermNode,
)

if TYPE_CHECKING:
    import networkx as nx

    from implicit_word_network.network.ranking import Weighting

EntityLike = EntityNode | int | tuple[str, str]
"""Ways to refer to an entity: node object, integer id or ``(text, label)``."""

TermLike = TermNode | int | tuple[str, str]
"""Ways to refer to a term: node object, integer id or ``(text, pos)``."""

_SAVE_FORMAT_VERSION = 2


def default_entity_normalizer(text: str) -> str:
    """Default entity name normalisation: collapse whitespace and lowercase."""
    return " ".join(text.split()).lower()


@contextmanager
def _bulk_allocation() -> Iterator[None]:
    """Suspend the cyclic garbage collector while building many small, acyclic objects.

    Allocating hundreds of thousands of result objects otherwise triggers
    generation-2 collections that scan every live object (for example the
    tokens of a large annotated corpus), which dominates query latency.
    """
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()


class ImplicitNetwork:
    """Implicit entity network of a document collection.

    Networks are built incrementally from ``AnnotatedDocument`` objects (see
    :meth:`add_documents`) and expose entity, term, sentence and document
    nodes together with aggregated entity–entity edges, instance-level
    cooccurrences, LOAD importance weights and ranking queries.

    Args:
        config: Construction parameters (window, decay, term filters).
        normalize_entity: Function mapping a mention surface form to the
            normalised name used for entity identity. Defaults to lowercasing
            and whitespace collapsing. Custom functions are not persisted by
            :meth:`save`.

    Example:
        ```python
        from implicit_word_network import GazetteerEntityExtractor, ImplicitNetwork

        extractor = GazetteerEntityExtractor({"PERSON": ["Feynman", "Schwinger"]})
        docs = extractor.annotate_all(["Feynman shared the prize with Schwinger."])

        network = ImplicitNetwork.from_documents(docs)
        for edge in network.edges():
            print(edge.source.text, edge.target.text, edge.weight)
        ```
    """

    def __init__(
        self,
        config: NetworkConfig | None = None,
        *,
        normalize_entity: Callable[[str], str] | None = None,
    ) -> None:
        self.config: NetworkConfig = config if config is not None else NetworkConfig()
        self._normalize: Callable[[str], str] = normalize_entity or default_entity_normalizer
        self._decay: DecayFunction = resolve_decay(self.config.decay, window=self.config.window)

        # Documents
        self._doc_ids: list[ID] = []
        self._doc_meta: list[dict[str, Any]] = []
        self._doc_index: dict[ID, int] = {}
        self._doc_sent_offsets = GrowableArray(np.int64)
        self._doc_sent_offsets.extend(np.zeros(1, dtype=np.int64))

        # Sentences
        self._sent_doc = GrowableArray(np.int32)
        self._sent_local = GrowableArray(np.int32)
        self._sent_start = GrowableArray(np.int32)
        self._sent_end = GrowableArray(np.int32)
        self._sent_text: list[str] = []

        # Vocabularies
        self._entity_index: dict[tuple[str, str], int] = {}
        self._entity_text: list[str] = []
        self._entity_norm: list[str] = []
        self._entity_label: list[str] = []
        self._label_index: dict[str, int] = {}
        self._entity_label_id = GrowableArray(np.int32)
        self._term_index: dict[tuple[str, str], int] = {}
        self._term_text: list[str] = []
        self._term_pos: list[str] = []

        # Mention table (sorted by global sentence id, then offset)
        self._m_doc = GrowableArray(np.int32)
        self._m_sent = GrowableArray(np.int64)
        self._m_entity = GrowableArray(np.int64)
        self._m_start = GrowableArray(np.int32)
        self._m_end = GrowableArray(np.int32)
        self._m_score = GrowableArray(np.float32)

        # Term occurrence table (sorted by global sentence id, then offset)
        self._t_sent = GrowableArray(np.int64)
        self._t_term = GrowableArray(np.int64)
        self._t_start = GrowableArray(np.int32)
        self._t_end = GrowableArray(np.int32)

        # Sparse matrices (lazily compacted)
        self._S_e = SparseAccumulator(np.int64)  # sentence × entity
        self._S_t = SparseAccumulator(np.int64)  # sentence × term
        self._W = SparseAccumulator(np.float64)  # entity × entity weights
        self._C = SparseAccumulator(np.int64)  # entity × entity counts
        self._ET = SparseAccumulator(np.int64)  # entity × term

        self._cache: dict[str, Any] = {}

    # =========================================================================
    # Construction
    # =========================================================================

    @classmethod
    def from_documents(
        cls,
        documents: Iterable[AnnotatedDocument],
        config: NetworkConfig | None = None,
        *,
        normalize_entity: Callable[[str], str] | None = None,
        show_progress: bool = False,
    ) -> ImplicitNetwork:
        """Build a network from annotated documents.

        Args:
            documents: Annotated documents.
            config: Construction parameters.
            normalize_entity: Entity name normalisation function.
            show_progress: Display a progress bar.
        """
        network = cls(config, normalize_entity=normalize_entity)
        network.add_documents(documents, show_progress=show_progress)
        return network

    def add_documents(
        self,
        documents: Iterable[AnnotatedDocument],
        *,
        show_progress: bool = False,
    ) -> ImplicitNetwork:
        """Add documents to the network, updating all nodes, tables and edges.

        Cooccurrences never cross document boundaries, so adding documents is
        purely additive: the contribution of the new batch is computed with
        vectorised sparse operations and accumulated into the existing
        matrices. Appends are amortised O(1) per row, so networks can grow by
        many small batches.

        Args:
            documents: Annotated documents that are not yet part of the network.
            show_progress: Display a progress bar.

        Returns:
            ``self`` for chaining.

        Raises:
            ValueError: If a document id is already present or annotations are
                inconsistent.
        """
        batch = list(documents)
        if not batch:
            return self

        n_docs0 = len(self._doc_ids)
        n_sent0 = self.n_sentences

        config = self.config
        include_punct = config.include_punctuation
        include_stop = config.include_stopwords
        term_pos_filter = config.term_pos
        use_lemma = config.use_lemma
        lowercase = config.lowercase_terms
        store_text = config.store_text
        normalize = self._normalize
        entity_index = self._entity_index
        entity_text = self._entity_text
        entity_norm = self._entity_norm
        entity_label = self._entity_label
        term_index = self._term_index
        term_text = self._term_text
        term_pos = self._term_pos

        sent_doc: list[int] = []
        sent_local: list[int] = []
        sent_start: list[int] = []
        sent_end: list[int] = []
        sent_text: list[str] = []
        m_doc: list[int] = []
        m_sent: list[int] = []
        m_entity: list[int] = []
        m_start: list[int] = []
        m_end: list[int] = []
        m_score: list[float] = []
        t_sent: list[int] = []
        t_term: list[int] = []
        t_start: list[int] = []
        t_end: list[int] = []
        new_ids: list[ID] = []
        new_meta: list[dict[str, Any]] = []
        new_offsets: list[int] = []
        seen: set[ID] = set()

        iterator = tqdm(batch, disable=not show_progress, desc="Building network", unit="doc")
        for k, document in enumerate(iterator):
            if document.id in self._doc_index or document.id in seen:
                raise ValueError(f"Document {document.id!r} is already part of the network")
            seen.add(document.id)
            doc_index = n_docs0 + k
            sent_base = n_sent0 + len(sent_doc)
            n_local = len(document.sentences)
            new_offsets.append(sent_base)

            text = document.text
            for sentence in document.sentences:
                sent_doc.append(doc_index)
                sent_local.append(sentence.index)
                sent_start.append(sentence.start)
                sent_end.append(sentence.end)
                if store_text:
                    sent_text.append(text[sentence.start : sentence.end])

            for mention in document.mentions:
                if not 0 <= mention.sentence < n_local:
                    raise ValueError(
                        f"Mention {mention.text!r} in document {document.id!r} refers to "
                        f"sentence {mention.sentence}, but the document has {n_local} sentences"
                    )
                key = (normalize(mention.text), mention.label)
                entity_id = entity_index.get(key)
                if entity_id is None:
                    entity_id = len(entity_text)
                    entity_index[key] = entity_id
                    entity_text.append(" ".join(mention.text.split()))
                    entity_norm.append(key[0])
                    entity_label.append(mention.label)
                m_doc.append(doc_index)
                m_sent.append(sent_base + mention.sentence)
                m_entity.append(entity_id)
                m_start.append(mention.start)
                m_end.append(mention.end)
                m_score.append(mention.score)

            tokens = document.tokens
            if not tokens:
                new_ids.append(document.id)
                new_meta.append(dict(document.meta))
                continue
            covered = document.entity_token_mask().tolist() if document.mentions else None
            for position, token in enumerate(tokens):
                if covered is not None and covered[position]:
                    continue
                if token.is_space:
                    continue
                if token.is_punct and not include_punct:
                    continue
                if token.is_stop and not include_stop:
                    continue
                pos = token.pos
                if term_pos_filter is not None and pos not in term_pos_filter:
                    continue
                base = token.lemma if (use_lemma and token.lemma) else token.text
                if lowercase:
                    base = base.lower()
                if not base.isprintable() or " " in base:
                    base = " ".join(base.split())
                if not base:
                    continue
                sentence_index = token.sentence
                if not 0 <= sentence_index < n_local:
                    raise ValueError(
                        f"Token {token.text!r} in document {document.id!r} refers to "
                        f"sentence {sentence_index}, but the document has {n_local} sentences"
                    )
                key = (base, pos)
                term_id = term_index.get(key)
                if term_id is None:
                    term_id = len(term_text)
                    term_index[key] = term_id
                    term_text.append(base)
                    term_pos.append(pos)
                t_sent.append(sent_base + sentence_index)
                t_term.append(term_id)
                t_start.append(token.start)
                t_end.append(token.end)

            new_ids.append(document.id)
            new_meta.append(dict(document.meta))

        # ---- Entity label ids (for type-aware queries)
        label_index = self._label_index
        new_label_ids = [
            label_index.setdefault(label, len(label_index))
            for label in entity_label[len(self._entity_label_id) :]
        ]
        self._entity_label_id.extend(np.asarray(new_label_ids, dtype=np.int32))

        n_ent = self.n_entities
        n_term = self.n_terms
        n_sent_new = len(sent_doc)

        # ---- Tables of the new batch (sorted by sentence, then offset)
        m_sent_arr = np.asarray(m_sent, dtype=np.int64)
        m_start_arr = np.asarray(m_start, dtype=np.int32)
        order = np.lexsort((m_start_arr, m_sent_arr))
        m_sent_arr = m_sent_arr[order]
        m_start_arr = m_start_arr[order]
        m_doc_arr = np.asarray(m_doc, dtype=np.int32)[order]
        m_entity_arr = np.asarray(m_entity, dtype=np.int64)[order]
        m_end_arr = np.asarray(m_end, dtype=np.int32)[order]
        m_score_arr = np.asarray(m_score, dtype=np.float32)[order]

        t_sent_arr = np.asarray(t_sent, dtype=np.int64)
        t_start_arr = np.asarray(t_start, dtype=np.int32)
        order = np.lexsort((t_start_arr, t_sent_arr))
        t_sent_arr = t_sent_arr[order]
        t_start_arr = t_start_arr[order]
        t_term_arr = np.asarray(t_term, dtype=np.int64)[order]
        t_end_arr = np.asarray(t_end, dtype=np.int32)[order]

        sent_doc_arr = np.asarray(sent_doc, dtype=np.int32)

        # ---- Incidence matrices of the new batch
        S_e_new = incidence_matrix(m_sent_arr - n_sent0, m_entity_arr, (n_sent_new, n_ent))
        S_t_new = incidence_matrix(t_sent_arr - n_sent0, t_term_arr, (n_sent_new, n_term))

        # ---- Entity–entity weights via Ω = Sᵀ K S (block of the new batch)
        rows, cols, delta = band_structure(sent_doc_arr, config.window)
        K_w = band_kernel(sent_doc_arr, config.window, self._decay(delta), rows, cols)
        K_c = band_kernel(
            sent_doc_arr, config.window, np.ones(delta.shape[0], dtype=np.int64), rows, cols
        )
        S_e_float = S_e_new.astype(np.float64)
        W_new = (S_e_float.T @ K_w @ S_e_float).tocsr()
        C_new = (S_e_new.T @ K_c @ S_e_new).tocsr()
        for matrix in (W_new, C_new):
            matrix.setdiag(0)
            matrix.eliminate_zeros()
        ET_new = (S_e_new.T @ S_t_new).tocsr()

        # ---- Accumulate
        self._S_e.resize((n_sent0 + n_sent_new, n_ent))
        self._S_e.add(m_sent_arr, m_entity_arr, np.ones(m_sent_arr.shape[0], dtype=np.int64))
        self._S_t.resize((n_sent0 + n_sent_new, n_term))
        self._S_t.add(t_sent_arr, t_term_arr, np.ones(t_sent_arr.shape[0], dtype=np.int64))
        self._W.resize((n_ent, n_ent))
        self._W.add_matrix(W_new)
        self._C.resize((n_ent, n_ent))
        self._C.add_matrix(C_new)
        self._ET.resize((n_ent, n_term))
        self._ET.add_matrix(ET_new)

        self._sent_doc.extend(sent_doc_arr)
        self._sent_local.extend(np.asarray(sent_local, dtype=np.int32))
        self._sent_start.extend(np.asarray(sent_start, dtype=np.int32))
        self._sent_end.extend(np.asarray(sent_end, dtype=np.int32))
        if store_text:
            self._sent_text.extend(sent_text)

        self._m_doc.extend(m_doc_arr)
        self._m_sent.extend(m_sent_arr)
        self._m_entity.extend(m_entity_arr)
        self._m_start.extend(m_start_arr)
        self._m_end.extend(m_end_arr)
        self._m_score.extend(m_score_arr)

        self._t_sent.extend(t_sent_arr)
        self._t_term.extend(t_term_arr)
        self._t_start.extend(t_start_arr)
        self._t_end.extend(t_end_arr)

        for doc_id in new_ids:
            self._doc_index[doc_id] = len(self._doc_ids)
            self._doc_ids.append(doc_id)
        self._doc_meta.extend(new_meta)
        # offsets[-1] already equals n_sent0 == new_offsets[0]; append the remaining starts
        self._doc_sent_offsets.extend(
            np.asarray(new_offsets[1:] + [n_sent0 + n_sent_new], dtype=np.int64)
        )
        self._cache.clear()
        return self

    # =========================================================================
    # Sizes
    # =========================================================================

    @property
    def n_documents(self) -> int:
        """Number of document nodes."""
        return len(self._doc_ids)

    @property
    def n_sentences(self) -> int:
        """Number of sentence nodes."""
        return len(self._sent_doc)

    @property
    def n_entities(self) -> int:
        """Number of entity nodes."""
        return len(self._entity_text)

    @property
    def n_terms(self) -> int:
        """Number of term nodes."""
        return len(self._term_text)

    @property
    def n_mentions(self) -> int:
        """Number of entity mentions."""
        return len(self._m_entity)

    @property
    def n_edges(self) -> int:
        """Number of (undirected) entity–entity edges."""
        return int(self._W.matrix.nnz // 2)

    @property
    def window(self) -> int:
        """Context window in sentences."""
        return self.config.window

    # =========================================================================
    # Matrices
    # =========================================================================

    @property
    def entity_entity_matrix(self) -> sparse.csr_matrix:
        """Symmetric ``n_entities × n_entities`` matrix of aggregated edge weights."""
        return self._W.matrix

    @property
    def entity_count_matrix(self) -> sparse.csr_matrix:
        """Symmetric ``n_entities × n_entities`` matrix of cooccurrence counts."""
        return self._C.matrix

    @property
    def entity_term_matrix(self) -> sparse.csr_matrix:
        """``n_entities × n_terms`` matrix of same-sentence cooccurrence counts."""
        return self._ET.matrix

    @property
    def sentence_entity_matrix(self) -> sparse.csr_matrix:
        """``n_sentences × n_entities`` matrix of mention counts."""
        return self._S_e.matrix

    @property
    def sentence_term_matrix(self) -> sparse.csr_matrix:
        """``n_sentences × n_terms`` matrix of term occurrence counts."""
        return self._S_t.matrix

    def sentence_document(self) -> NDArray[np.int32]:
        """Document index of every sentence (the document–sentence edges)."""
        return self._sent_doc.view

    def entity_counts(self) -> NDArray[np.int64]:
        """Number of mentions per entity."""
        counts = self._cache.get("entity_counts")
        if counts is None:
            counts = np.bincount(self._m_entity.view, minlength=self.n_entities).astype(np.int64)
            self._cache["entity_counts"] = counts
        return counts

    def term_counts(self) -> NDArray[np.int64]:
        """Number of occurrences per term."""
        counts = self._cache.get("term_counts")
        if counts is None:
            counts = np.bincount(self._t_term.view, minlength=self.n_terms).astype(np.int64)
            self._cache["term_counts"] = counts
        return counts

    def entity_label_ids(self) -> NDArray[np.int32]:
        """Integer type id per entity (see :meth:`entity_labels`)."""
        return self._entity_label_id.view

    def _label_id(self, label: str) -> int:
        return self._label_index.get(label, -1)

    def load_weight_matrix(self) -> sparse.csr_matrix:
        """Directed LOAD importance weights (see ``load_weight_matrix``)."""
        matrix = self._cache.get("load")
        if matrix is None:
            from implicit_word_network.network.ranking import load_weight_matrix

            matrix = load_weight_matrix(self)
            self._cache["load"] = matrix
        return matrix

    # ---------- Entity → mention index ----------

    def _mention_index(self) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        index = self._cache.get("mention_index")
        if index is None:
            entities = self._m_entity.view
            order = np.argsort(entities, kind="stable")
            bounds = np.searchsorted(entities[order], np.arange(self.n_entities + 1))
            index = (order, bounds)
            self._cache["mention_index"] = index
        return index

    def _mention_rows(self, entity_id: int) -> NDArray[np.int64]:
        order, bounds = self._mention_index()
        return order[bounds[entity_id] : bounds[entity_id + 1]]

    # =========================================================================
    # Nodes
    # =========================================================================

    def _resolve_entity(self, entity: EntityLike) -> int:
        if isinstance(entity, EntityNode):
            entity_id = entity.id
        elif isinstance(entity, tuple):
            text, label = entity
            key = (self._normalize(text), label)
            if key not in self._entity_index:
                raise KeyError(f"Unknown entity {entity!r}")
            return self._entity_index[key]
        else:
            entity_id = int(entity)
        if not 0 <= entity_id < self.n_entities:
            raise KeyError(f"Unknown entity id {entity_id}")
        return entity_id

    def _resolve_term(self, term: TermLike) -> int:
        if isinstance(term, TermNode):
            term_id = term.id
        elif isinstance(term, tuple):
            key = (term[0].lower() if self.config.lowercase_terms else term[0], term[1])
            if key not in self._term_index:
                raise KeyError(f"Unknown term {term!r}")
            return self._term_index[key]
        else:
            term_id = int(term)
        if not 0 <= term_id < self.n_terms:
            raise KeyError(f"Unknown term id {term_id}")
        return term_id

    def _entity_node(self, entity_id: int, counts: NDArray[np.int64] | None = None) -> EntityNode:
        cache: dict[int, EntityNode] = self._cache.setdefault("entity_nodes", {})
        node = cache.get(entity_id)
        if node is None:
            if counts is None:
                counts = self.entity_counts()
            node = EntityNode(
                id=entity_id,
                text=self._entity_text[entity_id],
                norm=self._entity_norm[entity_id],
                label=self._entity_label[entity_id],
                count=int(counts[entity_id]),
            )
            cache[entity_id] = node
        return node

    def _term_node(self, term_id: int, counts: NDArray[np.int64] | None = None) -> TermNode:
        cache: dict[int, TermNode] = self._cache.setdefault("term_nodes", {})
        node = cache.get(term_id)
        if node is None:
            if counts is None:
                counts = self.term_counts()
            node = TermNode(
                id=term_id,
                text=self._term_text[term_id],
                pos=self._term_pos[term_id],
                count=int(counts[term_id]),
            )
            cache[term_id] = node
        return node

    def entity(self, text: str, label: str) -> EntityNode | None:
        """Look up an entity by (surface) name and type; ``None`` if absent."""
        key = (self._normalize(text), label)
        entity_id = self._entity_index.get(key)
        return None if entity_id is None else self._entity_node(entity_id)

    def entity_by_id(self, entity_id: int) -> EntityNode:
        """Return the entity node with integer id ``entity_id``."""
        return self._entity_node(self._resolve_entity(entity_id))

    def entities(self, *, label: str | None = None) -> list[EntityNode]:
        """All entity nodes (optionally restricted to one type), ordered by id."""
        counts = self.entity_counts()
        return [
            self._entity_node(i, counts)
            for i in range(self.n_entities)
            if label is None or self._entity_label[i] == label
        ]

    def iter_entities(self, *, label: str | None = None) -> Iterator[EntityNode]:
        """Lazily iterate over entity nodes ordered by id."""
        counts = self.entity_counts()
        for i in range(self.n_entities):
            if label is None or self._entity_label[i] == label:
                yield self._entity_node(i, counts)

    def entity_labels(self) -> list[str]:
        """Distinct entity types present in the network."""
        return sorted(self._label_index)

    def top_entities(self, k: int = 10, *, label: str | None = None) -> list[EntityNode]:
        """The ``k`` most frequently mentioned entities."""
        counts = self.entity_counts()
        if label is not None:
            label_id = self._label_id(label)
            candidates = np.flatnonzero(self.entity_label_ids() == label_id)
        else:
            candidates = np.arange(self.n_entities)
        if candidates.size == 0 or k <= 0:
            return []
        order = candidates[np.argsort(-counts[candidates], kind="stable")[:k]]
        return [self._entity_node(int(i), counts) for i in order]

    def term(self, text: str, pos: str = "") -> TermNode | None:
        """Look up a term by normalised form and POS tag; ``None`` if absent."""
        try:
            return self._term_node(self._resolve_term((text, pos)))
        except KeyError:
            return None

    def term_by_id(self, term_id: int) -> TermNode:
        """Return the term node with integer id ``term_id``."""
        return self._term_node(self._resolve_term(term_id))

    def terms(self) -> list[TermNode]:
        """All term nodes ordered by id."""
        counts = self.term_counts()
        return [self._term_node(i, counts) for i in range(self.n_terms)]

    def top_terms(self, k: int = 10) -> list[TermNode]:
        """The ``k`` most frequent terms."""
        counts = self.term_counts()
        order = np.argsort(-counts, kind="stable")[: max(k, 0)]
        return [self._term_node(int(i), counts) for i in order]

    def document(self, index: int) -> DocumentRef:
        """Return the document node at position ``index``."""
        if not 0 <= index < self.n_documents:
            raise KeyError(f"Unknown document index {index}")
        offsets = self._doc_sent_offsets.view
        n_sentences = int(offsets[index + 1] - offsets[index])
        return DocumentRef(index, self._doc_ids[index], n_sentences, self._doc_meta[index])

    def document_by_id(self, doc_id: ID) -> DocumentRef:
        """Return the document node with identifier ``doc_id``."""
        if doc_id not in self._doc_index:
            raise KeyError(f"Unknown document id {doc_id!r}")
        return self.document(self._doc_index[doc_id])

    def documents(self) -> list[DocumentRef]:
        """All document nodes in insertion order."""
        return [self.document(i) for i in range(self.n_documents)]

    def sentence(self, sentence_id: int) -> SentenceRef:
        """Return the sentence node with global id ``sentence_id``."""
        cache: dict[int, SentenceRef] = self._cache.setdefault("sentences", {})
        ref = cache.get(sentence_id)
        if ref is None:
            if not 0 <= sentence_id < self.n_sentences:
                raise KeyError(f"Unknown sentence id {sentence_id}")
            ref = SentenceRef(
                id=sentence_id,
                document=self._doc_ids[int(self._sent_doc.view[sentence_id])],
                index=int(self._sent_local.view[sentence_id]),
                text=self._sent_text[sentence_id] if self.config.store_text else "",
            )
            cache[sentence_id] = ref
        return ref

    def sentences_of_document(self, doc: int | ID) -> list[SentenceRef]:
        """Sentence nodes of a document (by position or identifier)."""
        index = (
            doc if isinstance(doc, int) and 0 <= doc < self.n_documents else self._doc_index[doc]
        )
        offsets = self._doc_sent_offsets.view
        return [self.sentence(i) for i in range(int(offsets[index]), int(offsets[index + 1]))]

    # =========================================================================
    # Mentions
    # =========================================================================

    def _mention(self, row: int) -> Mention:
        return self._mentions_at(np.asarray([row], dtype=np.int64))[0]

    def _mentions_at(self, rows: NDArray[np.int64]) -> list[Mention]:
        """Build ``Mention`` objects for mention-table rows (bulk, cached nodes)."""
        if rows.shape[0] == 0:
            return []
        entity_node = self._entity_node
        sentence = self.sentence
        counts = self.entity_counts()
        with _bulk_allocation():
            return [
                Mention(
                    id=row,
                    entity=entity_node(ent, counts),
                    sentence=sentence(sent),
                    start=start,
                    end=end,
                    score=score,
                )
                for row, ent, sent, start, end, score in zip(
                    rows.tolist(),
                    self._m_entity.view[rows].tolist(),
                    self._m_sent.view[rows].tolist(),
                    self._m_start.view[rows].tolist(),
                    self._m_end.view[rows].tolist(),
                    self._m_score.view[rows].tolist(),
                )
            ]

    def mentions_of(self, entity: EntityLike) -> list[Mention]:
        """All mentions of an entity in corpus order."""
        return self._mentions_at(self._mention_rows(self._resolve_entity(entity)))

    def sentences_of(self, entity: EntityLike) -> list[SentenceRef]:
        """Distinct sentences mentioning an entity."""
        rows = self._mention_rows(self._resolve_entity(entity))
        sentence_ids = np.unique(self._m_sent.view[rows])
        return [self.sentence(int(i)) for i in sentence_ids]

    def mention(self, row: int) -> Mention:
        """Return the mention stored at row ``row`` of the mention table."""
        if not 0 <= row < self.n_mentions:
            raise KeyError(f"Unknown mention row {row}")
        return self._mention(row)

    # =========================================================================
    # Entity–entity edges
    # =========================================================================

    def weight(self, a: EntityLike, b: EntityLike) -> float:
        """Aggregated edge weight between two entities (``0.0`` if not connected)."""
        i, j = self._resolve_entity(a), self._resolve_entity(b)
        return float(self._W.matrix[i, j])

    def count(self, a: EntityLike, b: EntityLike) -> int:
        """Number of cooccurrences between two entities."""
        i, j = self._resolve_entity(a), self._resolve_entity(b)
        return int(self._C.matrix[i, j])

    def load_weight(self, x: EntityLike, y: EntityLike) -> float:
        """Directed LOAD importance of ``y`` for ``x`` (see ``load_weight_matrix``)."""
        i, j = self._resolve_entity(x), self._resolve_entity(y)
        return float(self.load_weight_matrix()[i, j])

    def edge_table(
        self,
        *,
        min_weight: float = 0.0,
        top_k: int | None = None,
        labels: Sequence[str] | None = None,
    ) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64], NDArray[np.int64]]:
        """Edges as arrays ``(source_ids, target_ids, weights, counts)`` with ``source < target``.

        Sorted by decreasing weight; ``top_k`` selects the heaviest edges with
        a partial sort. This is the allocation-free counterpart of
        :meth:`edges` for bulk analyses.
        """
        empty = (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.int64),
        )
        if self.n_entities == 0:
            return empty
        upper = sparse.triu(self._W.matrix, k=1).tocoo()
        rows = np.asarray(upper.row, dtype=np.int64)
        cols = np.asarray(upper.col, dtype=np.int64)
        values = np.asarray(upper.data, dtype=np.float64)
        keep = values >= min_weight if min_weight > 0 else np.ones(values.shape[0], dtype=bool)
        if labels is not None:
            allowed = np.zeros(len(self._label_index), dtype=bool)
            for label in labels:
                label_id = self._label_id(label)
                if label_id >= 0:
                    allowed[label_id] = True
            label_ids = self.entity_label_ids()
            keep &= allowed[label_ids[rows]] & allowed[label_ids[cols]]
        rows, cols, values = rows[keep], cols[keep], values[keep]
        if rows.shape[0] == 0:
            return empty
        if top_k is not None and 0 <= top_k < values.shape[0]:
            if top_k == 0:
                return empty
            partial = np.argpartition(-values, top_k - 1)[:top_k]
            rows, cols, values = rows[partial], cols[partial], values[partial]
        order = np.argsort(-values, kind="stable")
        rows, cols, values = rows[order], cols[order], values[order]
        counts = np.asarray(self._C.matrix[rows, cols], dtype=np.int64).ravel()
        return rows, cols, values, counts

    def edges(
        self,
        *,
        min_weight: float = 0.0,
        top_k: int | None = None,
        labels: Sequence[str] | None = None,
    ) -> list[EntityEdge]:
        """Aggregated entity–entity edges sorted by decreasing weight.

        Args:
            min_weight: Drop edges lighter than this.
            top_k: Keep only the heaviest ``top_k`` edges.
            labels: Keep only edges whose endpoints both have one of these
                entity types.
        """
        rows, cols, values, counts = self.edge_table(
            min_weight=min_weight, top_k=top_k, labels=labels
        )
        node_counts = self.entity_counts()
        return [
            EntityEdge(
                source=self._entity_node(int(r), node_counts),
                target=self._entity_node(int(c), node_counts),
                weight=float(w),
                count=int(n),
            )
            for r, c, w, n in zip(rows, cols, values, counts)
        ]

    def neighbors(
        self,
        entity: EntityLike,
        *,
        k: int | None = None,
        min_weight: float = 0.0,
        weighting: Weighting = "raw",
    ) -> list[tuple[EntityNode, float]]:
        """Adjacent entities with edge weights, heaviest first.

        Args:
            entity: The entity whose neighbourhood is returned.
            k: Number of neighbours (``None`` for all).
            min_weight: Drop neighbours below this weight.
            weighting: ``"raw"`` for ω = Σ exp(−δ) or ``"load"`` for the
                directed LOAD importance of the neighbour for ``entity``.
        """
        if weighting not in ("raw", "load"):
            raise ValueError("weighting must be 'raw' or 'load'")
        entity_id = self._resolve_entity(entity)
        matrix = self._W.matrix if weighting == "raw" else self.load_weight_matrix()
        row = matrix.getrow(entity_id).tocoo()
        cols = np.asarray(row.col, dtype=np.int64)
        values = np.asarray(row.data, dtype=np.float64)
        keep = values >= min_weight if min_weight > 0 else values > 0
        cols, values = cols[keep], values[keep]
        order = np.argsort(-values, kind="stable")
        if k is not None:
            order = order[: max(k, 0)]
        counts = self.entity_counts()
        return [(self._entity_node(int(cols[i]), counts), float(values[i])) for i in order]

    def _cooccurrence_rows(
        self, a: EntityLike, b: EntityLike
    ) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:
        i, j = self._resolve_entity(a), self._resolve_entity(b)
        if i == j:
            raise ValueError("Cooccurrences are only defined between two distinct entities")
        rows = np.concatenate([self._mention_rows(i), self._mention_rows(j)])
        empty = np.empty(0, dtype=np.int64)
        if rows.shape[0] == 0:
            return empty, empty, empty, np.empty(0, dtype=np.float64)
        rows.sort()
        m_doc = self._m_doc.view
        m_sent = self._m_sent.view
        m_entity = self._m_entity.view
        src, tgt = all_window_pairs(m_doc[rows], m_sent[rows], self.config.window)
        keep = m_entity[rows[src]] != m_entity[rows[tgt]]
        src, tgt = rows[src[keep]], rows[tgt[keep]]
        delta = (m_sent[tgt] - m_sent[src]).astype(np.int64)
        return src, tgt, delta, self._decay(delta)

    def cooccurrences(self, a: EntityLike, b: EntityLike) -> list[Cooccurrence]:
        """Instance-level cooccurrences of two entities (on-demand enumeration).

        Args:
            a: First entity.
            b: Second entity (must differ from ``a``).

        Returns:
            One ``Cooccurrence`` per pair of mentions inside the context
            window, in corpus order.
        """
        src, tgt, delta, weight = self._cooccurrence_rows(a, b)
        sources = self._mentions_at(src)
        targets = self._mentions_at(tgt)
        with _bulk_allocation():
            return [
                Cooccurrence(s, t, d, w)
                for s, t, d, w in zip(sources, targets, delta.tolist(), weight.tolist())
            ]

    def cooccurrence_table(self, a: EntityLike, b: EntityLike) -> CooccurrenceTable:
        """Cooccurrences of two entities as column arrays (see :meth:`all_cooccurrences`)."""
        src, tgt, delta, weight = self._cooccurrence_rows(a, b)
        return CooccurrenceTable(src, tgt, delta, weight)

    def all_cooccurrences(self) -> CooccurrenceTable:
        """All cooccurrence instances of the network as column arrays.

        The rows reference the mention table (see :meth:`mention`). This is
        the eager counterpart of :meth:`cooccurrences` and is used for exports
        and validation; its size grows quadratically with the number of
        mentions per context window. Use :meth:`iter_cooccurrences` to stream
        the pairs in bounded chunks.
        """
        chunks = list(self.iter_cooccurrences())
        if not chunks:
            empty = np.empty(0, dtype=np.int64)
            return CooccurrenceTable(empty, empty, empty, np.empty(0, dtype=np.float64))
        return CooccurrenceTable(
            np.concatenate([c.source for c in chunks]),
            np.concatenate([c.target for c in chunks]),
            np.concatenate([c.delta for c in chunks]),
            np.concatenate([c.weight for c in chunks]),
        )

    def iter_cooccurrences(self, *, chunk_size: int = 1_000_000) -> Iterator[CooccurrenceTable]:
        """Stream all cooccurrence instances in chunks of at most ``chunk_size`` pairs."""
        m_entity = self._m_entity.view
        m_sent = self._m_sent.view
        for src, tgt in window_pairs(
            self._m_doc.view, m_sent, self.config.window, chunk_size=chunk_size
        ):
            keep = m_entity[src] != m_entity[tgt]
            src, tgt = src[keep], tgt[keep]
            if src.shape[0] == 0:
                continue
            delta = (m_sent[tgt] - m_sent[src]).astype(np.int64)
            yield CooccurrenceTable(src, tgt, delta, self._decay(delta))

    # =========================================================================
    # Contexts
    # =========================================================================

    def context_of(self, cooccurrence: Cooccurrence) -> str:
        """Text of the sentences spanned by a cooccurrence (its context window)."""
        first, last = cooccurrence.sentence_span
        return self.context_of_span(first, last)

    def context_of_span(self, first: int, last: int) -> str:
        """Text of the global sentences ``first``..``last`` (inclusive)."""
        if not self.config.store_text:
            raise RuntimeError(
                "Contexts are unavailable: the network was built with store_text=False"
            )
        return " ".join(self._sent_text[first : last + 1])

    def contexts(self, a: EntityLike, b: EntityLike) -> list[str]:
        """Context texts of all cooccurrences of two entities (parallel to :meth:`cooccurrences`)."""
        src, tgt, _, _ = self._cooccurrence_rows(a, b)
        m_sent = self._m_sent.view
        first = np.minimum(m_sent[src], m_sent[tgt])
        last = np.maximum(m_sent[src], m_sent[tgt])
        return [self.context_of_span(int(lo), int(hi)) for lo, hi in zip(first, last)]

    # =========================================================================
    # Entity–term edges
    # =========================================================================

    def entity_terms(
        self, entity: EntityLike, *, k: int | None = None
    ) -> list[tuple[TermNode, int]]:
        """Terms sharing sentences with an entity, with cooccurrence counts (most frequent first)."""
        entity_id = self._resolve_entity(entity)
        row = self._ET.matrix.getrow(entity_id).tocoo()
        cols = np.asarray(row.col, dtype=np.int64)
        values = np.asarray(row.data, dtype=np.int64)
        order = np.argsort(-values, kind="stable")
        if k is not None:
            order = order[: max(k, 0)]
        counts = self.term_counts()
        return [(self._term_node(int(cols[i]), counts), int(values[i])) for i in order]

    def term_entities(
        self, term: TermLike, *, k: int | None = None
    ) -> list[tuple[EntityNode, int]]:
        """Entities sharing sentences with a term, with cooccurrence counts."""
        term_id = self._resolve_term(term)
        col = self._ET.matrix.getcol(term_id).tocoo()
        rows = np.asarray(col.row, dtype=np.int64)
        values = np.asarray(col.data, dtype=np.int64)
        order = np.argsort(-values, kind="stable")
        if k is not None:
            order = order[: max(k, 0)]
        counts = self.entity_counts()
        return [(self._entity_node(int(rows[i]), counts), int(values[i])) for i in order]

    # =========================================================================
    # Ranking queries (LOAD / EVELIN)
    # =========================================================================

    def rank_entities(
        self,
        query: EntityLike | Sequence[EntityLike],
        *,
        label: str | None = None,
        k: int | None = 10,
        weighting: Weighting = "load",
    ) -> list[tuple[EntityNode, float]]:
        """Rank entities related to one or more query entities (see ``rank_entities``)."""
        from implicit_word_network.network.ranking import rank_entities

        return rank_entities(self, query, label=label, k=k, weighting=weighting)

    def rank_sentences(
        self,
        query: EntityLike | Sequence[EntityLike],
        *,
        k: int | None = 10,
        n_terms: int = 10,
    ) -> list[tuple[SentenceRef, float]]:
        """Rank sentences describing the query entities (see ``rank_sentences``)."""
        from implicit_word_network.network.ranking import rank_sentences

        return rank_sentences(self, query, k=k, n_terms=n_terms)

    def rank_documents(
        self,
        query: EntityLike | Sequence[EntityLike],
        *,
        k: int | None = 10,
        n_terms: int = 10,
    ) -> list[tuple[DocumentRef, float]]:
        """Rank documents for the query entities (see ``rank_documents``)."""
        from implicit_word_network.network.ranking import rank_documents

        return rank_documents(self, query, k=k, n_terms=n_terms)

    # =========================================================================
    # Conversion & persistence
    # =========================================================================

    def to_networkx(self, **kwargs: Any) -> nx.Graph:
        """Convert to a NetworkX graph (see ``to_networkx``)."""
        from implicit_word_network.network.export import to_networkx

        return to_networkx(self, **kwargs)

    def to_dict(self, **kwargs: Any) -> dict[str, Any]:
        """JSON-serialisable node/edge lists (see ``to_dict``)."""
        from implicit_word_network.network.export import to_dict

        return to_dict(self, **kwargs)

    _ARRAY_FIELDS = (
        "doc_sent_offsets",
        "sent_doc",
        "sent_local",
        "sent_start",
        "sent_end",
        "entity_label_id",
        "m_doc",
        "m_sent",
        "m_entity",
        "m_start",
        "m_end",
        "m_score",
        "t_sent",
        "t_term",
        "t_start",
        "t_end",
    )
    _MATRIX_FIELDS = ("S_e", "S_t", "W", "C", "ET")

    def save(self, path: str | Path) -> Path:
        """Persist the network to a compressed ``.npz`` file.

        Args:
            path: Target path (``.npz`` is appended when missing).

        Returns:
            The written path.
        """
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_suffix(path.suffix + ".npz")
        meta = {
            "format": _SAVE_FORMAT_VERSION,
            "config": self.config.to_dict(),
            "doc_ids": self._doc_ids,
            "doc_meta": self._doc_meta,
            "entity_text": self._entity_text,
            "entity_norm": self._entity_norm,
            "entity_label": self._entity_label,
            "labels": sorted(self._label_index, key=self._label_index.__getitem__),
            "term_text": self._term_text,
            "term_pos": self._term_pos,
            "sent_text": self._sent_text,
        }
        arrays: dict[str, Any] = {"meta": np.array(json.dumps(meta, default=str))}
        for name in self._ARRAY_FIELDS:
            arrays[name] = getattr(self, f"_{name}").view
        for name in self._MATRIX_FIELDS:
            matrix = getattr(self, f"_{name}").matrix
            arrays[f"{name}_data"] = matrix.data
            arrays[f"{name}_indices"] = matrix.indices
            arrays[f"{name}_indptr"] = matrix.indptr
            arrays[f"{name}_shape"] = np.asarray(matrix.shape, dtype=np.int64)
        np.savez_compressed(path, **arrays)
        return path

    @classmethod
    def load(
        cls, path: str | Path, *, normalize_entity: Callable[[str], str] | None = None
    ) -> ImplicitNetwork:
        """Load a network written by :meth:`save`.

        Args:
            path: Path of the ``.npz`` file.
            normalize_entity: Entity normalisation function used when the
                network was built (needed for name lookups if it was custom).
        """
        with np.load(Path(path), allow_pickle=False) as archive:
            meta = json.loads(str(archive["meta"]))
            if meta.get("format") != _SAVE_FORMAT_VERSION:
                raise ValueError(f"Unsupported network file format {meta.get('format')!r}")
            network = cls(
                NetworkConfig.from_dict(meta["config"]), normalize_entity=normalize_entity
            )
            network._doc_ids = list(meta["doc_ids"])
            network._doc_meta = list(meta["doc_meta"])
            network._doc_index = {doc_id: i for i, doc_id in enumerate(network._doc_ids)}
            network._entity_text = list(meta["entity_text"])
            network._entity_norm = list(meta["entity_norm"])
            network._entity_label = list(meta["entity_label"])
            network._entity_index = {
                (norm, label): i
                for i, (norm, label) in enumerate(zip(network._entity_norm, network._entity_label))
            }
            network._label_index = {label: i for i, label in enumerate(meta["labels"])}
            network._term_text = list(meta["term_text"])
            network._term_pos = list(meta["term_pos"])
            network._term_index = {
                (text, pos): i
                for i, (text, pos) in enumerate(zip(network._term_text, network._term_pos))
            }
            network._sent_text = list(meta["sent_text"])
            for name in cls._ARRAY_FIELDS:
                setattr(network, f"_{name}", GrowableArray.from_array(archive[name]))
            for name in cls._MATRIX_FIELDS:
                shape = (int(archive[f"{name}_shape"][0]), int(archive[f"{name}_shape"][1]))
                matrix = sparse.csr_matrix(
                    (
                        archive[f"{name}_data"],
                        archive[f"{name}_indices"],
                        archive[f"{name}_indptr"],
                    ),
                    shape=shape,
                )
                setattr(network, f"_{name}", SparseAccumulator.from_matrix(matrix))
        return network

    # =========================================================================
    # Introspection
    # =========================================================================

    def summary(self) -> str:
        """Human-readable summary of the network size and configuration."""
        lines = [
            "ImplicitNetwork",
            f"  documents : {self.n_documents}",
            f"  sentences : {self.n_sentences}",
            f"  entities  : {self.n_entities}",
            f"  terms     : {self.n_terms}",
            f"  mentions  : {self.n_mentions}",
            f"  edges     : {self.n_edges}",
            f"  window    : {self.config.window} (decay={self.config.decay})",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"ImplicitNetwork(documents={self.n_documents}, sentences={self.n_sentences}, "
            f"entities={self.n_entities}, terms={self.n_terms}, edges={self.n_edges})"
        )
