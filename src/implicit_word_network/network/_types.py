# network/_types.py
"""Data classes describing nodes, edges and the configuration of implicit networks."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from implicit_word_network.document import ID

# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class NetworkConfig:
    """Parameters controlling how an implicit network is built.

    Attributes:
        window: Context window ``c`` in sentences. Two entity mentions
            cooccur when they appear in the same document at most ``window``
            sentences apart (``0`` restricts cooccurrence to one sentence).
        decay: Name of the weighting function applied to the sentence
            distance ``δ`` of a cooccurrence. ``"exponential"`` (the default
            from Spitz & Gertz) uses ``exp(-δ)``; ``"constant"`` counts
            cooccurrences; ``"inverse"`` uses ``1 / (1 + δ)``; ``"linear"``
            uses ``1 - δ / (window + 1)``. Custom functions can be added with
            ``register_decay``.
        include_stopwords: Keep stop words as term nodes.
        include_punctuation: Keep punctuation tokens as term nodes.
        term_pos: Restrict term nodes to these coarse POS tags (e.g.
            ``{"NOUN", "PROPN", "VERB", "ADJ"}``). ``None`` keeps every tag;
            note that the regex segmenter does not provide POS tags.
        use_lemma: Use the lemma (when available) instead of the surface form
            as term identity.
        lowercase_terms: Lowercase term identities.
        store_text: Keep sentence texts in the network. Required for
            cooccurrence contexts and contextual edge clustering.

    Example:
        ```python
        from implicit_word_network import NetworkConfig

        config = NetworkConfig(window=3, decay="exponential", term_pos={"NOUN", "PROPN"})
        ```
    """

    window: int = 2
    decay: str = "exponential"
    include_stopwords: bool = False
    include_punctuation: bool = False
    term_pos: frozenset[str] | None = None
    use_lemma: bool = True
    lowercase_terms: bool = True
    store_text: bool = True

    def __post_init__(self) -> None:
        if self.window < 0:
            raise ValueError("window must be >= 0")
        if self.term_pos is not None and not isinstance(self.term_pos, frozenset):
            object.__setattr__(self, "term_pos", frozenset(self.term_pos))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        data = asdict(self)
        data["term_pos"] = sorted(self.term_pos) if self.term_pos is not None else None
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NetworkConfig:
        """Rebuild a configuration from ``to_dict`` output."""
        term_pos: Iterable[str] | None = data.get("term_pos")
        payload = {**data, "term_pos": frozenset(term_pos) if term_pos is not None else None}
        return cls(**payload)


# =============================================================================
# Nodes
# =============================================================================


@dataclass(frozen=True, slots=True)
class EntityNode:
    """An entity node, unique by normalised name and type.

    Attributes:
        id: Integer index of the node inside the network.
        text: Surface form of the first mention seen.
        norm: Normalised name used for identity.
        label: Entity type.
        count: Number of mentions in the corpus.
    """

    id: int
    text: str
    norm: str
    label: str
    count: int

    @property
    def key(self) -> tuple[str, str]:
        """``(norm, label)`` identity tuple."""
        return (self.norm, self.label)


@dataclass(frozen=True, slots=True)
class TermNode:
    """A term node (non-entity word), unique by normalised form and POS tag.

    Attributes:
        id: Integer index of the term inside the network.
        text: Normalised form (lemma or lowercased surface).
        pos: Coarse part-of-speech tag (empty when unknown).
        count: Number of occurrences in the corpus.
    """

    id: int
    text: str
    pos: str
    count: int


@dataclass(frozen=True, slots=True)
class DocumentRef:
    """A document node.

    Attributes:
        index: Position of the document in the network.
        id: User-facing document identifier.
        n_sentences: Number of sentences.
        meta: Document metadata.
    """

    index: int
    id: ID
    n_sentences: int
    meta: dict[str, Any] = field(default_factory=dict, hash=False, compare=False)


@dataclass(frozen=True, slots=True)
class SentenceRef:
    """A sentence node.

    Attributes:
        id: Global sentence index inside the network.
        document: Identifier of the containing document.
        index: Position of the sentence inside its document.
        text: Sentence text (empty when texts are not stored).
    """

    id: int
    document: ID
    index: int
    text: str


# =============================================================================
# Edges / instances
# =============================================================================


@dataclass(frozen=True, slots=True)
class Mention:
    """One occurrence of an entity in a sentence.

    Attributes:
        id: Row index in the mention table.
        entity: The entity node.
        sentence: The containing sentence.
        start: Character offset in the document text.
        end: Character offset one past the last character.
        score: Extractor confidence.
    """

    id: int
    entity: EntityNode
    sentence: SentenceRef
    start: int
    end: int
    score: float


@dataclass(frozen=True, slots=True)
class Cooccurrence:
    """One cooccurrence of two entity mentions inside the context window.

    Attributes:
        source: First mention (earlier in the document).
        target: Second mention.
        delta: Sentence distance between the mentions.
        weight: Decayed weight contributed to the edge (e.g. ``exp(-delta)``).
    """

    source: Mention
    target: Mention
    delta: int
    weight: float

    @property
    def document(self) -> ID:
        """Identifier of the document containing both mentions."""
        return self.source.sentence.document

    @property
    def sentence_span(self) -> tuple[int, int]:
        """Global ids of the first and last sentence of the context."""
        first = min(self.source.sentence.id, self.target.sentence.id)
        last = max(self.source.sentence.id, self.target.sentence.id)
        return first, last


@dataclass(frozen=True, slots=True)
class EntityEdge:
    """Aggregated (implicit) edge between two entities.

    Attributes:
        source: Entity with the smaller id.
        target: Entity with the larger id.
        weight: Aggregated weight ``ω = Σ decay(δ)`` over all cooccurrences.
        count: Number of cooccurrence instances.
    """

    source: EntityNode
    target: EntityNode
    weight: float
    count: int


@dataclass(frozen=True)
class CooccurrenceTable:
    """Column-oriented table of all cooccurrence instances of a network.

    Attributes:
        source: Mention row indices of the first mention of every pair.
        target: Mention row indices of the second mention.
        delta: Sentence distances.
        weight: Decayed weights.
    """

    source: NDArray[np.int64]
    target: NDArray[np.int64]
    delta: NDArray[np.int64]
    weight: NDArray[np.float64]

    def __len__(self) -> int:
        return int(self.source.shape[0])
