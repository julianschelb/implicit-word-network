# annotation.py
"""Typed annotation layer produced by entity extractors.

An ``AnnotatedDocument`` holds the sentence segmentation, tokenisation and
entity mentions of one document, all expressed as character offsets into the
original text. It is the interface between the *extraction* stage and the
*network construction* stage: any extractor that produces
``AnnotatedDocument`` objects can feed an
``ImplicitNetwork``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from implicit_word_network.document import ID

# =============================================================================
# Building blocks
# =============================================================================


@dataclass(frozen=True, slots=True)
class Sentence:
    """A sentence as a character span of the document text.

    Attributes:
        index: Position of the sentence in the document (0-based).
        start: Character offset of the first character.
        end: Character offset one past the last character.
    """

    index: int
    start: int
    end: int

    def __len__(self) -> int:
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class Token:
    """A token with linguistic flags used for term selection.

    Attributes:
        text: Surface form.
        start: Character offset of the first character.
        end: Character offset one past the last character.
        sentence: Index of the sentence containing the token.
        is_stop: Whether the token is a stop word.
        is_punct: Whether the token is punctuation.
        is_space: Whether the token consists of whitespace only.
        pos: Coarse part-of-speech tag (empty when unknown).
        lemma: Lemma (empty when unknown; the lowercased text is used then).
    """

    text: str
    start: int
    end: int
    sentence: int
    is_stop: bool = False
    is_punct: bool = False
    is_space: bool = False
    pos: str = ""
    lemma: str = ""

    def __len__(self) -> int:
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class EntitySpan:
    """A raw entity prediction (character span) before sentence/token alignment.

    Span-based extractors (GLiNER, gazetteers, ...) return these; they are
    turned into ``EntityMention`` objects by ``align_spans``.

    Attributes:
        start: Character offset of the first character.
        end: Character offset one past the last character.
        label: Entity type.
        score: Confidence in ``[0, 1]``.
        text: Surface form (filled from the document text when empty).
    """

    start: int
    end: int
    label: str
    score: float = 1.0
    text: str = ""


@dataclass(frozen=True, slots=True)
class EntityMention:
    """An entity mention aligned to the sentence and token structure.

    Attributes:
        text: Surface form of the mention.
        label: Entity type (e.g. ``"PERSON"`` or ``"person"``).
        start: Character offset of the first character.
        end: Character offset one past the last character.
        sentence: Index of the sentence containing the mention.
        score: Extractor confidence in ``[0, 1]``.
        token_start: Index of the first covered token (``-1`` if unknown).
        token_end: Index one past the last covered token (``-1`` if unknown).
    """

    text: str
    label: str
    start: int
    end: int
    sentence: int
    score: float = 1.0
    token_start: int = -1
    token_end: int = -1

    def __len__(self) -> int:
        return self.end - self.start


# =============================================================================
# AnnotatedDocument
# =============================================================================


@dataclass(slots=True)
class AnnotatedDocument:
    """A document with sentence, token and entity annotations.

    Attributes:
        id: Document identifier (copied from the source ``Document``).
        text: Original text.
        sentences: Sentences in document order.
        tokens: Tokens in document order (each assigned to a sentence).
        mentions: Entity mentions in document order.
        meta: Document metadata.

    Example:
        ```python
        from implicit_word_network import GazetteerEntityExtractor

        extractor = GazetteerEntityExtractor({"PERSON": ["Feynman"]})
        doc = extractor.annotate_text("Feynman taught at Caltech. Feynman loved bongos.")
        print(doc.n_sentences)                 # 2
        print([m.text for m in doc.mentions])  # ["Feynman", "Feynman"]
        print(doc.sentence_text(1))            # "Feynman loved bongos."
        ```
    """

    id: ID
    text: str
    sentences: list[Sentence] = field(default_factory=list)
    tokens: list[Token] = field(default_factory=list)
    mentions: list[EntityMention] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    # ---------- Convenience ----------

    @property
    def n_sentences(self) -> int:
        """Number of sentences."""
        return len(self.sentences)

    def sentence_text(self, index: int) -> str:
        """Return the text of sentence ``index``."""
        sentence = self.sentences[index]
        return self.text[sentence.start : sentence.end]

    def tokens_in(self, index: int) -> list[Token]:
        """Return the tokens of sentence ``index``."""
        return [token for token in self.tokens if token.sentence == index]

    def mentions_in(self, index: int) -> list[EntityMention]:
        """Return the entity mentions of sentence ``index``."""
        return [mention for mention in self.mentions if mention.sentence == index]

    def entity_token_mask(self) -> np.ndarray:
        """Boolean mask over ``tokens`` that is ``True`` for tokens covered by a mention.

        Tokens covered by an entity mention are not used as term nodes when
        building a network. Mentions with token indices are applied directly;
        for the others coverage is decided by character overlap, so the method
        also works for mentions without token indices.
        """
        mask = np.zeros(len(self.tokens), dtype=bool)
        if not self.tokens or not self.mentions:
            return mask
        unaligned: list[EntityMention] = []
        for mention in self.mentions:
            if mention.token_start >= 0 and mention.token_end >= mention.token_start:
                mask[mention.token_start : mention.token_end] = True
            else:
                unaligned.append(mention)
        if unaligned:
            n = len(self.tokens)
            starts = np.fromiter((t.start for t in self.tokens), dtype=np.int64, count=n)
            ends = np.fromiter((t.end for t in self.tokens), dtype=np.int64, count=n)
            for mention in unaligned:
                mask |= (starts < mention.end) & (ends > mention.start)
        return mask

    def validate(self) -> None:
        """Check the internal consistency of offsets and indices.

        Raises:
            ValueError: If a sentence, token or mention has invalid offsets or
                refers to a sentence that does not exist.
        """
        n_chars = len(self.text)
        previous_end = 0
        for expected, sentence in enumerate(self.sentences):
            if sentence.index != expected:
                raise ValueError(f"Sentence index {sentence.index} != position {expected}")
            if not (0 <= sentence.start < sentence.end <= n_chars) or sentence.start < previous_end:
                raise ValueError(f"Invalid sentence span {sentence}")
            previous_end = sentence.end
        for token in self.tokens:
            if not (0 <= token.start < token.end <= n_chars):
                raise ValueError(f"Invalid token span {token}")
            if not (0 <= token.sentence < len(self.sentences)):
                raise ValueError(f"Token {token} refers to unknown sentence")
        for mention in self.mentions:
            if not (0 <= mention.start < mention.end <= n_chars):
                raise ValueError(f"Invalid mention span {mention}")
            if not (0 <= mention.sentence < len(self.sentences)):
                raise ValueError(f"Mention {mention} refers to unknown sentence")

    def __repr__(self) -> str:
        return (
            f"AnnotatedDocument(id={self.id!r}, sentences={len(self.sentences)}, "
            f"tokens={len(self.tokens)}, mentions={len(self.mentions)})"
        )


# =============================================================================
# Alignment
# =============================================================================


def align_spans(
    spans: Sequence[EntitySpan],
    sentences: Sequence[Sentence],
    tokens: Sequence[Token],
    *,
    text: str,
) -> list[EntityMention]:
    """Align character spans to sentences and tokens.

    Every span is assigned to the sentence containing its first character and
    to the range of tokens it overlaps. Empty spans are dropped and the result
    is sorted by document position.

    Args:
        spans: Raw entity spans.
        sentences: Sentences of the document (sorted, non-overlapping).
        tokens: Tokens of the document (sorted, non-overlapping).
        text: Document text (used to fill missing surface forms).

    Returns:
        Aligned mentions sorted by ``(start, end)``.

    Raises:
        ValueError: If spans are given but the document has no sentences.
    """
    valid = [span for span in spans if span.end > span.start]
    if not valid:
        return []
    if not sentences:
        raise ValueError("Cannot align entity spans: the document has no sentences")

    starts = np.fromiter((s.start for s in valid), dtype=np.int64, count=len(valid))
    ends = np.fromiter((s.end for s in valid), dtype=np.int64, count=len(valid))

    sentence_starts = np.fromiter(
        (s.start for s in sentences), dtype=np.int64, count=len(sentences)
    )
    sentence_idx = np.searchsorted(sentence_starts, starts, side="right") - 1
    sentence_idx = np.clip(sentence_idx, 0, len(sentences) - 1)

    if tokens:
        token_starts = np.fromiter((t.start for t in tokens), dtype=np.int64, count=len(tokens))
        token_ends = np.fromiter((t.end for t in tokens), dtype=np.int64, count=len(tokens))
        token_start = np.searchsorted(token_ends, starts, side="right")
        token_end = np.searchsorted(token_starts, ends, side="left")
        token_end = np.maximum(token_end, token_start)
    else:
        token_start = np.full(len(valid), -1, dtype=np.int64)
        token_end = np.full(len(valid), -1, dtype=np.int64)

    mentions = [
        EntityMention(
            text=span.text or text[span.start : span.end],
            label=span.label,
            start=span.start,
            end=span.end,
            sentence=int(sentence_idx[i]),
            score=float(span.score),
            token_start=int(token_start[i]),
            token_end=int(token_end[i]),
        )
        for i, span in enumerate(valid)
    ]
    mentions.sort(key=lambda m: (m.start, m.end))
    return mentions
