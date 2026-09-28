# segmentation/_base.py
"""Abstract base class for sentence segmenters / tokenisers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from typing import NamedTuple

from implicit_word_network.annotation import Sentence, Token


class Segmentation(NamedTuple):
    """Sentences and tokens of one document."""

    sentences: list[Sentence]
    tokens: list[Token]


class BaseSegmenter(ABC):
    """Abstract base class for sentence splitting and tokenisation.

    Segmenters provide the sentence and token structure that span-based
    entity extractors (GLiNER, gazetteers, ...) lack. Subclasses must
    implement ``segment``; ``segment_many`` may be overridden for
    batched processing.

    Available implementations:

    - ``RegexSegmenter`` — dependency-free regular-expression segmenter.
    - ``SpacySegmenter`` — spaCy pipeline (sentences, POS tags, lemmas,
      stop words).
    """

    @abstractmethod
    def segment(self, text: str) -> Segmentation:
        """Split ``text`` into sentences and tokens.

        Args:
            text: Document text.

        Returns:
            A ``Segmentation`` with sentences and tokens sorted by
            character offset. Every token belongs to exactly one sentence.
        """
        ...

    def segment_many(self, texts: Iterable[str], *, batch_size: int = 64) -> Iterator[Segmentation]:
        """Segment several texts, yielding one ``Segmentation`` per text.

        Args:
            texts: Document texts.
            batch_size: Hint for implementations that process texts in batches.
        """
        for text in texts:
            yield self.segment(text)
