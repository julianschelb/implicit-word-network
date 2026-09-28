# extraction/_base.py
"""Abstract base classes for entity extractors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Sequence

from tqdm import tqdm

from implicit_word_network._utils import batched
from implicit_word_network.annotation import (
    AnnotatedDocument,
    EntitySpan,
    Sentence,
    align_spans,
)
from implicit_word_network.document import ID, Document
from implicit_word_network.segmentation import BaseSegmenter, RegexSegmenter


def as_documents(documents: Iterable[Document | str]) -> Iterator[Document]:
    """Normalise an iterable of documents or strings into ``Document`` objects.

    Strings receive sequential integer identifiers.
    """
    for position, item in enumerate(documents):
        if isinstance(item, Document):
            yield item
        elif isinstance(item, str):
            yield Document(item, position)
        else:
            raise TypeError(f"Expected Document or str, got {type(item).__name__}")


class BaseEntityExtractor(ABC):
    """Abstract base class for entity extractors.

    An entity extractor turns raw documents into
    ``AnnotatedDocument`` objects that
    carry sentences, tokens and entity mentions. Subclasses must implement
    ``annotate``.

    Available implementations:

    - ``SpacyEntityExtractor`` — spaCy NER with a fixed label set.
    - ``GLiNEREntityExtractor`` — zero-shot NER with GLiNER (any labels).
    - ``GazetteerEntityExtractor`` — dictionary matching, no models needed.

    Extractors that only predict character spans should subclass
    ``SpanEntityExtractor`` instead, which handles segmentation and
    alignment.
    """

    @abstractmethod
    def annotate(
        self,
        documents: Iterable[Document | str],
        *,
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> Iterator[AnnotatedDocument]:
        """Annotate documents lazily, in corpus order.

        Args:
            documents: Documents (or raw strings) to annotate.
            batch_size: Number of documents processed per model call.
            show_progress: Whether to display a progress bar.

        Yields:
            One annotated document per input document.
        """
        ...

    def annotate_text(self, text: str, *, doc_id: ID = 0) -> AnnotatedDocument:
        """Annotate a single string.

        Args:
            text: Document text.
            doc_id: Identifier of the resulting document.
        """
        return next(iter(self.annotate([Document(text, doc_id)])))

    def annotate_all(
        self,
        documents: Iterable[Document | str],
        *,
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> list[AnnotatedDocument]:
        """Eager variant of ``annotate`` returning a list."""
        return list(self.annotate(documents, batch_size=batch_size, show_progress=show_progress))


class SpanEntityExtractor(BaseEntityExtractor):
    """Base class for extractors that predict character spans only.

    Sentence splitting and tokenisation are delegated to a
    ``BaseSegmenter``; predicted
    spans are aligned to the resulting structure with
    ``align_spans``. Subclasses must
    implement ``extract_spans``.

    Args:
        segmenter: Segmenter providing sentences and tokens. Defaults to the
            dependency-free ``RegexSegmenter``.

    Example:
        ```python
        from implicit_word_network import SpanEntityExtractor, EntitySpan

        class UpperCaseExtractor(SpanEntityExtractor):
            def extract_spans(self, texts, sentences):
                import re
                return [
                    [EntitySpan(m.start(), m.end(), "TERM") for m in re.finditer(r"\\b[A-Z]{2,}\\b", t)]
                    for t in texts
                ]

        doc = UpperCaseExtractor().annotate_text("NASA and ESA cooperate.")
        print([m.text for m in doc.mentions])  # ["NASA", "ESA"]
        ```
    """

    def __init__(self, *, segmenter: BaseSegmenter | None = None) -> None:
        self.segmenter: BaseSegmenter = segmenter if segmenter is not None else RegexSegmenter()

    @abstractmethod
    def extract_spans(
        self,
        texts: Sequence[str],
        sentences: Sequence[Sequence[Sentence]],
    ) -> list[list[EntitySpan]]:
        """Predict entity spans for a batch of texts.

        Args:
            texts: Document texts of the batch.
            sentences: Sentence boundaries of every text (parallel to
                ``texts``), useful for chunking long inputs.

        Returns:
            One list of ``EntitySpan``
            per text.
        """
        ...

    def annotate(
        self,
        documents: Iterable[Document | str],
        *,
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> Iterator[AnnotatedDocument]:
        progress = tqdm(disable=not show_progress, desc="Annotating", unit="doc")
        try:
            for batch in batched(as_documents(documents), batch_size):
                texts = [document.text for document in batch]
                segmentations = list(self.segmenter.segment_many(texts, batch_size=batch_size))
                spans = self.extract_spans(texts, [seg.sentences for seg in segmentations])
                if len(spans) != len(batch):
                    raise RuntimeError(
                        f"extract_spans returned {len(spans)} results for {len(batch)} texts"
                    )
                for document, segmentation, doc_spans in zip(batch, segmentations, spans):
                    mentions = align_spans(
                        doc_spans, segmentation.sentences, segmentation.tokens, text=document.text
                    )
                    yield AnnotatedDocument(
                        id=document.id,
                        text=document.text,
                        sentences=segmentation.sentences,
                        tokens=segmentation.tokens,
                        mentions=mentions,
                        meta=dict(document.meta),
                    )
                progress.update(len(batch))
        finally:
            progress.close()
