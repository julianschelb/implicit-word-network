# extraction/spacy.py
"""spaCy named-entity extractor."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from typing import Any

from tqdm import tqdm

from implicit_word_network.annotation import AnnotatedDocument, EntityMention
from implicit_word_network.document import Document
from implicit_word_network.extraction._base import BaseEntityExtractor, as_documents
from implicit_word_network.segmentation.spacy import (
    DEFAULT_SPACY_MODEL,
    load_spacy_model,
    spacy_doc_to_segmentation,
)

DEFAULT_SPACY_LABELS: tuple[str, ...] = ("PERSON", "ORG", "GPE", "NORP", "LOC", "WORK_OF_ART")
"""Default spaCy entity types (the ECCE defaults)."""


class SpacyEntityExtractor(BaseEntityExtractor):
    """Extract entities with a spaCy NER pipeline.

    Runs segmentation, tagging and NER in one batched ``nlp.pipe`` pass and
    keeps only entities of the requested types. Requires the ``spacy`` extra
    and an installed pipeline (``python -m spacy download en_core_web_sm``).

    Args:
        model: Installed pipeline name, or an already loaded ``Language``.
        labels: Entity types to keep, or ``None`` to keep every type.
        n_process: Number of processes for ``nlp.pipe``.
        disable: Additional pipeline components to disable when loading.
        max_length: Raise spaCy's character limit (default 1,000,000) for
            long documents.

    Example:
        ```python
        from implicit_word_network import SpacyEntityExtractor

        extractor = SpacyEntityExtractor("en_core_web_sm", labels=["PERSON", "ORG"])
        doc = extractor.annotate_text("Feynman received the Nobel Prize at Caltech.")
        print([(m.text, m.label) for m in doc.mentions])
        ```
    """

    def __init__(
        self,
        model: str | Any = DEFAULT_SPACY_MODEL,
        *,
        labels: Sequence[str] | None = DEFAULT_SPACY_LABELS,
        n_process: int = 1,
        disable: Sequence[str] = (),
        max_length: int | None = None,
    ) -> None:
        self._model_name: str | None = model if isinstance(model, str) else None
        self._nlp: Any | None = None if isinstance(model, str) else model
        self.labels: frozenset[str] | None = None if labels is None else frozenset(labels)
        self.n_process = n_process
        self.disable = tuple(disable)
        self.max_length = max_length

    @property
    def nlp(self) -> Any:
        """The underlying spaCy pipeline (loaded lazily)."""
        if self._nlp is None:
            assert self._model_name is not None
            self._nlp = load_spacy_model(
                self._model_name, disable=self.disable, max_length=self.max_length
            )
        return self._nlp

    def annotate(
        self,
        documents: Iterable[Document | str],
        *,
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> Iterator[AnnotatedDocument]:
        pairs = ((document.text, document) for document in as_documents(documents))
        stream = self.nlp.pipe(
            pairs, as_tuples=True, batch_size=batch_size, n_process=self.n_process
        )
        for spacy_doc, document in tqdm(
            stream, disable=not show_progress, desc="Annotating", unit="doc"
        ):
            segmentation = spacy_doc_to_segmentation(spacy_doc)
            mentions = [
                EntityMention(
                    text=ent.text,
                    label=ent.label_,
                    start=ent.start_char,
                    end=ent.end_char,
                    sentence=segmentation.tokens[ent.start].sentence,
                    score=1.0,
                    token_start=ent.start,
                    token_end=ent.end,
                )
                for ent in spacy_doc.ents
                if self.labels is None or ent.label_ in self.labels
            ]
            yield AnnotatedDocument(
                id=document.id,
                text=document.text,
                sentences=segmentation.sentences,
                tokens=segmentation.tokens,
                mentions=mentions,
                meta=dict(document.meta),
            )
