# segmentation/spacy.py
"""spaCy-based segmenter and shared spaCy helpers."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from typing import Any

from implicit_word_network._utils import require
from implicit_word_network.annotation import Sentence, Token
from implicit_word_network.segmentation._base import BaseSegmenter, Segmentation

DEFAULT_SPACY_MODEL = "en_core_web_sm"
"""spaCy pipeline used when none is given."""


def load_spacy_model(
    model: str, *, disable: Sequence[str] = (), max_length: int | None = None
) -> Any:
    """Load a spaCy pipeline, making sure it can split sentences.

    Args:
        model: Name of an installed spaCy pipeline (e.g. ``"en_core_web_sm"``).
        disable: Pipeline components to disable.
        max_length: Raise spaCy's ``nlp.max_length`` (default 1,000,000
            characters) to process longer documents.

    Returns:
        The loaded ``spacy.language.Language`` object.

    Raises:
        ImportError: If spaCy is not installed.
        OSError: If the pipeline is not installed (with download hint).
    """
    require("spacy", extra="spacy", purpose="spaCy segmentation/extraction")
    import spacy

    try:
        nlp = spacy.load(model, disable=list(disable))
    except OSError as error:  # pragma: no cover - depends on local installation
        raise OSError(
            f"spaCy pipeline {model!r} is not installed. "
            f"Download it with: python -m spacy download {model}"
        ) from error
    if not any(nlp.has_pipe(name) for name in ("parser", "senter", "sentencizer")):
        nlp.add_pipe("sentencizer")
    if max_length is not None:
        nlp.max_length = max(int(max_length), nlp.max_length)
    return nlp


def spacy_doc_to_segmentation(doc: Any) -> Segmentation:
    """Convert a ``spacy.tokens.Doc`` into a ``Segmentation``.

    Token positions are preserved, so spaCy token indices (e.g. ``ent.start``)
    index the returned token list directly.
    """
    sentences: list[Sentence] = []
    tokens: list[Token] = []
    for index, sent in enumerate(doc.sents):
        sentences.append(Sentence(index, sent.start_char, sent.end_char))
        for token in sent:
            tokens.append(
                Token(
                    text=token.text,
                    start=token.idx,
                    end=token.idx + len(token.text),
                    sentence=index,
                    is_stop=bool(token.is_stop),
                    is_punct=bool(token.is_punct),
                    is_space=bool(token.is_space),
                    pos=token.pos_,
                    lemma=token.lemma_,
                )
            )
    return Segmentation(sentences, tokens)


class SpacySegmenter(BaseSegmenter):
    """Sentence splitter and tokeniser backed by a spaCy pipeline.

    Provides sentences, part-of-speech tags, lemmas and stop word flags, which
    enable lemma-based term nodes and POS filtering in the network builder.
    Requires the ``spacy`` extra and an installed pipeline.

    Args:
        model: Installed spaCy pipeline name, or an already loaded
            ``Language`` object.
        disable: Components to disable when loading by name. NER is disabled
            by default because this class only segments.
        n_process: Number of processes for ``nlp.pipe``.
        max_length: Raise spaCy's character limit for long documents.

    Example:
        ```python
        from implicit_word_network import SpacySegmenter

        segmenter = SpacySegmenter("en_core_web_sm")
        sentences, tokens = segmenter.segment("Feynman taught at Caltech.")
        print(tokens[0].pos, tokens[0].lemma)  # "PROPN" "Feynman"
        ```
    """

    def __init__(
        self,
        model: str | Any = DEFAULT_SPACY_MODEL,
        *,
        disable: Sequence[str] = ("ner",),
        n_process: int = 1,
        max_length: int | None = None,
    ) -> None:
        self._model_name: str | None = model if isinstance(model, str) else None
        self._nlp: Any | None = None if isinstance(model, str) else model
        self.disable = tuple(disable)
        self.n_process = n_process
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

    def segment(self, text: str) -> Segmentation:
        return spacy_doc_to_segmentation(self.nlp(text))

    def segment_many(self, texts: Iterable[str], *, batch_size: int = 64) -> Iterator[Segmentation]:
        for doc in self.nlp.pipe(texts, batch_size=batch_size, n_process=self.n_process):
            yield spacy_doc_to_segmentation(doc)
