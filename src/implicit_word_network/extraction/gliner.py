# extraction/gliner.py
"""Zero-shot entity extraction with GLiNER."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from implicit_word_network._utils import require
from implicit_word_network.annotation import EntitySpan, Sentence
from implicit_word_network.extraction._base import SpanEntityExtractor
from implicit_word_network.segmentation import BaseSegmenter

DEFAULT_GLINER_MODEL = "gliner-community/gliner_medium-v2.5"
"""Default GLiNER checkpoint (GLiNER v2.5, Apache-2.0)."""

DEFAULT_GLINER_LABELS: tuple[str, ...] = ("person", "organization", "location", "work of art")
"""Default zero-shot labels, mirroring the spaCy defaults."""


class GLiNEREntityExtractor(SpanEntityExtractor):
    """Zero-shot entity extraction with a GLiNER model.

    GLiNER predicts spans for *arbitrary* natural-language labels, so the
    entity types of the network can be chosen freely (``"person"``,
    ``"disease"``, ``"ship"``, ...). Documents are split into chunks of whole
    sentences that fit the model's context window, and all chunks of a batch
    are scored in one batched call. Requires the ``gliner`` extra.

    Args:
        model: Hugging Face model id / local path, or an already loaded
            ``gliner.GLiNER`` instance. Defaults to
            ``gliner-community/gliner_medium-v2.5``.
        labels: Natural-language entity labels to extract.
        threshold: Minimum confidence for a predicted span.
        segmenter: Segmenter providing sentences and tokens (defaults to
            ``RegexSegmenter``;
            use ``SpacySegmenter``
            for lemmas and POS tags).
        device: Device passed to ``GLiNER.from_pretrained`` (``"cpu"``,
            ``"cuda"``, ``"mps"``).
        max_chunk_words: Maximum number of whitespace-delimited words per
            chunk sent to the model.
        batch_size: Number of chunks per forward pass.
        flat_ner: Disallow overlapping spans.
        multi_label: Allow several labels for the same span.
        label_map: Optional mapping to rename predicted labels
            (e.g. ``{"person": "PERSON"}``).
        load_kwargs: Extra keyword arguments for ``GLiNER.from_pretrained``.

    Example:
        ```python
        from implicit_word_network import GLiNEREntityExtractor

        extractor = GLiNEREntityExtractor(
            "gliner-community/gliner_medium-v2.5",
            labels=["person", "organization", "award"],
            threshold=0.4,
        )
        doc = extractor.annotate_text("Feynman received the Nobel Prize in 1965.")
        print([(m.text, m.label, round(m.score, 2)) for m in doc.mentions])
        ```
    """

    def __init__(
        self,
        model: str | Any = DEFAULT_GLINER_MODEL,
        *,
        labels: Sequence[str] = DEFAULT_GLINER_LABELS,
        threshold: float = 0.5,
        segmenter: BaseSegmenter | None = None,
        device: str = "cpu",
        max_chunk_words: int = 200,
        batch_size: int = 8,
        flat_ner: bool = True,
        multi_label: bool = False,
        label_map: Mapping[str, str] | None = None,
        load_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(segmenter=segmenter)
        if not labels:
            raise ValueError("At least one label is required")
        if max_chunk_words < 1:
            raise ValueError("max_chunk_words must be >= 1")
        self._model_name: str | None = model if isinstance(model, str) else None
        self._model: Any | None = None if isinstance(model, str) else model
        self.labels: list[str] = list(labels)
        self.threshold = threshold
        self.device = device
        self.max_chunk_words = max_chunk_words
        self.batch_size = batch_size
        self.flat_ner = flat_ner
        self.multi_label = multi_label
        self.label_map: dict[str, str] = dict(label_map or {})
        self.load_kwargs: dict[str, Any] = dict(load_kwargs or {})

    # ---------- Model handling ----------

    @property
    def model(self) -> Any:
        """The underlying GLiNER model (loaded lazily on first use)."""
        if self._model is None:
            require("gliner", extra="gliner", purpose="GLiNEREntityExtractor")
            from gliner import GLiNER

            assert self._model_name is not None
            kwargs: dict[str, Any] = {"map_location": self.device, **self.load_kwargs}
            self._model = GLiNER.from_pretrained(self._model_name, **kwargs)
        return self._model

    def _predict(self, chunks: Sequence[str]) -> list[list[dict[str, Any]]]:
        """Run batched inference and return raw GLiNER entity dicts per chunk."""
        model = self.model
        kwargs: dict[str, Any] = {
            "threshold": self.threshold,
            "flat_ner": self.flat_ner,
            "multi_label": self.multi_label,
            "batch_size": self.batch_size,
        }
        inference = getattr(model, "inference", None)
        if callable(inference):  # gliner >= 0.2.20
            result = inference(list(chunks), self.labels, **kwargs)
        else:  # pragma: no cover - older gliner releases
            result = model.batch_predict_entities(list(chunks), self.labels, **kwargs)
        return [list(entities) for entities in result]

    # ---------- Chunking ----------

    def _chunks(self, text: str, sentences: Sequence[Sentence]) -> list[tuple[int, int]]:
        """Group whole sentences into chunks of at most ``max_chunk_words`` words."""
        if not sentences:
            return [(0, len(text))] if text.strip() else []
        chunks: list[tuple[int, int]] = []
        chunk_start = sentences[0].start
        chunk_end = sentences[0].end
        words = _count_words(text, sentences[0])
        for sentence in sentences[1:]:
            n_words = _count_words(text, sentence)
            if words + n_words > self.max_chunk_words and words > 0:
                chunks.extend(self._split_long(text, chunk_start, chunk_end))
                chunk_start = sentence.start
                words = 0
            chunk_end = sentence.end
            words += n_words
        chunks.extend(self._split_long(text, chunk_start, chunk_end))
        return chunks

    def _split_long(self, text: str, start: int, end: int) -> list[tuple[int, int]]:
        """Split a chunk that exceeds ``max_chunk_words`` at whitespace."""
        segment = text[start:end]
        if len(segment.split()) <= self.max_chunk_words:
            return [(start, end)]
        pieces: list[tuple[int, int]] = []
        piece_start = start
        count = 0
        position = start
        while position < end:
            while position < end and text[position].isspace():
                position += 1
            word_start = position
            while position < end and not text[position].isspace():
                position += 1
            if position > word_start:
                count += 1
                if count > self.max_chunk_words:
                    pieces.append((piece_start, word_start))
                    piece_start = word_start
                    count = 1
        if piece_start < end:
            pieces.append((piece_start, end))
        return pieces

    # ---------- SpanEntityExtractor API ----------

    def extract_spans(
        self,
        texts: Sequence[str],
        sentences: Sequence[Sequence[Sentence]],
    ) -> list[list[EntitySpan]]:
        chunk_owner: list[tuple[int, int]] = []  # (text index, chunk start offset)
        chunk_texts: list[str] = []
        for index, (text, doc_sentences) in enumerate(zip(texts, sentences)):
            for start, end in self._chunks(text, doc_sentences):
                chunk = text[start:end]
                if not chunk.strip():
                    continue
                chunk_owner.append((index, start))
                chunk_texts.append(chunk)

        results: list[list[EntitySpan]] = [[] for _ in texts]
        if not chunk_texts:
            return results

        predictions = self._predict(chunk_texts)
        if len(predictions) != len(chunk_texts):
            raise RuntimeError(
                f"GLiNER returned {len(predictions)} results for {len(chunk_texts)} chunks"
            )
        for (index, offset), entities in zip(chunk_owner, predictions):
            text = texts[index]
            for entity in entities:
                start = offset + int(entity["start"])
                end = offset + int(entity["end"])
                if end <= start:
                    continue
                label = str(entity["label"])
                results[index].append(
                    EntitySpan(
                        start=start,
                        end=end,
                        label=self.label_map.get(label, label),
                        score=float(entity.get("score", 1.0)),
                        text=text[start:end],
                    )
                )
        return results


def _count_words(text: str, sentence: Sentence) -> int:
    return len(text[sentence.start : sentence.end].split())
