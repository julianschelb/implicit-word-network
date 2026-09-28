# extraction/gazetteer.py
"""Dictionary (gazetteer) based entity extractor."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence

from implicit_word_network.annotation import EntitySpan, Sentence
from implicit_word_network.extraction._base import SpanEntityExtractor
from implicit_word_network.segmentation import BaseSegmenter


class GazetteerEntityExtractor(SpanEntityExtractor):
    """Extract entities by matching a dictionary of known surface forms.

    Useful for closed vocabularies, for tests, and as a fully offline
    baseline. Longer surface forms win over shorter ones at the same
    position; matches never start or end inside a word.

    Args:
        gazetteer: Mapping from entity label to surface forms, e.g.
            ``{"PERSON": ["Richard Feynman", "Feynman"], "ORG": ["Caltech"]}``.
        segmenter: Segmenter providing sentences and tokens.
        case_sensitive: Whether matching respects letter case.

    Example:
        ```python
        from implicit_word_network import GazetteerEntityExtractor

        extractor = GazetteerEntityExtractor(
            {"PERSON": ["Feynman"], "ORG": ["Caltech", "Nobel Prize"]}
        )
        doc = extractor.annotate_text("Feynman worked at Caltech.")
        print([(m.text, m.label) for m in doc.mentions])
        # [("Feynman", "PERSON"), ("Caltech", "ORG")]
        ```
    """

    def __init__(
        self,
        gazetteer: Mapping[str, Iterable[str]],
        *,
        segmenter: BaseSegmenter | None = None,
        case_sensitive: bool = False,
    ) -> None:
        super().__init__(segmenter=segmenter)
        self.case_sensitive = case_sensitive
        self._labels: dict[str, str] = {}
        forms: list[str] = []
        for label, surfaces in gazetteer.items():
            for surface in surfaces:
                surface = surface.strip()
                if not surface:
                    continue
                self._labels[self._key(surface)] = label
                forms.append(surface)
        if not forms:
            raise ValueError("The gazetteer does not contain any surface forms")
        forms.sort(key=len, reverse=True)
        alternation = "|".join(re.escape(form) for form in forms)
        flags = 0 if case_sensitive else re.IGNORECASE
        self._pattern = re.compile(rf"(?<!\w)(?:{alternation})(?!\w)", flags)

    def _key(self, surface: str) -> str:
        return surface if self.case_sensitive else surface.lower()

    @property
    def labels(self) -> frozenset[str]:
        """Entity labels known to this extractor."""
        return frozenset(self._labels.values())

    def extract_spans(
        self,
        texts: Sequence[str],
        sentences: Sequence[Sequence[Sentence]],
    ) -> list[list[EntitySpan]]:
        results: list[list[EntitySpan]] = []
        for text in texts:
            spans = [
                EntitySpan(
                    start=match.start(),
                    end=match.end(),
                    label=self._labels[self._key(match.group(0))],
                    score=1.0,
                    text=match.group(0),
                )
                for match in self._pattern.finditer(text)
            ]
            results.append(spans)
        return results
