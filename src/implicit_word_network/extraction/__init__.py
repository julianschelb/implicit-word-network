# extraction/__init__.py
"""Entity extraction components.

All extractors implement ``BaseEntityExtractor`` and produce
``AnnotatedDocument`` objects.
"""

from implicit_word_network.extraction._base import (
    BaseEntityExtractor,
    SpanEntityExtractor,
    as_documents,
)
from implicit_word_network.extraction.gazetteer import GazetteerEntityExtractor
from implicit_word_network.extraction.gliner import (
    DEFAULT_GLINER_LABELS,
    DEFAULT_GLINER_MODEL,
    GLiNEREntityExtractor,
)
from implicit_word_network.extraction.spacy import DEFAULT_SPACY_LABELS, SpacyEntityExtractor

__all__ = [
    "BaseEntityExtractor",
    "SpanEntityExtractor",
    "GazetteerEntityExtractor",
    "GLiNEREntityExtractor",
    "SpacyEntityExtractor",
    "DEFAULT_GLINER_MODEL",
    "DEFAULT_GLINER_LABELS",
    "DEFAULT_SPACY_LABELS",
    "as_documents",
]
