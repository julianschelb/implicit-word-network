# segmentation/__init__.py
"""Sentence segmentation and tokenisation components."""

from implicit_word_network.segmentation._base import BaseSegmenter, Segmentation
from implicit_word_network.segmentation._stopwords import ENGLISH_STOPWORDS
from implicit_word_network.segmentation.regex import RegexSegmenter
from implicit_word_network.segmentation.spacy import (
    DEFAULT_SPACY_MODEL,
    SpacySegmenter,
    load_spacy_model,
    spacy_doc_to_segmentation,
)

__all__ = [
    "BaseSegmenter",
    "Segmentation",
    "RegexSegmenter",
    "SpacySegmenter",
    "ENGLISH_STOPWORDS",
    "DEFAULT_SPACY_MODEL",
    "load_spacy_model",
    "spacy_doc_to_segmentation",
]
