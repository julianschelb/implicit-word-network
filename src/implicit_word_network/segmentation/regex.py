# segmentation/regex.py
"""Dependency-free regular-expression segmenter."""

from __future__ import annotations

import re
from collections.abc import Collection

from implicit_word_network.annotation import Sentence, Token
from implicit_word_network.segmentation._base import BaseSegmenter, Segmentation
from implicit_word_network.segmentation._stopwords import ENGLISH_STOPWORDS

_DEFAULT_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?…])([\"'”’)\]]*)\s+|\n+")
_DEFAULT_TOKEN = re.compile(r"\w+(?:[-'’]\w+)*|[^\w\s]")


class RegexSegmenter(BaseSegmenter):
    """Sentence splitter and tokeniser based on regular expressions.

    This segmenter has no external dependencies and is fast, but it knows
    nothing about abbreviations, part-of-speech tags or lemmas. Use
    ``SpacySegmenter`` when
    linguistic quality matters.

    Args:
        stopwords: Stop word list (lowercase). Defaults to a compact English
            list; pass an empty collection to disable stop word flagging.
        sentence_boundary: Regular expression matching sentence boundaries.
            When the first group matches closing quotes/brackets they are
            kept with the preceding sentence.
        token_pattern: Regular expression matching one token.

    Example:
        ```python
        from implicit_word_network import RegexSegmenter

        segmenter = RegexSegmenter()
        sentences, tokens = segmenter.segment("Hello world. Second sentence!")
        print(len(sentences))  # 2
        print(tokens[0].text)  # "Hello"
        ```
    """

    def __init__(
        self,
        *,
        stopwords: Collection[str] | None = None,
        sentence_boundary: str | re.Pattern[str] | None = None,
        token_pattern: str | re.Pattern[str] | None = None,
    ) -> None:
        self.stopwords: frozenset[str] = (
            ENGLISH_STOPWORDS if stopwords is None else frozenset(w.lower() for w in stopwords)
        )
        self._boundary = (
            _DEFAULT_SENTENCE_BOUNDARY
            if sentence_boundary is None
            else re.compile(sentence_boundary)
        )
        self._token = _DEFAULT_TOKEN if token_pattern is None else re.compile(token_pattern)

    def segment(self, text: str) -> Segmentation:
        sentences = self._split_sentences(text)
        tokens: list[Token] = []
        for sentence in sentences:
            for match in self._token.finditer(text, sentence.start, sentence.end):
                surface = match.group(0)
                lowered = surface.lower()
                tokens.append(
                    Token(
                        text=surface,
                        start=match.start(),
                        end=match.end(),
                        sentence=sentence.index,
                        is_stop=lowered in self.stopwords,
                        is_punct=not any(ch.isalnum() for ch in surface),
                        is_space=False,
                    )
                )
        return Segmentation(sentences, tokens)

    def _split_sentences(self, text: str) -> list[Sentence]:
        sentences: list[Sentence] = []
        position = 0
        for match in self._boundary.finditer(text):
            closing = (match.group(1) or "") if self._boundary.groups else ""
            self._append(sentences, text, position, match.start() + len(closing))
            position = match.end()
        self._append(sentences, text, position, len(text))
        return sentences

    @staticmethod
    def _append(sentences: list[Sentence], text: str, start: int, end: int) -> None:
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if end > start:
            sentences.append(Sentence(len(sentences), start, end))
