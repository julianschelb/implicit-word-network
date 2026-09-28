"""
Unit tests for implicit_word_network.annotation (data classes and span alignment).
"""

import numpy as np
import pytest

from implicit_word_network.annotation import (
    AnnotatedDocument,
    EntityMention,
    EntitySpan,
    Sentence,
    Token,
    align_spans,
)
from implicit_word_network.segmentation import RegexSegmenter

TEXT = "Feynman worked at Caltech. Schwinger taught at Harvard."


@pytest.fixture
def segmentation():
    return RegexSegmenter().segment(TEXT)


class TestDataClasses:
    def test_lengths(self):
        assert len(Sentence(0, 0, 5)) == 5
        assert len(Token("ab", 3, 5, 0)) == 2
        assert len(EntityMention("ab", "X", 3, 5, 0)) == 2

    def test_token_defaults(self):
        token = Token("x", 0, 1, 0)
        assert not token.is_stop and not token.is_punct and not token.is_space
        assert token.pos == "" and token.lemma == ""

    def test_entity_span_defaults(self):
        span = EntitySpan(0, 3, "PERSON")
        assert span.score == 1.0 and span.text == ""


class TestAlignSpans:
    def test_assigns_sentences_and_tokens(self, segmentation):
        sentences, tokens = segmentation
        spans = [
            EntitySpan(TEXT.index("Schwinger"), TEXT.index("Schwinger") + 9, "PERSON", 0.8),
            EntitySpan(0, 7, "PERSON"),
        ]
        mentions = align_spans(spans, sentences, tokens, text=TEXT)
        assert [m.text for m in mentions] == ["Feynman", "Schwinger"]  # sorted by offset
        assert mentions[0].sentence == 0 and mentions[1].sentence == 1
        assert mentions[1].score == 0.8
        first = mentions[0]
        assert tokens[first.token_start].text == "Feynman"
        assert first.token_end == first.token_start + 1
        second = mentions[1]
        assert [t.text for t in tokens[second.token_start : second.token_end]] == ["Schwinger"]

    def test_multi_token_span(self, segmentation):
        sentences, tokens = segmentation
        start = TEXT.index("worked at Caltech")
        mention = align_spans([EntitySpan(start, start + 17, "X")], sentences, tokens, text=TEXT)[0]
        assert [t.text for t in tokens[mention.token_start : mention.token_end]] == [
            "worked",
            "at",
            "Caltech",
        ]

    def test_partial_token_overlap(self, segmentation):
        sentences, tokens = segmentation
        # "eynman wor" overlaps two tokens partially
        mention = align_spans([EntitySpan(1, 11, "X")], sentences, tokens, text=TEXT)[0]
        assert [t.text for t in tokens[mention.token_start : mention.token_end]] == [
            "Feynman",
            "worked",
        ]

    def test_empty_spans_dropped_and_text_filled(self, segmentation):
        sentences, tokens = segmentation
        mentions = align_spans(
            [EntitySpan(3, 3, "X"), EntitySpan(0, 7, "PERSON")], sentences, tokens, text=TEXT
        )
        assert len(mentions) == 1
        assert mentions[0].text == "Feynman"

    def test_no_spans(self, segmentation):
        assert align_spans([], *segmentation, text=TEXT) == []

    def test_no_sentences_raises(self):
        with pytest.raises(ValueError):
            align_spans([EntitySpan(0, 1, "X")], [], [], text="a")

    def test_no_tokens_gives_unknown_token_range(self):
        mention = align_spans([EntitySpan(0, 1, "X")], [Sentence(0, 0, 1)], [], text="a")[0]
        assert mention.token_start == -1 and mention.token_end == -1

    def test_span_in_whitespace_gets_empty_token_range(self):
        text = "a   b"
        tokens = [Token("a", 0, 1, 0), Token("b", 4, 5, 0)]
        mention = align_spans([EntitySpan(1, 3, "X")], [Sentence(0, 0, 5)], tokens, text=text)[0]
        assert mention.token_end == mention.token_start


class TestAnnotatedDocument:
    @pytest.fixture
    def doc(self, segmentation):
        sentences, tokens = segmentation
        spans = [EntitySpan(0, 7, "PERSON"), EntitySpan(18, 25, "ORG")]
        mentions = align_spans(spans, sentences, tokens, text=TEXT)
        return AnnotatedDocument("d", TEXT, sentences, tokens, mentions, {"k": "v"})

    def test_helpers(self, doc):
        assert doc.n_sentences == 2
        assert doc.sentence_text(1) == "Schwinger taught at Harvard."
        assert [t.text for t in doc.tokens_in(1)] == ["Schwinger", "taught", "at", "Harvard", "."]
        assert [m.text for m in doc.mentions_in(0)] == ["Feynman", "Caltech"]
        assert doc.mentions_in(1) == []
        assert "mentions=2" in repr(doc)

    def test_entity_token_mask_uses_token_indices(self, doc):
        mask = doc.entity_token_mask()
        covered = [t.text for t, flag in zip(doc.tokens, mask) if flag]
        assert covered == ["Feynman", "Caltech"]

    def test_entity_token_mask_falls_back_to_char_overlap(self, doc):
        doc.mentions = [EntityMention("Caltech", "ORG", 18, 25, 0)]  # no token indices
        mask = doc.entity_token_mask()
        assert [t.text for t, flag in zip(doc.tokens, mask) if flag] == ["Caltech"]

    def test_entity_token_mask_empty(self):
        assert AnnotatedDocument("d", "").entity_token_mask().shape == (0,)
        doc = AnnotatedDocument("d", "a", [Sentence(0, 0, 1)], [Token("a", 0, 1, 0)])
        assert not doc.entity_token_mask().any()

    def test_validate_ok(self, doc):
        doc.validate()

    @pytest.mark.parametrize(
        "mutation",
        [
            lambda d: d.sentences.__setitem__(1, Sentence(5, 27, 55)),
            lambda d: d.sentences.__setitem__(0, Sentence(0, 10, 5)),
            lambda d: d.tokens.append(Token("x", 0, 1, 9)),
            lambda d: d.tokens.append(Token("x", 0, 999, 0)),
            lambda d: d.mentions.append(EntityMention("x", "X", 0, 1, 9)),
            lambda d: d.mentions.append(EntityMention("x", "X", 5, 2, 0)),
        ],
    )
    def test_validate_errors(self, doc, mutation):
        mutation(doc)
        with pytest.raises(ValueError):
            doc.validate()

    def test_mask_dtype(self, doc):
        assert doc.entity_token_mask().dtype == np.bool_
