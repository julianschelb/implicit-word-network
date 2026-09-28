"""
Unit tests for implicit_word_network.segmentation.
"""

import pytest

from implicit_word_network.segmentation import (
    ENGLISH_STOPWORDS,
    BaseSegmenter,
    RegexSegmenter,
    Segmentation,
    SpacySegmenter,
    spacy_doc_to_segmentation,
)

# ============== ABC contract ==============


class TestBaseSegmenter:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BaseSegmenter()  # type: ignore[abstract]

    def test_segment_many_default(self):
        class Dummy(BaseSegmenter):
            def segment(self, text):
                return Segmentation([], [])

        assert list(Dummy().segment_many(["a", "b"])) == [
            Segmentation([], []),
            Segmentation([], []),
        ]


# ============== RegexSegmenter ==============


class TestRegexSegmenter:
    @pytest.fixture
    def segmenter(self):
        return RegexSegmenter()

    def test_sentence_splitting(self, segmenter):
        text = 'He said "Stop!" Then he left. Really?\nNew line here'
        sentences, _ = segmenter.segment(text)
        assert [text[s.start : s.end] for s in sentences] == [
            'He said "Stop!"',
            "Then he left.",
            "Really?",
            "New line here",
        ]
        assert [s.index for s in sentences] == [0, 1, 2, 3]

    def test_whitespace_trimmed(self, segmenter):
        sentences, tokens = segmenter.segment("   Hello world.   ")
        assert len(sentences) == 1
        assert sentences[0].start == 3 and sentences[0].end == 15
        assert [t.text for t in tokens] == ["Hello", "world", "."]

    def test_empty_and_blank_text(self, segmenter):
        assert segmenter.segment("") == Segmentation([], [])
        assert segmenter.segment("  \n ") == Segmentation([], [])

    def test_token_offsets_match_text(self, segmenter):
        text = "Feynman's diagrams (1948) changed physics. Truly."
        sentences, tokens = segmenter.segment(text)
        for token in tokens:
            assert text[token.start : token.end] == token.text
            assert sentences[token.sentence].start <= token.start < sentences[token.sentence].end
        assert [t.text for t in tokens][:4] == ["Feynman's", "diagrams", "(", "1948"]

    def test_flags(self, segmenter):
        _, tokens = segmenter.segment("The cat sat.")
        by_text = {t.text: t for t in tokens}
        assert by_text["The"].is_stop and not by_text["cat"].is_stop
        assert by_text["."].is_punct and not by_text["cat"].is_punct
        assert all(not t.is_space for t in tokens)
        assert all(t.pos == "" and t.lemma == "" for t in tokens)

    def test_custom_stopwords(self):
        segmenter = RegexSegmenter(stopwords=["cat"])
        _, tokens = segmenter.segment("The cat sat.")
        by_text = {t.text: t for t in tokens}
        assert by_text["cat"].is_stop and not by_text["The"].is_stop
        assert RegexSegmenter(stopwords=[]).stopwords == frozenset()

    def test_custom_patterns(self):
        segmenter = RegexSegmenter(sentence_boundary=r";\s*", token_pattern=r"\S+")
        sentences, tokens = segmenter.segment("a b; c d")
        assert len(sentences) == 2
        assert [t.text for t in tokens] == ["a", "b", "c", "d"]

    def test_segment_many(self, segmenter):
        results = list(segmenter.segment_many(["One. Two.", "Three."]))
        assert [len(r.sentences) for r in results] == [2, 1]

    def test_default_stopwords_are_lowercase(self):
        assert all(w == w.lower() for w in ENGLISH_STOPWORDS)
        assert "the" in ENGLISH_STOPWORDS


# ============== SpacySegmenter ==============


@pytest.mark.spacy
class TestSpacySegmenter:
    def test_segment_provides_linguistic_features(self, spacy_nlp):
        segmenter = SpacySegmenter(spacy_nlp)
        text = "Feynman taught physics at Caltech. He loved bongos."
        sentences, tokens = segmenter.segment(text)
        assert len(sentences) == 2
        for token in tokens:
            assert text[token.start : token.end] == token.text
        by_text = {t.text: t for t in tokens}
        assert by_text["taught"].lemma == "teach"
        assert by_text["taught"].pos == "VERB"
        assert by_text["He"].is_stop
        assert by_text["."].is_punct

    def test_token_indices_match_spacy(self, spacy_nlp):
        doc = spacy_nlp("Feynman taught physics. He loved bongos.")
        segmentation = spacy_doc_to_segmentation(doc)
        assert len(segmentation.tokens) == len(doc)
        assert [t.text for t in segmentation.tokens] == [t.text for t in doc]

    def test_segment_many(self, spacy_nlp):
        segmenter = SpacySegmenter(spacy_nlp)
        results = list(segmenter.segment_many(["One. Two.", "Three."]))
        assert [len(r.sentences) for r in results] == [2, 1]

    def test_lazy_loading_by_name(self):
        pytest.importorskip("spacy")
        segmenter = SpacySegmenter("en_core_web_sm")
        assert segmenter._nlp is None
        try:
            nlp = segmenter.nlp
        except OSError:  # pragma: no cover
            pytest.skip("en_core_web_sm not installed")
        assert nlp.has_pipe("parser") or nlp.has_pipe("senter") or nlp.has_pipe("sentencizer")
        assert not nlp.has_pipe("ner")


@pytest.mark.spacy
class TestSpacyMaxLength:
    def test_max_length_is_raised(self):
        pytest.importorskip("spacy")
        from implicit_word_network.segmentation.spacy import load_spacy_model

        try:
            nlp = load_spacy_model("en_core_web_sm", disable=("ner",), max_length=5_000_000)
        except OSError:  # pragma: no cover
            pytest.skip("en_core_web_sm not installed")
        assert nlp.max_length >= 5_000_000
        segmenter = SpacySegmenter("en_core_web_sm", max_length=2_000_000)
        assert segmenter.max_length == 2_000_000
