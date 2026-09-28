"""
Unit tests for implicit_word_network.extraction.

- BaseEntityExtractor / SpanEntityExtractor (ABC contract)
- GazetteerEntityExtractor (offline)
- GLiNEREntityExtractor (mocked model)
- SpacyEntityExtractor (requires spaCy + model)
"""

import re
from unittest.mock import patch

import pytest

from implicit_word_network import Corpus, Document
from implicit_word_network.annotation import EntitySpan
from implicit_word_network.extraction import (
    DEFAULT_GLINER_LABELS,
    BaseEntityExtractor,
    GazetteerEntityExtractor,
    GLiNEREntityExtractor,
    SpacyEntityExtractor,
    SpanEntityExtractor,
    as_documents,
)
from implicit_word_network.segmentation import RegexSegmenter
from tests.conftest import fake_gliner_inference

# ============== ABC contract ==============


class TestBaseClasses:
    def test_cannot_instantiate_abcs(self):
        with pytest.raises(TypeError):
            BaseEntityExtractor()  # type: ignore[abstract]
        with pytest.raises(TypeError):
            SpanEntityExtractor()  # type: ignore[abstract]

    def test_custom_span_extractor(self):
        class UpperCaseExtractor(SpanEntityExtractor):
            def extract_spans(self, texts, sentences):
                return [
                    [
                        EntitySpan(m.start(), m.end(), "ACRONYM")
                        for m in re.finditer(r"\b[A-Z]{2,}\b", t)
                    ]
                    for t in texts
                ]

        doc = UpperCaseExtractor().annotate_text("NASA and ESA cooperate. Not IBM.", doc_id="x")
        assert doc.id == "x"
        assert [(m.text, m.label, m.sentence) for m in doc.mentions] == [
            ("NASA", "ACRONYM", 0),
            ("ESA", "ACRONYM", 0),
            ("IBM", "ACRONYM", 1),
        ]
        assert isinstance(UpperCaseExtractor().segmenter, RegexSegmenter)

    def test_span_count_mismatch_raises(self):
        class Broken(SpanEntityExtractor):
            def extract_spans(self, texts, sentences):
                return []

        with pytest.raises(RuntimeError):
            Broken().annotate_text("a b.")

    def test_annotate_all_and_batches(self, gazetteer, texts):
        extractor = GazetteerEntityExtractor(gazetteer)
        one = extractor.annotate_all(texts, batch_size=1)
        many = extractor.annotate_all(texts, batch_size=10, show_progress=True)
        assert [d.mentions for d in one] == [d.mentions for d in many]
        assert [d.id for d in one] == [0, 1, 2, 3]


class TestAsDocuments:
    def test_strings_get_sequential_ids(self):
        docs = list(as_documents(["a", Document("b", "x"), "c"]))
        assert [d.id for d in docs] == [0, "x", 2]

    def test_rejects_other_types(self):
        with pytest.raises(TypeError):
            list(as_documents([1]))  # type: ignore[list-item]

    def test_corpus_passthrough(self):
        corpus = Corpus.from_texts(["a"], ids=["k"])
        assert [d.id for d in as_documents(corpus)] == ["k"]


# ============== GazetteerEntityExtractor ==============


class TestGazetteerEntityExtractor:
    def test_matches_with_labels(self, extractor):
        doc = extractor.annotate_text("Feynman worked at Caltech. Tomonaga lived in Tokyo.")
        assert [(m.text, m.label, m.sentence) for m in doc.mentions] == [
            ("Feynman", "PERSON", 0),
            ("Caltech", "ORG", 0),
            ("Tomonaga", "PERSON", 1),
            ("Tokyo", "LOC", 1),
        ]
        assert all(m.score == 1.0 for m in doc.mentions)

    def test_longest_match_wins(self, extractor):
        doc = extractor.annotate_text("Richard Feynman was here.")
        assert [m.text for m in doc.mentions] == ["Richard Feynman"]

    def test_word_boundaries(self, extractor):
        doc = extractor.annotate_text("Feynmanish ideas; feynman.")
        assert [m.text for m in doc.mentions] == ["feynman"]

    def test_case_insensitive_by_default(self, gazetteer):
        doc = GazetteerEntityExtractor(gazetteer).annotate_text("FEYNMAN and caltech")
        assert [(m.text, m.label) for m in doc.mentions] == [
            ("FEYNMAN", "PERSON"),
            ("caltech", "ORG"),
        ]

    def test_case_sensitive(self, gazetteer):
        doc = GazetteerEntityExtractor(gazetteer, case_sensitive=True).annotate_text(
            "FEYNMAN and Caltech"
        )
        assert [m.text for m in doc.mentions] == ["Caltech"]

    def test_empty_gazetteer_raises(self):
        with pytest.raises(ValueError):
            GazetteerEntityExtractor({"X": ["", "  "]})

    def test_labels_property(self, extractor):
        assert extractor.labels == {"PERSON", "ORG", "LOC"}

    def test_documents_without_entities(self, extractor):
        doc = extractor.annotate_text("Nothing here.")
        assert doc.mentions == [] and doc.n_sentences == 1

    def test_metadata_is_copied(self, extractor):
        doc = extractor.annotate_all([Document("Feynman.", "d", {"k": 1})])[0]
        assert doc.meta == {"k": 1}


# ============== GLiNEREntityExtractor (mocked) ==============


class TestGLiNEREntityExtractor:
    def test_extracts_with_mock_model(self, mock_gliner_model):
        extractor = GLiNEREntityExtractor(mock_gliner_model, labels=["person", "organization"])
        doc = extractor.annotate_text("Feynman worked at Caltech. Schwinger was at Harvard.")
        assert [(m.text, m.label, m.sentence) for m in doc.mentions] == [
            ("Feynman", "person", 0),
            ("Caltech", "organization", 0),
            ("Schwinger", "person", 1),
            ("Harvard", "organization", 1),
        ]
        assert doc.mentions[0].score == pytest.approx(0.9)
        mock_gliner_model.inference.assert_called_once()
        args, kwargs = mock_gliner_model.inference.call_args
        assert args[1] == ["person", "organization"]
        assert kwargs["threshold"] == 0.5 and kwargs["flat_ner"] and not kwargs["multi_label"]
        assert kwargs["batch_size"] == 8

    def test_threshold_and_label_map(self, mock_gliner_model):
        extractor = GLiNEREntityExtractor(
            mock_gliner_model,
            labels=["person"],
            threshold=0.8,
            label_map={"person": "PERSON"},
        )
        doc = extractor.annotate_text("Feynman and Schwinger.")
        # "Feynman" scores 0.9 (len 7 > 6); "Schwinger" 0.9 too -> both kept; short names dropped
        assert [(m.text, m.label) for m in doc.mentions] == [
            ("Feynman", "PERSON"),
            ("Schwinger", "PERSON"),
        ]

    def test_chunking_preserves_offsets(self, mock_gliner_model):
        sentences = [f"Sentence {i} mentions Feynman here." for i in range(12)]
        text = " ".join(sentences)
        extractor = GLiNEREntityExtractor(mock_gliner_model, labels=["person"], max_chunk_words=12)
        doc = extractor.annotate_text(text)
        assert len(doc.mentions) == 12
        for mention in doc.mentions:
            assert text[mention.start : mention.end] == "Feynman"
        assert [m.sentence for m in doc.mentions] == list(range(12))
        # several chunks were sent in one batched call
        (chunks, _), _ = mock_gliner_model.inference.call_args
        assert len(chunks) > 1
        assert all(len(c.split()) <= 12 for c in chunks)

    def test_long_sentence_is_split_by_words(self, mock_gliner_model):
        text = "Feynman " + "word " * 30 + "Schwinger"
        extractor = GLiNEREntityExtractor(mock_gliner_model, labels=["person"], max_chunk_words=10)
        chunks = extractor._chunks(text, RegexSegmenter().segment(text).sentences)
        assert len(chunks) > 1
        assert [text[s:e] for s, e in chunks] and "".join(text[s:e] for s, e in chunks).replace(
            " ", ""
        ) == text.replace(" ", "")
        doc = extractor.annotate_text(text)
        assert [m.text for m in doc.mentions] == ["Feynman", "Schwinger"]

    def test_batches_documents(self, mock_gliner_model, texts):
        extractor = GLiNEREntityExtractor(mock_gliner_model, labels=["person", "organization"])
        docs = extractor.annotate_all(texts, batch_size=2)
        assert len(docs) == 4
        assert mock_gliner_model.inference.call_count == 2
        assert docs[3].mentions == []

    def test_fallback_to_batch_predict_entities(self):
        class OldModel:
            inference = None  # not callable

            def batch_predict_entities(self, texts, labels, **kwargs):
                return fake_gliner_inference(texts, labels, **kwargs)

        extractor = GLiNEREntityExtractor(OldModel(), labels=["person"])
        doc = extractor.annotate_text("Feynman was here.")
        assert [m.text for m in doc.mentions] == ["Feynman"]

    def test_result_count_mismatch(self, mock_gliner_model):
        mock_gliner_model.inference.side_effect = lambda texts, labels, **kw: []
        with pytest.raises(RuntimeError):
            GLiNEREntityExtractor(mock_gliner_model).annotate_text("Feynman.")

    def test_empty_document(self, mock_gliner_model):
        doc = GLiNEREntityExtractor(mock_gliner_model).annotate_text("   ")
        assert doc.mentions == [] and doc.n_sentences == 0
        mock_gliner_model.inference.assert_not_called()

    def test_validation(self, mock_gliner_model):
        with pytest.raises(ValueError):
            GLiNEREntityExtractor(mock_gliner_model, labels=[])
        with pytest.raises(ValueError):
            GLiNEREntityExtractor(mock_gliner_model, max_chunk_words=0)

    def test_defaults(self):
        extractor = GLiNEREntityExtractor()
        assert extractor.labels == list(DEFAULT_GLINER_LABELS)
        assert extractor._model is None

    def test_missing_dependency_message(self):
        extractor = GLiNEREntityExtractor("some/model")
        with patch("importlib.util.find_spec", return_value=None):
            with pytest.raises(ImportError, match=r"implicit-word-network\[gliner\]"):
                _ = extractor.model

    def test_lazy_load_calls_from_pretrained(self, mock_gliner_model):
        extractor = GLiNEREntityExtractor(
            "some/model", device="cpu", load_kwargs={"max_length": 384}
        )
        fake_module = type("gliner", (), {})()
        fake_gliner_class = type(
            "GLiNER", (), {"from_pretrained": staticmethod(lambda *a, **k: (a, k))}
        )
        fake_module.GLiNER = fake_gliner_class
        with patch("importlib.util.find_spec", return_value=object()):
            with patch.dict("sys.modules", {"gliner": fake_module}):
                args, kwargs = extractor.model
        assert args == ("some/model",)
        assert kwargs == {"map_location": "cpu", "max_length": 384}


# ============== SpacyEntityExtractor ==============


@pytest.mark.spacy
class TestSpacyEntityExtractor:
    def test_extracts_default_labels(self, spacy_nlp):
        extractor = SpacyEntityExtractor(spacy_nlp)
        doc = extractor.annotate_text(
            "Richard Feynman received the Nobel Prize in 1965. He worked at Caltech in Pasadena."
        )
        labels = {m.label for m in doc.mentions}
        assert labels <= {"PERSON", "ORG", "GPE", "NORP", "LOC", "WORK_OF_ART"}
        assert "PERSON" in labels
        for mention in doc.mentions:
            assert doc.text[mention.start : mention.end] == mention.text
            covered = [t.text for t in doc.tokens[mention.token_start : mention.token_end]]
            assert " ".join(covered).replace(" ", "") == mention.text.replace(" ", "")
            assert doc.sentences[mention.sentence].start <= mention.start

    def test_labels_none_keeps_everything(self, spacy_nlp):
        extractor = SpacyEntityExtractor(spacy_nlp, labels=None)
        doc = extractor.annotate_text("Feynman received the prize in 1965.")
        assert any(m.label == "DATE" for m in doc.mentions)

    def test_batch_annotation(self, spacy_nlp, texts):
        docs = SpacyEntityExtractor(spacy_nlp).annotate_all(texts, batch_size=2)
        assert [d.id for d in docs] == [0, 1, 2, 3]
        assert all(d.n_sentences > 0 for d in docs)


@pytest.mark.spacy
def test_spacy_extractor_max_length():
    pytest.importorskip("spacy")
    extractor = SpacyEntityExtractor("en_core_web_sm", max_length=3_000_000)
    try:
        nlp = extractor.nlp
    except OSError:  # pragma: no cover
        pytest.skip("en_core_web_sm not installed")
    assert nlp.max_length >= 3_000_000
