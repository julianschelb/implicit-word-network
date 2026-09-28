"""
Unit tests for implicit_word_network.document.
"""

import pytest

from implicit_word_network.document import Corpus, Document

# =================== DOCUMENT ===================


class TestDocument:
    def test_creation(self):
        doc = Document("Hello", id="a", meta={"k": 1})
        assert doc.text == "Hello"
        assert doc.id == "a"
        assert doc.meta == {"k": 1}
        assert len(doc) == 5

    def test_defaults(self):
        doc = Document("x")
        assert doc.id == 0
        assert doc.meta == {}

    def test_repr(self):
        assert "Document(id='a', len=5)" == repr(Document("Hello", "a"))

    def test_rejects_non_string(self):
        with pytest.raises(TypeError):
            Document(123)  # type: ignore[arg-type]


# =================== CORPUS ===================


class TestCorpusConstruction:
    def test_from_texts_assigns_sequential_ids(self):
        corpus = Corpus.from_texts(["a", "b"])
        assert corpus.ids() == [0, 1]
        assert corpus.texts() == ["a", "b"]
        assert len(corpus) == 2

    def test_from_texts_with_ids(self):
        corpus = Corpus.from_texts(["a", "b"], ids=["x", "y"])
        assert corpus.ids() == ["x", "y"]
        assert corpus.get("y").text == "b"

    def test_mixed_documents_and_strings(self):
        corpus = Corpus([Document("a", 5), "b"])
        assert corpus.ids() == [5, 1]

    def test_auto_id_skips_taken_ids(self):
        corpus = Corpus([Document("a", 1)])
        corpus.add("b")
        corpus.add("c")
        assert corpus.ids() == [1, 2, 3]

    def test_duplicate_id_raises(self):
        corpus = Corpus([Document("a", "x")])
        with pytest.raises(ValueError, match="Duplicate"):
            corpus.add(Document("b", "x"))

    def test_add_string_with_meta(self):
        corpus = Corpus()
        doc = corpus.add("text", doc_id="d", meta={"year": 1965})
        assert doc.id == "d"
        assert corpus.get("d").meta == {"year": 1965}

    def test_from_records(self):
        records = [{"text": " a ", "doc_id": "r1", "year": 1}, {"text": "b", "doc_id": "r2"}]
        corpus = Corpus.from_records(records, id_key="doc_id")
        assert corpus.ids() == ["r1", "r2"]
        assert corpus.get("r1").text == "a"
        assert corpus.get("r1").meta == {"year": 1}

    def test_from_records_missing_text(self):
        with pytest.raises(ValueError):
            Corpus.from_records([{"body": "x"}])

    def test_from_txt_lines(self, tmp_path):
        path = tmp_path / "docs.txt"
        path.write_text("first\n\nsecond\n   \nthird\n", encoding="utf-8")
        corpus = Corpus.from_txt(path)
        assert corpus.texts() == ["first", "second", "third"]

    def test_from_txt_paragraphs(self, tmp_path):
        path = tmp_path / "docs.txt"
        path.write_text("line one\nline two\n\nsecond doc", encoding="utf-8")
        corpus = Corpus.from_txt(path, delimiter="\n\n")
        assert corpus.texts() == ["line one\nline two", "second doc"]

    def test_from_csv(self, tmp_path):
        path = tmp_path / "docs.csv"
        path.write_text("doc_id,text,source\na,Hello world,wiki\nb,Bye,news\n", encoding="utf-8")
        corpus = Corpus.from_csv(path, id_column="doc_id")
        assert corpus.ids() == ["a", "b"]
        assert corpus.get("a").text == "Hello world"
        assert corpus.get("a").meta == {"source": "wiki"}

    def test_from_csv_default_ids_and_trailing_comma(self, tmp_path):
        path = tmp_path / "docs.csv"
        path.write_text('text,\n"Hello, world",\nBye,\n', encoding="utf-8")
        corpus = Corpus.from_csv(path)
        assert corpus.ids() == [0, 1]
        assert corpus[0].text == "Hello, world"
        assert corpus[0].meta == {}

    def test_from_tsv(self, tmp_path):
        path = tmp_path / "docs.tsv"
        path.write_text("text\tid\nHello\tx\n", encoding="utf-8")
        corpus = Corpus.from_csv(path, id_column="id", delimiter="\t")
        assert corpus.get("x").text == "Hello"

    def test_from_csv_missing_columns(self, tmp_path):
        path = tmp_path / "docs.csv"
        path.write_text("body\nx\n", encoding="utf-8")
        with pytest.raises(ValueError, match="no column 'text'"):
            Corpus.from_csv(path)
        path.write_text("text\nx\n", encoding="utf-8")
        with pytest.raises(ValueError, match="no column 'id'"):
            Corpus.from_csv(path, id_column="id")


class TestCorpusAccess:
    def test_iteration_and_indexing(self):
        corpus = Corpus.from_texts(["a", "b", "c"])
        assert [d.text for d in corpus] == ["a", "b", "c"]
        assert corpus[1].text == "b"
        assert 1 in corpus
        assert "missing" not in corpus

    def test_get_unknown(self):
        with pytest.raises(KeyError):
            Corpus().get("nope")

    def test_head(self):
        corpus = Corpus.from_texts(["a", "b", "c"])
        assert [d.text for d in corpus.head(2)] == ["a", "b"]
        assert corpus.head(0) == []

    def test_repr_and_meta(self):
        corpus = Corpus.from_texts(["a"], meta={"name": "demo"})
        assert "documents=1" in repr(corpus)
        assert corpus.meta == {"name": "demo"}
