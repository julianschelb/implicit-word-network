"""
Tests for bundled example data.
"""

from implicit_word_network import load_example_corpus


def test_load_example_corpus():
    corpus = load_example_corpus()
    assert len(corpus) == 6
    assert corpus.ids() == list(range(6))
    assert all(len(doc.text) > 100 for doc in corpus)
    assert "Feynman" in corpus[0].text
