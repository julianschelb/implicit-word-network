"""
Integration tests that download and run real models.

Deselected by default (see ``addopts`` in pyproject.toml); run with::

    pytest -m integration
"""

import pytest

from implicit_word_network import (
    GLiNEREntityExtractor,
    ImplicitNetworkPipeline,
    load_example_corpus,
)

pytestmark = pytest.mark.integration


def test_gliner_v25_small_end_to_end():
    pytest.importorskip("gliner")
    extractor = GLiNEREntityExtractor(
        "gliner-community/gliner_small-v2.5",
        labels=["person", "organization", "award", "location"],
        threshold=0.4,
    )
    corpus = load_example_corpus()
    network = ImplicitNetworkPipeline(extractor, window=2).run(corpus.head(2))
    assert network.n_entities > 0 and network.n_edges > 0
    persons = [e for e in network.entities(label="person")]
    assert any("feynman" in e.norm for e in persons)
    assert all(0.4 <= m.score <= 1.0 for m in network.mentions_of(persons[0]))
