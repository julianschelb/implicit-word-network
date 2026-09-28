"""
Integration tests for the end-to-end pipeline (offline extractors and mocked GLiNER).
"""

import pytest

from implicit_word_network import (
    ContextualEdgeClusterer,
    Corpus,
    GazetteerEntityExtractor,
    GLiNEREntityExtractor,
    ImplicitNetwork,
    ImplicitNetworkPipeline,
    NetworkConfig,
    SpacyEntityExtractor,
    build_network,
)


class TestImplicitNetworkPipeline:
    def test_run_matches_manual_construction(self, extractor, corpus, annotated_docs):
        pipeline = ImplicitNetworkPipeline(extractor, window=2)
        network = pipeline.run(corpus)
        manual = ImplicitNetwork.from_documents(annotated_docs, NetworkConfig(window=2))
        assert repr(network) == repr(manual)
        assert abs(network.entity_entity_matrix - manual.entity_entity_matrix).sum() == 0

    def test_accepts_raw_strings_and_small_chunks(self, extractor, texts):
        pipeline = ImplicitNetworkPipeline(extractor, config=NetworkConfig(window=1))
        network = pipeline.run(texts, batch_size=1, chunk_size=1, show_progress=True)
        assert network.n_documents == 4 and network.config.window == 1

    def test_config_and_overrides_are_exclusive(self, extractor):
        with pytest.raises(ValueError):
            ImplicitNetworkPipeline(extractor, config=NetworkConfig(), window=3)

    def test_extends_existing_network(self, extractor, texts):
        pipeline = ImplicitNetworkPipeline(extractor)
        network = pipeline.run(texts[:2])
        same = pipeline.run(Corpus.from_texts(texts[2:], ids=["c", "d"]), network=network)
        assert same is network and network.n_documents == 4

    def test_annotate_only(self, extractor, texts):
        docs = list(ImplicitNetworkPipeline(extractor).annotate(texts))
        assert len(docs) == 4

    def test_cluster(self, extractor, texts):
        pipeline = ImplicitNetworkPipeline(extractor)
        network = pipeline.run(texts)
        with pytest.raises(RuntimeError):
            pipeline.cluster(network)
        pipeline.clusterer = ContextualEdgeClusterer()
        clusters = pipeline.cluster(network, top_k=2)
        assert len(clusters) == 2

    def test_build_network_helper(self, extractor, texts):
        network = build_network(texts, extractor=extractor, window=0)
        assert network.config.window == 0
        assert network.weight(("Feynman", "PERSON"), ("Nobel Prize", "ORG")) == pytest.approx(1.0)

    def test_gliner_end_to_end(self, mock_gliner_model, texts):
        extractor = GLiNEREntityExtractor(
            mock_gliner_model,
            labels=["person", "organization", "location"],
            label_map={"person": "PERSON", "organization": "ORG", "location": "LOC"},
        )
        network = ImplicitNetworkPipeline(extractor, window=2).run(texts)
        gazetteer = build_network(
            texts,
            extractor=GazetteerEntityExtractor(
                {
                    "PERSON": ["Richard Feynman", "Feynman", "Schwinger", "Tomonaga"],
                    "ORG": ["Caltech", "Nobel Prize", "Harvard", "Columbia", "Rogers Commission"],
                    "LOC": ["Tokyo"],
                }
            ),
            window=2,
        )
        assert {e.key for e in network.entities()} == {e.key for e in gazetteer.entities()}
        assert network.weight(("Feynman", "PERSON"), ("Schwinger", "PERSON")) == pytest.approx(
            gazetteer.weight(("Feynman", "PERSON"), ("Schwinger", "PERSON"))
        )

    @pytest.mark.spacy
    def test_default_extractor_is_spacy(self, spacy_nlp, texts):
        pipeline = ImplicitNetworkPipeline(window=1)
        assert isinstance(pipeline.extractor, SpacyEntityExtractor)
        pipeline = ImplicitNetworkPipeline(SpacyEntityExtractor(spacy_nlp), window=1)
        network = pipeline.run(texts)
        assert network.n_entities > 0 and network.n_edges > 0
        top = network.top_entities(1)[0]
        assert network.neighbors(top)
        assert {e.label for e in network.entities()} <= {
            "PERSON",
            "ORG",
            "GPE",
            "NORP",
            "LOC",
            "WORK_OF_ART",
        }
