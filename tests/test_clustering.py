"""
Unit tests for implicit_word_network.context (embedders, DBSCAN, edge clustering).
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from implicit_word_network import (
    BagOfWordsEmbedder,
    BaseContextEmbedder,
    ContextualEdgeClusterer,
    EdgeContextCluster,
    GazetteerEntityExtractor,
    ImplicitNetwork,
    NetworkConfig,
    SentenceTransformerEmbedder,
)
from implicit_word_network.context import cosine_distances, dbscan, euclidean_distances

# ============== Distances ==============


class TestDistances:
    def test_cosine(self):
        x = np.array([[1, 0], [0, 1], [2, 0], [0, 0]], dtype=float)
        d = cosine_distances(x)
        assert d.shape == (4, 4)
        assert d[0, 2] == pytest.approx(0.0)
        assert d[0, 1] == pytest.approx(1.0)
        assert d[0, 3] == pytest.approx(1.0)  # zero vector
        assert np.allclose(d, d.T) and np.all(np.diag(d) == 0)

    def test_euclidean(self):
        x = np.array([[0, 0], [3, 4]], dtype=float)
        d = euclidean_distances(x)
        assert d[0, 1] == pytest.approx(5.0) and d[1, 1] == 0.0


# ============== DBSCAN ==============


class TestDBSCAN:
    def test_two_clusters(self):
        points = np.array([[0, 0], [0.1, 0], [0, 0.1], [5, 5], [5.1, 5]])
        labels = dbscan(euclidean_distances(points), eps=0.5, min_samples=1)
        assert labels.tolist() == [0, 0, 0, 1, 1]

    def test_noise_and_border(self):
        # 0,1,2 dense core; 3 is a border point of 2; 4 is isolated noise
        d = np.array(
            [
                [0, 0.1, 0.1, 1.0, 1.0],
                [0.1, 0, 0.1, 1.0, 1.0],
                [0.1, 0.1, 0, 0.4, 1.0],
                [1.0, 1.0, 0.4, 0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 0],
            ]
        )
        labels = dbscan(d, eps=0.5, min_samples=3)
        assert labels.tolist() == [0, 0, 0, 0, -1]

    def test_all_noise(self):
        d = np.array([[0, 1], [1, 0]], dtype=float)
        assert dbscan(d, eps=0.5, min_samples=2).tolist() == [-1, -1]

    def test_labels_in_order_of_first_appearance(self):
        points = np.array([[9, 9], [0, 0], [9.1, 9], [0.1, 0]])
        labels = dbscan(euclidean_distances(points), eps=0.5, min_samples=1)
        assert labels.tolist() == [0, 1, 0, 1]

    def test_single_point_and_empty(self):
        assert dbscan(np.zeros((1, 1)), eps=0.1).tolist() == [0]
        assert dbscan(np.zeros((0, 0)), eps=0.1).shape == (0,)

    def test_validation(self):
        with pytest.raises(ValueError):
            dbscan(np.zeros((2, 3)), eps=0.1)
        with pytest.raises(ValueError):
            dbscan(np.zeros((2, 2)), eps=-1)
        with pytest.raises(ValueError):
            dbscan(np.zeros((2, 2)), eps=0.1, min_samples=0)


# ============== Embedders ==============


class TestBagOfWordsEmbedder:
    def test_shape_and_normalisation(self):
        embedder = BagOfWordsEmbedder(n_features=64)
        vectors = embedder.encode(
            ["Feynman received the Nobel Prize", "Feynman received the Nobel Prize", ""]
        )
        assert vectors.shape == (3, 64) and vectors.dtype == np.float32
        assert np.linalg.norm(vectors[0]) == pytest.approx(1.0)
        assert np.allclose(vectors[0], vectors[1])
        assert np.all(vectors[2] == 0)

    def test_stopwords_ignored(self):
        embedder = BagOfWordsEmbedder(n_features=64)
        assert np.allclose(embedder.encode(["the prize"])[0], embedder.encode(["prize"])[0])
        custom = BagOfWordsEmbedder(n_features=64, stopwords=["prize"])
        assert np.all(custom.encode(["prize"])[0] == 0)

    def test_similarity_reflects_overlap(self):
        embedder = BagOfWordsEmbedder(n_features=512)
        vectors = embedder.encode(
            ["nobel prize physics", "nobel prize chemistry", "space shuttle disaster"]
        )
        d = cosine_distances(vectors)
        assert d[0, 1] < d[0, 2]

    def test_validation(self):
        with pytest.raises(ValueError):
            BagOfWordsEmbedder(n_features=0)

    def test_abc(self):
        with pytest.raises(TypeError):
            BaseContextEmbedder()  # type: ignore[abstract]


class TestSentenceTransformerEmbedder:
    def test_uses_loaded_model(self):
        model = MagicMock()
        model.encode.return_value = np.ones((2, 4))
        embedder = SentenceTransformerEmbedder(model)
        vectors = embedder.encode(["a", "b"], batch_size=7)
        assert vectors.shape == (2, 4) and vectors.dtype == np.float32
        model.encode.assert_called_once()
        _, kwargs = model.encode.call_args
        assert (
            kwargs["batch_size"] == 7
            and kwargs["normalize_embeddings"]
            and kwargs["convert_to_numpy"]
        )

    def test_empty_input(self):
        assert SentenceTransformerEmbedder(MagicMock()).encode([]).shape == (0, 0)

    def test_missing_dependency(self):
        with patch("importlib.util.find_spec", return_value=None):
            with pytest.raises(ImportError, match=r"\[embeddings\]"):
                _ = SentenceTransformerEmbedder("some/model").model


# ============== ContextualEdgeClusterer ==============

CONTEXT_TEXTS = [
    "Feynman received the Nobel Prize for quantum electrodynamics. The Nobel Prize honoured Feynman.",
    "Feynman won the Nobel Prize award for physics research.",
    "Feynman served on the Rogers Commission investigating the shuttle. The Rogers Commission report cited Feynman.",
    "Feynman joined the Rogers Commission panel after the disaster.",
]


@pytest.fixture
def context_network():
    extractor = GazetteerEntityExtractor(
        {"PERSON": ["Feynman"], "ORG": ["Nobel Prize", "Rogers Commission"]}
    )
    return ImplicitNetwork.from_documents(
        extractor.annotate_all(CONTEXT_TEXTS), NetworkConfig(window=2)
    )


class TestContextualEdgeClusterer:
    def test_cluster_edge_groups_similar_contexts(self, context_network):
        clusterer = ContextualEdgeClusterer(BagOfWordsEmbedder(n_features=512), eps=0.6)
        clusters = clusterer.cluster_edge(
            context_network, ("Feynman", "PERSON"), ("Nobel Prize", "ORG")
        )
        edge_weight = context_network.weight(("Feynman", "PERSON"), ("Nobel Prize", "ORG"))
        edge_count = context_network.count(("Feynman", "PERSON"), ("Nobel Prize", "ORG"))
        assert sum(c.size for c in clusters) == edge_count
        assert sum(c.weight for c in clusters) == pytest.approx(edge_weight)
        assert all(isinstance(c, EdgeContextCluster) for c in clusters)
        assert all(len(c.contexts) == c.size for c in clusters)
        assert all(c.centroid is not None and c.centroid.shape == (512,) for c in clusters)
        assert [c.label for c in clusters] == list(range(len(clusters)))

    def test_no_centroids(self, context_network):
        clusterer = ContextualEdgeClusterer(eps=0.9, keep_centroids=False)
        clusters = clusterer.cluster_edge(
            context_network, ("Feynman", "PERSON"), ("Nobel Prize", "ORG")
        )
        assert all(c.centroid is None for c in clusters)

    def test_tight_eps_separates_every_context(self, context_network):
        clusterer = ContextualEdgeClusterer(eps=0.0)
        clusters = clusterer.cluster_edge(
            context_network, ("Feynman", "PERSON"), ("Rogers Commission", "ORG")
        )
        contexts = context_network.contexts(("Feynman", "PERSON"), ("Rogers Commission", "ORG"))
        assert len(clusters) == len(set(contexts))  # identical contexts share a cluster
        assert sum(c.size for c in clusters) == len(contexts)

    def test_single_cooccurrence_and_unconnected(self, extractor):
        network = ImplicitNetwork.from_documents(
            extractor.annotate_all(["Feynman met Tomonaga. Tokyo is far."]), NetworkConfig(window=0)
        )
        clusterer = ContextualEdgeClusterer()
        clusters = clusterer.cluster_edge(network, ("Feynman", "PERSON"), ("Tomonaga", "PERSON"))
        assert len(clusters) == 1 and clusters[0].size == 1 and clusters[0].label == 0
        assert clusterer.cluster_edge(network, ("Feynman", "PERSON"), ("Tokyo", "LOC")) == []

    def test_cluster_edges_batched_matches_single(self, context_network):
        embedder = BagOfWordsEmbedder(n_features=256)
        clusterer = ContextualEdgeClusterer(embedder, eps=0.6, batch_size=1)  # forces many flushes
        results = clusterer.cluster_edges(context_network, show_progress=True)
        assert set(results) == {(e.source.id, e.target.id) for e in context_network.edges()}
        for (a, b), clusters in results.items():
            single = ContextualEdgeClusterer(embedder, eps=0.6).cluster_edge(context_network, a, b)
            assert [c.size for c in clusters] == [c.size for c in single]
            assert [c.contexts for c in clusters] == [c.contexts for c in single]

    def test_cluster_edges_with_explicit_pairs(self, context_network):
        clusterer = ContextualEdgeClusterer(eps=0.5)
        pairs = [(("Rogers Commission", "ORG"), ("Feynman", "PERSON"))]
        results = clusterer.cluster_edges(context_network, pairs)
        feynman = context_network.entity("Feynman", "PERSON").id
        rogers = context_network.entity("Rogers Commission", "ORG").id
        assert list(results) == [(min(feynman, rogers), max(feynman, rogers))]

    def test_cluster_edges_filters(self, context_network):
        clusterer = ContextualEdgeClusterer()
        assert len(clusterer.cluster_edges(context_network, top_k=1)) == 1
        assert clusterer.cluster_edges(context_network, min_weight=1e9) == {}

    def test_euclidean_metric(self, context_network):
        clusterer = ContextualEdgeClusterer(metric="euclidean", eps=0.8)
        clusters = clusterer.cluster_edge(
            context_network, ("Feynman", "PERSON"), ("Nobel Prize", "ORG")
        )
        assert sum(c.size for c in clusters) == context_network.count(
            ("Feynman", "PERSON"), ("Nobel Prize", "ORG")
        )

    def test_noise_cluster(self, context_network):
        clusterer = ContextualEdgeClusterer(eps=0.0, min_samples=2)
        clusters = clusterer.cluster_edge(
            context_network, ("Feynman", "PERSON"), ("Nobel Prize", "ORG")
        )
        assert -1 in {c.label for c in clusters}
        assert sum(c.size for c in clusters) == context_network.count(
            ("Feynman", "PERSON"), ("Nobel Prize", "ORG")
        )

    def test_invalid_metric(self):
        with pytest.raises(ValueError):
            ContextualEdgeClusterer(metric="manhattan")  # type: ignore[arg-type]
