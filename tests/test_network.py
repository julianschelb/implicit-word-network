"""
Unit tests for implicit_word_network.network.ImplicitNetwork.

The vectorised construction is validated against naive pure-Python
reference implementations (tests/helpers.py).
"""

import math

import numpy as np
import pytest
from scipy import sparse

from implicit_word_network import (
    AnnotatedDocument,
    Document,
    EntityMention,
    GazetteerEntityExtractor,
    ImplicitNetwork,
    NetworkConfig,
    Sentence,
    Token,
)
from implicit_word_network.network import EntityNode, register_decay
from tests.helpers import default_norm, naive_edge_weights, naive_entity_terms


def weights_by_key(network):
    """Map ((norm, label), (norm, label)) -> (weight, count) from the network matrices."""
    result = {}
    for edge in network.edges():
        pair = (min(edge.source.key, edge.target.key), max(edge.source.key, edge.target.key))
        result[pair] = (edge.weight, edge.count)
    return result


# ============== Configuration ==============


class TestNetworkConfig:
    def test_defaults(self):
        config = NetworkConfig()
        assert config.window == 2 and config.decay == "exponential"
        assert not config.include_stopwords and config.store_text

    def test_validation(self):
        with pytest.raises(ValueError):
            NetworkConfig(window=-1)

    def test_term_pos_coerced(self):
        config = NetworkConfig(term_pos={"NOUN", "VERB"})
        assert isinstance(config.term_pos, frozenset)

    def test_roundtrip(self):
        config = NetworkConfig(window=3, decay="constant", term_pos=["NOUN"], use_lemma=False)
        assert NetworkConfig.from_dict(config.to_dict()) == config
        assert NetworkConfig.from_dict(NetworkConfig().to_dict()) == NetworkConfig()

    def test_unknown_decay_fails_at_construction(self):
        with pytest.raises(KeyError):
            ImplicitNetwork(NetworkConfig(decay="nope"))


# ============== Construction & correctness ==============


class TestConstruction:
    def test_sizes(self, network, annotated_docs):
        assert network.n_documents == 4
        assert network.n_sentences == sum(d.n_sentences for d in annotated_docs)
        assert network.n_mentions == sum(len(d.mentions) for d in annotated_docs)
        assert network.n_entities == 10
        assert network.n_terms > 0
        assert network.n_edges == len(network.edges())
        assert "documents=4" in repr(network)
        assert "entities  : 10" in network.summary()

    @pytest.mark.parametrize("window", [0, 1, 2, 3, 10])
    @pytest.mark.parametrize("decay", ["exponential", "constant", "inverse", "linear"])
    def test_edge_weights_match_reference(self, annotated_docs, window, decay):
        network = ImplicitNetwork.from_documents(
            annotated_docs, NetworkConfig(window=window, decay=decay)
        )
        decay_fn = {
            "exponential": lambda d: math.exp(-d),
            "constant": lambda d: 1.0,
            "inverse": lambda d: 1 / (1 + d),
            "linear": lambda d: 1 - d / (window + 1),
        }[decay]
        expected = naive_edge_weights(annotated_docs, window=window, decay=decay_fn)
        actual = weights_by_key(network)
        assert set(actual) == set(expected)
        for pair, (weight, count) in expected.items():
            assert actual[pair][0] == pytest.approx(weight)
            assert actual[pair][1] == count

    def test_matrices_are_symmetric_with_zero_diagonal(self, network):
        W = network.entity_entity_matrix
        C = network.entity_count_matrix
        assert abs(W - W.T).sum() == 0
        assert abs(C - C.T).sum() == 0
        assert W.diagonal().sum() == 0 and C.diagonal().sum() == 0
        assert sparse.triu(W, k=1).sum() == pytest.approx(network.all_cooccurrences().weight.sum())
        assert sparse.triu(C, k=1).sum() == len(network.all_cooccurrences())

    def test_entity_terms_match_reference(self, network, annotated_docs):
        expected = naive_entity_terms(annotated_docs)
        actual = {}
        for entity in network.entities():
            for term, count in network.entity_terms(entity):
                actual[(entity.key, (term.text, term.pos))] = count
        assert actual == expected
        assert network.entity_term_matrix.sum() == sum(expected.values())

    def test_entity_counts_and_mentions(self, network):
        feynman = network.entity("Feynman", "PERSON")
        assert feynman is not None
        assert feynman.count == 2  # "Richard Feynman" is a distinct entity
        assert network.entity_counts()[feynman.id] == 2
        mentions = network.mentions_of(feynman)
        assert len(mentions) == 2
        assert all(m.entity == feynman for m in mentions)
        doc0 = network.document(0)
        assert all(m.sentence.document == doc0.id for m in mentions)
        assert [m.sentence.index for m in mentions] == [1, 2]

    def test_entity_lookup_normalises(self, network):
        assert network.entity(" FEYNMAN ", "PERSON") == network.entity("feynman", "PERSON")
        assert network.entity("feynman", "ORG") is None
        assert network.entity_by_id(0).id == 0
        with pytest.raises(KeyError):
            network.entity_by_id(999)
        with pytest.raises(KeyError):
            network.weight(("unknown", "PERSON"), 0)

    def test_entities_and_labels(self, network):
        assert set(network.entity_labels()) == {"PERSON", "ORG", "LOC"}
        persons = network.entities(label="PERSON")
        assert {e.text for e in persons} == {"Richard Feynman", "Feynman", "Schwinger", "Tomonaga"}
        top = network.top_entities(2)
        assert len(top) == 2 and top[0].count >= top[1].count
        assert network.top_entities(1, label="LOC")[0].label == "LOC"
        assert isinstance(top[0], EntityNode)

    def test_terms(self, network):
        terms = network.terms()
        assert all(t.count >= 1 for t in terms)
        assert network.term("worked") is not None
        assert network.term("the") is None  # stop word
        assert network.term("nope") is None
        top = network.top_terms(3)
        assert top[0].count >= top[-1].count
        assert network.term_by_id(top[0].id) == top[0]
        with pytest.raises(KeyError):
            network.term_by_id(-1)
        entities = network.term_entities(network.term("worked"))
        assert {e.text for e, _ in entities} >= {"Richard Feynman", "Caltech"}

    def test_term_counts_match_sentence_term_matrix(self, network):
        counts = network.term_counts()
        assert counts.sum() == network.sentence_term_matrix.sum()
        assert network.entity_counts().sum() == network.sentence_entity_matrix.sum()


# ============== Term filters ==============


def make_doc(doc_id, tokens_spec, mentions_spec, text=None):
    """Build an AnnotatedDocument from (text, sentence, pos, lemma, is_stop, is_punct) tuples."""
    tokens = []
    offset = 0
    for spec in tokens_spec:
        surface, sentence, pos, lemma, is_stop, is_punct = spec
        tokens.append(
            Token(
                surface,
                offset,
                offset + len(surface),
                sentence,
                is_stop=is_stop,
                is_punct=is_punct,
                pos=pos,
                lemma=lemma,
            )
        )
        offset += len(surface) + 1
    n_sent = max(t.sentence for t in tokens) + 1
    sentences = []
    for s in range(n_sent):
        sent_tokens = [t for t in tokens if t.sentence == s]
        sentences.append(Sentence(s, sent_tokens[0].start, sent_tokens[-1].end))
    full_text = text or " ".join(t.text for t in tokens)
    mentions = []
    for surface, label, sentence in mentions_spec:
        token = next(t for t in tokens if t.text == surface and t.sentence == sentence)
        mentions.append(
            EntityMention(
                surface,
                label,
                token.start,
                token.end,
                sentence,
                1.0,
                tokens.index(token),
                tokens.index(token) + 1,
            )
        )
    return AnnotatedDocument(doc_id, full_text, sentences, tokens, mentions)


class TestTermFilters:
    @pytest.fixture
    def doc(self):
        return make_doc(
            "d",
            [
                ("Feynman", 0, "PROPN", "Feynman", False, False),
                ("taught", 0, "VERB", "teach", False, False),
                ("the", 0, "DET", "the", True, False),
                ("Students", 0, "NOUN", "student", False, False),
                (".", 0, "PUNCT", ".", False, True),
            ],
            [("Feynman", "PERSON", 0)],
        )

    def terms(self, doc, **kwargs):
        network = ImplicitNetwork.from_documents([doc], NetworkConfig(**kwargs))
        return {(t.text, t.pos) for t in network.terms()}

    def test_defaults_use_lemma_and_drop_stop_punct_entities(self, doc):
        assert self.terms(doc) == {("teach", "VERB"), ("student", "NOUN")}

    def test_include_stopwords_and_punctuation(self, doc):
        assert self.terms(doc, include_stopwords=True, include_punctuation=True) == {
            ("teach", "VERB"),
            ("student", "NOUN"),
            ("the", "DET"),
            (".", "PUNCT"),
        }

    def test_surface_forms(self, doc):
        assert self.terms(doc, use_lemma=False) == {("taught", "VERB"), ("students", "NOUN")}
        assert self.terms(doc, use_lemma=False, lowercase_terms=False) == {
            ("taught", "VERB"),
            ("Students", "NOUN"),
        }

    def test_pos_filter(self, doc):
        assert self.terms(doc, term_pos={"NOUN"}) == {("student", "NOUN")}

    def test_space_tokens_ignored(self):
        doc = make_doc("d", [("a", 0, "X", "", False, False)], [])
        doc.tokens.append(Token(" ", 1, 2, 0, is_space=True))
        network = ImplicitNetwork.from_documents([doc])
        assert {t.text for t in network.terms()} == {"a"}


# ============== Edges, neighbours, cooccurrences ==============


class TestEdges:
    def test_edges_sorted_and_filtered(self, network):
        edges = network.edges()
        weights = [e.weight for e in edges]
        assert weights == sorted(weights, reverse=True)
        assert all(e.source.id < e.target.id for e in edges)
        heavy = network.edges(min_weight=1.0)
        assert all(e.weight >= 1.0 for e in heavy) and len(heavy) < len(edges)
        assert len(network.edges(top_k=2)) == 2
        assert network.edges(top_k=0) == []
        persons = network.edges(labels=["PERSON"])
        assert persons and all(e.source.label == e.target.label == "PERSON" for e in persons)

    def test_weight_and_count(self, network):
        feynman = network.entity("Feynman", "PERSON")
        prize = network.entity("Nobel Prize", "ORG")
        assert network.weight(feynman, prize) == pytest.approx(1.0 + math.exp(-1))  # δ=0 and δ=1
        assert network.count(feynman, prize) == 2
        assert network.weight(feynman, feynman) == 0.0
        assert network.weight(("Tokyo", "LOC"), ("Harvard", "ORG")) == 0.0
        assert network.weight(feynman.id, prize.id) == network.weight(prize, feynman)

    def test_neighbors(self, network):
        feynman = network.entity("Feynman", "PERSON")
        neighbors = network.neighbors(feynman)
        assert neighbors and neighbors[0][1] >= neighbors[-1][1]
        assert all(w == network.weight(feynman, node) for node, w in neighbors)
        assert len(network.neighbors(feynman, k=1)) == 1
        assert all(w >= 1.0 for _, w in network.neighbors(feynman, min_weight=1.0))
        assert network.neighbors(("Tokyo", "LOC"))[0][0].text == "Tomonaga"

    def test_cooccurrences_consistent_with_matrices(self, network):
        for edge in network.edges():
            coocs = network.cooccurrences(edge.source, edge.target)
            assert len(coocs) == edge.count
            assert sum(c.weight for c in coocs) == pytest.approx(edge.weight)
            for c in coocs:
                assert {c.source.entity.id, c.target.entity.id} == {edge.source.id, edge.target.id}
                assert c.delta == abs(c.source.sentence.id - c.target.sentence.id) <= network.window
                assert c.weight == pytest.approx(math.exp(-c.delta))
                assert c.document == c.source.sentence.document == c.target.sentence.document
                assert c.source.id < c.target.id

    def test_cooccurrences_order_independent(self, network):
        a, b = ("Feynman", "PERSON"), ("Schwinger", "PERSON")
        assert network.cooccurrences(a, b) == network.cooccurrences(b, a)
        assert network.cooccurrences(("Tokyo", "LOC"), ("Harvard", "ORG")) == []

    def test_cooccurrences_same_entity_raises(self, network):
        with pytest.raises(ValueError):
            network.cooccurrences(0, 0)

    def test_contexts(self, network):
        feynman, prize = ("Feynman", "PERSON"), ("Nobel Prize", "ORG")
        contexts = network.contexts(feynman, prize)
        coocs = network.cooccurrences(feynman, prize)
        assert len(contexts) == len(coocs) == 2
        for cooc, context in zip(coocs, contexts):
            first, last = cooc.sentence_span
            assert context == " ".join(network.sentence(i).text for i in range(first, last + 1))
            assert "Nobel Prize" in context and "Feynman" in context
            assert network.context_of(cooc) == context

    def test_contexts_require_stored_text(self, annotated_docs):
        network = ImplicitNetwork.from_documents(annotated_docs, NetworkConfig(store_text=False))
        assert network.sentence(0).text == ""
        with pytest.raises(RuntimeError):
            network.contexts(("Feynman", "PERSON"), ("Nobel Prize", "ORG"))

    def test_all_cooccurrences_table(self, network):
        table = network.all_cooccurrences()
        assert len(table) == sparse.triu(network.entity_count_matrix, k=1).sum()
        assert np.all(table.delta <= network.window)
        assert np.all(network._m_entity.view[table.source] != network._m_entity.view[table.target])
        assert np.allclose(table.weight, np.exp(-table.delta))


# ============== Structure nodes ==============


class TestStructureNodes:
    def test_documents(self, network, annotated_docs):
        docs = network.documents()
        assert [d.id for d in docs] == [0, 1, 2, 3]
        assert docs[0].n_sentences == annotated_docs[0].n_sentences
        assert network.document_by_id(2).index == 2
        with pytest.raises(KeyError):
            network.document(9)
        with pytest.raises(KeyError):
            network.document_by_id("nope")

    def test_sentences(self, network, annotated_docs):
        assert network.sentence(0).text == annotated_docs[0].sentence_text(0)
        by_doc = network.sentences_of_document(1)
        assert [s.index for s in by_doc] == list(range(annotated_docs[1].n_sentences))
        assert [s.text for s in by_doc] == [
            annotated_docs[1].sentence_text(i) for i in range(len(by_doc))
        ]
        assert network.sentences_of_document(1) == network.sentences_of_document(
            network.document(1).id
        )
        assert network.sentence_document().tolist() == [
            s.document for s in [network.sentence(i) for i in range(network.n_sentences)]
        ]
        with pytest.raises(KeyError):
            network.sentence(-1)

    def test_sentences_of_entity(self, network):
        sentences = network.sentences_of(("Schwinger", "PERSON"))
        assert all("Schwinger" in s.text for s in sentences)
        assert [s.id for s in sentences] == sorted({s.id for s in sentences})

    def test_mention_rows(self, network):
        mention = network.mention(0)
        text = network.sentence(mention.sentence.id).text
        assert mention.entity.norm in text.lower()
        with pytest.raises(KeyError):
            network.mention(network.n_mentions)


# ============== Incremental updates ==============


class TestIncremental:
    def test_incremental_equals_batch(self, annotated_docs):
        batch = ImplicitNetwork.from_documents(annotated_docs)
        incremental = ImplicitNetwork()
        incremental.add_documents(annotated_docs[:1]).add_documents([]).add_documents(
            annotated_docs[1:3]
        )
        incremental.add_documents(annotated_docs[3:])
        assert incremental.n_documents == batch.n_documents
        assert incremental.n_entities == batch.n_entities and incremental.n_terms == batch.n_terms
        for name in (
            "entity_entity_matrix",
            "entity_count_matrix",
            "entity_term_matrix",
            "sentence_entity_matrix",
            "sentence_term_matrix",
        ):
            assert abs(getattr(incremental, name) - getattr(batch, name)).sum() == 0
        assert weights_by_key(incremental) == weights_by_key(batch)
        assert [d.id for d in incremental.documents()] == [d.id for d in batch.documents()]
        assert incremental.sentences_of_document(3) == batch.sentences_of_document(3)
        assert incremental.contexts(
            ("Feynman", "PERSON"), ("Schwinger", "PERSON")
        ) == batch.contexts(("Feynman", "PERSON"), ("Schwinger", "PERSON"))

    def test_new_entities_extend_existing_matrices(self, extractor):
        network = ImplicitNetwork(NetworkConfig(window=1))
        network.add_documents(extractor.annotate_all(["Feynman met Schwinger."]))
        assert network.n_entities == 2 and network.n_edges == 1
        network.add_documents(
            extractor.annotate_all([Document("Tomonaga met Schwinger in Tokyo.", "second")])
        )
        assert network.n_entities == 4
        assert network.weight(("Feynman", "PERSON"), ("Schwinger", "PERSON")) == pytest.approx(1.0)
        assert network.weight(("Tomonaga", "PERSON"), ("Tokyo", "LOC")) == pytest.approx(1.0)
        assert network.entity("Schwinger", "PERSON").count == 2

    def test_duplicate_document_raises(self, network, annotated_docs):
        with pytest.raises(ValueError, match="already part"):
            network.add_documents(annotated_docs[:1])
        fresh = ImplicitNetwork()
        with pytest.raises(ValueError):
            fresh.add_documents([annotated_docs[0], annotated_docs[0]])

    def test_invalid_sentence_index_raises(self):
        doc = AnnotatedDocument(
            "d",
            "Feynman.",
            [Sentence(0, 0, 8)],
            [Token("Feynman", 0, 7, 0)],
            [EntityMention("Feynman", "PERSON", 0, 7, 3)],
        )
        with pytest.raises(ValueError, match="sentence 3"):
            ImplicitNetwork.from_documents([doc])
        doc = AnnotatedDocument(
            "d", "Feynman.", [Sentence(0, 0, 8)], [Token("Feynman", 0, 7, 2)], []
        )
        with pytest.raises(ValueError):
            ImplicitNetwork.from_documents([doc])

    def test_empty_network(self):
        network = ImplicitNetwork()
        assert (
            network.n_documents == network.n_entities == network.n_edges == network.n_mentions == 0
        )
        assert network.edges() == [] and network.entities() == [] and network.terms() == []
        assert network.entity_counts().shape == (0,) and network.term_counts().shape == (0,)
        assert len(network.all_cooccurrences()) == 0
        assert network.top_entities() == []
        assert network.summary()

    def test_documents_without_mentions(self, extractor):
        network = ImplicitNetwork.from_documents(extractor.annotate_all(["Nothing here. At all."]))
        assert network.n_documents == 1 and network.n_sentences == 2
        assert network.n_entities == 0 and network.n_terms > 0
        assert network.edges() == []

    def test_documents_without_sentences(self):
        network = ImplicitNetwork.from_documents([AnnotatedDocument("empty", "")])
        assert network.n_documents == 1 and network.document(0).n_sentences == 0


# ============== Normalisation, decay & persistence ==============


class TestCustomisation:
    def test_custom_normalizer(self, extractor):
        docs = extractor.annotate_all(["Feynman met feynman."])
        default = ImplicitNetwork.from_documents(docs)
        assert default.n_entities == 1
        identity = ImplicitNetwork.from_documents(docs, normalize_entity=lambda s: s)
        assert identity.n_entities == 2
        assert identity.entity("feynman", "PERSON").text == "feynman"

    def test_registered_decay(self, annotated_docs):
        register_decay("square", lambda d: 1.0 / (1.0 + d.astype(float)) ** 2)
        network = ImplicitNetwork.from_documents(annotated_docs, NetworkConfig(decay="square"))
        expected = naive_edge_weights(annotated_docs, window=2, decay=lambda d: 1 / (1 + d) ** 2)
        for pair, (weight, _) in expected.items():
            assert weights_by_key(network)[pair][0] == pytest.approx(weight)

    def test_save_and_load(self, network, tmp_path):
        path = network.save(tmp_path / "net")
        assert path.suffix == ".npz" and path.exists()
        loaded = ImplicitNetwork.load(path)
        assert repr(loaded) == repr(network)
        assert loaded.config == network.config
        for name in ("entity_entity_matrix", "entity_count_matrix", "entity_term_matrix"):
            assert abs(getattr(loaded, name) - getattr(network, name)).sum() == 0
        assert weights_by_key(loaded) == weights_by_key(network)
        assert loaded.entity("Feynman", "PERSON") == network.entity("Feynman", "PERSON")
        assert loaded.term("worked") == network.term("worked")
        assert loaded.documents() == network.documents()
        assert loaded.contexts(("Feynman", "PERSON"), ("Schwinger", "PERSON")) == network.contexts(
            ("Feynman", "PERSON"), ("Schwinger", "PERSON")
        )
        # loaded networks can be extended further
        loaded.add_documents(
            GazetteerEntityExtractor({"PERSON": ["Dirac"]}).annotate_all(
                [Document("Dirac wrote a book.", "extra")]
            )
        )
        assert loaded.n_documents == network.n_documents + 1

    def test_load_rejects_unknown_format(self, network, tmp_path):
        path = network.save(tmp_path / "net.npz")
        archive = dict(np.load(path))
        archive["meta"] = np.array('{"format": 99}')
        np.savez(path, **archive)
        with pytest.raises(ValueError):
            ImplicitNetwork.load(path)

    def test_default_norm_helper(self):
        assert default_norm("  Richard   Feynman ") == "richard feynman"
