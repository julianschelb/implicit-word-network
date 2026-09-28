"""
Validation of the vectorised engine against two independent oracles:

1. A literal, loop-based implementation of the LOAD / implicit-network
   definitions (Spitz & Gertz 2016; Spitz, Almasian & Gertz 2017; ECCE 2022):
   nodes, all five edge classes and the weights ω = Σ exp(−δ) and
   ω_LOAD(x, y) = log(|Y| / |N(x) ∩ Y|) · ω(x, y).
2. The original 0.0.x implementation (``tests/legacy_reference.py``).

Random corpora are generated with Hypothesis so that windows, repeated
mentions, empty sentences and documents without entities are all covered.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from implicit_word_network import ImplicitNetwork, NetworkConfig
from implicit_word_network.annotation import AnnotatedDocument, EntityMention, Sentence, Token
from tests import legacy_reference as legacy

LABELS = ("PER", "ORG", "LOC")
ENTITY_NAMES = ("Feynman", "Schwinger", "Tomonaga", "Caltech", "Harvard", "Tokyo", "Nobel Prize")
TERMS = ("physics", "prize", "lecture", "quantum", "diagram", "the", ".")

# =============================================================================
# Random corpus strategy
# =============================================================================


@st.composite
def annotated_documents(draw: st.DrawFn) -> list[AnnotatedDocument]:
    n_docs = draw(st.integers(1, 4))
    documents = []
    for doc_index in range(n_docs):
        n_sentences = draw(st.integers(0, 6))
        tokens: list[Token] = []
        mentions: list[EntityMention] = []
        sentences: list[Sentence] = []
        pieces: list[str] = []
        position = 0
        for s in range(n_sentences):
            start = position
            n_items = draw(st.integers(1, 6))
            for _ in range(n_items):
                is_entity = draw(st.booleans())
                if is_entity:
                    name = draw(st.sampled_from(ENTITY_NAMES))
                    label = draw(st.sampled_from(LABELS))
                    surface = name.upper() if draw(st.booleans()) else name
                    words = surface.split()
                    first = position
                    for word in words:
                        tokens.append(Token(word, position, position + len(word), s, pos="PROPN"))
                        pieces.append(word)
                        position += len(word) + 1
                    mentions.append(
                        EntityMention(
                            surface,
                            label,
                            first,
                            position - 1,
                            s,
                            1.0,
                            len(tokens) - len(words),
                            len(tokens),
                        )
                    )
                else:
                    word = draw(st.sampled_from(TERMS))
                    tokens.append(
                        Token(
                            word,
                            position,
                            position + len(word),
                            s,
                            is_stop=word == "the",
                            is_punct=word == ".",
                            pos="NOUN",
                        )
                    )
                    pieces.append(word)
                    position += len(word) + 1
            sentences.append(Sentence(s, start, position - 1))
        text = " ".join(pieces)
        documents.append(AnnotatedDocument(doc_index, text, sentences, tokens, mentions))
    return documents


# =============================================================================
# Literal implementation of the definitions
# =============================================================================


def norm(text: str) -> str:
    return " ".join(text.split()).lower()


def literal_model(docs, window: int):
    """Return dictionaries describing every node and edge class of the IEN/LOAD graph."""
    entities: dict[tuple[str, str], int] = {}  # key -> mention count
    terms: dict[tuple[str, str], int] = {}
    sentence_entity: dict[tuple[int, tuple[str, str]], int] = {}
    sentence_term: dict[tuple[int, tuple[str, str]], int] = {}
    document_sentence: set[tuple[int, int]] = set()
    entity_term: dict[tuple[tuple[str, str], tuple[str, str]], int] = {}
    entity_entity: dict[tuple[tuple[str, str], tuple[str, str]], tuple[float, int]] = {}
    global_sentence = 0
    for doc_index, doc in enumerate(docs):
        covered = doc.entity_token_mask()
        sentence_ids = {}
        for sentence in doc.sentences:
            sentence_ids[sentence.index] = global_sentence
            document_sentence.add((doc_index, global_sentence))
            global_sentence += 1
        per_sentence_entities: dict[int, list[tuple[str, str]]] = {}
        per_sentence_terms: dict[int, list[tuple[str, str]]] = {}
        for mention in doc.mentions:
            key = (norm(mention.text), mention.label)
            entities[key] = entities.get(key, 0) + 1
            per_sentence_entities.setdefault(mention.sentence, []).append(key)
            sk = (sentence_ids[mention.sentence], key)
            sentence_entity[sk] = sentence_entity.get(sk, 0) + 1
        for position, token in enumerate(doc.tokens):
            if covered[position] or token.is_stop or token.is_punct or token.is_space:
                continue
            key = (token.text.lower(), token.pos)
            terms[key] = terms.get(key, 0) + 1
            per_sentence_terms.setdefault(token.sentence, []).append(key)
            sk = (sentence_ids[token.sentence], key)
            sentence_term[sk] = sentence_term.get(sk, 0) + 1
        # entity–term edges: every (mention, term occurrence) pair in the same sentence
        for s, ents in per_sentence_entities.items():
            for e in ents:
                for t in per_sentence_terms.get(s, []):
                    entity_term[(e, t)] = entity_term.get((e, t), 0) + 1
        # entity–entity edges: every pair of mentions of distinct entities within the window
        flat = [(m.sentence, (norm(m.text), m.label)) for m in doc.mentions]
        for i in range(len(flat)):
            for j in range(i + 1, len(flat)):
                (s_i, e_i), (s_j, e_j) = flat[i], flat[j]
                delta = abs(s_i - s_j)
                if e_i == e_j or delta > window:
                    continue
                pair = (min(e_i, e_j), max(e_i, e_j))
                w, n = entity_entity.get(pair, (0.0, 0))
                entity_entity[pair] = (w + math.exp(-delta), n + 1)
    return {
        "entities": entities,
        "terms": terms,
        "sentence_entity": sentence_entity,
        "sentence_term": sentence_term,
        "document_sentence": document_sentence,
        "entity_term": entity_term,
        "entity_entity": entity_entity,
    }


def literal_load_weights(model):
    """ω_LOAD(x, y) = log(|Y| / |N(x) ∩ Y|) · ω(x, y) for every ordered pair."""
    neighbours: dict[tuple[str, str], set[tuple[str, str]]] = {}
    for (a, b), (w, _) in model["entity_entity"].items():
        neighbours.setdefault(a, set()).add(b)
        neighbours.setdefault(b, set()).add(a)
    type_size: dict[str, int] = {}
    for _, label in model["entities"]:
        type_size[label] = type_size.get(label, 0) + 1
    result = {}
    for (a, b), (w, _) in model["entity_entity"].items():
        for x, y in ((a, b), (b, a)):
            same_type = sum(1 for n in neighbours[x] if n[1] == y[1])
            result[(x, y)] = math.log(type_size[y[1]] / same_type) * w
    return result


def network_model(network: ImplicitNetwork):
    """Extract the same dictionaries from an ImplicitNetwork."""
    key_of = {e.id: e.key for e in network.entities()}
    term_key = {t.id: (t.text, t.pos) for t in network.terms()}
    model = {
        "entities": {e.key: e.count for e in network.entities()},
        "terms": {(t.text, t.pos): t.count for t in network.terms()},
        "document_sentence": {
            (int(d), s) for s, d in enumerate(network.sentence_document().tolist())
        },
    }
    se = network.sentence_entity_matrix.tocoo()
    model["sentence_entity"] = {
        (int(r), key_of[int(c)]): int(v) for r, c, v in zip(se.row, se.col, se.data)
    }
    stm = network.sentence_term_matrix.tocoo()
    model["sentence_term"] = {
        (int(r), term_key[int(c)]): int(v) for r, c, v in zip(stm.row, stm.col, stm.data)
    }
    et = network.entity_term_matrix.tocoo()
    model["entity_term"] = {
        (key_of[int(r)], term_key[int(c)]): int(v) for r, c, v in zip(et.row, et.col, et.data)
    }
    model["entity_entity"] = {
        (min(e.source.key, e.target.key), max(e.source.key, e.target.key)): (e.weight, e.count)
        for e in network.edges()
    }
    return model


def assert_models_equal(actual, expected):
    for name in ("entities", "terms", "sentence_entity", "sentence_term", "entity_term"):
        assert actual[name] == expected[name], name
    assert actual["document_sentence"] == expected["document_sentence"]
    assert set(actual["entity_entity"]) == set(expected["entity_entity"])
    for pair, (weight, count) in expected["entity_entity"].items():
        assert actual["entity_entity"][pair][0] == pytest.approx(weight)
        assert actual["entity_entity"][pair][1] == count


# =============================================================================
# Tests against the literal definitions
# =============================================================================


class TestLiteralDefinitions:
    @given(docs=annotated_documents(), window=st.integers(0, 4))
    @settings(max_examples=120, deadline=None)
    def test_all_node_and_edge_classes(self, docs, window):
        network = ImplicitNetwork.from_documents(
            docs, NetworkConfig(window=window, use_lemma=False)
        )
        assert_models_equal(network_model(network), literal_model(docs, window))

    @given(docs=annotated_documents(), window=st.integers(0, 3))
    @settings(max_examples=60, deadline=None)
    def test_load_weights(self, docs, window):
        network = ImplicitNetwork.from_documents(
            docs, NetworkConfig(window=window, use_lemma=False)
        )
        expected = literal_load_weights(literal_model(docs, window))
        matrix = network.load_weight_matrix().tocoo()
        key_of = {e.id: e.key for e in network.entities()}
        actual = {
            (key_of[int(r)], key_of[int(c)]): float(v)
            for r, c, v in zip(matrix.row, matrix.col, matrix.data)
        }
        # zero factors (log 1) are dropped by the sparse representation
        expected = {k: v for k, v in expected.items() if v != 0.0}
        assert set(actual) == set(expected)
        for pair, value in expected.items():
            assert actual[pair] == pytest.approx(value)
            assert network.load_weight(pair[0], pair[1]) == pytest.approx(value)

    @given(docs=annotated_documents(), window=st.integers(0, 3))
    @settings(max_examples=60, deadline=None)
    def test_incremental_equals_literal(self, docs, window):
        network = ImplicitNetwork(NetworkConfig(window=window, use_lemma=False))
        for doc in docs:
            network.add_documents([doc])
        assert_models_equal(network_model(network), literal_model(docs, window))
        # cooccurrence enumeration agrees with the aggregated matrices
        for edge in network.edges():
            coocs = network.cooccurrences(edge.source, edge.target)
            assert len(coocs) == edge.count
            assert sum(c.weight for c in coocs) == pytest.approx(edge.weight)


# =============================================================================
# Tests against the original 0.0.x implementation
# =============================================================================


def legacy_model(docs, window: int):
    """Run the legacy ``buildGraph`` and express its output like ``literal_model``."""
    V, Ep = legacy.buildGraph(legacy.to_legacy_corpus(docs), window, show_progress=False)
    model = {
        "entities": {
            (v["text"].lower(), v["entity_type"]): len(v["instances"]) for v in V["entities"]
        },
        "terms": {(v["text"].lower(), v["pos"]): len(v["instances"]) for v in V["terms"]},
    }
    offsets = {}
    global_sentence = 0
    for s in V["sentences"]:
        offsets[(s["d_id"], s["s_id"])] = global_sentence
        global_sentence += 1
    model["document_sentence"] = {
        (e["vertex_1"]["d_id"], offsets[(e["vertex_2"]["d_id"], e["vertex_2"]["s_id"])])
        for e in Ep[("d", "s")]
    }
    model["sentence_entity"] = {
        (
            offsets[(e["vertex_1"]["d_id"], e["vertex_1"]["s_id"])],
            (e["vertex_2"]["text"].lower(), e["vertex_2"]["entity_type"]),
        ): len(e["instances"])
        for e in Ep[("s", "e")]
    }
    # NOTE: the legacy ``Ep[("s", "t")]`` instance lists are corrupted by list
    # aliasing in ``extendTerms`` (the list of the first sentence containing a
    # term is later extended with every further instance of that term), so the
    # sentence–term counts are derived from the term-node instances instead.
    sentence_term: dict = {}
    for v in V["terms"]:
        for instance in v["instances"]:
            key = (offsets[(instance["d_id"], instance["s_id"])], (v["text"].lower(), v["pos"]))
            sentence_term[key] = sentence_term.get(key, 0) + 1
    model["sentence_term"] = sentence_term
    assert set(model["sentence_term"]) == {
        (
            offsets[(e["vertex_1"]["d_id"], e["vertex_1"]["s_id"])],
            (e["vertex_2"]["text"].lower(), e["vertex_2"]["pos"]),
        )
        for e in Ep[("s", "t")]
    }
    model["entity_term"] = {
        (
            (e["vertex_1"]["text"], e["vertex_1"]["entity_type"]),
            (e["vertex_2"]["text"], e["vertex_2"]["pos"]),
        ): len(e["instances"])
        for e in Ep[("e", "t")]
    }
    entity_entity = {}
    for e in Ep[("e", "e")]:
        a = (e["vertex_1"]["text"], e["vertex_1"]["entity_type"])
        b = (e["vertex_2"]["text"], e["vertex_2"]["entity_type"])
        pair = (min(a, b), max(a, b))
        weight, count = entity_entity.get(pair, (0.0, 0))
        entity_entity[pair] = (
            weight + sum(i["w"] for i in e["instances"]),
            count + len(e["instances"]),
        )
    model["entity_entity"] = entity_entity
    return model


class TestLegacyImplementation:
    @given(docs=annotated_documents(), window=st.integers(0, 4))
    @settings(max_examples=100, deadline=None)
    def test_matches_original_build_graph(self, docs, window):
        network = ImplicitNetwork.from_documents(
            docs, NetworkConfig(window=window, use_lemma=False)
        )
        assert_models_equal(network_model(network), legacy_model(docs, window))

    def test_literal_and_legacy_agree(self, annotated_docs):
        assert_models_equal(legacy_model(annotated_docs, 2), literal_model(annotated_docs, 2))


# =============================================================================
# Ranking queries against the EVELIN formulas
# =============================================================================


class TestRankingFormulas:
    @pytest.fixture
    def network(self, annotated_docs):
        return ImplicitNetwork.from_documents(annotated_docs, NetworkConfig(window=2))

    def test_single_entity_ranking_is_normalised_load_weight(self, network):
        feynman = network.entity("Feynman", "PERSON")
        results = network.rank_entities(feynman, k=None)
        load = network.load_weight_matrix()
        row = load.getrow(feynman.id).toarray().ravel()
        expected = {i: w for i, w in enumerate(row) if w != 0 and i != feynman.id}
        w_max = max(expected.values())
        assert {n.id: pytest.approx(s) for n, s in results} == {
            i: pytest.approx(w / w_max) for i, w in expected.items()
        }
        assert [s for _, s in results] == sorted((s for _, s in results), reverse=True)
        raw = network.rank_entities(feynman, k=None, weighting="raw")
        assert {n.id for n, _ in raw} == {n.id for n, _ in network.neighbors(feynman)}

    def test_multi_entity_ranking_cohesion_plus_sum(self, network):
        query = [("Feynman", "PERSON"), ("Schwinger", "PERSON")]
        results = network.rank_entities(query, k=None, weighting="raw")
        ids = [network._resolve_entity(q) for q in query]
        W = network.entity_entity_matrix.toarray()
        totals = {
            x: sum(W[q, x] for q in ids)
            for x in range(network.n_entities)
            if x not in ids and any(W[q, x] > 0 for q in ids)
        }
        s_max = max(totals.values())
        expected = {x: (sum(1 for q in ids if W[q, x] > 0) - 1) + totals[x] / s_max for x in totals}
        assert {n.id: pytest.approx(s) for n, s in results} == {
            x: pytest.approx(v) for x, v in expected.items()
        }
        assert all(0 <= s <= len(query) for _, s in results)

    def test_label_filter_and_k(self, network):
        results = network.rank_entities(("Feynman", "PERSON"), label="ORG", k=1)
        assert len(results) == 1 and results[0][0].label == "ORG"
        assert network.rank_entities(("Tokyo", "LOC"), label="LOC") == []  # no other LOC entity

    def test_sentence_ranking(self, network):
        query = [("Feynman", "PERSON"), ("Nobel Prize", "ORG")]
        results = network.rank_sentences(query, k=None, n_terms=3)
        assert results and all(
            any(q[0].lower() in s.text.lower() for q in query) for s, _ in results
        )
        top_sentence, top_score = results[0]
        assert "Feynman" in top_sentence.text and "Nobel Prize" in top_sentence.text
        assert top_score >= 2.0  # both query entities present
        assert [s for _, s in results] == sorted((s for _, s in results), reverse=True)

    def test_document_ranking(self, network):
        results = network.rank_documents(("Schwinger", "PERSON"), k=None)
        assert [d.id for d, _ in results][0] in (0, 1, 2)
        assert all(score >= 1.0 for _, score in results)
        assert network.rank_documents(("Schwinger", "PERSON"), k=1)[0] == results[0]

    def test_query_validation(self, network):
        with pytest.raises(ValueError):
            network.rank_entities([])
        with pytest.raises(ValueError):
            network.rank_entities(("Feynman", "PERSON"), weighting="nope")  # type: ignore[arg-type]
        with pytest.raises(KeyError):
            network.rank_sentences(("Nobody", "PERSON"))


class TestStorage:
    def test_growable_array(self):
        from implicit_word_network.network import GrowableArray

        arr = GrowableArray(np.int32, capacity=2)
        arr.extend(np.array([1, 2, 3]))
        arr.extend(np.array([], dtype=np.int32))
        arr.extend(np.arange(100))
        assert len(arr) == 103 and arr.view.tolist() == [1, 2, 3] + list(range(100))
        assert arr.dtype == np.int32
        copy = GrowableArray.from_array(arr.view)
        assert copy.view.tolist() == arr.view.tolist()

    def test_sparse_accumulator(self):
        from implicit_word_network.network import SparseAccumulator

        acc = SparseAccumulator(np.float64, (2, 2))
        acc.add(np.array([0, 0, 1]), np.array([1, 1, 0]), np.array([1.0, 2.0, 5.0]))
        acc.resize((3, 3))
        acc.add(np.array([2]), np.array([2]), np.array([7.0]))
        dense = acc.matrix.toarray()
        assert dense.tolist() == [[0, 3, 0], [5, 0, 0], [0, 0, 7]]
        assert acc.nnz == 3
        with pytest.raises(ValueError):
            acc.resize((1, 1))
        acc.add(np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([]))
        assert acc.matrix.shape == (3, 3)
        wrapped = SparseAccumulator.from_matrix(acc.matrix)
        wrapped.resize((4, 4))
        assert wrapped.matrix.shape == (4, 4) and wrapped.matrix[2, 2] == 7.0
