"""
Unit tests for implicit_word_network.network.export.
"""

import json

import networkx as nx
import pytest

from implicit_word_network.network import (
    entity_node_id,
    term_node_id,
    to_dict,
    to_edgelist,
    to_json,
    to_networkx,
    to_pandas,
)


class TestToNetworkx:
    def test_entity_graph(self, network):
        graph = network.to_networkx()
        assert isinstance(graph, nx.Graph)
        assert graph.number_of_nodes() == network.n_entities
        assert graph.number_of_edges() == network.n_edges
        assert graph.graph["window"] == 2
        feynman = network.entity("Feynman", "PERSON")
        node = graph.nodes[entity_node_id(feynman)]
        assert node == {
            "kind": "entity",
            "text": "Feynman",
            "norm": "feynman",
            "label": "PERSON",
            "count": 2,
        }
        for u, v, data in graph.edges(data=True):
            assert data["kind"] == "entity-entity" and data["weight"] > 0 and data["count"] >= 1

    def test_filters(self, network):
        graph = network.to_networkx(min_weight=1.0, include_isolated=False)
        assert all(d > 0 for _, d in graph.degree())
        assert all(data["weight"] >= 1.0 for _, _, data in graph.edges(data=True))
        persons = network.to_networkx(labels=["PERSON"])
        assert {d["label"] for _, d in persons.nodes(data=True)} == {"PERSON"}
        assert to_networkx(network, top_k=1).number_of_edges() == 1

    def test_terms(self, network):
        graph = network.to_networkx(include_terms=True, max_terms_per_entity=2, min_term_count=1)
        term_nodes = [n for n, d in graph.nodes(data=True) if d["kind"] == "term"]
        assert term_nodes
        assert all(n.endswith("|term") for n in term_nodes)
        term_edges = [d for _, _, d in graph.edges(data=True) if d["kind"] == "entity-term"]
        assert term_edges and all(d["weight"] == d["count"] for d in term_edges)
        assert (
            network.to_networkx(include_terms=True, min_term_count=99).number_of_nodes()
            == network.n_entities
        )

    def test_node_ids(self, network):
        feynman = network.entity("Feynman", "PERSON")
        assert entity_node_id(feynman) == "feynman|PERSON"
        term = network.term("worked")
        assert term_node_id(term) == "worked||term"


class TestPlainExports:
    def test_to_dict(self, network):
        data = to_dict(network, top_k=3)
        assert data["config"]["window"] == 2 and data["n_documents"] == 4
        assert len(data["nodes"]) == network.n_entities and len(data["edges"]) == 3
        node_ids = {n["id"] for n in data["nodes"]}
        assert all(e["source"] in node_ids and e["target"] in node_ids for e in data["edges"])
        assert network.to_dict(labels=["LOC"])["nodes"] == [
            {"id": "tokyo|LOC", "text": "Tokyo", "norm": "tokyo", "label": "LOC", "count": 1}
        ]

    def test_to_json(self, network, tmp_path):
        path = to_json(network, tmp_path / "net.json", min_weight=1.0)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert all(e["weight"] >= 1.0 for e in data["edges"])

    def test_to_edgelist(self, network):
        rows = to_edgelist(network, top_k=2)
        assert len(rows) == 2
        assert set(rows[0]) == {
            "source",
            "source_label",
            "target",
            "target_label",
            "weight",
            "count",
        }

    def test_to_pandas(self, network):
        pytest.importorskip("pandas")
        nodes, edges = to_pandas(network)
        assert len(nodes) == network.n_entities and len(edges) == network.n_edges
        assert list(edges.columns) == ["source", "target", "weight", "count"]
