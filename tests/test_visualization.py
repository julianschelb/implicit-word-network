"""
Tests for plotting helpers (skipped without matplotlib).
"""

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from implicit_word_network import plot_network  # noqa: E402


class TestPlotNetwork:
    @pytest.mark.parametrize("layout", ["spring", "kamada_kawai", "circular"])
    def test_returns_axes(self, network, layout):
        ax = plot_network(network, layout=layout, top_k=5)
        assert ax is not None
        assert len(ax.collections) >= 1
        matplotlib.pyplot.close("all")

    def test_from_graph_and_save(self, network, tmp_path):
        graph = network.to_networkx(include_isolated=False)
        output = tmp_path / "net.png"
        plot_network(graph, output_path=str(output), with_labels=False)
        assert output.exists() and output.stat().st_size > 0
        matplotlib.pyplot.close("all")

    def test_missing_dependency_message(self, network):
        from unittest.mock import patch

        with patch("importlib.util.find_spec", return_value=None):
            with pytest.raises(ImportError, match=r"\[viz\]"):
                plot_network(network)


class TestPlotOptions:
    def test_legend_and_label_order(self, network):
        ax = plot_network(network, label_order=["ORG", "PERSON", "LOC"], top_k=None)
        legend = ax.get_legend()
        assert legend is not None
        assert [t.get_text() for t in legend.get_texts()] == ["ORG", "PERSON", "LOC"]
        matplotlib.pyplot.close("all")

    def test_no_legend_single_type(self, network):
        ax = plot_network(network, labels=["PERSON"], top_k=None, legend=True)
        assert ax.get_legend() is None
        matplotlib.pyplot.close("all")

    def test_max_labels_and_title(self, network):
        ax = plot_network(network, max_labels=2, title="Demo", top_k=None)
        assert len(ax.texts) == 2
        assert ax.get_title(loc="left") == "Demo"
        matplotlib.pyplot.close("all")

    def test_min_component_size_and_empty(self, extractor):
        from implicit_word_network import ImplicitNetwork, NetworkConfig

        docs = extractor.annotate_all(
            ["Feynman met Schwinger.", "Tomonaga worked in Tokyo with Caltech people."]
        )
        net = ImplicitNetwork.from_documents(docs, NetworkConfig(window=0))
        ax = plot_network(net, min_component_size=3, top_k=None)
        assert len(ax.collections) >= 1 and len(ax.texts) == 3  # only the 3-node component
        empty = plot_network(net, min_weight=1e9)
        assert any("no edges" in t.get_text() for t in empty.texts)
        matplotlib.pyplot.close("all")

    def test_labels_do_not_overlap(self, network):
        from matplotlib.transforms import Bbox

        ax = plot_network(network, top_k=None, font_size=9)
        ax.figure.canvas.draw()
        renderer = ax.figure.canvas.get_renderer()
        boxes = [t.get_window_extent(renderer) for t in ax.texts]
        overlaps = [
            (a, b)
            for i, a in enumerate(boxes)
            for b in boxes[i + 1 :]
            if Bbox.intersection(a, b) is not None
            and Bbox.intersection(a, b).width > 2
            and Bbox.intersection(a, b).height > 2
        ]
        assert not overlaps
        matplotlib.pyplot.close("all")
