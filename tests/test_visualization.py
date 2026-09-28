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
