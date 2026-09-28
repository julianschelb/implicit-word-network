"""
Tests for the command-line interface (offline, gazetteer extractor).
"""

import json

import networkx as nx
import numpy as np
import pytest

from implicit_word_network import ImplicitNetwork
from implicit_word_network.cli import build_parser, main


@pytest.fixture
def inputs(tmp_path, texts, gazetteer):
    text_file = tmp_path / "docs.txt"
    text_file.write_text("\n".join(texts), encoding="utf-8")
    csv_file = tmp_path / "docs.csv"
    csv_file.write_text(
        "doc_id,body\n" + "\n".join(f'{i},"{t}"' for i, t in enumerate(texts)) + "\n",
        encoding="utf-8",
    )
    gaz_file = tmp_path / "gaz.json"
    gaz_file.write_text(json.dumps(gazetteer), encoding="utf-8")
    return text_file, csv_file, gaz_file


def run(*args):
    return main([str(a) for a in args])


class TestBuild:
    @pytest.mark.parametrize("suffix", ["json", "csv", "graphml", "gexf", "npz"])
    def test_output_formats(self, inputs, tmp_path, suffix):
        text_file, _, gaz_file = inputs
        output = tmp_path / f"net.{suffix}"
        assert (
            run(
                "build",
                text_file,
                "-o",
                output,
                "--extractor",
                "gazetteer",
                "--gazetteer",
                gaz_file,
                "-q",
            )
            == 0
        )
        assert output.exists()
        if suffix == "json":
            data = json.loads(output.read_text(encoding="utf-8"))
            assert data["n_documents"] == 4 and data["edges"]
        elif suffix == "csv":
            lines = output.read_text(encoding="utf-8").splitlines()
            assert lines[0] == "source,source_label,target,target_label,weight,count"
            assert len(lines) > 1
        elif suffix == "graphml":
            assert nx.read_graphml(output).number_of_nodes() == 10
        elif suffix == "gexf":
            assert nx.read_gexf(output).number_of_nodes() == 10
        else:
            assert ImplicitNetwork.load(output).n_documents == 4

    def test_csv_input_and_filters(self, inputs, tmp_path):
        _, csv_file, gaz_file = inputs
        output = tmp_path / "net.json"
        run(
            "build",
            csv_file,
            "-o",
            output,
            "--extractor",
            "gazetteer",
            "--gazetteer",
            gaz_file,
            "--text-column",
            "body",
            "--id-column",
            "doc_id",
            "--top-k",
            "2",
            "--window",
            "1",
            "--decay",
            "constant",
            "-q",
        )
        data = json.loads(output.read_text(encoding="utf-8"))
        assert len(data["edges"]) == 2 and data["config"]["window"] == 1
        assert all(float(e["weight"]).is_integer() for e in data["edges"])

    def test_include_terms(self, inputs, tmp_path):
        text_file, _, gaz_file = inputs
        output = tmp_path / "net.graphml"
        run(
            "build",
            text_file,
            "-o",
            output,
            "--extractor",
            "gazetteer",
            "--gazetteer",
            gaz_file,
            "--include-terms",
            "-q",
        )
        assert nx.read_graphml(output).number_of_nodes() > 10

    def test_unsupported_format(self, inputs, tmp_path):
        text_file, _, gaz_file = inputs
        with pytest.raises(SystemExit):
            run(
                "build",
                text_file,
                "-o",
                tmp_path / "net.xyz",
                "--extractor",
                "gazetteer",
                "--gazetteer",
                gaz_file,
                "-q",
            )

    def test_gazetteer_requires_file(self, inputs, tmp_path):
        text_file, _, _ = inputs
        with pytest.raises(SystemExit):
            run("build", text_file, "-o", tmp_path / "net.json", "--extractor", "gazetteer", "-q")

    def test_progress_message(self, inputs, tmp_path, capsys):
        text_file, _, gaz_file = inputs
        run(
            "build",
            text_file,
            "-o",
            tmp_path / "net.json",
            "--extractor",
            "gazetteer",
            "--gazetteer",
            gaz_file,
        )
        assert "Wrote" in capsys.readouterr().err


class TestSummary:
    def test_prints_statistics(self, inputs, capsys):
        text_file, _, gaz_file = inputs
        assert (
            run(
                "summary",
                text_file,
                "--extractor",
                "gazetteer",
                "--gazetteer",
                gaz_file,
                "--top",
                "3",
                "-q",
            )
            == 0
        )
        out = capsys.readouterr().out
        assert "documents : 4" in out and "Top 3 entities" in out and "Schwinger" in out


class TestParser:
    def test_requires_command(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args([])

    def test_defaults(self):
        args = build_parser().parse_args(["build", "x.txt", "-o", "y.json"])
        assert args.extractor == "spacy" and args.window == 2 and args.decay == "exponential"

    def test_make_extractor_variants(self, inputs):
        from implicit_word_network.cli import make_extractor
        from implicit_word_network.extraction import GLiNEREntityExtractor, SpacyEntityExtractor

        _, _, gaz_file = inputs
        parser = build_parser()
        gliner = make_extractor(
            parser.parse_args(
                ["summary", "x", "--extractor", "gliner", "--labels", "person", "ship"]
            )
        )
        assert isinstance(gliner, GLiNEREntityExtractor) and gliner.labels == ["person", "ship"]
        spacy = make_extractor(parser.parse_args(["summary", "x", "--model", "en_core_web_md"]))
        assert isinstance(spacy, SpacyEntityExtractor) and spacy._model_name == "en_core_web_md"
        assert np is not None
