# cli.py
"""Command-line interface: ``implicit-word-network``."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from implicit_word_network.document import Corpus
from implicit_word_network.extraction import BaseEntityExtractor
from implicit_word_network.network import (
    ImplicitNetwork,
    NetworkConfig,
    available_decays,
    to_edgelist,
    to_json,
)
from implicit_word_network.pipeline import ImplicitNetworkPipeline


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="implicit-word-network",
        description="Extract implicit entity networks from text corpora.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Build a network and write it to a file")
    _add_input_options(build)
    build.add_argument(
        "-o", "--output", required=True, help="Output file (.json, .graphml, .gexf, .csv or .npz)"
    )
    build.add_argument("--min-weight", type=float, default=0.0, help="Drop edges below this weight")
    build.add_argument("--top-k", type=int, default=None, help="Keep only the k heaviest edges")
    build.add_argument(
        "--include-terms", action="store_true", help="Add term nodes (GraphML/GEXF only)"
    )

    summary = subparsers.add_parser("summary", help="Build a network and print statistics")
    _add_input_options(summary)
    summary.add_argument("--top", type=int, default=10, help="Number of entities/edges to list")
    return parser


def _add_input_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", help="Text file (one document per line) or CSV/TSV file")
    parser.add_argument("--text-column", default="text", help="CSV column holding the text")
    parser.add_argument("--id-column", default=None, help="CSV column holding document ids")
    parser.add_argument("--delimiter", default=None, help="CSV delimiter (default: by extension)")
    parser.add_argument(
        "--extractor",
        choices=("spacy", "gliner", "gazetteer"),
        default="spacy",
        help="Entity extractor to use",
    )
    parser.add_argument("--model", default=None, help="spaCy pipeline or GLiNER model id")
    parser.add_argument("--labels", nargs="+", default=None, help="Entity labels to extract")
    parser.add_argument("--threshold", type=float, default=0.5, help="GLiNER confidence threshold")
    parser.add_argument("--gazetteer", default=None, help="JSON file {label: [surface forms]}")
    parser.add_argument("--device", default="cpu", help="Device for GLiNER (cpu, cuda, mps)")
    parser.add_argument("--window", type=int, default=2, help="Context window in sentences")
    parser.add_argument("--decay", default="exponential", choices=available_decays() + ["linear"])
    parser.add_argument("--include-stopwords", action="store_true", help="Keep stop words as terms")
    parser.add_argument("--batch-size", type=int, default=32, help="Documents per extractor call")
    parser.add_argument("-q", "--quiet", action="store_true", help="Hide progress bars")


def load_corpus(args: argparse.Namespace) -> Corpus:
    """Load the input corpus described by the CLI arguments."""
    path = Path(args.input)
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        delimiter = args.delimiter or ("\t" if suffix == ".tsv" else ",")
        return Corpus.from_csv(
            path, text_column=args.text_column, id_column=args.id_column, delimiter=delimiter
        )
    return Corpus.from_txt(path)


def make_extractor(args: argparse.Namespace) -> BaseEntityExtractor:
    """Instantiate the extractor described by the CLI arguments."""
    if args.extractor == "gazetteer":
        if not args.gazetteer:
            raise SystemExit("--gazetteer FILE is required for the gazetteer extractor")
        gazetteer = json.loads(Path(args.gazetteer).read_text(encoding="utf-8"))
        from implicit_word_network.extraction.gazetteer import GazetteerEntityExtractor

        return GazetteerEntityExtractor(gazetteer)
    if args.extractor == "gliner":
        from implicit_word_network.extraction.gliner import (
            DEFAULT_GLINER_LABELS,
            DEFAULT_GLINER_MODEL,
            GLiNEREntityExtractor,
        )

        return GLiNEREntityExtractor(
            args.model or DEFAULT_GLINER_MODEL,
            labels=args.labels or DEFAULT_GLINER_LABELS,
            threshold=args.threshold,
            device=args.device,
        )
    from implicit_word_network.extraction.spacy import (
        DEFAULT_SPACY_LABELS,
        DEFAULT_SPACY_MODEL,
        SpacyEntityExtractor,
    )

    return SpacyEntityExtractor(
        args.model or DEFAULT_SPACY_MODEL, labels=args.labels or DEFAULT_SPACY_LABELS
    )


def build_network_from_args(args: argparse.Namespace) -> ImplicitNetwork:
    """Run the pipeline for the given CLI arguments."""
    config = NetworkConfig(
        window=args.window, decay=args.decay, include_stopwords=args.include_stopwords
    )
    pipeline = ImplicitNetworkPipeline(make_extractor(args), config=config)
    return pipeline.run(load_corpus(args), batch_size=args.batch_size, show_progress=not args.quiet)


def write_output(network: ImplicitNetwork, args: argparse.Namespace) -> Path:
    """Write the network in the format implied by the output file extension."""
    output = Path(args.output)
    suffix = output.suffix.lower()
    export_kwargs: dict[str, Any] = {"min_weight": args.min_weight, "top_k": args.top_k}
    if suffix == ".json":
        to_json(network, output, **export_kwargs)
    elif suffix in {".graphml", ".gexf"}:
        import networkx as nx

        graph = network.to_networkx(include_terms=args.include_terms, **export_kwargs)
        if suffix == ".graphml":
            nx.write_graphml(graph, output)
        else:
            nx.write_gexf(graph, output)
    elif suffix == ".csv":
        rows = to_edgelist(network, **export_kwargs)
        with output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["source", "source_label", "target", "target_label", "weight", "count"],
            )
            writer.writeheader()
            writer.writerows(rows)
    elif suffix == ".npz":
        network.save(output)
    else:
        raise SystemExit(
            f"Unsupported output format {suffix!r} (use .json, .graphml, .gexf, .csv or .npz)"
        )
    return output


def print_summary(network: ImplicitNetwork, top: int, stream: Any = None) -> None:
    """Print network statistics, top entities and top edges."""
    stream = stream or sys.stdout
    print(network.summary(), file=stream)
    print(f"\nTop {top} entities:", file=stream)
    for node in network.top_entities(top):
        print(f"  {node.count:6d}  {node.text} [{node.label}]", file=stream)
    print(f"\nTop {top} edges:", file=stream)
    for edge in network.edges(top_k=top):
        print(
            f"  {edge.weight:8.3f}  {edge.source.text} [{edge.source.label}] -- "
            f"{edge.target.text} [{edge.target.label}]  (n={edge.count})",
            file=stream,
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of the ``implicit-word-network`` command."""
    parser = build_parser()
    args = parser.parse_args(argv)
    network = build_network_from_args(args)
    if args.command == "build":
        output = write_output(network, args)
        if not args.quiet:
            print(f"Wrote {output} ({network!r})", file=sys.stderr)
    else:
        print_summary(network, args.top)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
