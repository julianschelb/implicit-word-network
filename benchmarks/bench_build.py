"""Benchmark network construction on synthetic corpora.

Usage::

    python benchmarks/bench_build.py --docs 2000 --sentences 30 --profile
"""

from __future__ import annotations

import argparse
import cProfile
import pstats
import time

from synthetic import synthetic_corpus

from implicit_word_network import ImplicitNetwork, NetworkConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", type=int, default=2000)
    parser.add_argument("--sentences", type=int, default=30)
    parser.add_argument("--tokens", type=int, default=15)
    parser.add_argument("--entities", type=int, default=20_000)
    parser.add_argument("--terms", type=int, default=50_000)
    parser.add_argument("--window", type=int, default=2)
    parser.add_argument("--chunk", type=int, default=0, help="add documents in chunks of this size")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--queries", action="store_true", help="also time common queries")
    args = parser.parse_args()

    t0 = time.perf_counter()
    docs = synthetic_corpus(
        args.docs,
        sentences_per_doc=args.sentences,
        tokens_per_sentence=args.tokens,
        n_entities=args.entities,
        n_terms=args.terms,
    )
    n_tokens = sum(len(d.tokens) for d in docs)
    n_mentions = sum(len(d.mentions) for d in docs)
    print(
        f"generated {len(docs)} docs, {n_tokens:,} tokens, {n_mentions:,} mentions in {time.perf_counter() - t0:.1f}s"
    )

    config = NetworkConfig(window=args.window)

    def build() -> ImplicitNetwork:
        network = ImplicitNetwork(config)
        if args.chunk > 0:
            for start in range(0, len(docs), args.chunk):
                network.add_documents(docs[start : start + args.chunk])
        else:
            network.add_documents(docs)
        return network

    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()
    t0 = time.perf_counter()
    network = build()
    elapsed = time.perf_counter() - t0
    if args.profile:
        profiler.disable()
    print(f"built {network!r} in {elapsed:.2f}s ({n_tokens / elapsed:,.0f} tokens/s)")

    if args.queries:
        top = network.top_entities(1)[0]
        t0 = time.perf_counter()
        edges = network.edges(top_k=100)
        t_edges = time.perf_counter() - t0
        t0 = time.perf_counter()
        network.neighbors(top, k=10)
        t_nb = time.perf_counter() - t0
        t0 = time.perf_counter()
        coocs = network.cooccurrences(edges[0].source, edges[0].target)
        t_co = time.perf_counter() - t0
        t0 = time.perf_counter()
        for edge in edges[:50]:
            network.cooccurrences(edge.source, edge.target)
        t_co50 = time.perf_counter() - t0
        t0 = time.perf_counter()
        network.entity_terms(top, k=10)
        t_terms = time.perf_counter() - t0
        t0 = time.perf_counter()
        network.mentions_of(top)
        t_men = time.perf_counter() - t0
        print(
            f"queries: edges(top_k=100) {t_edges * 1000:.1f}ms | neighbors {t_nb * 1000:.1f}ms | "
            f"cooccurrences(1 edge, {len(coocs)} inst.) {t_co * 1000:.1f}ms | 50 edges {t_co50 * 1000:.1f}ms | "
            f"entity_terms {t_terms * 1000:.1f}ms | mentions_of {t_men * 1000:.1f}ms"
        )

    if args.profile:
        stats = pstats.Stats(profiler)
        stats.sort_stats("cumulative").print_stats(22)


if __name__ == "__main__":
    main()
