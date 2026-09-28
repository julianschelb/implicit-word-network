"""Compare the 0.1 engine against the original 0.0.x ``buildGraph`` on the same data.

Usage::

    python benchmarks/bench_legacy.py --docs 300
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from synthetic import synthetic_corpus  # noqa: E402
from tests import legacy_reference as legacy  # noqa: E402

from implicit_word_network import ImplicitNetwork, NetworkConfig  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", type=int, default=300)
    parser.add_argument("--sentences", type=int, default=30)
    parser.add_argument("--entities", type=int, default=2000)
    parser.add_argument("--window", type=int, default=2)
    args = parser.parse_args()

    docs = synthetic_corpus(args.docs, sentences_per_doc=args.sentences, n_entities=args.entities)
    n_tokens = sum(len(d.tokens) for d in docs)
    print(f"{len(docs)} docs, {n_tokens:,} tokens, {sum(len(d.mentions) for d in docs):,} mentions")

    t0 = time.perf_counter()
    network = ImplicitNetwork.from_documents(
        docs, NetworkConfig(window=args.window, use_lemma=False)
    )
    t_new = time.perf_counter() - t0
    print(f"0.1 engine     : {t_new:6.2f}s  ({network!r})")

    corpus = legacy.to_legacy_corpus(docs)
    t0 = time.perf_counter()
    V, Ep = legacy.buildGraph(corpus, args.window, show_progress=False)
    t_old = time.perf_counter() - t0
    print(
        f"0.0.x buildGraph: {t_old:6.2f}s  (entities={len(V['entities'])}, e-e edges={len(Ep[('e', 'e')])})"
    )
    print(f"speed-up: {t_old / t_new:.1f}x")


if __name__ == "__main__":
    main()
