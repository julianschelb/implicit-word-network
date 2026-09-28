# Performance & Scaling

The 0.0.x prototype stored every token as a Python dictionary, enumerated
entity pairs with nested loops and quadratic list membership tests, and called
the embedding model once per edge. Version 0.1 replaces this with
column-oriented tables and sparse linear algebra.

## Architecture for scale

| Concern | Design |
|---|---|
| Entity–entity weights | sparse triple product `Ω = Sᵀ K S` per added batch (sentence × entity incidence, banded sentence-distance kernel) |
| Cooccurrence counts, entity–term counts | the same products with a 0/1 kernel and `S_eᵀ S_t` |
| Growing corpora | append-only columns with geometric capacity growth (`GrowableArray`) and lazily compacted sparse accumulators, so 1 000 small updates cost about the same as one big build |
| Large entity vocabularies | integer ids everywhere; an entity → mention index built once per update makes per-entity queries independent of the corpus size |
| Long documents | instance pairs are enumerated with `searchsorted` + ragged range expansion in bounded chunks (`iter_cooccurrences(chunk_size=...)`), never per token |
| Query objects | node and sentence objects are cached; `edge_table` / `cooccurrence_table` return NumPy arrays for bulk work |
| Context embedding | contexts of many edges are pooled into model batches (`ContextualEdgeClusterer.cluster_edges`) |
| Extraction | spaCy `nlp.pipe` with `n_process`, GLiNER sentence-aligned chunks scored in batches; `max_length` for very long documents |

The remaining Python work per document is one pass over its tokens for
vocabulary lookups; everything downstream is NumPy/SciPy.

## Benchmarks

Synthetic corpora (`benchmarks/synthetic.py`, Zipf-distributed vocabularies,
≈1.5 mentions and 15 tokens per sentence), single core of an Apple M-series
laptop, network construction only (extraction is model-bound and excluded):

| Corpus | Tokens | Mentions | Entities | Edges | Build time | Throughput |
|---|---|---|---|---|---|---|
| 2 000 docs × 30 sentences | 1.0 M | 90 k | 12 k | 129 k | 0.7 s | 1.5 M tokens/s |
| same, added in 200 batches of 10 | 1.0 M | 90 k | 12 k | 129 k | 0.85 s | 1.2 M tokens/s |
| 10 000 docs × 30 sentences | 5.0 M | 450 k | 52 k | 558 k | 3.7 s | 1.4 M tokens/s |
| 200 docs × 2 000 sentences (long documents) | 6.8 M | 600 k | 5 k | 495 k | 5.1 s | 1.3 M tokens/s |

Against the original implementation on 300 documents (150 k tokens):

| Implementation | Time |
|---|---|
| 0.0.x `buildGraph` | 3.5 s |
| 0.1 `ImplicitNetwork.from_documents` | 0.14 s (26× faster) |

The gap widens with corpus size because the legacy pair enumeration is
quadratic in the number of mentions per document.

Typical query latencies on the 5 M-token network: `neighbors` ≈ 3 ms,
`edges(top_k=100)` ≈ 70 ms (partial sort over 558 k edges),
`cooccurrences` on the busiest edge (45 k instances) ≈ 0.3 s, dominated by
building the Python result objects; use `cooccurrence_table` for arrays.

Reproduce with:

```bash
python benchmarks/bench_build.py --docs 2000 --queries
python benchmarks/bench_build.py --docs 10000 --chunk 500 --entities 100000 --terms 200000
python benchmarks/bench_build.py --docs 200 --sentences 2000 --entities 5000
python benchmarks/bench_legacy.py --docs 300
```

## Memory

- Mentions and term occurrences are stored as NumPy int32/int64 columns, not
  per-token objects.
- Aggregated edges live in CSR matrices; instance-level cooccurrences are
  *not* materialised by default. `cooccurrences(a, b)` enumerates the pairs of
  one edge on demand, `iter_cooccurrences` streams all pairs in chunks and
  `all_cooccurrences` materialises everything only when asked.
- `NetworkConfig(store_text=False)` drops sentence texts when contexts are not
  needed.
- `ImplicitNetworkPipeline.run(chunk_size=...)` adds documents to the network
  in chunks, so annotated documents of a large corpus never have to be held in
  memory all at once.

## Throughput tips

- Extraction dominates end-to-end runtime. Use `batch_size` and, for spaCy,
  `n_process`; for GLiNER, use a GPU (`device="cuda"`) and a larger
  `batch_size`.
- Prefer `SpacyEntityExtractor` (one pass) over GLiNER + `SpacySegmenter`
  (two passes) when spaCy's label set suffices.
- Keep the window small (`window=2` is the ECCE default); the exponential
  decay makes distant cooccurrences negligible while the number of pairs grows
  with the window.
- Build once, `save()` the network and `load()` it in analysis sessions.

## Validation

See [Validation](validation.md): the engine is tested against a literal
implementation of the published definitions and against the original
0.0.x code on random corpora.
