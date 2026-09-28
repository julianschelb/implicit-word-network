# CHANGELOG

<!-- version list -->

## v0.1.0 (2026-09-28)

Complete rewrite of the package as a typed, modular library.

### Features

- New high-level API: `Corpus` / `Document` ingestion, pluggable entity
  extractors, `ImplicitNetwork` container and `ImplicitNetworkPipeline`.
- `BaseEntityExtractor` interface with three implementations:
  `SpacyEntityExtractor`, `GLiNEREntityExtractor` (zero-shot, GLiNER v2.5
  checkpoints) and the offline `GazetteerEntityExtractor`.
- Pluggable sentence segmentation (`RegexSegmenter`, `SpacySegmenter`).
- Vectorised network construction: entity–entity weights are computed with a
  sparse triple product `Ω = Sᵀ K S` and can be updated incrementally with
  `ImplicitNetwork.add_documents`.
- Instance-level cooccurrences and contexts on demand (`cooccurrences`,
  `contexts`), plus the eager `all_cooccurrences` table.
- Contextual implicit entity networks: `ContextualEdgeClusterer` with a
  NumPy DBSCAN, pooled/batched context embedding and pluggable embedders
  (`BagOfWordsEmbedder`, `SentenceTransformerEmbedder`).
- Export to NetworkX, JSON, edge lists, pandas and GraphML/GEXF; persistence
  with `save` / `load`.
- Command-line interface `implicit-word-network build|summary`.
- Configurable decay functions (`exponential`, `constant`, `inverse`,
  `linear`, custom via `register_decay`).
- LOAD importance weights (`load_weight`, `load_weight_matrix`,
  `neighbors(weighting="load")`) and the EVELIN query model
  (`rank_entities`, `rank_sentences`, `rank_documents`).
- Bulk accessors `edge_table`, `cooccurrence_table`, `iter_cooccurrences`,
  `iter_entities`; `SpacyEntityExtractor(max_length=...)` for long documents.

### Performance

- Append-only columns with geometric growth and lazily compacted sparse
  accumulators: incremental updates are amortised O(1) per row.
- Entity → mention index, cached node/sentence objects, partial sort for
  `top_k` edges and GC-free bulk construction of result objects.
- About 1.4 M tokens/s network construction on a laptop core and a 25×
  speed-up over the 0.0.x `buildGraph` on 300 documents (see the
  performance page of the documentation).

### Validation

- Property-based tests compare all node and edge classes, weights, LOAD
  weights and rankings with a literal implementation of the published
  definitions and with the original 0.0.x implementation
  (`tests/test_reference_model.py`).

### Breaking changes

- The legacy camelCase functions (`readDocuments`, `parseDocuments`,
  `createCorpMat`, `buildGraph`, `clusterEdges`, `convertToNetworkX`,
  `plotNetwork`) were removed. See the migration notes in the documentation.
- `scikit-learn`, `matplotlib` and `networkx_viewer` are no longer required
  dependencies; `matplotlib` is available through the `viz` extra.

### Tooling

- `pyproject.toml` with `poetry-core` build backend and optional extras
  (`spacy`, `gliner`, `embeddings`, `viz`, `pandas`, `all`, `test`, `docs`,
  `dev`).
- Ruff, mypy, pytest (+hypothesis) and pre-commit configuration.
- GitHub Actions: CI (lint, type check, tests on Python 3.10–3.13, build),
  MkDocs Material documentation deployed to GitHub Pages, and PyPI releases
  via Trusted Publishing on version tags.

## v0.0.x

Original research prototype used by [ECCE](https://doi.org/10.1145/3487553.3524237).
