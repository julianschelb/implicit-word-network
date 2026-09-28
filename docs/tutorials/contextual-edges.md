# Contextual Edges (CIEN)

An implicit network aggregates *all* cooccurrences of two entities into one
edge. When the same pair appears in different contexts, the aggregated edge
hides that ambiguity. Contextual implicit entity networks (CIEN) split an edge
into one sub-edge per context cluster. See [Theory](../theory.md#contextual-implicit-entity-networks-cien).

## On-demand clustering of one edge

```python
from implicit_word_network import ContextualEdgeClusterer

clusterer = ContextualEdgeClusterer(eps=0.25, min_samples=1, metric="cosine")
clusters = clusterer.cluster_edge(network, ("Feynman", "PERSON"), ("Nobel Prize", "ORG"))

for cluster in clusters:
    print(f"cluster {cluster.label}: {cluster.size} cooccurrences, weight {cluster.weight:.2f}")
    print("  ", cluster.contexts[0][:100])
```

Each `EdgeContextCluster` holds its member `Cooccurrence` objects, their
context texts, the summed weight (the sub-edge weight) and the mean embedding.
The cluster weights add up to the original edge weight.

## Embedders

Contexts are embedded by a `BaseContextEmbedder`:

- `BagOfWordsEmbedder` (default) — hashed bag of words, no dependencies.
  Groups contexts by lexical overlap; use a larger `eps` (0.4–0.7).
- `SentenceTransformerEmbedder` — neural sentence embeddings as in ECCE
  (`sentence-transformers/multi-qa-distilbert-cos-v1`); requires the
  `embeddings` extra. Works well with the ECCE defaults (`eps=0.25`).

```python
from implicit_word_network import SentenceTransformerEmbedder

embedder = SentenceTransformerEmbedder("sentence-transformers/multi-qa-distilbert-cos-v1", device="cuda")
clusterer = ContextualEdgeClusterer(embedder, eps=0.25)
```

Custom embedders only need an `encode(texts, batch_size=...) -> np.ndarray`
method.

## Precomputing many edges

`cluster_edges` pools the contexts of many edges and embeds them in batches of
`batch_size` contexts, which is far faster than one model call per edge:

```python
results = clusterer.cluster_edges(network, min_weight=1.0, show_progress=True)
for (a, b), clusters in results.items():
    source, target = network.entity_by_id(a), network.entity_by_id(b)
    print(source.text, target.text, [c.size for c in clusters])
```

Pass explicit `pairs=[...]` to cluster a selection, or `top_k=` for the
heaviest edges only. Through the pipeline:

```python
pipeline = ImplicitNetworkPipeline(extractor, clusterer=clusterer, window=2)
network = pipeline.run(corpus)
clusters = pipeline.cluster(network, top_k=100)
```

## DBSCAN

The clustering step is a small NumPy DBSCAN operating on a precomputed
distance matrix (`implicit_word_network.context.dbscan`), so no scikit-learn
dependency is needed. `min_samples=1` (the ECCE setting) means every context
belongs to a cluster; with larger values, outlying contexts are collected in a
noise cluster with label `-1`.
