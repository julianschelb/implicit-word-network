# API Reference

This section provides detailed documentation for the Python API, auto-generated from source code docstrings.

## Core Modules

### Document Module

The [Document](document.md) module provides corpus ingestion:

- `Document` - A raw text with an identifier and metadata
- `Corpus` - Ordered collection of documents (from texts, TXT or CSV files)
- `load_example_corpus` - Bundled example data

### Annotation Module

The [Annotation](annotation.md) module defines the typed output of extractors:

- `AnnotatedDocument` - Sentences, tokens and mentions of one document
- `Sentence`, `Token`, `EntitySpan`, `EntityMention`
- `align_spans` - Map character spans to sentences and tokens

### Segmentation Module

The [Segmentation](segmentation.md) module splits text into sentences and tokens:

- `BaseSegmenter` - Abstract interface
- `RegexSegmenter` - Dependency-free segmenter
- `SpacySegmenter` - spaCy-backed segmenter (POS tags, lemmas)

### Extraction Module

The [Extraction](extraction.md) module provides entity extractors:

- `BaseEntityExtractor` - Abstract interface (`annotate()`)
- `SpanEntityExtractor` - Base class for span-only extractors
- `SpacyEntityExtractor` - spaCy NER
- `GLiNEREntityExtractor` - Zero-shot NER with GLiNER v2.5
- `GazetteerEntityExtractor` - Dictionary matching

### Network Module

The [Network](network.md) module builds and stores implicit networks:

- `ImplicitNetwork` - Nodes, edges, cooccurrences, contexts, persistence
- `NetworkConfig` - Window, decay and term filters
- `EntityNode`, `TermNode`, `SentenceRef`, `DocumentRef`, `Mention`, `Cooccurrence`, `EntityEdge`
- `rank_entities`, `rank_sentences`, `rank_documents`, `load_weight_matrix` - LOAD/EVELIN queries
- `to_networkx`, `to_dict`, `to_json`, `to_edgelist`, `to_pandas` - Exports
- `register_decay` - Custom decay functions

### Context Module

The [Context](context.md) module implements contextual implicit entity networks:

- `ContextualEdgeClusterer` - DBSCAN clustering of edge contexts
- `EdgeContextCluster` - One context cluster of an edge
- `BaseContextEmbedder`, `BagOfWordsEmbedder`, `SentenceTransformerEmbedder`
- `dbscan` - DBSCAN on a distance matrix

### Pipeline Module

The [Pipeline](pipeline.md) module composes the stages:

- `ImplicitNetworkPipeline` - Extractor + network (+ clusterer)
- `build_network` - One-call convenience wrapper

### Visualization Module

The [Visualization](visualization.md) module provides `plot_network`.

## Quick Reference

```python
from implicit_word_network import Corpus, GLiNEREntityExtractor, ImplicitNetworkPipeline

corpus = Corpus.from_txt("documents.txt")
network = ImplicitNetworkPipeline(GLiNEREntityExtractor(), window=2).run(corpus)

network.edges(top_k=10)
network.neighbors(("Feynman", "person"))
network.contexts(("Feynman", "person"), ("Caltech", "organization"))
network.to_networkx()
network.save("network.npz")
```
