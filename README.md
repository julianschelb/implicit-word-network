# Implicit Word Network

[![CI](https://github.com/julianschelb/implicit-word-network/actions/workflows/ci.yml/badge.svg)](https://github.com/julianschelb/implicit-word-network/actions/workflows/ci.yml)
[![Docs](https://github.com/julianschelb/implicit-word-network/actions/workflows/docs.yml/badge.svg)](https://julianschelb.github.io/implicit-word-network/)
[![PyPI](https://img.shields.io/pypi/v/implicit-word-network.svg)](https://pypi.org/project/implicit-word-network/)
[![Python](https://img.shields.io/pypi/pyversions/implicit-word-network.svg)](https://pypi.org/project/implicit-word-network/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Implicit Word Network** extracts and explores *(contextual) implicit entity
networks* from text corpora: cooccurrence graphs over named entities and terms
in which relations are weighted by the sentence distance of their mentions.
The model was introduced by Spitz & Gertz and powers entity-centric corpus
exploration tools such as [ECCE](https://doi.org/10.1145/3487553.3524237).

Documentation: <https://julianschelb.github.io/implicit-word-network/>

![Entity network extracted from the bundled example corpus](docs/assets/example-network.png)

## Features

- **Pluggable entity extraction** – spaCy NER, zero-shot NER with
  [GLiNER v2.5](https://huggingface.co/gliner-community/gliner_medium-v2.5),
  offline gazetteers, or your own `BaseEntityExtractor`.
- **Vectorised network construction** – entity–entity weights
  `ω(v, w) = Σ exp(−δ)` are computed with sparse matrix products and updated
  incrementally as documents are added.
- **Provenance** – every edge can be expanded into its cooccurrence instances
  and the sentence contexts they stem from.
- **LOAD/EVELIN queries** – directed importance weights
  `log(|Y| / |N(x) ∩ Y|) · ω` and ranking of related entities, summarising
  sentences and documents for one or more query entities.
- **Contextual edges (CIEN)** – cluster the contexts of an edge with DBSCAN
  over bag-of-words or sentence-transformer embeddings, on demand or batched.
- **Interoperability** – export to NetworkX, JSON, CSV edge lists, pandas,
  GraphML/GEXF; save and reload full networks.
- **Validated** – property-based tests check the engine against a literal
  implementation of the published definitions and against the original
  0.0.x code; the vectorised builder is 25× faster than the original and
  processes about 1.4 M tokens/s.
- **Typed, tested, documented** – Python 3.10+, mypy-clean, offline test
  suite, MkDocs Material docs with an executed example notebook.

## Installation

```bash
pip install implicit-word-network

# entity extraction backends (choose one or more)
pip install "implicit-word-network[spacy]" && python -m spacy download en_core_web_sm
pip install "implicit-word-network[gliner]"

# optional: neural context embeddings, plotting, pandas export, everything
pip install "implicit-word-network[embeddings]"
pip install "implicit-word-network[viz]"
pip install "implicit-word-network[all]"
```

## Quick Start

```python
from implicit_word_network import Corpus, GLiNEREntityExtractor, ImplicitNetworkPipeline

# 1. Load documents (one per line; CSV and Python lists work too)
corpus = Corpus.from_txt("documents.txt")

# 2. Choose an extractor – GLiNER accepts arbitrary labels
extractor = GLiNEREntityExtractor(
    "gliner-community/gliner_medium-v2.5",
    labels=["person", "organization", "location", "award"],
    threshold=0.5,
)

# 3. Build the network (window = cooccurrence distance in sentences)
pipeline = ImplicitNetworkPipeline(extractor, window=2)
network = pipeline.run(corpus, show_progress=True)
print(network.summary())

# 4. Explore
for edge in network.edges(top_k=10):
    print(f"{edge.weight:6.2f}  {edge.source.text} -- {edge.target.text}")

feynman = network.entity("Feynman", "person")
network.neighbors(feynman, k=5)                       # strongest related entities
network.entity_terms(feynman, k=10)                   # characteristic terms
network.contexts(feynman, ("Nobel Prize", "award"))   # where do they cooccur?
network.rank_entities([feynman, ("Nobel Prize", "award")], k=5)   # LOAD/EVELIN ranking
network.rank_sentences(feynman, k=3)                  # summarising sentences

# 5. Export
graph = network.to_networkx(min_weight=0.5)
network.save("network.npz")
```

With spaCy instead of GLiNER:

```python
from implicit_word_network import SpacyEntityExtractor

extractor = SpacyEntityExtractor("en_core_web_sm", labels=["PERSON", "ORG", "GPE", "NORP", "LOC", "WORK_OF_ART"])
```

### Contextual implicit entity networks

Split an edge into context clusters (Spitz & Gertz, 2018; ECCE):

```python
from implicit_word_network import ContextualEdgeClusterer, SentenceTransformerEmbedder

clusterer = ContextualEdgeClusterer(SentenceTransformerEmbedder(), eps=0.25)
for cluster in clusterer.cluster_edge(network, feynman, ("Nobel Prize", "award")):
    print(cluster.size, round(cluster.weight, 2), cluster.contexts[0][:80])
```

### Custom extractors

```python
import re
from implicit_word_network import EntitySpan, SpanEntityExtractor

class HashtagExtractor(SpanEntityExtractor):
    def extract_spans(self, texts, sentences):
        return [[EntitySpan(m.start(), m.end(), "HASHTAG") for m in re.finditer(r"#\w+", t)] for t in texts]
```

## Examples and Benchmarks

- [`examples/example.ipynb`](examples/example.ipynb) – executed, offline walkthrough of the whole API
- [`examples/example.py`](examples/example.py), [`examples/example_gliner.py`](examples/example_gliner.py) – spaCy and GLiNER scripts
- [`benchmarks/`](benchmarks/) – synthetic corpora, construction throughput and the comparison with 0.0.x

## Command-Line Interface

```bash
implicit-word-network build documents.txt -o network.json --extractor spacy --window 2
implicit-word-network build documents.csv -o network.graphml --extractor gliner --labels person company
implicit-word-network summary documents.txt --top 15
```

## Theoretical Background

1. Spitz, A. & Gertz, M. (2016). Terms over LOAD: Leveraging Named Entities for Cross-Document Extraction and Summarization of Events. *SIGIR '16*. https://doi.org/10.1145/2911451.2911529
2. Spitz, A. & Gertz, M. (2018). Exploring Entity-centric Networks in Entangled News Streams. *WWW '18 Companion*. https://doi.org/10.1145/3184558.3188726
3. Spitz, A. (2019). Implicit Entity Networks: A Versatile Document Model. Heidelberg University. https://doi.org/10.11588/HEIDOK.00026328
4. Schelb, J., Ehrmann, M., Romanello, M. & Spitz, A. (2022). ECCE: Entity-centric Corpus Exploration Using Contextual Implicit Networks. *WWW '22 Companion*. https://doi.org/10.1145/3487553.3524237

## Development

See [DEVELOPMENT.md](DEVELOPMENT.md) for setup, testing, linting, docs and the
release process.

```bash
pip install -e ".[dev,spacy]"
poe check   # ruff, mypy, pytest
```

## License

MIT – see [LICENSE](LICENSE).
