# Getting Started

This guide walks through installing the package and building a first network.

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Installing from PyPI

The core package depends only on NumPy, SciPy, NetworkX and tqdm. Entity
extraction models are optional extras:

| Extra | Installs | Enables |
|---|---|---|
| `spacy` | spaCy | `SpacyEntityExtractor`, `SpacySegmenter` (lemmas, POS tags) |
| `gliner` | gliner, torch | `GLiNEREntityExtractor` (zero-shot NER, GLiNER v2.5) |
| `embeddings` | sentence-transformers | `SentenceTransformerEmbedder` for contextual edges |
| `viz` | matplotlib | `plot_network` |
| `pandas` | pandas | `to_pandas` |
| `all` | everything above | |
| `test`, `docs`, `dev` | tooling | |

```bash
pip install "implicit-word-network[spacy]"
python -m spacy download en_core_web_sm
```

### Installing from Source

```bash
git clone https://github.com/julianschelb/implicit-word-network.git
cd implicit-word-network
pip install -e ".[dev,spacy]"
```

## Basic Concepts

The package is organised as a pipeline of pluggable components:

```
Corpus ──▶ BaseEntityExtractor ──▶ AnnotatedDocument ──▶ ImplicitNetwork ──▶ (ContextualEdgeClusterer)
```

1. A **Corpus** is an ordered collection of raw **Document** objects.
2. An **entity extractor** annotates every document with sentences, tokens
   and entity mentions (an **AnnotatedDocument**).
3. The **ImplicitNetwork** aggregates mentions into entity, term, sentence and
   document nodes and computes weighted entity–entity edges.
4. Optionally, a **ContextualEdgeClusterer** splits edges by cooccurrence
   context (contextual implicit entity networks).

### Loading a Corpus

```python
from implicit_word_network import Corpus, Document

corpus = Corpus.from_texts(["First document ...", "Second document ..."])
corpus = Corpus.from_txt("documents.txt")                  # one document per line
corpus = Corpus.from_txt("documents.txt", delimiter="\n\n")  # blank-line separated
corpus = Corpus.from_csv("documents.csv", text_column="text", id_column="doc_id")
corpus.add(Document("Another document.", id="extra", meta={"source": "manual"}))
```

### Extracting Entities

```python
from implicit_word_network import SpacyEntityExtractor, GLiNEREntityExtractor

# Fixed label set, fast, needs a spaCy pipeline
extractor = SpacyEntityExtractor("en_core_web_sm", labels=["PERSON", "ORG", "GPE"])

# Zero-shot: any natural-language labels
extractor = GLiNEREntityExtractor(
    "gliner-community/gliner_medium-v2.5",
    labels=["person", "organization", "city", "award"],
    threshold=0.5,
)

doc = extractor.annotate_text("Feynman received the Nobel Prize at Caltech.")
print([(m.text, m.label, m.sentence) for m in doc.mentions])
```

### Building the Network

```python
from implicit_word_network import ImplicitNetworkPipeline

pipeline = ImplicitNetworkPipeline(extractor, window=2)   # window in sentences
network = pipeline.run(corpus, show_progress=True)

print(network.summary())
network.top_entities(10)                     # most frequent entities
network.edges(top_k=10)                      # strongest relations
network.neighbors(("Feynman", "PERSON"))     # adjacent entities with weights
network.contexts(("Feynman", "PERSON"), ("Nobel Prize", "ORG"))  # provenance
network.entity_terms(("Feynman", "PERSON"), k=10)                # characteristic terms
network.rank_entities(("Feynman", "PERSON"), k=10)               # LOAD-weighted related entities
network.rank_sentences(("Feynman", "PERSON"), k=5)                # summarising sentences
```

### Exporting

```python
graph = network.to_networkx(min_weight=0.5)      # networkx.Graph
network.to_dict()                                 # JSON-serialisable nodes/edges
network.save("network.npz")                       # full state, reload with ImplicitNetwork.load
```

## Next Steps

- [Theory](theory.md) explains the model and its parameters.
- [Building Networks](tutorials/building-networks.md) covers the network API in depth.
- [Querying Networks](tutorials/querying.md) covers LOAD weights and entity/sentence/document ranking.
- [Entity Extractors](tutorials/entity-extractors.md) shows how to switch or write extractors.
- [Contextual Edges](tutorials/contextual-edges.md) covers CIEN edge clustering.
