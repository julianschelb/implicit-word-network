# Building Networks

This tutorial covers the `ImplicitNetwork` API: configuration, construction,
querying and persistence. The examples use the offline
`GazetteerEntityExtractor` so they run without models; swap in any other
extractor for real corpora.

```python
from implicit_word_network import GazetteerEntityExtractor, ImplicitNetwork, NetworkConfig

texts = [
    "Richard Feynman worked at Caltech. Feynman shared the Nobel Prize with Schwinger "
    "and Tomonaga. Later, Feynman joined the Rogers Commission.",
    "Schwinger studied at Columbia and later taught at Harvard.",
]
extractor = GazetteerEntityExtractor({
    "PERSON": ["Richard Feynman", "Feynman", "Schwinger", "Tomonaga"],
    "ORG": ["Caltech", "Nobel Prize", "Rogers Commission", "Columbia", "Harvard"],
})
docs = extractor.annotate_all(texts)
```

## Configuration

`NetworkConfig` controls how mentions become nodes and edges:

```python
config = NetworkConfig(
    window=2,                 # cooccurrence window in sentences
    decay="exponential",      # exp(-δ); or "constant", "inverse", "linear", custom
    include_stopwords=False,  # stop words are not term nodes
    include_punctuation=False,
    term_pos={"NOUN", "PROPN", "VERB", "ADJ"},  # needs a POS-tagging segmenter (spaCy)
    use_lemma=True,           # term identity = lemma when available
    store_text=True,          # keep sentence texts (needed for contexts)
)
network = ImplicitNetwork.from_documents(docs, config)
```

Custom decay functions operate on a NumPy array of sentence distances:

```python
import numpy as np
from implicit_word_network import register_decay

register_decay("gaussian", lambda d: np.exp(-(d.astype(float) ** 2) / 2.0))
network = ImplicitNetwork.from_documents(docs, NetworkConfig(decay="gaussian"))
```

## Incremental construction

Cooccurrences never cross document boundaries, so networks can grow document
by document (or batch by batch) with identical results to a one-shot build:

```python
network = ImplicitNetwork(NetworkConfig(window=2))
network.add_documents(docs[:1])
network.add_documents(docs[1:])   # new entities/terms extend the vocabularies
```

Document ids must be unique; adding a document twice raises `ValueError`.

## Nodes

```python
network.n_documents, network.n_sentences, network.n_entities, network.n_terms

feynman = network.entity("feynman", "PERSON")   # lookup is case/whitespace-insensitive
network.entities(label="ORG")                    # all organisations
network.top_entities(5)                          # by mention count
network.entity_labels()                          # ["ORG", "PERSON"]

network.terms(), network.top_terms(10), network.term("worked")
network.documents(), network.document_by_id(0), network.sentences_of_document(0)
```

Entities are referenced interchangeably by `EntityNode`, integer id or
`(text, label)` tuple in every query method.

## Edges

```python
for edge in network.edges(min_weight=0.5, top_k=10, labels=["PERSON", "ORG"]):
    print(edge.source.text, edge.target.text, edge.weight, edge.count)

network.weight(feynman, ("Nobel Prize", "ORG"))   # aggregated ω
network.count(feynman, ("Nobel Prize", "ORG"))    # number of cooccurrences
network.neighbors(feynman, k=5)                   # [(EntityNode, weight), ...]
```

### Instance level: cooccurrences and contexts

Every aggregated edge is backed by cooccurrence instances that are enumerated
on demand:

```python
for cooc in network.cooccurrences(feynman, ("Schwinger", "PERSON")):
    print(cooc.delta, round(cooc.weight, 3), cooc.document, cooc.sentence_span)
    print(network.context_of(cooc))     # text of the spanned sentences

network.contexts(feynman, ("Schwinger", "PERSON"))   # all contexts at once
network.mentions_of(feynman)                          # every mention with offsets
network.sentences_of(feynman)                         # sentences mentioning the entity
```

`all_cooccurrences()` returns the complete instance table as NumPy arrays for
bulk analyses.

### Entity–term edges

```python
network.entity_terms(feynman, k=10)        # [(TermNode, shared sentences), ...]
network.term_entities(("prize", ""), k=5)  # entities sharing sentences with a term
```

## Matrices

The sparse matrices behind the network are exposed for custom analyses
(SciPy CSR):

| Property | Shape | Content |
|---|---|---|
| `entity_entity_matrix` | entities × entities | aggregated weights ω |
| `entity_count_matrix` | entities × entities | cooccurrence counts |
| `entity_term_matrix` | entities × terms | same-sentence counts |
| `sentence_entity_matrix` | sentences × entities | mention counts |
| `sentence_term_matrix` | sentences × terms | occurrence counts |
| `sentence_document()` | sentences | document index per sentence |

## Export and persistence

```python
graph = network.to_networkx(min_weight=0.5, include_terms=True, max_terms_per_entity=5)
data = network.to_dict(top_k=100)          # JSON-serialisable
from implicit_word_network.network import to_json, to_edgelist, to_pandas
to_json(network, "network.json")
nodes, edges = to_pandas(network)          # needs the pandas extra

network.save("network.npz")
network = ImplicitNetwork.load("network.npz")
```

Saved networks contain the full state (tables, vocabularies, matrices and
sentence texts), so loaded networks can be queried and extended further.
