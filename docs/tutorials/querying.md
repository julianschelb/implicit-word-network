# Querying Networks

Beyond raw edge weights, the package implements the LOAD/EVELIN query model
(see [Theory](../theory.md#load-importance-weights)) for entity-centric
exploration: *which entities, sentences and documents matter for a given set
of query entities?*

```python
from implicit_word_network import GLiNEREntityExtractor, ImplicitNetworkPipeline, load_example_corpus

network = ImplicitNetworkPipeline(
    GLiNEREntityExtractor(labels=["person", "organization", "award", "location"]), window=2
).run(load_example_corpus())
```

## Directed importance (LOAD weights)

```python
feynman = network.entity("Feynman", "person")

network.neighbors(feynman, k=5)                    # raw ω = Σ exp(−δ), symmetric
network.neighbors(feynman, k=5, weighting="load")  # log(|Y| / |N(x) ∩ Y|) · ω, directed
network.load_weight(feynman, ("Nobel Prize", "award"))
network.load_weight_matrix()                       # sparse n_entities × n_entities
```

LOAD weights down-weight neighbour types that are connected to almost every
entity of that type (an IDF-like effect), which makes them better suited for
ranking than the raw cooccurrence weight.

## Entity queries

```python
# single query entity: neighbours ranked by normalised LOAD weight, in [0, 1]
network.rank_entities(feynman, k=10)
network.rank_entities(feynman, label="award", k=3)

# multiple query entities: cohesion (how many query entities the candidate
# connects to, minus one) plus the normalised weight sum, in [0, |Q|]
network.rank_entities([feynman, ("Nobel Prize", "award")], k=10)
```

Use `weighting="raw"` to rank by the symmetric weight instead.

## Sentence and document queries (summarisation and provenance)

```python
for sentence, score in network.rank_sentences([feynman, ("Nobel Prize", "award")], k=5):
    print(round(score, 2), sentence.document, sentence.text[:100])

for document, score in network.rank_documents(feynman, k=3):
    print(round(score, 2), document.id, document.meta)
```

Sentence scores add the number of query entities contained in the sentence
(cohesion) and the fraction of the query entities' most important terms
(`n_terms` per entity, from the entity–term edges) that the sentence
contains. Document scores propagate the sentence scores as in EVELIN's page
ranking.

## Custom analyses on the matrices

All rankings are thin layers over the sparse matrices, which are available
for your own scoring functions:

```python
W = network.entity_entity_matrix        # symmetric weights
L = network.load_weight_matrix()        # directed LOAD weights
S_e = network.sentence_entity_matrix    # sentence × entity
S_t = network.sentence_term_matrix      # sentence × term
ET = network.entity_term_matrix         # entity × term
labels = network.entity_label_ids()     # type id per entity
```

For example, entity centrality with NetworkX:

```python
import networkx as nx
graph = network.to_networkx(min_weight=0.5)
central = nx.pagerank(graph, weight="weight")
```
