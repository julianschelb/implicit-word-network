# Migrating from 0.0.x

Version 0.1 is a rewrite; the camelCase functions of the prototype were
replaced by a typed object API. The table maps old calls to their
replacements.

| 0.0.x | 0.1 |
|---|---|
| `wn.readDocuments(path)` | `Corpus.from_txt(path)` |
| `sp.load(...)` + `wn.parseDocuments(D, entity_types, nlp=nlp)` | `SpacyEntityExtractor(model, labels=entity_types).annotate_all(corpus)` |
| `wn.createCorpMat(D_parsed)` | not needed (annotations are typed `AnnotatedDocument`s) |
| `wn.buildGraph(D_mat, c)` | `ImplicitNetwork.from_documents(docs, NetworkConfig(window=c))` |
| `V["entities"]`, `Ep[("e", "e")]` | `network.entities()`, `network.edges()` |
| `wn.clusterEdges(Ep, D_mat, model=...)` | `ContextualEdgeClusterer(SentenceTransformerEmbedder(...)).cluster_edges(network)` |
| `wn.convertToNetworkX(V, Ep)` | `network.to_networkx()` |
| `wn.plotNetwork(G, mode="show")` | `plot_network(network, show=True)` (extra `viz`) |

```python
# 0.0.x
D = wn.readDocuments("data.txt")
D_parsed = wn.parseDocuments(D, ["PERSON", "ORG"], nlp=spacy.load("en_core_web_sm"))
V, Ep = wn.buildGraph(wn.createCorpMat(D_parsed), c=2)
G = wn.convertToNetworkX(V, Ep)

# 0.1
from implicit_word_network import Corpus, SpacyEntityExtractor, build_network
network = build_network(
    Corpus.from_txt("data.txt"),
    extractor=SpacyEntityExtractor("en_core_web_sm", labels=["PERSON", "ORG"]),
    window=2,
)
G = network.to_networkx()
```

Behavioural differences:

- Edge weights follow the literature exactly (`ω = Σ exp(-δ)`); the
  prototype's NetworkX export additionally halved the weights.
- Compound entities come from the extractor's spans instead of merging
  `I`-tagged tokens.
- The interactive `networkx_viewer` mode was dropped; use NetworkX/matplotlib
  or export to GraphML/GEXF for Gephi.
