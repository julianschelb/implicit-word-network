# Examples

All examples live in the [`examples/`](https://github.com/julianschelb/implicit-word-network/tree/main/examples) folder of the repository.

| File | What it shows | Requirements |
|---|---|---|
| [`example.ipynb`](https://github.com/julianschelb/implicit-word-network/blob/main/examples/example.ipynb) | Executed notebook: extraction, network construction, exploration, LOAD/EVELIN ranking, CIEN edge clustering, export | none (offline gazetteer) |
| [`example.py`](https://github.com/julianschelb/implicit-word-network/blob/main/examples/example.py) | End-to-end script with spaCy NER, provenance, clustering and plotting | `spacy` extra + `en_core_web_sm`, `viz` extra for the plot |
| [`example_gliner.py`](https://github.com/julianschelb/implicit-word-network/blob/main/examples/example_gliner.py) | Zero-shot entity types with GLiNER v2.5 | `gliner` extra (downloads a checkpoint) |
| [`example_data.csv`](https://github.com/julianschelb/implicit-word-network/blob/main/examples/example_data.csv) | The bundled example corpus (also available via `load_example_corpus()`) | |

## Minimal end-to-end script

```python
from implicit_word_network import Corpus, SpacyEntityExtractor, ImplicitNetworkPipeline

corpus = Corpus.from_txt("documents.txt")
network = ImplicitNetworkPipeline(SpacyEntityExtractor(), window=2).run(corpus, show_progress=True)

print(network.summary())
for edge in network.edges(top_k=10):
    print(f"{edge.weight:6.2f}  {edge.source.text} -- {edge.target.text}")

network.save("network.npz")
```

## Benchmarks

The [`benchmarks/`](https://github.com/julianschelb/implicit-word-network/tree/main/benchmarks) folder contains a synthetic corpus generator and scripts to time network construction and to compare with the original 0.0.x implementation; see [Performance & Scaling](tutorials/performance.md).
