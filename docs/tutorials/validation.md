# Validation

The vectorised engine is validated against two independent oracles in the
test suite (`tests/test_reference_model.py`), on random corpora generated with
[Hypothesis](https://hypothesis.readthedocs.io/) that cover repeated mentions,
mixed entity types, documents without entities, empty sentences and windows
from 0 to 4 sentences.

## Oracle 1: the formal definitions

A literal, loop-based implementation of the model as described by Spitz &
Gertz (SIGIR 2016), Spitz, Almasian & Gertz (EVELIN, WWW 2017) and Schelb et
al. (ECCE, WWW 2022):

- entity nodes unique by normalised name and type, term nodes by word and POS;
- document–sentence, sentence–entity and sentence–term containment edges;
- entity–term edges for every pair of mention and term occurrence inside the
  same sentence;
- entity–entity edges for every pair of mentions of distinct entities within
  the window, aggregated as $\omega = \sum \exp(-\delta)$ with instance
  counts;
- the directed LOAD weights $\log(|Y| / |N(x) \cap Y|)\,\omega(x, y)$;
- the EVELIN ranking formulas $r = c + s$ for entities, sentences and
  documents.

All five sparse matrices of the network, the aggregated edge weights, the
LOAD weights and the rankings are compared exactly (up to floating-point
tolerance) with this reference, both for one-shot construction and for
document-by-document incremental updates.

## Oracle 2: the original implementation

The 0.0.x `buildGraph` function that backed ECCE is vendored unchanged into
`tests/legacy_reference.py` (minus its scikit-learn clustering part). Every
node class, every edge class and every aggregated weight of the new engine is
compared with its output on the same random corpora. The two oracles are also
checked against each other.

## What is intentionally different from 0.0.x

- The NetworkX export no longer halves the aggregated weights.
- Compound entities come from extractor spans instead of merging `I`-tagged
  tokens, so mention boundaries follow the extractor exactly.
- Term identity uses lemmas when the segmenter provides them
  (`NetworkConfig(use_lemma=False)` restores surface forms).

## Running the validation

```bash
pytest tests/test_reference_model.py -q
python benchmarks/bench_legacy.py --docs 300   # also reports the speed-up
```
