# Entity Extractors

Entity extraction is fully decoupled from network construction. Every
extractor implements `BaseEntityExtractor.annotate()` and yields
`AnnotatedDocument` objects carrying sentences, tokens and entity mentions
as character offsets.

## spaCy

```python
from implicit_word_network import SpacyEntityExtractor

extractor = SpacyEntityExtractor(
    "en_core_web_sm",
    labels=["PERSON", "ORG", "GPE", "NORP", "LOC", "WORK_OF_ART"],  # None keeps all types
    n_process=1,
)
```

spaCy provides sentences, POS tags, lemmas and stop-word flags in a single
batched pass (`nlp.pipe`), which enables lemma-based term nodes and POS
filtering (`NetworkConfig.term_pos`). The default label set matches ECCE.

## GLiNER (zero-shot)

[GLiNER](https://github.com/urchade/GLiNER) predicts spans for arbitrary
natural-language labels. The default checkpoint is
`gliner-community/gliner_medium-v2.5`; `gliner_small-v2.5` and
`gliner_large-v2.5` trade speed for accuracy.

```python
from implicit_word_network import GLiNEREntityExtractor, SpacySegmenter

extractor = GLiNEREntityExtractor(
    "gliner-community/gliner_medium-v2.5",
    labels=["person", "organization", "location", "award", "scientific theory"],
    threshold=0.5,           # minimum confidence
    device="cuda",           # or "cpu" / "mps"
    max_chunk_words=200,     # documents are split into sentence-aligned chunks
    batch_size=8,            # chunks per forward pass
    label_map={"person": "PERSON"},  # optional renaming
    segmenter=SpacySegmenter("en_core_web_sm"),  # lemmas/POS for term nodes (optional)
)
```

Long documents are split into chunks of whole sentences that fit the model's
context window; all chunks of a document batch are scored in one batched
`inference` call and offsets are mapped back to the document. Predicted
scores are kept on every `EntityMention`.

!!! tip "Choosing labels"
    GLiNER labels are free text. Short, concrete nouns work best
    (`"company"`, `"disease"`, `"ship"`). Because entity identity is the pair
    (name, label), keep the label set stable across a corpus.

## Gazetteer (offline)

`GazetteerEntityExtractor` matches a dictionary of known surface forms. It
needs no models and is ideal for closed vocabularies, unit tests and quick
experiments:

```python
from implicit_word_network import GazetteerEntityExtractor

extractor = GazetteerEntityExtractor(
    {"PERSON": ["Richard Feynman", "Feynman"], "ORG": ["Caltech"]},
    case_sensitive=False,
)
```

Longer surface forms win at the same position and matches respect word
boundaries.

## Segmenters

Span-based extractors (GLiNER, gazetteer, custom) delegate sentence splitting
and tokenisation to a `BaseSegmenter`:

- `RegexSegmenter` (default) — dependency-free; provides sentences, tokens,
  stop-word and punctuation flags, but no POS tags or lemmas.
- `SpacySegmenter` — full linguistic features; pass it as `segmenter=` to any
  span extractor.

## Writing a custom extractor

Most custom extractors only need to predict character spans; subclass
`SpanEntityExtractor` and implement `extract_spans()`:

```python
import re
from implicit_word_network import EntitySpan, SpanEntityExtractor

class HashtagExtractor(SpanEntityExtractor):
    def extract_spans(self, texts, sentences):
        return [
            [EntitySpan(m.start(), m.end(), "HASHTAG") for m in re.finditer(r"#\w+", text)]
            for text in texts
        ]

doc = HashtagExtractor().annotate_text("Loving #physics and #Feynman.")
print([(m.text, m.label) for m in doc.mentions])
```

`texts` and `sentences` arrive per batch, so model calls can be batched. For
full control (e.g. a tagger that also segments), subclass
`BaseEntityExtractor` and implement `annotate()` directly, building
`AnnotatedDocument` objects yourself; `align_spans()` helps mapping spans to
sentences and tokens.

## Inspecting annotations

```python
doc = extractor.annotate_text("Feynman taught at Caltech. He loved bongos.")
doc.n_sentences, doc.sentence_text(1)
[(m.text, m.label, m.sentence, m.score) for m in doc.mentions]
[(t.text, t.pos, t.lemma, t.is_stop) for t in doc.tokens_in(0)]
doc.validate()   # raises on inconsistent offsets
```
