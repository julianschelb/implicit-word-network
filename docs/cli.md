# CLI Reference

The package installs the `implicit-word-network` command.

## `build`

Build a network and write it to a file. The output format is inferred from
the file extension: `.json` (nodes + edges), `.csv` (edge list), `.graphml` /
`.gexf` (NetworkX, optionally with term nodes) or `.npz` (full network state
for `ImplicitNetwork.load`).

```bash
implicit-word-network build documents.txt -o network.json
implicit-word-network build documents.csv -o network.graphml \
  --text-column body --id-column doc_id \
  --extractor gliner --model gliner-community/gliner_medium-v2.5 \
  --labels person organization location --threshold 0.4 --device cuda \
  --window 2 --decay exponential --min-weight 0.5 --include-terms
implicit-word-network build documents.txt -o network.npz --extractor gazetteer --gazetteer gaz.json
```

Input files: plain text (one document per non-empty line) or CSV/TSV with a
text column.

| Option | Description |
|---|---|
| `--extractor {spacy,gliner,gazetteer}` | Entity extractor (default `spacy`) |
| `--model` | spaCy pipeline or GLiNER model id |
| `--labels L [L ...]` | Entity types to keep / zero-shot labels |
| `--threshold` | GLiNER confidence threshold |
| `--gazetteer FILE` | JSON `{label: [surface forms]}` for the gazetteer extractor |
| `--device` | GLiNER device (`cpu`, `cuda`, `mps`) |
| `--window` | Context window in sentences (default 2) |
| `--decay` | `exponential`, `constant`, `inverse`, `linear` |
| `--include-stopwords` | Keep stop words as term nodes |
| `--min-weight`, `--top-k` | Edge filters for the export |
| `--include-terms` | Add term nodes (GraphML/GEXF) |
| `--batch-size` | Documents per extractor call |
| `-q`, `--quiet` | Hide progress bars |

## `summary`

Build a network and print statistics, the most frequent entities and the
strongest edges:

```bash
implicit-word-network summary documents.txt --extractor spacy --top 15
```
