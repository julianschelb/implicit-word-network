# Contributing

Thanks for considering a contribution! This project follows a lightweight
GitHub flow.

## Ways to contribute

- **Bug reports and feature requests** via the issue templates.
- **Documentation** improvements (the `docs/` folder and docstrings).
- **New extractors, embedders or exports** that implement the abstract base
  classes (`BaseEntityExtractor`, `BaseSegmenter`, `BaseContextEmbedder`).
- **Performance work**: see `benchmarks/` for reproducible synthetic
  workloads.

## Development setup

```bash
git clone https://github.com/julianschelb/implicit-word-network.git
cd implicit-word-network
pip install -e ".[dev,spacy,viz,pandas]"
python -m spacy download en_core_web_sm
pre-commit install && pre-commit install --hook-type commit-msg
```

See [DEVELOPMENT.md](DEVELOPMENT.md) for the full tool reference.

## Pull requests

1. Create a branch from `main`.
2. Add or update tests; keep the offline test suite offline (mock models).
3. Run `poe check` (ruff, mypy, pytest) and `poe docs-build` if you touched
   documentation or docstrings.
4. Use [Conventional Commits](https://www.conventionalcommits.org/) messages
   (`feat:`, `fix:`, `docs:`, `perf:`, `test:`, `ci:`, `chore:`).
5. Open the pull request against `main`; CI must pass before merging.

## Code style

- Python 3.10+, fully type-annotated public API (mypy runs in CI).
- Google-style docstrings with an `Example:` block for public classes.
- Optional dependencies are imported lazily inside the class/function that
  needs them and reported through `_utils.require`.
- Vectorise with NumPy/SciPy; avoid per-token Python loops in the network
  layer.

## Reporting security issues

Please see [SECURITY.md](SECURITY.md).
