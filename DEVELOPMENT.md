# Development Guide

This document covers setup and workflows for contributing to Implicit Word Network.

## Prerequisites

- Python 3.10 - 3.13
- pip (or uv / poetry)

## Installation

```bash
# Clone the repository
git clone https://github.com/julianschelb/implicit-word-network.git
cd implicit-word-network

# Install in editable mode with development dependencies
pip install -e ".[dev]"

# Optional extras used by parts of the test suite
pip install -e ".[dev,spacy,viz,pandas]"
python -m spacy download en_core_web_sm
```

The `gliner` and `embeddings` extras are only needed to run real models; the
standard test suite mocks them.

## Running Tests

```bash
# Run all (offline) tests
poe test

# With coverage report
poe test-cov

# Integration tests that download real models (GLiNER v2.5)
poe test-integration
```

Tests marked `spacy` are skipped automatically when spaCy or `en_core_web_sm`
is not installed.

## Linting & Formatting

The project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting.

```bash
poe lint          # ruff check
poe format        # ruff format
poe format-check  # ruff format --check
```

## Type Checking

[Mypy](https://mypy.readthedocs.io/) runs in strict-ish mode over the package:

```bash
poe typecheck
```

## Everything at once

```bash
poe check   # lint, format-check, typecheck, test
```

## Pre-commit Hooks

```bash
pre-commit install
pre-commit install --hook-type commit-msg
pre-commit run --all-files
```

## Building Documentation

```bash
poe docs          # live-reload dev server
poe docs-build    # one-off strict build into site/
```

## Continuous Integration

| Workflow | Trigger | What it does |
|---|---|---|
| `ci.yml` | push / PR | Ruff lint + format check, mypy (3.12), pytest on Python 3.10–3.13, distribution build |
| `docs.yml` | push / PR | `mkdocs build --strict`; deploys to GitHub Pages on pushes to `main`/`master` |
| `release.yml` | after CI on `main` | semantic-release: version bump, changelog, tag, GitHub release; then PyPI publish via Trusted Publishing |

## Semantic Versioning & Releases

The project uses [Conventional Commits](https://www.conventionalcommits.org/)
and [python-semantic-release](https://python-semantic-release.readthedocs.io/)
for fully automated version bumps and changelog generation, exactly like
LociSimiles.

### Commit Message Format

```
<type>(<optional scope>): <description>

[optional body]

[optional footer(s)]
```

| Prefix | Example | Version Bump |
|---|---|---|
| `fix:` | `fix: handle empty document` | Patch (0.1.0 → 0.1.1) |
| `feat:` | `feat: add GraphML export` | Minor (0.1.0 → 0.2.0) |
| `feat!:` or `BREAKING CHANGE:` | `feat!: rename pipeline API` | Major |
| `perf:` | `perf: cache mention index` | Patch |
| `docs:`, `chore:`, `test:`, `refactor:`, `ci:` | | No release |

A pre-commit hook validates commit messages (`pre-commit install --hook-type commit-msg`).

### How Releases Work

1. Merge a PR into `main` (or push to it).
2. CI runs all checks (lint, typecheck, tests, build).
3. If CI passes, the **Release** workflow runs automatically.
4. `semantic-release` analyses the commits since the last tag. If there are
   releasable commits it bumps the version in `pyproject.toml` and
   `__init__.py`, updates `CHANGELOG.md`, commits, tags `vX.Y.Z`, creates a
   GitHub Release with the changelog and attaches the built distribution.
5. The distribution is published to PyPI through
   [Trusted Publishing](https://docs.pypi.org/trusted-publishers/) (OIDC),
   so no PyPI token is stored anywhere.

One-time setup on PyPI (project → Publishing → add a trusted publisher):
owner `julianschelb`, repository `implicit-word-network`, workflow
`release.yml`, environment `pypi`.

### Manual Version Check

```bash
semantic-release version --print   # preview the next version (dry run)
```

## Quick Reference

| Task | Command |
|---|---|
| Run tests | `poe test` |
| Lint | `poe lint` |
| Format | `poe format` |
| Type check | `poe typecheck` |
| All checks | `poe check` |
| All pre-commit hooks | `pre-commit run --all-files` |
| Serve docs | `poe docs` |
| Build docs | `poe docs-build` |
