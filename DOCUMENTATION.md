# Documentation Guide

This project uses [MkDocs](https://www.mkdocs.org/) with the [Material theme](https://squidfunk.github.io/mkdocs-material/) and [mkdocstrings](https://mkdocstrings.github.io/) for auto-generated API docs. The site is deployed to GitHub Pages by the `Docs` workflow on every push to `main`.

## Quick Commands

```bash
poe docs          # Serve locally at http://127.0.0.1:8000
poe docs-build    # Strict build into site/
poe docs-deploy   # Deploy to GitHub Pages by hand
```

## Layout

| Path | Content |
|------|---------|
| `docs/index.md` | Homepage |
| `docs/getting-started.md` | Installation & first network |
| `docs/theory.md` | The IEN / LOAD / CIEN model and formulas |
| `docs/tutorials/*.md` | Task-oriented guides (building, querying, extractors, CIEN, performance, validation, migration) |
| `docs/cli.md` | CLI reference |
| `docs/api/*.md` | API reference, generated from docstrings |
| `docs/development.md`, `docs/changelog.md` | Snippets of `DEVELOPMENT.md` / `CHANGELOG.md` |
| `docs/assets/` | Images |

## API Documentation

API pages contain `::: package.module.Object` directives. To document a new
public object:

1. Write a Google-style docstring with `Args:`, `Returns:`, `Raises:` and an
   `Example:` block.
2. Add a `:::` directive to the matching page in `docs/api/`.
3. Run `poe docs-build`; the build is strict and fails on broken references.

```python
def my_function(param: str) -> int:
    """Short description.

    Args:
        param: Description of parameter.

    Returns:
        Description of return value.
    """
```

## Adding Pages

1. Create the Markdown file under `docs/`.
2. Add it to `nav` in `mkdocs.yml`.

Math is rendered with MathJax (`pymdownx.arithmatex`), so `$...$` and
`$$...$$` blocks work in every page.
