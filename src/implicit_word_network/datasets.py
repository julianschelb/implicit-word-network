# datasets.py
"""Bundled example data."""

from __future__ import annotations

from importlib import resources

from implicit_word_network.document import Corpus


def load_example_corpus() -> Corpus:
    """Load the bundled example corpus (six English Wikipedia-style paragraphs about physicists).

    Returns:
        A ``Corpus`` with integer ids.

    Example:
        ```python
        from implicit_word_network import load_example_corpus

        corpus = load_example_corpus()
        print(len(corpus), corpus[0].text[:60])
        ```
    """
    package = resources.files("implicit_word_network.data.examples")
    with resources.as_file(package.joinpath("example_data.csv")) as path:
        return Corpus.from_csv(path, text_column="text")
