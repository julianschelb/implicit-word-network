"""Naive pure-Python reference implementations used to validate the vectorised code."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

from implicit_word_network.annotation import AnnotatedDocument


def default_norm(text: str) -> str:
    return " ".join(text.split()).lower()


def naive_edge_weights(
    docs: Sequence[AnnotatedDocument],
    *,
    window: int,
    decay: Callable[[int], float] = lambda d: math.exp(-d),
    normalize: Callable[[str], str] = default_norm,
) -> dict[tuple[tuple[str, str], tuple[str, str]], tuple[float, int]]:
    """Aggregate ω(v, w) = Σ decay(δ) with explicit O(n²) loops per document."""
    result: dict[tuple[tuple[str, str], tuple[str, str]], tuple[float, int]] = {}
    for doc in docs:
        mentions = [(m.sentence, (normalize(m.text), m.label)) for m in doc.mentions]
        for i in range(len(mentions)):
            for j in range(i + 1, len(mentions)):
                s_i, key_i = mentions[i]
                s_j, key_j = mentions[j]
                delta = abs(s_i - s_j)
                if delta > window or key_i == key_j:
                    continue
                pair = (min(key_i, key_j), max(key_i, key_j))
                weight, count = result.get(pair, (0.0, 0))
                result[pair] = (weight + decay(delta), count + 1)
    return result


def naive_entity_terms(
    docs: Sequence[AnnotatedDocument],
    *,
    include_stopwords: bool = False,
    include_punctuation: bool = False,
    use_lemma: bool = True,
    normalize: Callable[[str], str] = default_norm,
) -> dict[tuple[tuple[str, str], tuple[str, str]], int]:
    """Count entity–term pairs inside the same sentence with explicit loops."""
    result: dict[tuple[tuple[str, str], tuple[str, str]], int] = {}
    for doc in docs:
        covered = doc.entity_token_mask()
        for sentence in doc.sentences:
            entities = [
                (normalize(m.text), m.label) for m in doc.mentions if m.sentence == sentence.index
            ]
            terms = []
            for position, token in enumerate(doc.tokens):
                if token.sentence != sentence.index or covered[position] or token.is_space:
                    continue
                if token.is_punct and not include_punctuation:
                    continue
                if token.is_stop and not include_stopwords:
                    continue
                base = token.lemma if (use_lemma and token.lemma) else token.text
                terms.append((base.lower(), token.pos))
            for entity in entities:
                for term in terms:
                    result[(entity, term)] = result.get((entity, term), 0) + 1
    return result
