"""
Shared fixtures for implicit_word_network tests.
"""

from __future__ import annotations

import re
from unittest.mock import MagicMock

import pytest

from implicit_word_network import (
    Corpus,
    GazetteerEntityExtractor,
    ImplicitNetwork,
    NetworkConfig,
)

# ============== CORPUS FIXTURES ==============

TEXTS = [
    (
        "Richard Feynman worked at Caltech. Feynman shared the Nobel Prize with Schwinger "
        "and Tomonaga. Later, Feynman joined the Rogers Commission. The commission investigated "
        "the disaster."
    ),
    (
        "Schwinger studied at Columbia. He later taught at Harvard, where Schwinger supervised "
        "many students. Harvard remained his home."
    ),
    "Tomonaga worked in Tokyo. Tomonaga and Schwinger never met at Caltech.",
    "Nothing interesting happens here. Just plain sentences without entities.",
]

GAZETTEER = {
    "PERSON": ["Richard Feynman", "Feynman", "Schwinger", "Tomonaga"],
    "ORG": ["Caltech", "Nobel Prize", "Rogers Commission", "Columbia", "Harvard"],
    "LOC": ["Tokyo"],
}


@pytest.fixture
def texts() -> list[str]:
    """Small English corpus with known entities."""
    return list(TEXTS)


@pytest.fixture
def corpus(texts) -> Corpus:
    """Corpus built from the sample texts."""
    return Corpus.from_texts(texts)


@pytest.fixture
def gazetteer() -> dict[str, list[str]]:
    """Gazetteer covering the entities of the sample texts."""
    return {label: list(forms) for label, forms in GAZETTEER.items()}


@pytest.fixture
def extractor(gazetteer) -> GazetteerEntityExtractor:
    """Offline gazetteer extractor."""
    return GazetteerEntityExtractor(gazetteer)


@pytest.fixture
def annotated_docs(extractor, corpus):
    """Annotated sample documents."""
    return extractor.annotate_all(corpus)


@pytest.fixture
def network(annotated_docs) -> ImplicitNetwork:
    """Network built from the sample documents with the default window."""
    return ImplicitNetwork.from_documents(annotated_docs, NetworkConfig(window=2))


# ============== MOCK MODEL FIXTURES ==============

_MOCK_LEXICON = {
    "person": ["Richard Feynman", "Feynman", "Schwinger", "Tomonaga"],
    "organization": ["Caltech", "Nobel Prize", "Harvard", "Columbia", "Rogers Commission"],
    "location": ["Tokyo"],
}


def fake_gliner_inference(texts, labels, **kwargs):
    """Deterministic stand-in for ``GLiNER.inference`` (character offsets per chunk)."""
    threshold = kwargs.get("threshold", 0.5)
    results = []
    for text in texts:
        entities = []
        for label in labels:
            for form in _MOCK_LEXICON.get(label, []):
                for match in re.finditer(rf"(?<!\w){re.escape(form)}(?!\w)", text):
                    score = 0.9 if len(form) > 6 else 0.6
                    if score < threshold:
                        continue
                    entities.append(
                        {
                            "start": match.start(),
                            "end": match.end(),
                            "text": match.group(0),
                            "label": label,
                            "score": score,
                        }
                    )
        # longest match wins on overlaps (flat NER)
        entities.sort(key=lambda e: (e["start"], -(e["end"] - e["start"])))
        flat = []
        last_end = -1
        for entity in entities:
            if entity["start"] >= last_end:
                flat.append(entity)
                last_end = entity["end"]
        results.append(flat)
    return results


@pytest.fixture
def mock_gliner_model() -> MagicMock:
    """MagicMock mimicking a loaded ``gliner.GLiNER`` model."""
    model = MagicMock(name="GLiNER")
    model.inference.side_effect = fake_gliner_inference
    return model


# ============== OPTIONAL DEPENDENCY FIXTURES ==============


@pytest.fixture(scope="session")
def spacy_nlp():
    """Loaded ``en_core_web_sm`` pipeline, or skip."""
    spacy = pytest.importorskip("spacy")
    try:
        return spacy.load("en_core_web_sm")
    except OSError:  # pragma: no cover - depends on local installation
        pytest.skip("spaCy model en_core_web_sm is not installed")
