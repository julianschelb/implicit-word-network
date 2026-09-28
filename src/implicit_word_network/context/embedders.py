# context/embedders.py
"""Context embedders used for contextual implicit entity networks (CIEN)."""

from __future__ import annotations

import re
import zlib
from abc import ABC, abstractmethod
from collections.abc import Collection, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from implicit_word_network._utils import require
from implicit_word_network.segmentation import ENGLISH_STOPWORDS

DEFAULT_SENTENCE_TRANSFORMER = "sentence-transformers/multi-qa-distilbert-cos-v1"
"""Sentence-transformer used by ECCE for edge contexts."""

_TOKEN = re.compile(r"\w+")


class BaseContextEmbedder(ABC):
    """Abstract base class for text embedders.

    Embedders map cooccurrence contexts (the sentences spanned by two entity
    mentions) to dense vectors so that
    ``ContextualEdgeClusterer`` can group
    parallel edges by context. Subclasses must implement ``encode``.

    Available implementations:

    - ``BagOfWordsEmbedder`` — hashed, L2-normalised bag of words; no
      dependencies, deterministic, good for tests and lexical similarity.
    - ``SentenceTransformerEmbedder`` — neural sentence embeddings via
      ``sentence-transformers``.
    """

    @abstractmethod
    def encode(self, texts: Sequence[str], *, batch_size: int = 32) -> NDArray[np.float32]:
        """Embed texts.

        Args:
            texts: Input texts.
            batch_size: Batch size hint for model-based implementations.

        Returns:
            A ``(len(texts), dim)`` float32 array.
        """
        ...


class BagOfWordsEmbedder(BaseContextEmbedder):
    """Hashed bag-of-words embedder (dependency-free baseline).

    Tokens are lowercased, stop words removed and hashed into
    ``n_features`` buckets; rows are L2-normalised so that cosine similarity
    equals the dot product.

    Args:
        n_features: Dimensionality of the hashed space.
        stopwords: Words to ignore (defaults to English stop words).
    """

    def __init__(self, n_features: int = 2048, *, stopwords: Collection[str] | None = None) -> None:
        if n_features < 1:
            raise ValueError("n_features must be >= 1")
        self.n_features = n_features
        self.stopwords: frozenset[str] = (
            ENGLISH_STOPWORDS if stopwords is None else frozenset(w.lower() for w in stopwords)
        )

    def encode(self, texts: Sequence[str], *, batch_size: int = 32) -> NDArray[np.float32]:
        matrix = np.zeros((len(texts), self.n_features), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in _TOKEN.findall(text.lower()):
                if token in self.stopwords:
                    continue
                matrix[row, zlib.crc32(token.encode("utf-8")) % self.n_features] += 1.0
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        np.divide(matrix, norms, out=matrix, where=norms > 0)
        return matrix


class SentenceTransformerEmbedder(BaseContextEmbedder):
    """Neural sentence embeddings via ``sentence-transformers``.

    Requires the ``embeddings`` extra. The model is loaded lazily on first use.

    Args:
        model: Model name or path, or a loaded ``SentenceTransformer``.
        device: Torch device (``None`` lets the library choose).
        normalize: L2-normalise embeddings (recommended with cosine distance).

    Example:
        ```python
        from implicit_word_network import SentenceTransformerEmbedder

        embedder = SentenceTransformerEmbedder("sentence-transformers/multi-qa-distilbert-cos-v1")
        vectors = embedder.encode(["Feynman received the Nobel Prize."])
        print(vectors.shape)  # (1, 768)
        ```
    """

    def __init__(
        self,
        model: str | Any = DEFAULT_SENTENCE_TRANSFORMER,
        *,
        device: str | None = None,
        normalize: bool = True,
    ) -> None:
        self._model_name: str | None = model if isinstance(model, str) else None
        self._model: Any | None = None if isinstance(model, str) else model
        self.device = device
        self.normalize = normalize

    @property
    def model(self) -> Any:
        """The underlying ``SentenceTransformer`` (loaded lazily)."""
        if self._model is None:
            require(
                "sentence_transformers", extra="embeddings", purpose="SentenceTransformerEmbedder"
            )
            from sentence_transformers import SentenceTransformer

            assert self._model_name is not None
            self._model = SentenceTransformer(self._model_name, device=self.device)
        return self._model

    def encode(self, texts: Sequence[str], *, batch_size: int = 32) -> NDArray[np.float32]:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        vectors = self.model.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)
