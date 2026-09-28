# pipeline.py
"""High-level pipeline: documents → entity extraction → implicit network."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from typing import Any

from implicit_word_network._utils import batched
from implicit_word_network.annotation import AnnotatedDocument
from implicit_word_network.context.clustering import ContextualEdgeClusterer, EdgeContextCluster
from implicit_word_network.document import Document
from implicit_word_network.extraction import BaseEntityExtractor
from implicit_word_network.network import ImplicitNetwork, NetworkConfig


class ImplicitNetworkPipeline:
    """End-to-end construction of (contextual) implicit entity networks.

    The pipeline composes three pluggable stages:

    1. an **entity extractor** (``BaseEntityExtractor``)
       that annotates raw documents,
    2. the **network builder** (``ImplicitNetwork``
       with a ``NetworkConfig``) that
       aggregates cooccurrences into weighted edges, and
    3. an optional **edge clusterer**
       (``ContextualEdgeClusterer``) that
       splits edges by cooccurrence context.

    Args:
        extractor: Entity extractor. Defaults to
            ``SpacyEntityExtractor``
            (requires the ``spacy`` extra).
        config: Network construction parameters. Mutually exclusive with
            keyword overrides such as ``window=3``.
        clusterer: Optional contextual edge clusterer used by ``cluster``.
        normalize_entity: Custom entity-name normalisation.
        **config_overrides: Fields of ``NetworkConfig``.

    Example:
        ```python
        from implicit_word_network import Corpus, GLiNEREntityExtractor, ImplicitNetworkPipeline

        corpus = Corpus.from_txt("news.txt")
        pipeline = ImplicitNetworkPipeline(
            GLiNEREntityExtractor(labels=["person", "organization", "location"]),
            window=2,
        )
        network = pipeline.run(corpus, show_progress=True)
        print(network.summary())
        for edge in network.edges(top_k=5):
            print(edge.source.text, "—", edge.target.text, round(edge.weight, 2))
        ```
    """

    def __init__(
        self,
        extractor: BaseEntityExtractor | None = None,
        *,
        config: NetworkConfig | None = None,
        clusterer: ContextualEdgeClusterer | None = None,
        normalize_entity: Callable[[str], str] | None = None,
        **config_overrides: Any,
    ) -> None:
        if config is not None and config_overrides:
            raise ValueError("Pass either `config` or keyword overrides, not both")
        self.config: NetworkConfig = (
            config if config is not None else NetworkConfig(**config_overrides)
        )
        self._extractor: BaseEntityExtractor | None = extractor
        self.clusterer: ContextualEdgeClusterer | None = clusterer
        self.normalize_entity = normalize_entity

    @property
    def extractor(self) -> BaseEntityExtractor:
        """The entity extractor (a spaCy extractor is created lazily by default)."""
        if self._extractor is None:
            from implicit_word_network.extraction.spacy import SpacyEntityExtractor

            self._extractor = SpacyEntityExtractor()
        return self._extractor

    def annotate(
        self,
        documents: Iterable[Document | str],
        *,
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> Iterator[AnnotatedDocument]:
        """Run only the extraction stage."""
        return self.extractor.annotate(
            documents, batch_size=batch_size, show_progress=show_progress
        )

    def run(
        self,
        documents: Iterable[Document | str],
        *,
        batch_size: int = 32,
        chunk_size: int = 256,
        show_progress: bool = False,
        network: ImplicitNetwork | None = None,
    ) -> ImplicitNetwork:
        """Extract entities and build (or extend) an implicit network.

        Args:
            documents: A ``Corpus``, or any
                iterable of documents / strings.
            batch_size: Documents per extractor call.
            chunk_size: Documents per network update (bounds peak memory).
            show_progress: Display a progress bar.
            network: Existing network to extend instead of creating a new one.

        Returns:
            The resulting network.
        """
        if network is None:
            network = ImplicitNetwork(self.config, normalize_entity=self.normalize_entity)
        annotated = self.annotate(documents, batch_size=batch_size, show_progress=show_progress)
        for chunk in batched(annotated, max(chunk_size, 1)):
            network.add_documents(chunk)
        return network

    def cluster(
        self, network: ImplicitNetwork, **kwargs: Any
    ) -> dict[tuple[int, int], list[EdgeContextCluster]]:
        """Cluster edge contexts with the configured clusterer (see ``ContextualEdgeClusterer.cluster_edges``)."""
        if self.clusterer is None:
            raise RuntimeError("No clusterer configured; pass `clusterer=` to the pipeline")
        return self.clusterer.cluster_edges(network, **kwargs)


def build_network(
    documents: Iterable[Document | str],
    *,
    extractor: BaseEntityExtractor | None = None,
    config: NetworkConfig | None = None,
    show_progress: bool = False,
    **config_overrides: Any,
) -> ImplicitNetwork:
    """Convenience wrapper: build an implicit network in one call.

    Args:
        documents: Corpus, documents or raw strings.
        extractor: Entity extractor (defaults to spaCy).
        config: Network parameters, or pass fields as keyword arguments.
        show_progress: Display a progress bar.
        **config_overrides: Fields of ``NetworkConfig``.

    Example:
        ```python
        from implicit_word_network import build_network, GazetteerEntityExtractor

        network = build_network(
            ["Feynman met Schwinger in Stockholm."],
            extractor=GazetteerEntityExtractor({"PERSON": ["Feynman", "Schwinger"], "LOC": ["Stockholm"]}),
            window=1,
        )
        ```
    """
    pipeline = ImplicitNetworkPipeline(extractor, config=config, **config_overrides)
    return pipeline.run(documents, show_progress=show_progress)
