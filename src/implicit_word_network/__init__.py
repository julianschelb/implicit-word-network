"""Implicit Word Network - (contextual) implicit entity networks from text corpora."""

__version__ = "0.1.0"

from implicit_word_network.annotation import (
    AnnotatedDocument,
    EntityMention,
    EntitySpan,
    Sentence,
    Token,
    align_spans,
)
from implicit_word_network.context import (
    BagOfWordsEmbedder,
    BaseContextEmbedder,
    ContextualEdgeClusterer,
    EdgeContextCluster,
    SentenceTransformerEmbedder,
)
from implicit_word_network.datasets import load_example_corpus
from implicit_word_network.document import Corpus, Document
from implicit_word_network.extraction import (
    BaseEntityExtractor,
    GazetteerEntityExtractor,
    GLiNEREntityExtractor,
    SpacyEntityExtractor,
    SpanEntityExtractor,
)
from implicit_word_network.network import (
    Cooccurrence,
    EntityEdge,
    EntityNode,
    ImplicitNetwork,
    Mention,
    NetworkConfig,
    SentenceRef,
    TermNode,
    register_decay,
    to_networkx,
)
from implicit_word_network.pipeline import ImplicitNetworkPipeline, build_network
from implicit_word_network.segmentation import BaseSegmenter, RegexSegmenter, SpacySegmenter
from implicit_word_network.visualization import plot_network

__all__ = [
    "__version__",
    # Ingestion
    "Document",
    "Corpus",
    "load_example_corpus",
    # Annotation
    "AnnotatedDocument",
    "Sentence",
    "Token",
    "EntitySpan",
    "EntityMention",
    "align_spans",
    # Segmentation
    "BaseSegmenter",
    "RegexSegmenter",
    "SpacySegmenter",
    # Extraction
    "BaseEntityExtractor",
    "SpanEntityExtractor",
    "GazetteerEntityExtractor",
    "SpacyEntityExtractor",
    "GLiNEREntityExtractor",
    # Network
    "ImplicitNetwork",
    "NetworkConfig",
    "EntityNode",
    "TermNode",
    "SentenceRef",
    "Mention",
    "Cooccurrence",
    "EntityEdge",
    "register_decay",
    "to_networkx",
    # Context (CIEN)
    "BaseContextEmbedder",
    "BagOfWordsEmbedder",
    "SentenceTransformerEmbedder",
    "ContextualEdgeClusterer",
    "EdgeContextCluster",
    # Pipeline
    "ImplicitNetworkPipeline",
    "build_network",
    # Visualisation
    "plot_network",
]
