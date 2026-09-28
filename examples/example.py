"""End-to-end example: build and inspect an implicit entity network.

Requires the ``spacy`` extra and the ``en_core_web_sm`` model::

    pip install "implicit-word-network[spacy]"
    python -m spacy download en_core_web_sm
"""

from implicit_word_network import (
    ContextualEdgeClusterer,
    ImplicitNetworkPipeline,
    SpacyEntityExtractor,
    load_example_corpus,
)

# Importing data (one document per row of the bundled CSV)
corpus = load_example_corpus()

# Entity types to keep and context window (in sentences)
extractor = SpacyEntityExtractor("en_core_web_sm", labels=["PERSON", "LOC", "GPE", "NORP", "ORG", "WORK_OF_ART"])
pipeline = ImplicitNetworkPipeline(extractor, window=2)

# Building the network
network = pipeline.run(corpus, show_progress=True)
print(network.summary())

# Strongest relations
for edge in network.edges(top_k=10):
    print(f"{edge.weight:6.2f}  {edge.source.text} [{edge.source.label}] -- {edge.target.text} [{edge.target.label}]")

# Provenance: where do two entities cooccur?
feynman = network.entity("Feynman", "PERSON")
if feynman is not None:
    neighbour, _ = network.neighbors(feynman, k=1)[0]
    for context in network.contexts(feynman, neighbour)[:3]:
        print("-", context[:120])

    # Contextual edges (CIEN): split the edge by context similarity
    clusters = ContextualEdgeClusterer(eps=0.5).cluster_edge(network, feynman, neighbour)
    print(f"{len(clusters)} context cluster(s) for {feynman.text} -- {neighbour.text}")

# Export to NetworkX / plot (needs the ``viz`` extra)
graph = network.to_networkx(min_weight=0.5)
print(graph)
try:
    from implicit_word_network import plot_network

    plot_network(network, top_k=50, output_path="network.png")
except ImportError as error:
    print(error)
