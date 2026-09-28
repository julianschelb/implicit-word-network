"""Zero-shot entity networks with GLiNER v2.5.

Requires the ``gliner`` extra (downloads ``gliner-community/gliner_medium-v2.5``)::

    pip install "implicit-word-network[gliner]"
"""

from implicit_word_network import GLiNEREntityExtractor, ImplicitNetworkPipeline, load_example_corpus

extractor = GLiNEREntityExtractor(
    "gliner-community/gliner_medium-v2.5",
    labels=["person", "organization", "award", "scientific concept", "location"],
    threshold=0.4,
    device="cpu",  # or "cuda" / "mps"
)

network = ImplicitNetworkPipeline(extractor, window=2).run(load_example_corpus(), show_progress=True)
print(network.summary())
for edge in network.edges(top_k=10):
    print(f"{edge.weight:6.2f}  {edge.source.text} [{edge.source.label}] -- {edge.target.text} [{edge.target.label}]")
