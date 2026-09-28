"""Render the example network figure used in the documentation.

Runs offline: entities come from a curated gazetteer with alias merging
(``Richard Phillips Feynman`` → ``Feynman``, ``Nobel's`` → ``Alfred Nobel``, ...).
Requires the ``viz`` extra::

    pip install "implicit-word-network[viz]"
    python examples/example_plot.py docs/assets/example-network.png
"""

from __future__ import annotations

import sys

from implicit_word_network import (
    GazetteerEntityExtractor,
    ImplicitNetworkPipeline,
    load_example_corpus,
    plot_network,
)

GAZETTEER = {
    "Person": [
        "Richard Phillips Feynman",
        "Richard Feynman",
        "Feynman",
        "Julian Schwinger",
        "Schwinger",
        "Shin'ichirō Tomonaga",
        "Tomonaga",
        "Alfred Nobel",
        "Nobel's",
        "Nobel",
        "Albert Einstein",
        "Einstein",
        "Max Planck",
        "Planck",
        "Wilhelm Röntgen",
        "Röntgen",
        "Ragnar Sohlman",
        "Rudolf Lilljequist",
        "King Oscar II",
        "Ralph Leighton",
        "James Gleick",
        "Richard C. Tolman",
    ],
    "Organisation": [
        "Nobel Prize",
        "The Nobel Prize",
        "Nobel Prizes",
        "Nobel Peace Prize",
        "Nobel Foundation",
        "The Nobel Foundation",
        "Nobel Foundation's",
        "Royal Swedish Academy of Sciences",
        "Royal Swedish Academy",
        "Swedish Academy",
        "Norwegian Nobel Committee",
        "Karolinska Institutet",
        "California Institute of Technology",
        "Caltech",
        "Rogers Commission",
        "Storting",
        "Norwegian Parliament",
        "Physics World",
    ],
    "Location": [
        "Stockholm",
        "Paris",
        "Oslo",
        "Sweden",
        "Norway",
        "Tuva",
        "Germany",
        "United States",
    ],
}

ALIASES = {
    "richard phillips feynman": "feynman",
    "richard feynman": "feynman",
    "julian schwinger": "schwinger",
    "shin'ichirō tomonaga": "tomonaga",
    "nobel": "alfred nobel",
    "nobel's": "alfred nobel",
    "einstein": "albert einstein",
    "planck": "max planck",
    "röntgen": "wilhelm röntgen",
    "the nobel prize": "nobel prize",
    "nobel prizes": "nobel prize",
    "the nobel foundation": "nobel foundation",
    "nobel foundation's": "nobel foundation",
    "royal swedish academy": "royal swedish academy of sciences",
    "caltech": "california institute of technology",
    "norwegian parliament": "storting",
}

DISPLAY = {
    "feynman": "Feynman",
    "schwinger": "Schwinger",
    "tomonaga": "Tomonaga",
    "alfred nobel": "Alfred Nobel",
    "albert einstein": "Einstein",
    "max planck": "Planck",
    "wilhelm röntgen": "Röntgen",
    "royal swedish academy of sciences": "Royal Swedish Academy",
    "california institute of technology": "Caltech",
    "norwegian nobel committee": "Nobel Committee",
    "king oscar ii": "King Oscar II",
}


def normalize(text: str) -> str:
    key = " ".join(text.split()).lower()
    return ALIASES.get(key, key)


def main(output: str = "example-network.png") -> None:
    extractor = GazetteerEntityExtractor(GAZETTEER)
    pipeline = ImplicitNetworkPipeline(extractor, window=2, normalize_entity=normalize)
    network = pipeline.run(load_example_corpus())
    print(network.summary())

    graph = network.to_networkx(min_weight=0.3, include_isolated=False)
    for _, data in graph.nodes(data=True):
        fallback = data["text"].removeprefix("The ").removeprefix("the ").removesuffix("'s")
        data["text"] = DISPLAY.get(data["norm"], fallback)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, ax = plt.subplots(figsize=(11, 7))
    plot_network(
        graph,
        label_order=["Person", "Organisation", "Location"],
        min_component_size=3,
        seed=11,
        font_size=9,
        ax=ax,
        output_path=output,
    )
    print("wrote", output)


if __name__ == "__main__":
    main(*sys.argv[1:])
