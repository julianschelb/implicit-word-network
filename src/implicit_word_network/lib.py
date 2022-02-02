# ---------------------------------------------------------------------------- #
#                               Helper Functions                               #
# ---------------------------------------------------------------------------- #
import networkx as nx
import matplotlib.pyplot as plt
from networkx_viewer import Viewer


def readDocuments(path):
    """
    Read a text document file into a list of strings.
    """

    D = []
    with open(path, encoding="utf8") as file:
        for line in file:
            if line.rstrip() != "":
                D.append(line.rstrip())
    return D


def convertToNetworkX(V, Ep):
    """
    Converts a list of nodes and edges to a NetworkX graph object.
    """

    # Init graph object
    G = nx.MultiGraph()

    V = V["entities"]

    for v in V:

        v_id = v["text"].lower() + "_" + v["entity_type"]
        w = len(v["instances"])
        G.add_node(v_id, weight=w)

    # Only entity <-> entity edges
    Ep = Ep[("e", "e")]

    for ep in Ep:
        v1 = ep["vertex_1"]
        v2 = ep["vertex_2"]

        v1_id = v1["text"].lower() + "_" + v1["entity_type"]
        v2_id = v2["text"].lower() + "_" + v2["entity_type"]
        w = sum(i["w"] for i in ep["instances"]) * 0.5

        G.add_edge(v1_id, v2_id, weight=w)

    return G


def plotNetwork(G, mode="show", output_path="graph.png"):
    """
    Plots a NetworkX graph object.
    """

    # Extract weights and positions
    # edges, weights = zip(*nx.get_edge_attributes(G, "weight").items())
    pos = nx.spring_layout(G)

    # Plot using the draw function
    if mode == "show" or mode == "save":

        nx.draw(
            G,
            pos,
            # with_labels=True,
            # edgelist=edges,
            # edge_color=weights,
            # width=5,
            # edge_cmap=plt.cm.Blues,
        )

        if mode == "show":
            plt.show()
        else:
            plt.savefig(output_path)

    # Plot using nteractive viewer
    elif mode == "interactive":

        # Start interactive viewer
        app = Viewer(G)
        app.mainloop()

    # Or raise an error
    else:
        raise Exception("Unknown mode. Possible values: show, save and interactive")
