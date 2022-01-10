#
# ---------------------------------------------------------------------------- #
#                              Build Word Network                              #
# ---------------------------------------------------------------------------- #
#
# Extracting implicit word networks from a user specified
# set of documents as described in:
#
#   1. Spitz, “Implicit Entity Networks: A Versatile Document Model.”
#   2. Spitz and Gertz, “Exploring Entity-Centric Networks in Entangled News Streams.”
#

# ---------------------------------- Imports --------------------------------- #

import math
from sklearn.cluster import DBSCAN
from tqdm import tqdm

# -------------------------------- Build Graph ------------------------------- #


def getEntitiesInDoc(d_id, D):
    """
    Return all tokens marked as entities in document.
    """

    d = D[d_id]  # Current document
    E_d = {}  # Init entity dic

    # Iterate over all sentences and tokens
    # in document to find entity instances
    for s_id in d.keys():
        s = d[s_id]

        for t_id in s.keys():
            t = s[t_id]

            if t["is_entity"]:
                # Add instance of entity to dictionary of entities
                e_key = (t["text"].lower(), t["entity_type"])
                if e_key not in E_d:
                    E_d[e_key] = []
                E_d[e_key].append(t)

    return E_d


def getSentencesInDoc(d_id, D):
    """
    Return all sentences in document.
    """

    d = D[d_id]  # Current document
    S_d = [{"d_id": d_id, "s_id": s_id, "type": "s"} for s_id in d.keys()]

    return S_d


def getTermsInSent(s):
    """
    Return all terms in sentence.
    """

    T_s = {}  # Init token dic

    # Iterate over all tokens in sentence
    # to find terms (non entities)
    for t_id in s.keys():
        t = s[t_id]

        if not t["is_entity"]:
            # Add instance of token to dictionary
            e_key = (t["text"].lower(), t["pos"])
            if e_key not in T_s:
                T_s[e_key] = []
            T_s[e_key].append(t)

    return T_s


def getEntsInSent(s):
    """
    Return all entities in sentence.
    """

    E_s = {}  # Init entities dic

    # Iterate over all tokens in sentence
    # to find entities
    for t_id in s.keys():
        t = s[t_id]

        if t["is_entity"]:
            # Add instance of entities to dictionary
            e_key = (t["text"].lower(), t["entity_type"])
            if e_key not in E_s:
                E_s[e_key] = []
            E_s[e_key].append(t)

    return E_s


def extendEntities(V_e, E_d):
    """
    Merge two dictionaries of entity nodes. If necessary the instance lists are combinded.
    """

    # If entity already present
    # add list of instances
    for e_key in E_d:
        if e_key not in V_e:
            V_e[e_key] = E_d[e_key]
        else:
            V_e[e_key].extend(E_d[e_key])


def extendTerms(V_t, T_s):
    """
    Merge two dictionaries of term nodes. If necessary the instance lists are combinded.
    """

    # If token already present
    # add new instances
    for e_key in T_s:
        if e_key not in V_t:
            V_t[e_key] = T_s[e_key]
        else:
            V_t[e_key].extend(T_s[e_key])


def linkDocToSent(d, s, Ep_d_s):
    """
    Create edge between docuement and sentence.
    """

    ep_key = (d["d_id"], s["d_id"], s["s_id"])
    Ep_d_s[ep_key] = {"w": 1}


def linkEntToSent(E_s, Ep_s_e):
    """
    Create edge between entity and sentence.
    """

    for t_id in E_s.keys():

        t = E_s[t_id]
        ep_key = (t[0]["d_id"], t[0]["s_id"], t[0]["text"].lower(), t[0]["entity_type"])
        Ep_s_e[ep_key] = t


def linkTokenToSent(T_s, Ep_s_t):
    """
    Create edge between token and sentence.
    """

    for t_id in T_s.keys():

        t = T_s[t_id]
        ep_key = (t[0]["d_id"], t[0]["s_id"], t[0]["text"].lower(), t[0]["pos"])
        Ep_s_t[ep_key] = t


def linkEntToTerm(E_s, T_s, Ep_e_t):
    """
    Create edge between entity and term.
    """

    # Iterate over every entity and term in sentence
    for t_s_id, e_s_id in (
        (t_s_id, e_s_id) for t_s_id in T_s.keys() for e_s_id in E_s.keys()
    ):
        t = T_s[t_s_id]
        e = E_s[e_s_id]

        # Iterate over every instance of term and entity
        for t_i, e_i in ((t_i, e_i) for t_i in t for e_i in e):

            e_key = (
                e_i["text"].lower(),
                e_i["entity_type"],
                t_i["text"].lower(),
                t_i["pos"],
            )
            e_instance = {"entity": e_i, "term": t_i}
            if e_key not in Ep_e_t:
                Ep_e_t[e_key] = []
            Ep_e_t[e_key].append(e_instance)


def linkEntToEnt(E_d, Ep_e_e, c):
    """
    Create edge between entity and entity.
    """

    # Iterate over every entity in document
    e_1_done = []  # remember already visited entities
    for e_1_id, e_2_id in (
        (e_1_id, e_2_id) for e_1_id in E_d.keys() for e_2_id in E_d.keys()
    ):
        e_1 = E_d[e_1_id]
        e_2 = E_d[e_2_id]
        e_1_done.append(e_1)

        if e_2 not in e_1_done:

            e_key = (
                e_1[0]["text"].lower(),
                e_1[0]["entity_type"],
                e_2[0]["text"].lower(),
                e_2[0]["entity_type"],
            )

            # Iterate over every instance of both entities
            for e_1_i, e_2_i in ((e_1_i, e_2_i) for e_1_i in e_1 for e_2_i in e_2):

                delta = abs(e_1_i["s_id"] - e_2_i["s_id"])

                if delta <= c:
                    w = math.exp(-delta)
                    e_instance = {"entity_1": e_1_i, "entity_2": e_2_i, "w": w}
                    if e_key not in Ep_e_e:
                        Ep_e_e[e_key] = []
                    Ep_e_e[e_key].append(e_instance)


def combineVerticies(V_d, V_s, V_e, V_t):
    """
    Combine type specific node dictionaries into one.
    """

    # Add document and sentence nodes
    V = {"documents": V_d, "sentences": V_s, "entities": [], "terms": []}

    # Add entity nodes
    for v in V_e.values():
        V["entities"].append(
            {
                "text": v[0]["text"],
                "entity_type": v[0]["entity_type"],
                "type": "e",
                "instances": v,
            }
        )

    # Add term nodes
    for v in V_t.values():
        V["terms"].append(
            {"text": v[0]["text"], "pos": v[0]["pos"], "type": "t", "instances": v,}
        )

    return V


def combindeEdges(Ep_d_s, Ep_s_e, Ep_s_t, Ep_e_t, Ep_e_e):
    """
    Combine type specific edge dictionaries into one.
    """

    # List of edges per node types
    Ep = {
        ("d", "s"): [],
        ("s", "e"): [],
        ("s", "t"): [],
        ("e", "t"): [],
        ("e", "e"): [],
    }

    # Add document-sentence edges
    for key in Ep_d_s:

        e = Ep_d_s[key]
        e_new = {
            "vertex_1": {"type": "d", "d_id": key[0],},
            "vertex_2": {"type": "s", "d_id": key[1], "s_id": key[2]},
            "w": e["w"],
        }
        Ep[("d", "s")].append(e_new)

    # Add sentence-entity edges
    for key in Ep_s_e:

        e = Ep_s_e[key]
        e_new = {
            "vertex_1": {"type": "s", "d_id": key[0], "s_id": key[1]},
            "vertex_2": {
                "type": "e",
                "text": e[0]["text"],
                "entity_type": e[0]["entity_type"],
            },
            "instances": e,
        }
        Ep[("s", "e")].append(e_new)

    # Add sentence-term edges
    for key in Ep_s_t:

        e = Ep_s_t[key]
        e_new = {
            "vertex_1": {"type": "s", "d_id": key[0], "s_id": key[1]},
            "vertex_2": {"type": "t", "text": e[0]["text"], "pos": e[0]["pos"],},
            "instances": e,
        }
        Ep[("s", "t")].append(e_new)

    # Add entity-term edges
    for key in Ep_e_t:

        e = Ep_e_t[key]
        e_new = {
            "vertex_1": {"type": "e", "text": key[0], "entity_type": key[1],},
            "vertex_2": {"type": "t", "text": key[2], "pos": key[3],},
            "instances": e,
        }
        Ep[("e", "t")].append(e_new)

    # Add entity-entity edges
    for key in Ep_e_e:

        e = Ep_e_e[key]
        e_new = {
            "vertex_1": {"type": "e", "text": key[0], "entity_type": key[1],},
            "vertex_2": {"type": "e", "text": key[2], "entity_type": key[3],},
            "instances": e,
        }
        Ep[("e", "e")].append(e_new)

    return Ep


def buildGraph(D, c, show_progress=True):
    """
    Extract implicit word network from corpus. Returns nodes and edges.
    """

    # List of edges per node type
    Ep_d_s = {}
    Ep_s_e = {}
    Ep_s_t = {}
    Ep_e_t = {}
    Ep_e_e = {}

    # List of nodes per node type
    V_d = []
    V_s = []
    V_e = {}
    V_t = {}

    # Iterate over all documents
    for d_id in tqdm(D, desc="Documents", disable=(not show_progress)):

        d = D[d_id]
        d_i = {"d_id": d_id, "type": "d"}
        S_d_i = getSentencesInDoc(d_id, D)  # Sentences in d
        E_d = getEntitiesInDoc(d_id, D)  # Entities in d

        # Extend list of nodes
        V_d.append(d_i)
        V_s.extend(S_d_i)
        extendEntities(V_e, E_d)

        # Iterate over all sentences in document
        for s_i in S_d_i:

            s = d[s_i["s_id"]]

            # Link document to sentence
            linkDocToSent(d_i, s_i, Ep_d_s)

            # Identify and entities in sentence
            # and link them to sentence node
            E_s = getEntsInSent(s)
            linkEntToSent(E_s, Ep_s_e)

            # Identify terms in sentence
            # and extend list of token nodes
            # and link them to sentence node
            T_s = getTermsInSent(s)
            extendTerms(V_t, T_s)
            linkTokenToSent(T_s, Ep_s_t)

            # Link term to entity
            linkEntToTerm(E_s, T_s, Ep_e_t)

        # Link entity to entity
        linkEntToEnt(E_d, Ep_e_e, c)

    # Construct final nodes and edges lists
    V = combineVerticies(V_d, V_s, V_e, V_t)
    Ep = combindeEdges(Ep_d_s, Ep_s_e, Ep_s_t, Ep_e_t, Ep_e_e)

    return V, Ep


# ------------------------------- Cluster Edges ------------------------------ #


def constructEdgeContexts(e, D):
    """
    Return the context of two entities occurred together.
    """

    contexts = []

    # Iterate over all edge instances
    for e_i in e["instances"]:

        e_1 = e_i["entity_1"]
        e_2 = e_i["entity_2"]
        d = D[e_1["d_id"]]  # Both entities are in the same document
        s_id_from = min(e_1["s_id"], e_2["s_id"])
        s_id_to = max(e_1["s_id"], e_2["s_id"])

        # Build context string out of individual tokens
        context = []
        for s_id in range(s_id_from, s_id_to + 1):
            s = d[s_id]
            for t_id in s:
                context.append(s[t_id]["text"])

        contexts.append(" ".join(context))

    return contexts


def getContextEmbeddings(contexts, model):
    """
    Use BERT to compute context embeddings.
    """

    cont_embeddings = model.encode(contexts, show_progress_bar=False)
    return cont_embeddings


def clusterContextEmbeddings(embeddings, metric, eps, min_samples):
    """
    Use DBSCAN to cluster context embeddings.
    """

    clustering = DBSCAN(metric=metric, eps=eps, min_samples=min_samples).fit(embeddings)
    return clustering.labels_


def groupEdgeInstances(instances, clusters):
    """
    Sort edge instances into clusters.
    """

    # Iterate over all edge instances
    # to group them according to clusters
    e_i_clustered = {}
    for i, e_i in enumerate(instances):
        cluster = clusters[i]
        if cluster not in e_i_clustered:
            e_i_clustered[cluster] = []
        e_i_clustered[cluster].append(e_i)

    return [e_i for e_i in e_i_clustered.values()]


def clusterEdges(
    Ep, D, model="", metric="cosine", eps=0.25, min_samples=1, show_progress=True
):
    """
    Cluster edge instances between entities by using DBSCAN on embeddings of the context two entities occurred together.
    """

    Ep_clustered = Ep.copy()

    # Iterate over all edges
    for e in tqdm(Ep_clustered[("e", "e")], desc="Edges", disable=(not show_progress)):

        if len(e["instances"]) > 1:  # Clustering if mulltiple edge instances

            # Conctruct context for each edge instance
            # compute embeddings and cluster them
            contexts = constructEdgeContexts(e, D)
            embeddings = getContextEmbeddings(contexts, model)
            clusters = clusterContextEmbeddings(embeddings, metric, eps, min_samples)
            instances_clustered = groupEdgeInstances(e["instances"], clusters)

        else:
            # If edge has only one instance
            # one cluster with one instance
            # is created instead of clustering
            instances_clustered = [e["instances"]]

        # Attach clustered instances to edge
        e["instances_clustered"] = instances_clustered

    return Ep_clustered
