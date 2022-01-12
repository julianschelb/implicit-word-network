# ---------------------------------------------------------------------------- #
#                            Import and Parse Corpus                           #
# ---------------------------------------------------------------------------- #

from tqdm import tqdm


def parseDocument(d_id=0, d="", entity_types=[], nlp=None):
    """
    Parse a document. Document is sentencized and tokenized.
    Also, Entities are marked.
    """

    d_parsed = []
    # SpaCy token object: https://spacy.io/api/token
    nlp_d = nlp(d)
    S = nlp_d.sents

    # Iterate over sentences
    for s_id, s in enumerate(S):

        # Iterate over tokens
        for t_id, t in enumerate(s):

            # Check if punctuation
            is_punct = t.pos_ == "PUNCT"
            is_entity = t.ent_iob_ != "O" and (
                (t.ent_type_ in entity_types) or (len(entity_types) == 0)
            )

            t_parsed = {
                "d_id": d_id,
                "s_id": s_id,
                "t_id": t_id,
                "text": t.text,
                "pos": t.pos_,
                "is_stopword": t.is_stop,
                "is_punctuation": is_punct,
                "is_entity": is_entity,
                "entity_type": t.ent_type_,
            }

            d_parsed.append(t_parsed)

    return d_parsed


def parseDocuments(D=[], entity_types=[], show_progress=True, nlp=None):
    """
    Parse a list of documents.
    Documents are sentencized and tokenized.
    Also, Entities are marked.
    """

    D_parsed = []

    # Iterate over documents
    for d_id, d in enumerate(tqdm(D, desc="Documents", disable=(not show_progress))):

        d_parsed = parseDocument(d_id, d, entity_types, nlp)
        D_parsed = D_parsed + d_parsed

    return D_parsed


def createCorpMat(D=[], remove_stopwords=True, show_progress=True):
    """
    Convert parsing results from a flat list of tokens into a nested dictionary.
    """

    D_mat = {}

    d_id_prev = -1
    s_id_prev = -1

    # iterate over all tokens in flat list
    for t in tqdm(D, desc="Tokens", disable=(not show_progress)):

        d_id = t["d_id"]
        s_id = t["s_id"]
        t_id = t["t_id"]

        # init emtpy document dict
        # if not present already
        if d_id_prev != d_id:
            if d_id not in D_mat:
                D_mat[d_id] = {}

        # init emtpy sentence dict
        # if not present already
        if s_id_prev != s_id:
            if s_id not in D_mat[d_id]:
                D_mat[d_id][s_id] = {}

        # store token in mat
        if (not t["is_stopword"] and not t["is_punctuation"]) or not remove_stopwords:
            D_mat[d_id][s_id][t_id] = t

    return D_mat
