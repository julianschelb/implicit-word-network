# ---------------------------------------------------------------------------- #
#                               Helper Functions                               #
# ---------------------------------------------------------------------------- #


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
