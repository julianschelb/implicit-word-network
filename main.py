import spacy as sp
import src.implicit_word_network as wn

# Path to text file
path = "./example/example_data.csv"

# Entities to search for in corpus
entity_types = ["PERSON", "LOC", "NORP", "ORG", "WORK_OF_ART"]

c = 2  # Cut-off parameter

# Importing data ...
D = wn.readDocuments(path)

# Parsing data ...
nlp = sp.load("en_core_web_sm")
D_parsed = wn.parseDocuments(D, entity_types, nlp=nlp)

# Converting parsing results ...
D_mat = wn.createCorpMat(D_parsed)

# Building graph ...
V, Ep = wn.buildGraph(D_mat, c)

# Convert to NetworkX object ...
G = wn.convertToNetworkX(V, Ep)

# Plot Graph
wn.plotNetwork(G, mode="show")
