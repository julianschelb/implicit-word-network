# Implicit Word Network

## Introduction
This python package can be used to extract context-enriched implicit word networks as described by Spitz and Gertz. The theoretical background is explained in the following publications:

   1. Spitz, A. (2019). Implicit Entity Networks: A Versatile Document Model. Heidelberg University Library. https://doi.org/10.11588/HEIDOK.00026328
   2. Spitz, A., & Gertz, M. (2018). Exploring Entity-centric Networks in Entangled News Streams. In Companion of the The Web Conference 2018 on The Web Conference 2018 - WWW ’18. Companion of the The Web Conference 2018. ACM Press. https://doi.org/10.1145/3184558.3188726

## Dependencies

This project uses models from the spaCy and sentence_transformers package. These packages are not installed automatically. You can use the following commands to install them.

```console
pip install sentence_transformers
pip install spacy
python -m spacy download en_core_web_sm
```

## Example Usage

```python

import spacy as sp
import implicit_word_network as wn

# Path to text file
path = "data.txt"

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

```