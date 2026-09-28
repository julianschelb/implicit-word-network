# Theory

This page summarises the models implemented by the package. For the full
treatment see Spitz (2019), *Implicit Entity Networks: A Versatile Document
Model*, and the papers listed on the [home page](index.md#citation).

## Implicit Entity Networks (IEN)

An implicit entity network is a joint representation of the **entities**,
**terms**, **sentences** and **documents** of a corpus. Its nodes are:

| Node class | Identity | Package type |
|---|---|---|
| Document | corpus id | `DocumentRef` |
| Sentence | document + position | `SentenceRef` |
| Entity | normalised name **and** entity type | `EntityNode` |
| Term | normalised word (lemma) **and** POS tag | `TermNode` |

Edges encode containment (document–sentence), mentions (sentence–entity,
sentence–term), same-sentence cooccurrence (entity–term) and, most
importantly, **entity–entity cooccurrence**.

Two entity mentions cooccur when they appear in the same document at most $c$
sentences apart ($c$ is the *context window*, `NetworkConfig.window`). Each
cooccurrence instance $i$ contributes a weight that decays with its sentence
distance $\delta_i$; all instances between entities $v$ and $w$ are aggregated
into a single edge weight

$$
\omega(v, w) = \sum_{i \in I_{v,w}} \exp\left(-\delta_i(v, w)\right).
$$

Because the weights decay exponentially, restricting cooccurrences to a small
window (ECCE uses $c = 2$) loses almost nothing. The decay function is
configurable (`NetworkConfig.decay`): `exponential` (default), `constant`
(plain counts), `inverse`, `linear`, or a custom function registered with
`register_decay`.

### Vectorised construction

The package computes $\omega$ for all entity pairs at once. With $S$ the
sparse *sentence × entity* mention-count matrix and $K$ the banded
*sentence × sentence* kernel with $K_{s,s'} = \exp(-|s - s'|)$ for sentences
of the same document at most $c$ apart,

$$
\Omega = S^{\top} K S
$$

contains the aggregated weights of all entity pairs (its diagonal, the
self-cooccurrence, is discarded). Entity–term counts are the sparse product
$S_e^{\top} S_t$. Because cooccurrences never cross document boundaries, the
contribution of new documents is a separate matrix that is simply added to
$\Omega$, which is what makes `ImplicitNetwork.add_documents` incremental.

### LOAD importance weights

The raw weight $\omega$ is symmetric. The LOAD model (Spitz & Gertz, 2016;
Spitz, Almasian & Gertz, 2017) derives from it a *directed* importance of an
entity $y$ of type $Y$ for an entity $x$:

$$
\omega_{\text{LOAD}}(x, y) = \log\frac{|Y|}{|N(x) \cap Y|} \sum_{i \in I_{x,y}} \exp\left(-\delta_i(x, y)\right),
$$

where $N(x)$ is the neighbourhood of $x$. The logarithmic factor plays the
role of an inverse document frequency: a neighbour type in which $x$ is
connected to almost every entity contributes little. The package exposes both
weightings (`ImplicitNetwork.weight` / `load_weight`,
`neighbors(weighting="raw" | "load")`, `load_weight_matrix()`).

### Ranking queries

EVELIN (Spitz, Almasian & Gertz, 2017) answers queries $\langle X \mid Q, n \rangle$
(rank nodes of class $X$ for query entities $Q$) with two-component scores
$r = c + s$:

| Target | Cohesion $c$ | Score $s$ |
|---|---|---|
| Entities | $\lvert N(x) \cap Q \rvert - 1$ | $\sum_{q \in Q} \omega(q, x) / s_{\max}$ (single query: $\omega(q, x) / \omega_{\max}$) |
| Sentences | number of query entities in the sentence | fraction of the top-$k$ terms of the query entities contained in the sentence |
| Documents | $\max$ sentence cohesion | normalised sum of sentence scores |

These are implemented by `rank_entities`, `rank_sentences` and
`rank_documents`, all vectorised over the sparse matrices.

## Contextual Implicit Entity Networks (CIEN)

Aggregating all cooccurrences of two entities into a single edge conflates
relations that arise in different contexts (a person and an organisation may
cooccur in reports about an award *and* about a scandal). Contextual implicit
entity networks (Spitz & Gertz, 2018) keep parallel edges apart:

1. Every cooccurrence instance is attributed with a **context embedding**
   $\kappa$ computed from the text of its context window (the sentences
   spanned by the two mentions).
2. The embeddings of one entity pair are **clustered** with a density-based
   algorithm (DBSCAN with cosine distance, $\varepsilon = 0.25$ and
   `min_samples = 1` in ECCE), since the number of contexts is unknown a priori.
3. Only instances with similar contexts are aggregated, so the pair retains
   one edge per context cluster.

ECCE replaced the averaged word2vec contexts of the original CIEN with
sentence-transformer embeddings and proposed **on-demand** clustering: only
the edges a user explores are clustered. `ContextualEdgeClusterer.cluster_edge`
implements the on-demand variant; `cluster_edges` batches the embedding of
many edges for precomputation.

## References

- Spitz, A. & Gertz, M. (2016). Terms over LOAD: Leveraging Named Entities for
  Cross-Document Extraction and Summarization of Events. *SIGIR '16*.
- Spitz, A. & Gertz, M. (2018). Exploring Entity-centric Networks in Entangled
  News Streams. *WWW '18 Companion*.
- Spitz, A. (2019). *Implicit Entity Networks: A Versatile Document Model*.
  Doctoral thesis, Heidelberg University.
- Schelb, J., Ehrmann, M., Romanello, M. & Spitz, A. (2022). ECCE:
  Entity-centric Corpus Exploration Using Contextual Implicit Networks.
  *WWW '22 Companion*.
