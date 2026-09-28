"""Synthetic annotated corpora for benchmarks (no NLP models required).

Documents are generated with Zipf-distributed entity and term vocabularies so
that a few entities are very frequent (many cooccurrences per edge) while most
are rare, which mirrors real corpora.
"""

from __future__ import annotations

import numpy as np

from implicit_word_network.annotation import AnnotatedDocument, EntityMention, Sentence, Token

LABELS = ("PERSON", "ORG", "LOC", "DATE")
STOPWORDS = ("the", "and", "of", "in", "a", "to")


def synthetic_corpus(
    n_docs: int = 1000,
    *,
    sentences_per_doc: int = 30,
    tokens_per_sentence: int = 15,
    mentions_per_sentence: float = 1.5,
    n_entities: int = 20_000,
    n_terms: int = 50_000,
    zipf: float = 1.2,
    seed: int = 0,
) -> list[AnnotatedDocument]:
    """Generate ``n_docs`` annotated documents with consistent character offsets."""
    rng = np.random.default_rng(seed)
    entity_labels = rng.integers(0, len(LABELS), size=n_entities)
    entity_words = rng.integers(1, 3, size=n_entities)  # one- or two-word names

    def zipf_ids(size: int, vocab: int) -> np.ndarray:
        return (rng.zipf(zipf, size=size) - 1) % vocab

    documents = []
    for d in range(n_docs):
        pieces: list[str] = []
        sentences: list[Sentence] = []
        tokens: list[Token] = []
        mentions: list[EntityMention] = []
        position = 0
        n_mentions = rng.poisson(mentions_per_sentence, size=sentences_per_doc)
        for s in range(sentences_per_doc):
            sentence_start = position
            n_ent = int(n_mentions[s])
            n_tok = tokens_per_sentence
            slots = sorted(rng.choice(n_tok, size=min(n_ent, n_tok), replace=False).tolist())
            ent_ids = zipf_ids(len(slots), n_entities)
            term_ids = zipf_ids(n_tok, n_terms)
            slot_iter = iter(zip(slots, ent_ids))
            next_slot = next(slot_iter, None)
            for t in range(n_tok):
                if next_slot is not None and next_slot[0] == t:
                    entity = int(next_slot[1])
                    words = [f"Ent{entity}"] + (["Name"] if entity_words[entity] == 2 else [])
                    start = position
                    for word in words:
                        pieces.append(word)
                        tokens.append(Token(word, position, position + len(word), s, pos="PROPN"))
                        position += len(word) + 1
                    end = position - 1
                    mentions.append(
                        EntityMention(
                            " ".join(words),
                            LABELS[entity_labels[entity]],
                            start,
                            end,
                            s,
                            1.0,
                            len(tokens) - len(words),
                            len(tokens),
                        )
                    )
                    next_slot = next(slot_iter, None)
                    continue
                if t % 5 == 4:
                    word = STOPWORDS[int(term_ids[t]) % len(STOPWORDS)]
                    is_stop = True
                else:
                    word = f"term{int(term_ids[t])}"
                    is_stop = False
                pieces.append(word)
                tokens.append(
                    Token(
                        word,
                        position,
                        position + len(word),
                        s,
                        is_stop=is_stop,
                        pos="NOUN",
                        lemma=word,
                    )
                )
                position += len(word) + 1
            # sentence-final punctuation
            pieces[-1] = pieces[-1] + "."
            tokens.append(Token(".", position - 1, position, s, is_punct=True, pos="PUNCT"))
            sentences.append(Sentence(s, sentence_start, position))
            position += 1
        text = " ".join(pieces)
        documents.append(AnnotatedDocument(d, text, sentences, tokens, mentions))
    return documents


if __name__ == "__main__":
    docs = synthetic_corpus(3)
    docs[0].validate()
    print(docs[0])
    print(docs[0].text[:200])
