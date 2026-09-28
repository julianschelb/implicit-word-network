# document.py
"""Corpus ingestion: ``Document`` and ``Corpus``.

A ``Corpus`` is an ordered collection of ``Document`` objects and is
the input of every entity extractor and of ``ImplicitNetworkPipeline``.
Corpora can be built from Python iterables, plain-text files (one document per
line or paragraph) or CSV files with a text column.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ID = str | int
"""Type of document identifiers (strings or integers)."""

# =============================================================================
# Document
# =============================================================================


@dataclass(slots=True)
class Document:
    """A single raw text document.

    Attributes:
        text: Raw text content of the document.
        id: Unique identifier of the document inside its corpus.
        meta: Optional dictionary of document-level metadata (e.g. title,
            publication date).

    Example:
        ```python
        from implicit_word_network import Document

        doc = Document("Feynman received the Nobel Prize in 1965.", id="feynman")
        print(doc.id)         # "feynman"
        print(len(doc))       # number of characters
        ```
    """

    text: str
    id: ID = 0
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError(f"Document text must be a str, got {type(self.text).__name__}")

    def __len__(self) -> int:
        return len(self.text)

    def __repr__(self) -> str:
        return f"Document(id={self.id!r}, len={len(self.text)})"


# =============================================================================
# Corpus
# =============================================================================


class Corpus:
    """Ordered collection of ``Document`` objects.

    Documents can be given as ``Document`` instances or as plain strings,
    in which case sequential integer identifiers are assigned. Document
    identifiers must be unique inside a corpus.

    Attributes:
        meta: Optional dictionary of corpus-level metadata.

    Example:
        ```python
        from implicit_word_network import Corpus

        corpus = Corpus.from_texts(["First document.", "Second document."])
        print(len(corpus))          # 2
        print(corpus[0].text)       # "First document."
        print(corpus.ids())         # [0, 1]

        # Load one document per non-empty line
        corpus = Corpus.from_txt("documents.txt")

        # Load from CSV (needs a text column; other columns become metadata)
        corpus = Corpus.from_csv("documents.csv", text_column="text", id_column="doc_id")
        ```
    """

    def __init__(
        self,
        documents: Iterable[Document | str] = (),
        *,
        meta: Mapping[str, Any] | None = None,
    ) -> None:
        self.meta: dict[str, Any] = dict(meta or {})
        self._documents: list[Document] = []
        self._index: dict[ID, int] = {}
        for document in documents:
            self.add(document)

    # ---------- Constructors ----------

    @classmethod
    def from_texts(
        cls,
        texts: Iterable[str],
        *,
        ids: Iterable[ID] | None = None,
        meta: Mapping[str, Any] | None = None,
    ) -> Corpus:
        """Build a corpus from an iterable of raw strings.

        Args:
            texts: Document texts.
            ids: Optional identifiers, parallel to ``texts``. Sequential
                integers are used when omitted.
            meta: Optional corpus-level metadata.

        Returns:
            A new corpus with one document per text.
        """
        corpus = cls(meta=meta)
        if ids is None:
            for text in texts:
                corpus.add(text)
        else:
            for text, doc_id in zip(texts, ids):
                corpus.add(Document(text, doc_id))
        return corpus

    @classmethod
    def from_txt(
        cls,
        path: str | Path,
        *,
        encoding: str = "utf-8",
        delimiter: str = "\n",
        meta: Mapping[str, Any] | None = None,
    ) -> Corpus:
        """Load a plain-text file with one document per delimited block.

        Empty blocks are skipped. With the default delimiter every non-empty
        line becomes a document; use ``"\\n\\n"`` to split on blank lines.

        Args:
            path: Path to the text file.
            encoding: File encoding.
            delimiter: String separating documents.
            meta: Optional corpus-level metadata.

        Returns:
            A new corpus with sequential integer identifiers.
        """
        raw = Path(path).read_text(encoding=encoding)
        texts = (block.strip() for block in raw.split(delimiter))
        return cls.from_texts((text for text in texts if text), meta=meta)

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        *,
        text_column: str = "text",
        id_column: str | None = None,
        encoding: str = "utf-8",
        delimiter: str = ",",
        meta: Mapping[str, Any] | None = None,
    ) -> Corpus:
        """Load a corpus from a delimited file.

        Every row becomes a document. Columns other than ``text_column`` and
        ``id_column`` are stored as document metadata.

        Args:
            path: Path to the CSV/TSV file.
            text_column: Name of the column holding the document text.
            id_column: Optional name of the column holding document ids.
            encoding: File encoding.
            delimiter: Field delimiter (``"\\t"`` for TSV files).
            meta: Optional corpus-level metadata.

        Returns:
            A new corpus.

        Raises:
            ValueError: If a required column is missing.
        """
        with Path(path).open(encoding=encoding, newline="") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            fieldnames = reader.fieldnames or []
            if text_column not in fieldnames:
                raise ValueError(
                    f"CSV file {str(path)!r} has no column {text_column!r} "
                    f"(available: {fieldnames})"
                )
            if id_column is not None and id_column not in fieldnames:
                raise ValueError(f"CSV file {str(path)!r} has no column {id_column!r}")
            records = list(reader)
        return cls.from_records(records, text_key=text_column, id_key=id_column, meta=meta)

    @classmethod
    def from_records(
        cls,
        records: Iterable[Mapping[str, Any]],
        *,
        text_key: str = "text",
        id_key: str | None = None,
        meta: Mapping[str, Any] | None = None,
    ) -> Corpus:
        """Build a corpus from dictionaries (e.g. rows of a DataFrame).

        Args:
            records: Mappings with at least a ``text_key`` entry.
            text_key: Key holding the document text.
            id_key: Optional key holding the document id.
            meta: Optional corpus-level metadata.

        Returns:
            A new corpus. Remaining keys are stored as document metadata.
        """
        corpus = cls(meta=meta)
        for position, record in enumerate(records):
            text = record.get(text_key)
            if text is None:
                raise ValueError(f"Record {position} has no {text_key!r} entry")
            doc_id: ID = record[id_key] if id_key is not None else position
            doc_meta = {
                str(key): value
                for key, value in record.items()
                if key not in (text_key, id_key) and key not in (None, "")
            }
            corpus.add(Document(str(text).strip(), doc_id, doc_meta))
        return corpus

    # ---------- Mutation ----------

    def add(
        self,
        document: Document | str,
        *,
        doc_id: ID | None = None,
        meta: Mapping[str, Any] | None = None,
    ) -> Document:
        """Append a document to the corpus.

        Args:
            document: A ``Document`` or a raw text string.
            doc_id: Identifier to assign when ``document`` is a string. The
                next free integer is used when omitted.
            meta: Metadata to attach when ``document`` is a string.

        Returns:
            The stored ``Document``.

        Raises:
            ValueError: If the identifier already exists in the corpus.
        """
        if isinstance(document, str):
            if doc_id is None:
                doc_id = self._next_free_id()
            document = Document(document, doc_id, dict(meta or {}))
        if document.id in self._index:
            raise ValueError(f"Duplicate document id {document.id!r}")
        self._index[document.id] = len(self._documents)
        self._documents.append(document)
        return document

    def _next_free_id(self) -> int:
        candidate = len(self._documents)
        while candidate in self._index:
            candidate += 1
        return candidate

    # ---------- Dunder helpers ----------

    def __len__(self) -> int:
        return len(self._documents)

    def __iter__(self) -> Iterator[Document]:
        return iter(self._documents)

    def __getitem__(self, position: int) -> Document:
        return self._documents[position]

    def __contains__(self, doc_id: object) -> bool:
        return doc_id in self._index

    def __repr__(self) -> str:
        return f"Corpus(documents={len(self)}, meta={self.meta})"

    # ---------- Convenience ----------

    def get(self, doc_id: ID) -> Document:
        """Return the document with identifier ``doc_id``."""
        try:
            return self._documents[self._index[doc_id]]
        except KeyError:
            raise KeyError(f"No document with id {doc_id!r}") from None

    def ids(self) -> list[ID]:
        """Return document identifiers in corpus order."""
        return [document.id for document in self._documents]

    def texts(self) -> list[str]:
        """Return document texts in corpus order."""
        return [document.text for document in self._documents]

    def head(self, n: int = 5) -> list[Document]:
        """Return the first ``n`` documents in corpus order."""
        if n <= 0:
            return []
        return self._documents[:n]
