# network/_storage.py
"""Append-only storage primitives used by :class:`~implicit_word_network.network.ImplicitNetwork`.

Both classes trade a little bookkeeping for amortised O(1) appends, so that
networks can grow by thousands of small batches without the quadratic cost of
re-concatenating arrays or re-adding sparse matrices on every update.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import sparse


class GrowableArray:
    """1-D NumPy array with geometric capacity growth.

    ``extend`` copies new values into a pre-allocated buffer that doubles when
    full; ``view`` returns the filled prefix without copying.
    """

    __slots__ = ("_buffer", "_size")

    def __init__(self, dtype: Any, capacity: int = 64) -> None:
        self._buffer: NDArray[Any] = np.empty(max(capacity, 1), dtype=dtype)
        self._size = 0

    @classmethod
    def from_array(cls, values: NDArray[Any]) -> GrowableArray:
        """Wrap an existing array (copied)."""
        instance = cls(values.dtype, capacity=max(values.shape[0], 1))
        instance.extend(values)
        return instance

    def extend(self, values: NDArray[Any]) -> None:
        """Append ``values`` (converted to the buffer dtype)."""
        n = int(values.shape[0])
        if n == 0:
            return
        needed = self._size + n
        if needed > self._buffer.shape[0]:
            capacity = max(needed, 2 * self._buffer.shape[0])
            buffer = np.empty(capacity, dtype=self._buffer.dtype)
            buffer[: self._size] = self._buffer[: self._size]
            self._buffer = buffer
        self._buffer[self._size : needed] = values
        self._size = needed

    @property
    def view(self) -> NDArray[Any]:
        """The filled part of the buffer (a view, not a copy)."""
        return self._buffer[: self._size]

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._buffer.dtype

    def __len__(self) -> int:
        return self._size


class SparseAccumulator:
    """Sum of sparse COO contributions with lazy CSR compaction.

    Contributions are appended as ``(rows, cols, data)`` triplets; the CSR
    matrix is only rebuilt (summing duplicates) when :attr:`matrix` is read or
    when the pending triplets outgrow the compacted matrix, which keeps both
    the per-update cost and the memory overhead bounded.
    """

    __slots__ = ("_cols", "_csr", "_data", "_dtype", "_pending", "_rows", "_shape")

    def __init__(self, dtype: Any, shape: tuple[int, int] = (0, 0)) -> None:
        self._dtype = np.dtype(dtype)
        self._shape = shape
        self._csr: sparse.csr_matrix = sparse.csr_matrix(shape, dtype=self._dtype)
        self._rows: list[NDArray[Any]] = []
        self._cols: list[NDArray[Any]] = []
        self._data: list[NDArray[Any]] = []
        self._pending = 0

    @classmethod
    def from_matrix(cls, matrix: sparse.csr_matrix | sparse.coo_matrix) -> SparseAccumulator:
        """Wrap an existing sparse matrix."""
        csr = matrix.tocsr()
        instance = cls(csr.dtype, (int(csr.shape[0]), int(csr.shape[1])))
        instance._csr = csr
        return instance

    @property
    def shape(self) -> tuple[int, int]:
        return self._shape

    @property
    def nnz(self) -> int:
        return int(self._csr.nnz) + self._pending

    def resize(self, shape: tuple[int, int]) -> None:
        """Grow the logical shape (never shrinks)."""
        if shape[0] < self._shape[0] or shape[1] < self._shape[1]:
            raise ValueError("SparseAccumulator cannot shrink")
        self._shape = shape

    def add(self, rows: NDArray[Any], cols: NDArray[Any], data: NDArray[Any]) -> None:
        """Add ``data[k]`` at ``(rows[k], cols[k])``; duplicates are summed."""
        n = int(rows.shape[0])
        if n == 0:
            return
        self._rows.append(np.asarray(rows, dtype=np.int64))
        self._cols.append(np.asarray(cols, dtype=np.int64))
        self._data.append(np.asarray(data, dtype=self._dtype))
        self._pending += n
        if self._pending > max(2 * int(self._csr.nnz), 1_000_000):
            self._compact()

    def add_matrix(self, matrix: sparse.csr_matrix | sparse.coo_matrix) -> None:
        """Add a whole sparse matrix (its shape must fit the logical shape)."""
        coo = matrix.tocoo()
        self.add(coo.row, coo.col, coo.data)

    def _compact(self) -> None:
        if not self._pending:
            if self._csr.shape != self._shape:
                self._csr = _grow_csr(self._csr, self._shape)
            return
        base = self._csr.tocoo()
        rows = np.concatenate([base.row.astype(np.int64)] + self._rows)
        cols = np.concatenate([base.col.astype(np.int64)] + self._cols)
        data = np.concatenate([base.data.astype(self._dtype)] + self._data)
        self._csr = sparse.coo_matrix((data, (rows, cols)), shape=self._shape).tocsr()
        self._csr.sum_duplicates()
        self._rows, self._cols, self._data, self._pending = [], [], [], 0

    @property
    def matrix(self) -> sparse.csr_matrix:
        """The compacted CSR matrix."""
        if self._pending or self._csr.shape != self._shape:
            self._compact()
        return self._csr


def _grow_csr(matrix: sparse.csr_matrix, shape: tuple[int, int]) -> sparse.csr_matrix:
    indptr = matrix.indptr
    if shape[0] > matrix.shape[0]:
        pad = np.full(shape[0] - matrix.shape[0], indptr[-1], dtype=indptr.dtype)
        indptr = np.concatenate([indptr, pad])
    return sparse.csr_matrix((matrix.data, matrix.indices, indptr), shape=shape)
