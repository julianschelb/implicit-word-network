# network/_ops.py
"""Vectorised building blocks for network construction.

All functions operate on NumPy arrays / SciPy sparse matrices and avoid
per-token Python loops. They are the performance core of
``ImplicitNetwork``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

DecayFunction = Callable[[NDArray[np.int64]], NDArray[np.float64]]
"""Maps an integer array of sentence distances to float weights."""

# =============================================================================
# Decay functions
# =============================================================================

_DECAY_FUNCTIONS: dict[str, DecayFunction] = {
    "exponential": lambda delta: np.exp(-delta.astype(np.float64)),
    "constant": lambda delta: np.ones(delta.shape, dtype=np.float64),
    "inverse": lambda delta: 1.0 / (1.0 + delta.astype(np.float64)),
}


def register_decay(name: str, function: DecayFunction) -> None:
    """Register a custom decay function under ``name``.

    The function receives a 1-D ``int64`` array of sentence distances and must
    return a ``float64`` array of the same shape.

    Example:
        ```python
        import numpy as np
        from implicit_word_network.network import register_decay

        register_decay("gaussian", lambda d: np.exp(-(d.astype(float) ** 2) / 2))
        ```
    """
    if not name:
        raise ValueError("decay name must not be empty")
    _DECAY_FUNCTIONS[name] = function


def available_decays() -> list[str]:
    """Names of all registered decay functions."""
    return sorted(_DECAY_FUNCTIONS)


def resolve_decay(name: str, *, window: int) -> DecayFunction:
    """Return the decay function registered under ``name``.

    ``"linear"`` is resolved relative to ``window`` as ``1 - δ / (window + 1)``.

    Raises:
        KeyError: If no such decay function exists.
    """
    if name == "linear":
        scale = float(window + 1)
        return lambda delta: 1.0 - delta.astype(np.float64) / scale
    try:
        return _DECAY_FUNCTIONS[name]
    except KeyError:
        raise KeyError(
            f"Unknown decay {name!r}; available: {available_decays() + ['linear']}"
        ) from None


# =============================================================================
# Ragged range expansion
# =============================================================================


def ragged_ranges(
    starts: NDArray[np.int64], stops: NDArray[np.int64]
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Expand half-open integer ranges into flat ``(owner, value)`` arrays.

    For every ``k`` the values ``starts[k], ..., stops[k] - 1`` are emitted with
    owner ``k``. Empty ranges contribute nothing.

    Args:
        starts: Range starts.
        stops: Range stops (exclusive).

    Returns:
        ``owner`` (index ``k`` of the range) and ``value`` arrays.
    """
    starts = np.asarray(starts, dtype=np.int64)
    stops = np.asarray(stops, dtype=np.int64)
    counts = np.maximum(stops - starts, 0)
    total = int(counts.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    owner = np.repeat(np.arange(counts.shape[0], dtype=np.int64), counts)
    offsets = np.cumsum(counts) - counts
    within = np.arange(total, dtype=np.int64) - np.repeat(offsets, counts)
    value = np.repeat(starts, counts) + within
    return owner, value


# =============================================================================
# Window pairs
# =============================================================================


def _window_key(
    group: NDArray[np.int64], position: NDArray[np.int64], window: int
) -> NDArray[np.int64]:
    """Composite sort key such that ``key + window`` never crosses a group."""
    if position.shape[0] == 0:
        return position
    offset = int(position.min())
    stride = int(position.max()) - offset + window + 2
    key = group * stride + (position - offset)
    if np.any(np.diff(key) < 0):
        raise ValueError("Input must be sorted by (group, position)")
    return key


def window_pairs(
    group: NDArray[np.integer],
    position: NDArray[np.integer],
    window: int,
    *,
    chunk_size: int = 1_000_000,
) -> Iterator[tuple[NDArray[np.int64], NDArray[np.int64]]]:
    """Enumerate index pairs ``(i, j)`` with ``i < j`` inside a positional window.

    Two rows pair up when they belong to the same group (document) and
    ``0 <= position[j] - position[i] <= window`` (sentence distance). The
    input must be sorted by ``(group, position)``. Pairs are produced in
    chunks of at most ``chunk_size`` to bound memory.

    Args:
        group: Group id per row (e.g. document index).
        position: Position per row (e.g. sentence index).
        window: Maximum positional distance.
        chunk_size: Maximum number of pairs per yielded chunk.

    Yields:
        Arrays ``(i, j)`` of row indices.
    """
    group64 = np.asarray(group, dtype=np.int64)
    position64 = np.asarray(position, dtype=np.int64)
    n = group64.shape[0]
    if n == 0:
        return
    if window < 0:
        raise ValueError("window must be >= 0")
    key = _window_key(group64, position64, window)
    hi = np.searchsorted(key, key + window, side="right")
    lo = np.arange(1, n + 1, dtype=np.int64)
    counts = hi - lo
    cumulative = np.cumsum(counts)
    start = 0
    while start < n:
        base = int(cumulative[start - 1]) if start > 0 else 0
        end = int(np.searchsorted(cumulative, base + chunk_size, side="right"))
        end = min(max(end, start + 1), n)
        owner, value = ragged_ranges(lo[start:end], hi[start:end])
        if owner.shape[0]:
            yield owner + start, value
        start = end


def all_window_pairs(
    group: NDArray[np.integer], position: NDArray[np.integer], window: int
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Eager variant of ``window_pairs`` returning concatenated arrays."""
    chunks = list(window_pairs(group, position, window))
    if not chunks:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate([c[0] for c in chunks]), np.concatenate([c[1] for c in chunks])


# =============================================================================
# Sparse helpers
# =============================================================================


def same_group_pairs(
    group_a: NDArray[np.integer], group_b: NDArray[np.integer]
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Pair every row of ``a`` with every row of ``b`` sharing the same group value.

    Both inputs must be sorted ascending.
    """
    a = np.asarray(group_a, dtype=np.int64)
    b = np.asarray(group_b, dtype=np.int64)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    starts = np.searchsorted(b, a, side="left")
    stops = np.searchsorted(b, a, side="right")
    return ragged_ranges(starts, stops)


def incidence_matrix(
    rows: NDArray[np.integer],
    cols: NDArray[np.integer],
    shape: tuple[int, int],
    *,
    dtype: type = np.int64,
) -> sparse.csr_matrix:
    """Sparse count matrix with one increment per ``(row, col)`` pair."""
    data = np.ones(np.asarray(rows).shape[0], dtype=dtype)
    matrix = sparse.coo_matrix((data, (rows, cols)), shape=shape)
    return matrix.tocsr()


def band_structure(
    group: NDArray[np.integer], window: int
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """Sparsity pattern of the sentence-distance kernel.

    For consecutive positions ``s`` and ``s + d`` (``0 <= d <= window``) that
    belong to the same group, both ``(s, s + d)`` and ``(s + d, s)`` are
    emitted (the diagonal once).

    Args:
        group: Group id per position, sorted so that groups are contiguous.
        window: Maximum distance.

    Returns:
        ``rows``, ``cols`` and ``delta`` arrays.
    """
    group64 = np.asarray(group, dtype=np.int64)
    n = group64.shape[0]
    rows: list[NDArray[np.int64]] = []
    cols: list[NDArray[np.int64]] = []
    deltas: list[NDArray[np.int64]] = []
    for d in range(window + 1):
        if d >= n:
            break
        idx = np.arange(n - d, dtype=np.int64)
        same = group64[idx] == group64[idx + d]
        idx = idx[same]
        if idx.shape[0] == 0:
            continue
        rows.append(idx)
        cols.append(idx + d)
        deltas.append(np.full(idx.shape[0], d, dtype=np.int64))
        if d > 0:
            rows.append(idx + d)
            cols.append(idx)
            deltas.append(np.full(idx.shape[0], d, dtype=np.int64))
    if not rows:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, empty
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(deltas)


def band_kernel(
    group: NDArray[np.integer],
    window: int,
    values: NDArray[np.floating] | NDArray[np.integer],
    rows: NDArray[np.int64],
    cols: NDArray[np.int64],
) -> sparse.csr_matrix:
    """Assemble a square kernel matrix from a ``band_structure`` pattern."""
    n = np.asarray(group).shape[0]
    return sparse.coo_matrix((values, (rows, cols)), shape=(n, n)).tocsr()


def grow(matrix: sparse.csr_matrix, n_rows: int, n_cols: int) -> sparse.csr_matrix:
    """Return a copy of a CSR matrix padded with empty rows/columns."""
    matrix = matrix.tocsr()
    old_rows, old_cols = matrix.shape
    if n_rows < old_rows or n_cols < old_cols:
        raise ValueError("grow() cannot shrink a matrix")
    indptr = matrix.indptr
    if n_rows > old_rows:
        pad = np.full(n_rows - old_rows, indptr[-1], dtype=indptr.dtype)
        indptr = np.concatenate([indptr, pad])
    return sparse.csr_matrix((matrix.data, matrix.indices, indptr), shape=(n_rows, n_cols))
