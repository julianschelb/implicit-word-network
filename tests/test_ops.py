"""
Unit tests for the vectorised primitives in implicit_word_network.network._ops.
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import sparse

from implicit_word_network.network._ops import (
    all_window_pairs,
    available_decays,
    band_kernel,
    band_structure,
    grow,
    incidence_matrix,
    ragged_ranges,
    register_decay,
    resolve_decay,
    same_group_pairs,
    window_pairs,
)


def naive_window_pairs(group, position, window):
    pairs = set()
    n = len(group)
    for i in range(n):
        for j in range(i + 1, n):
            if group[i] == group[j] and 0 <= position[j] - position[i] <= window:
                pairs.add((i, j))
    return pairs


sorted_rows = st.lists(
    st.tuples(st.integers(0, 3), st.integers(0, 6)), min_size=0, max_size=40
).map(sorted)


class TestRaggedRanges:
    def test_basic(self):
        owner, value = ragged_ranges(np.array([0, 5, 2]), np.array([2, 5, 4]))
        assert owner.tolist() == [0, 0, 2, 2]
        assert value.tolist() == [0, 1, 2, 3]

    def test_empty(self):
        owner, value = ragged_ranges(np.array([], dtype=np.int64), np.array([], dtype=np.int64))
        assert owner.shape == (0,) and value.shape == (0,)

    def test_negative_ranges_ignored(self):
        owner, value = ragged_ranges(np.array([3]), np.array([1]))
        assert owner.shape == (0,)


class TestWindowPairs:
    @given(rows=sorted_rows, window=st.integers(0, 4))
    @settings(max_examples=150, deadline=None)
    def test_matches_naive(self, rows, window):
        group = np.array([g for g, _ in rows], dtype=np.int64)
        position = np.array([p for _, p in rows], dtype=np.int64)
        i, j = all_window_pairs(group, position, window)
        assert np.all(i < j)
        assert set(zip(i.tolist(), j.tolist())) == naive_window_pairs(group, position, window)

    def test_chunking_matches_eager(self):
        rng = np.random.default_rng(0)
        group = np.sort(rng.integers(0, 5, size=300))
        position = np.concatenate(
            [np.sort(rng.integers(0, 30, size=(group == g).sum())) for g in range(5)]
        )
        i, j = all_window_pairs(group, position, 3)
        chunks = list(window_pairs(group, position, 3, chunk_size=17))
        assert len(chunks) > 1
        ci = np.concatenate([c[0] for c in chunks])
        cj = np.concatenate([c[1] for c in chunks])
        assert ci.tolist() == i.tolist() and cj.tolist() == j.tolist()

    def test_empty_input(self):
        i, j = all_window_pairs(np.array([], dtype=np.int64), np.array([], dtype=np.int64), 2)
        assert i.shape == (0,) and j.shape == (0,)

    def test_unsorted_raises(self):
        with pytest.raises(ValueError):
            list(window_pairs(np.array([0, 0]), np.array([3, 1]), 2))
        with pytest.raises(ValueError):
            list(window_pairs(np.array([1, 0]), np.array([0, 0]), 2))

    def test_negative_window_raises(self):
        with pytest.raises(ValueError):
            list(window_pairs(np.array([0]), np.array([0]), -1))

    def test_window_zero_same_position_only(self):
        i, j = all_window_pairs(np.array([0, 0, 0]), np.array([1, 1, 2]), 0)
        assert list(zip(i.tolist(), j.tolist())) == [(0, 1)]

    def test_large_positions(self):
        i, j = all_window_pairs(np.array([7, 7, 8]), np.array([10**9, 10**9 + 1, 10**9 + 1]), 1)
        assert list(zip(i.tolist(), j.tolist())) == [(0, 1)]


class TestSameGroupPairs:
    @given(
        a=st.lists(st.integers(0, 5), max_size=20).map(sorted),
        b=st.lists(st.integers(0, 5), max_size=20).map(sorted),
    )
    @settings(max_examples=100, deadline=None)
    def test_matches_naive(self, a, b):
        ia, ib = same_group_pairs(np.array(a, dtype=np.int64), np.array(b, dtype=np.int64))
        expected = {(i, j) for i, x in enumerate(a) for j, y in enumerate(b) if x == y}
        assert set(zip(ia.tolist(), ib.tolist())) == expected

    def test_empty(self):
        ia, ib = same_group_pairs(np.array([1]), np.array([], dtype=np.int64))
        assert ia.shape == (0,)


class TestBandStructure:
    def test_pattern(self):
        rows, cols, delta = band_structure(np.array([0, 0, 0, 1, 1]), window=1)
        entries = sorted(zip(rows.tolist(), cols.tolist(), delta.tolist()))
        assert entries == [
            (0, 0, 0),
            (0, 1, 1),
            (1, 0, 1),
            (1, 1, 0),
            (1, 2, 1),
            (2, 1, 1),
            (2, 2, 0),
            (3, 3, 0),
            (3, 4, 1),
            (4, 3, 1),
            (4, 4, 0),
        ]

    def test_window_larger_than_group(self):
        rows, cols, delta = band_structure(np.array([0, 0]), window=5)
        assert len(rows) == 4

    def test_empty(self):
        rows, cols, delta = band_structure(np.array([], dtype=np.int64), window=2)
        assert rows.shape == (0,)

    def test_kernel_is_symmetric(self):
        group = np.array([0, 0, 0, 0, 1, 1])
        rows, cols, delta = band_structure(group, window=2)
        kernel = band_kernel(group, 2, np.exp(-delta.astype(float)), rows, cols)
        dense = kernel.toarray()
        assert np.allclose(dense, dense.T)
        assert dense[0, 2] == pytest.approx(np.exp(-2))
        assert dense[0, 3] == 0.0  # beyond window
        assert dense[3, 4] == 0.0  # different group


class TestSparseHelpers:
    def test_incidence_matrix_counts_duplicates(self):
        matrix = incidence_matrix(np.array([0, 0, 1]), np.array([2, 2, 0]), (2, 3))
        assert matrix.toarray().tolist() == [[0, 0, 2], [1, 0, 0]]

    def test_grow(self):
        matrix = sparse.csr_matrix(np.array([[1, 0], [0, 2]]))
        grown = grow(matrix, 3, 4)
        assert grown.shape == (3, 4)
        assert grown.toarray()[:2, :2].tolist() == [[1, 0], [0, 2]]
        assert grown.toarray()[2].sum() == 0
        with pytest.raises(ValueError):
            grow(matrix, 1, 2)


class TestDecay:
    def test_builtin_decays(self):
        delta = np.array([0, 1, 2])
        assert np.allclose(resolve_decay("exponential", window=2)(delta), np.exp(-delta))
        assert np.allclose(resolve_decay("constant", window=2)(delta), 1.0)
        assert np.allclose(resolve_decay("inverse", window=2)(delta), [1, 0.5, 1 / 3])
        assert np.allclose(resolve_decay("linear", window=2)(delta), [1, 2 / 3, 1 / 3])

    def test_unknown(self):
        with pytest.raises(KeyError):
            resolve_decay("nope", window=1)

    def test_register(self):
        register_decay("half", lambda d: 0.5 ** d.astype(float))
        assert "half" in available_decays()
        assert np.allclose(resolve_decay("half", window=1)(np.array([0, 1, 2])), [1, 0.5, 0.25])
        with pytest.raises(ValueError):
            register_decay("", lambda d: d)
