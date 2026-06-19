"""Unit tests for shapediscover.filtered_complex.FilteredComplex."""

import numpy as np

from shapediscover.filtered_complex import FilteredComplex


def _triangle():
    # 3 vertices (born at 1.0) and 3 edges (born at 0.5) of a triangle
    simplices = [
        np.array([[0], [1], [2]]),
        np.array([[0, 1], [1, 2], [0, 2]]),
    ]
    births = [np.array([1.0, 1.0, 1.0]), np.array([0.5, 0.5, 0.5])]
    return FilteredComplex(simplices, births)


def test_cut_keeps_simplices_at_or_above_threshold():
    fc = _triangle()
    above = fc.cut(0.7)
    np.testing.assert_array_equal(above[0], [[0], [1], [2]])  # vertices kept
    assert above[1].shape[0] == 0  # edges (born at 0.5) dropped
    assert fc.cut(0.3)[1].shape[0] == 3  # at a lower threshold, edges kept


def test_to_simplex_tree_counts():
    st = _triangle().to_simplex_tree(log_normalization=False)
    assert st.num_simplices() == 6  # 3 vertices + 3 edges
    assert st.dimension() == 1
