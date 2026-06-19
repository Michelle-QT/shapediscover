"""Unit tests for the pure helpers in shapediscover.fuzzy_cover."""

import numpy as np

from shapediscover.fuzzy_cover import (
    fuzzy_cover_from_kmeans,
    fuzzy_cover_to_filtered_complex,
    fuzzy_cover_to_weighted_edges,
    fuzzy_cover_with_labels_to_fractions,
    simplex_to_psimplex_numpy,
    standard_intersection,
    threshold_fuzzy_cover,
    volume_intersection,
)

# a small fuzzy cover: 3 cover elements over 4 points
COVER = np.array(
    [
        [1.0, 0.5, 0.0, 0.0],
        [0.0, 0.5, 1.0, 0.0],
        [0.0, 0.0, 0.5, 1.0],
    ]
)


def test_simplex_to_psimplex_numpy_normalizes_columns():
    out = simplex_to_psimplex_numpy(np.array([[3.0, 0.0], [4.0, 1.0]]), p=2)
    np.testing.assert_allclose(out, [[0.6, 0.0], [0.8, 1.0]])
    np.testing.assert_allclose(np.linalg.norm(out, ord=2, axis=0), [1.0, 1.0])


def test_threshold_fuzzy_cover_binarizes():
    thresholded, kept = threshold_fuzzy_cover(COVER, 0.6)
    np.testing.assert_array_equal(
        thresholded, [[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    )
    np.testing.assert_array_equal(kept, [True, True, True])


def test_threshold_fuzzy_cover_drops_all_zero_rows():
    cover = np.array([[0.9, 0.0], [0.1, 0.2]])  # second row never reaches 0.5
    thresholded, kept = threshold_fuzzy_cover(cover, 0.5)
    np.testing.assert_array_equal(thresholded, [[1, 0]])
    np.testing.assert_array_equal(kept, [True, False])


def test_intersections():
    a = np.array([0.2, 0.8, 0.5])
    b = np.array([0.5, 0.1, 0.5])
    assert standard_intersection(a, b) == 0.5  # max of pointwise min
    assert volume_intersection(a, b) == 0.8  # L1 of pointwise min


def test_fuzzy_cover_to_weighted_edges():
    edges, weights = fuzzy_cover_to_weighted_edges(COVER, min_weight=0)
    # only adjacent cover elements overlap (0-1 and 1-2); 0-2 has zero overlap
    np.testing.assert_array_equal(edges, [[0, 1], [1, 2]])
    np.testing.assert_allclose(weights, [0.5, 0.5])


def test_fuzzy_cover_to_filtered_complex_births():
    fc = fuzzy_cover_to_filtered_complex(COVER, max_dimension=1)
    np.testing.assert_array_equal(fc._simplices[0], [[0], [1], [2]])
    np.testing.assert_allclose(fc._births[0], [1.0, 1.0, 1.0])
    np.testing.assert_array_equal(fc._simplices[1], [[0, 1], [0, 2], [1, 2]])
    # vertex birth = max membership; edge birth = max pointwise min overlap
    np.testing.assert_allclose(fc._births[1], [0.5, 0.0, 0.5])


def test_fuzzy_cover_from_kmeans_is_one_hot():
    pts = np.array([[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [5.1, 5.0]])
    membership = fuzzy_cover_from_kmeans(pts, 2, random_state=0)
    assert membership.shape == (2, 4)
    # each point belongs to exactly one cluster
    np.testing.assert_array_equal(membership.sum(axis=0), [1, 1, 1, 1])


def test_fuzzy_cover_with_labels_to_fractions():
    fractions = fuzzy_cover_with_labels_to_fractions(COVER, np.array([0, 1, 0, 1]))
    np.testing.assert_allclose(
        fractions, [[2 / 3, 1 / 3], [2 / 3, 1 / 3], [1 / 3, 2 / 3]]
    )
    # each row is a distribution over labels
    np.testing.assert_allclose(fractions.sum(axis=1), 1.0)
