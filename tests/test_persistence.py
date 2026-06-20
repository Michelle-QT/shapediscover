"""Unit tests for the persistence path: flattening and nerve persistence."""

import numpy as np

from shapediscover.fuzzy_cover import fuzzy_cover_to_filtered_complex
from shapediscover.persistence_based_clustering import persistence_based_flattening


def _path_adjacency_list(n):
    """Adjacency list (flat_neighbors, start_end) for the path 0-1-...-(n-1)."""
    neighbors = [[i - 1, i + 1] for i in range(n)]
    neighbors[0] = [1]
    neighbors[-1] = [n - 2]
    flat = np.array([x for ns in neighbors for x in ns], dtype=int)
    start_end, start = [], 0
    for ns in neighbors:
        start_end.append([start, start + len(ns)])
        start += len(ns)
    return flat, np.array(start_end, dtype=int)


def test_flattening_unimodal_has_no_extra_clusters():
    # a single peak -> only the global-max cluster, so nothing is returned
    clusters, deaths = persistence_based_flattening(
        _path_adjacency_list(5), np.array([1.0, 2.0, 3.0, 2.0, 1.0]), threshold=0.5
    )
    assert clusters == []
    assert deaths == []


def test_flattening_bimodal_returns_the_lower_peak():
    # peaks at vertex 0 (height 5) and 2 (height 4), valley at 1 (height 0)
    clusters, deaths = persistence_based_flattening(
        _path_adjacency_list(3), np.array([5.0, 0.0, 4.0]), threshold=0.5
    )
    assert len(clusters) == 1
    np.testing.assert_array_equal(clusters[0], [2])  # the lower peak
    assert deaths[0] == 0.0  # it dies at the valley height


def test_flattening_threshold_suppresses_low_prominence_peak():
    # with a threshold above the second peak's prominence (4), it is not split off
    clusters, _ = persistence_based_flattening(
        _path_adjacency_list(3), np.array([5.0, 0.0, 4.0]), threshold=4.5
    )
    assert clusters == []


def test_nerve_persistence_of_a_4cycle_cover():
    # 4 cover elements overlapping in a ring -> nerve is a 4-cycle -> H1 = 1
    cover = np.array(
        [
            [1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5],  # A overlaps B, D
            [0.0, 0.5, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0],  # B overlaps A, C
            [0.0, 0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0],  # C overlaps B, D
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 0.5],  # D overlaps C, A
        ]
    )
    st = fuzzy_cover_to_filtered_complex(cover, max_dimension=2).to_simplex_tree(
        log_normalization=False
    )
    # the cover elements have no common triple intersection, so the nerve is the
    # bare 4-cycle: its loop is an essential (never-dying) class. gudhi skips
    # homology in the top dimension unless persistence_dim_max is set.
    st.persistence(persistence_dim_max=True)
    h1 = st.persistence_intervals_in_dimension(1)
    assert h1.shape[0] == 1  # exactly one loop in the nerve
    assert np.isinf(h1[0, 1])  # the loop never fills in (no triple intersections)
    h0 = st.persistence_intervals_in_dimension(0)
    assert int(np.sum(np.isinf(h0[:, 1]))) == 1  # one connected component survives
