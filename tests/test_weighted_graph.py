"""Unit tests for shapediscover.weighted_graph.WeightedGraph."""

import numpy as np
import scipy.sparse as sps

from shapediscover.weighted_graph import WeightedGraph, graph_from_pointcloud


def test_direct_construction_recomputes_adjacency_list():
    # a triangle, constructed directly without a precomputed adjacency list
    adjacency = sps.csr_matrix(np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float))
    graph = WeightedGraph(adjacency)
    flat, start_end = graph.efficient_adjacency_list()  # must not raise
    for i in range(3):
        neighbors = set(flat[start_end[i, 0] : start_end[i, 1]].tolist())
        assert neighbors == {0, 1, 2} - {i}


def test_graph_from_pointcloud_knn_provides_adjacency_list():
    points = np.linspace(0.0, 1.0, 40).reshape(-1, 1)
    graph = graph_from_pointcloud(points, n_neighbors=5, algorithm="knn")
    flat, start_end = graph.efficient_adjacency_list()
    assert start_end.shape == (40, 2)
    assert flat.shape[0] == 40 * 5
