import numpy as np

from .fuzzy_cover import (
    fuzzy_cover_to_weighted_edges,
    threshold_fuzzy_cover,
)


def nerve_layout(fuzzy_cover, threshold, method=None, seed=0):
    try:
        import networkx as nx
    except ImportError as e:
        raise ImportError(
            "Nerve layout/visualization requires networkx. Install it with "
            "`pip install networkx` (the package's 'viz' extra)."
        ) from e

    # turn fuzzy cover to cover
    thresholded_fuzzy_cover, non_zero_indices = threshold_fuzzy_cover(
        fuzzy_cover, threshold
    )

    n_vertices = thresholded_fuzzy_cover.shape[0]

    # compute edge weights
    at_least_one_point = 1
    edges, intersection_sizes = fuzzy_cover_to_weighted_edges(
        thresholded_fuzzy_cover, min_weight=at_least_one_point
    )

    # build networkx graph
    weights = np.full_like(intersection_sizes, 1)
    edges_weights = [
        (i, j, {"weight": str(weight)}) for (i, j), weight in zip(edges, weights)
    ]
    graph = nx.Graph()
    graph.add_nodes_from(list(range(n_vertices)))
    graph.add_edges_from(edges_weights)

    initial_positions_dict = nx.spectral_layout(graph)
    final_positions_dict = nx.spring_layout(
        graph, pos=initial_positions_dict, iterations=100, seed=seed
    )

    final_positions_array = np.array(
        [final_positions_dict[i] for i in range(len(final_positions_dict))]
    )

    return final_positions_array, thresholded_fuzzy_cover
