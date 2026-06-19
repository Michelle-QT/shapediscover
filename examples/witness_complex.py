import numpy as np
import scipy as sp
import numba as nb
import gudhi


def witness_complex(pointcloud, landmarks, max_dimension, nu=0):
    n_vertices = landmarks.shape[0]
    n_points = pointcloud.shape[0]
    n_edges = (n_vertices * (n_vertices - 1)) // 2
    edges = np.zeros((n_edges, 2), dtype=int)
    edges_births = np.full(n_edges, np.inf)

    if nu == 0:
        relaxation = np.zeros(n_points)
    else:
        all_dists = sp.spatial.distance_matrix(pointcloud, landmarks)
        relaxation = np.partition(all_dists, nu, axis=1)[:, nu]

    @nb.jit(nopython=True)
    def _witness_complex_main_loop(
        pointcloud, landmarks, n_vertices, n_points, edges, edges_births
    ):
        k = 0
        for i in range(n_vertices):
            for j in range(i + 1, n_vertices):
                edges[k, 0], edges[k, 1] = i, j
                for x_index in range(n_points):
                    birth_according_to_x = max(
                        max(
                            np.linalg.norm(landmarks[i] - pointcloud[x_index]),
                            np.linalg.norm(landmarks[j] - pointcloud[x_index]),
                        )
                        - relaxation[x_index],
                        0,
                    )
                    if birth_according_to_x < edges_births[k]:
                        edges_births[k] = birth_according_to_x
                k += 1

    _witness_complex_main_loop(
        pointcloud, landmarks, n_vertices, n_points, edges, edges_births
    )

    simplex_tree = gudhi.SimplexTree()
    vertices = np.arange(n_vertices).reshape(-1, 1)
    simplex_tree.insert_batch(vertices.T, np.zeros(n_vertices))
    simplex_tree.insert_batch(edges.T, edges_births)
    simplex_tree.expansion(max_dimension)

    return simplex_tree
