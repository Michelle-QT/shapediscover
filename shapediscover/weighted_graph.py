from sklearn.neighbors import BallTree
import scipy as sp
import numpy as np
import numba as nb
import warnings
from warnings import warn
from sklearn.decomposition import TruncatedSVD


class WeightedGraph:
    def __init__(
        self, adjacency_matrix, flat_neighbors=None, flat_neighbors_start_end=None,
        bandwidth=None,
    ):
        """
        Assumptions:
            - Simple, undirected, weighted graph, without loops, encoded as weighted symmetric adjacency matrix.
            - Input is scipy sparse matrix.

        ``flat_neighbors`` / ``flat_neighbors_start_end`` optionally provide a
        precomputed adjacency list (see ``efficient_adjacency_list``); when
        omitted it is derived from the adjacency matrix on first use.

        ``bandwidth`` is a length scale of the neighborhood graph (the median
        neighbor distance); it is used to make the Dirichlet-energy losses
        (geometry, regularity) density-invariant by dividing by ``bandwidth**2``
        (so ``mean((phi_a - phi_b)/h)**2`` estimates ``|grad phi|**2``). ``None``
        when the graph was not built from a point cloud.
        """
        self._adjacency_matrix = adjacency_matrix
        self.flat_neighbors_ = flat_neighbors
        self.flat_neighbors_start_end_ = flat_neighbors_start_end
        self._bandwidth = bandwidth

    def bandwidth(self):
        return self._bandwidth

    def adjacency_matrix(self):
        return self._adjacency_matrix

    def n_vertices(self):
        return self.adjacency_matrix().shape[0]

    # def total_edge_weight(self):
    #    return self.adjacency_matrix().sum() / 2

    def coboundary_matrix_and_edge_weights(self):

        @nb.jit(nopython=True)
        def _weighted_coboundary_matrix_main_loop(
            entries,
            rows,
            columns,
            coboundary_matrix_row,
            coboundary_matrix_col,
            coboundary_matrix_data,
            edge_weights,
        ):
            i = 0
            for entry, row, column in zip(entries, rows, columns):
                if row < column:
                    coboundary_matrix_row[2 * i] = i
                    coboundary_matrix_col[2 * i] = row
                    coboundary_matrix_data[2 * i] = 1
                    coboundary_matrix_row[2 * i + 1] = i
                    coboundary_matrix_col[2 * i + 1] = column
                    coboundary_matrix_data[2 * i + 1] = -1
                    edge_weights[i] = entry
                    i += 1

        adjacency_matrix = sp.sparse.coo_array(self.adjacency_matrix())
        entries = adjacency_matrix.data

        # entries = np.sqrt(entries)

        rows = adjacency_matrix.row
        columns = adjacency_matrix.col

        n_edges = entries.shape[0] // 2

        # coboundary_matrix = sp.sparse.lil_array((n_edges, n_points))
        coboundary_matrix_row = np.zeros(2 * n_edges, dtype=int)
        coboundary_matrix_col = np.zeros(2 * n_edges, dtype=int)
        coboundary_matrix_data = np.zeros(2 * n_edges)
        edge_weights = np.zeros(n_edges)

        _weighted_coboundary_matrix_main_loop(
            entries,
            rows,
            columns,
            coboundary_matrix_row,
            coboundary_matrix_col,
            coboundary_matrix_data,
            edge_weights,
        )

        return (
            sp.sparse.coo_matrix(
                (coboundary_matrix_data, (coboundary_matrix_row, coboundary_matrix_col))
            ),
            edge_weights,
        )

    def efficient_adjacency_list(self):
        if (
            self.flat_neighbors_ is not None
            and self.flat_neighbors_start_end_ is not None
        ):
            return self.flat_neighbors_, self.flat_neighbors_start_end_
        else:
            n_points = self.n_vertices()
            adjacency_matrix = self.adjacency_matrix()
            neighbors = [adjacency_matrix[:, [i]].nonzero()[0] for i in range(n_points)]
            flat_neighbors = np.array([n for ns in neighbors for n in ns], dtype=int)
            neighbors_lenghts = [len(ns) for ns in neighbors]
            neighbors_start_end = []
            current_start = 0
            for i in range(n_points):
                current_end = current_start + neighbors_lenghts[i]
                neighbors_start_end.append([current_start, current_end])
                current_start = current_end
            self.flat_neighbors_ = flat_neighbors
            self.flat_neighbors_start_end_ = np.array(neighbors_start_end, dtype=int)
            return self.flat_neighbors_, self.flat_neighbors_start_end_

    # Function laplacian_eigenfunctions taken from UMAP's codebase: https://github.com/lmcinnes/umap
    # Authors: McInnes, Leland and Healy, John and Saul, Nathaniel and Grossberger, Lukas
    # Commit and line: https://github.com/lmcinnes/umap/blob/d4d4c4aeb96e0d2296b5098d9dc9736de79e4e96/umap/spectral.py#L395
    def laplacian_eigenfunctions(
        self,
        dim,
        init="random",
        random_state=None,
        method=None,
        tol=0.0,
        maxiter=0,
    ):
        graph = self.adjacency_matrix()

        # symmetric normalized Laplacian L = I - D^{-1/2} A D^{-1/2}; this assumes
        # A has no self-loops (zero diagonal), which graph_from_pointcloud enforces
        sqrt_deg = np.sqrt(np.asarray(graph.sum(axis=0)).squeeze())
        I = sp.sparse.identity(graph.shape[0], dtype=np.float64)
        D = sp.sparse.spdiags(1.0 / sqrt_deg, 0, graph.shape[0], graph.shape[0])
        L = I - D * graph * D
        if not sp.sparse.issparse(L):
            L = np.asarray(L)

        k = dim + 1
        num_lanczos_vectors = max(2 * k + 1, int(np.sqrt(graph.shape[0])))
        gen = (
            random_state
            if isinstance(random_state, (np.random.Generator, np.random.RandomState))
            else np.random.default_rng(seed=random_state)
        )
        if not method:
            method = "eigsh" if L.shape[0] < 2000000 else "lobpcg"

        try:
            if init == "random":
                X = gen.normal(size=(L.shape[0], k))
            elif init == "tsvd":
                X = TruncatedSVD(
                    n_components=k,
                    random_state=random_state,
                ).fit_transform(L)
            else:
                raise ValueError(
                    "The init parameter must be either 'random' or 'tsvd': "
                    f"{init} is invalid."
                )
            # For such a normalized Laplacian, the first eigenvector is always
            # proportional to sqrt(degrees). We thus replace the first t-SVD guess
            # with the exact value.
            X[:, 0] = sqrt_deg / np.linalg.norm(sqrt_deg)

            if method == "eigsh":
                eigenvalues, eigenvectors = sp.sparse.linalg.eigsh(
                    L,
                    k,
                    which="SM",
                    ncv=num_lanczos_vectors,
                    tol=tol or 1e-4,
                    v0=np.ones(L.shape[0]),
                    maxiter=maxiter or graph.shape[0] * 5,
                )
            elif method == "lobpcg":
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        category=UserWarning,
                        message=r"(?ms).*not reaching the requested tolerance",
                        action="error",
                    )
                    eigenvalues, eigenvectors = sp.sparse.linalg.lobpcg(
                        L,
                        np.asarray(X),
                        largest=False,
                        tol=tol or 1e-4,
                        maxiter=maxiter or 5 * graph.shape[0],
                    )
            else:
                raise ValueError("Method should either be None, 'eigsh' or 'lobpcg'")

            order = np.argsort(eigenvalues)[0:k]
            return eigenvectors[:, order]
        except Exception:
            warn(
                "Spectral initialisation failed! The eigenvector solver\n"
                "failed. This is likely due to too small an eigengap. Consider\n"
                "adding some noise or jitter to your data.\n\n"
                "Falling back to random initialisation!"
            )
            return gen.uniform(low=-10.0, high=10.0, size=(graph.shape[0], dim))


def graph_from_pointcloud(
    pointcloud, n_neighbors, algorithm="knn", metric="euclidean", random_state=None,
    delta=1.0, cknn_weighted=False,
):
    n_points = pointcloud.shape[0]
    # bandwidth: the median neighbor distance, a length scale used to make the
    # Dirichlet-energy losses density-invariant (see WeightedGraph.bandwidth);
    # each branch fills it from the neighbor distances it already computes.
    bandwidth = None
    if algorithm == "knn":
        ball_tree = BallTree(pointcloud, metric=metric)
        neighbor_distances, neighbor_indices = ball_tree.query(pointcloud, n_neighbors)
        bandwidth = float(np.median(neighbor_distances[:, 1:]))  # col 0 is self (0)
        adjacency_matrix = sp.sparse.lil_array((n_points, n_points))
        for i in range(n_points):
            adjacency_matrix[i, neighbor_indices[i]] = 1
            adjacency_matrix[i, i] = 0
        adjacency_matrix = sp.sparse.coo_array(adjacency_matrix)
        adjacency_matrix = adjacency_matrix.maximum(adjacency_matrix.T)
        flat_neighbors = neighbor_indices.flatten()

    elif algorithm == "umap":
        import umap

        adjacency_matrix, _, _ = umap.umap_.fuzzy_simplicial_set(
            pointcloud, n_neighbors, random_state=random_state, metric=metric
        )
        knn_indices, knn_distances, _ = umap.umap_.nearest_neighbors(
            pointcloud,
            n_neighbors,
            metric=metric,
            metric_kwds={},
            angular=False,
            random_state=random_state,
        )
        bandwidth = float(np.median(knn_distances[:, 1:]))  # col 0 is self (0)
        flat_neighbors = np.array(knn_indices, dtype=int).flatten()

    elif algorithm == "cknn":
        # Continuous k-NN (Berry and Sauer, 2019): connect i, j iff
        # d(i, j) < delta * sqrt(d_k(i) * d_k(j)), where d_k(i) is the distance to
        # i's n_neighbors-th neighbor. The local scale d_k adapts to density, so
        # the graph is self-tuning and far less sensitive than a fixed-radius or
        # fixed-knn graph (the central "no single knn works" finding). Edges are
        # gathered from a candidate neighborhood and filtered by the condition;
        # each point's nearest neighbor is always kept (a connectivity floor that
        # avoids isolated vertices, which would break the normalized Laplacian).
        from sklearn.neighbors import NearestNeighbors

        n_candidates = int(min(n_points - 1, max(4 * n_neighbors, 20)))
        nn = NearestNeighbors(n_neighbors=n_candidates + 1, metric=metric).fit(pointcloud)
        cand_dists, cand_idx = nn.kneighbors(pointcloud)  # column 0 is the point itself
        d_k = cand_dists[:, n_neighbors]  # distance to the k-th neighbor (col 0 = self)
        bandwidth = float(np.median(cand_dists[:, 1 : n_neighbors + 1]))
        src = np.repeat(np.arange(n_points), n_candidates)
        dst = cand_idx[:, 1:].ravel()
        dd = cand_dists[:, 1:].ravel()
        keep = dd ** 2 < (delta ** 2) * d_k[src] * d_k[dst]
        src, dst = src[keep], dst[keep]
        nn1 = cand_idx[:, 1]  # nearest neighbor of each point (connectivity floor)
        src = np.concatenate([src, np.arange(n_points)])
        dst = np.concatenate([dst, nn1])
        adjacency_matrix = sp.sparse.csr_matrix(
            (np.ones(len(src)), (src, dst)), shape=(n_points, n_points)
        )
        adjacency_matrix.data[:] = 1.0  # csr summed duplicate candidate/floor edges
        adjacency_matrix = adjacency_matrix.maximum(adjacency_matrix.T)  # symmetric 0/1
        if cknn_weighted:
            # self-tuning Gaussian weights (Zelnik-Manor and Perona) on the CkNN
            # connectivity: w(i,j) = exp(-d(i,j)^2 / (d_k(i) d_k(j))). The fixed
            # and adaptive *unweighted* graphs both fail the tori while the
            # weighted umap graph recovers them, so the weights are what matter.
            coo = adjacency_matrix.tocoo()
            diff = pointcloud[coo.row] - pointcloud[coo.col]
            dij2 = np.einsum("ij,ij->i", diff, diff)
            w = np.exp(-dij2 / (d_k[coo.row] * d_k[coo.col]))
            adjacency_matrix = sp.sparse.csr_matrix(
                (w, (coo.row, coo.col)), shape=(n_points, n_points)
            )
        flat_neighbors = None  # variable degree: derive the adjacency list from the matrix

    else:
        raise Exception("Algorithm not recognized", algorithm)

    # the rest of the pipeline (in particular the normalized Laplacian in
    # laplacian_eigenfunctions, which assumes no self-loops) relies on the graph
    # having no self-edges; the algorithms produce a zero diagonal, so check it
    if (adjacency_matrix.diagonal() != 0).any():
        raise ValueError(
            "neighborhood graph has self-edges (nonzero adjacency diagonal); "
            "the normalized Laplacian assumes none."
        )

    # Variable-degree graphs (cknn) have no uniform-stride flat adjacency list, so
    # let WeightedGraph derive it from the adjacency matrix on first use.
    if flat_neighbors is None:
        return WeightedGraph(adjacency_matrix, bandwidth=bandwidth)

    # the flat adjacency list is stored with a uniform stride of n_neighbors
    # NOTE: each vertex's knn block includes the vertex itself (at position 0)
    # and the range below ends at start + n_neighbors - 1, so the topological
    # loss's union-find effectively skips both the self-entry (it is always
    # excluded by the ranks[y] < hind guard in persistence_based_flattening) and
    # the farthest neighbor. Tightening this would change the topology-loss
    # neighbor set and thus results, so it is left to the benchmark-driven R&D
    # phase (see notes/TODO.md).
    starts = np.arange(0, n_points * n_neighbors, n_neighbors, dtype=int)
    flat_neighbors_start_end = np.zeros((n_points, 2), dtype=int)
    flat_neighbors_start_end[:, 0] = starts
    flat_neighbors_start_end[:, 1] = starts + n_neighbors - 1

    return WeightedGraph(
        adjacency_matrix,
        flat_neighbors=flat_neighbors,
        flat_neighbors_start_end=flat_neighbors_start_end,
        bandwidth=bandwidth,
    )

