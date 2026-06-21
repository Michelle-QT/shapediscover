import torch
import numpy as np
import numba as nb
from scipy.special import comb
from sklearn.preprocessing import OneHotEncoder, Normalizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from .utils import scipy_sparse_matrix_to_torch_sparse
from .persistence_based_clustering import persistence_based_flattening
from .filtered_complex import FilteredComplex


class FuzzyCoverLossFunction:
    def __init__(
        self,
        graph,
        weights=None,
        probabilities=None,
        log=False,
        random_state=None,
        density_normalize_power=0.0,
        measure_point_weights=None,
        regularity_point_weights=None,
    ):
        # instance-local RNG for the stochastic loss sampling (no global side effect)
        self._rng = np.random.default_rng(random_state)

        # Optional per-point weighting of the measure loss (curvature-adaptive
        # cover): mass_a = sum_x w(x) phi_a(x) instead of sum_x phi_a(x). With w
        # large in curved regions, elements there are penalized more and shrink.
        # ``None`` (default) is uniform weighting, exactly the historical measure.
        self._measure_point_weights = None
        if measure_point_weights is not None:
            self._measure_point_weights = torch.tensor(
                np.asarray(measure_point_weights), requires_grad=False
            ).to(torch.float32)

        # Density normalization of the Dirichlet-energy losses (geometry,
        # regularity): divide by the graph bandwidth raised to
        # ``density_normalize_power`` (h = median neighbor distance). The
        # fixed-function Dirichlet scaling is h^2 (so mean((phi_a-phi_b)/h)^2
        # estimates |grad phi|^2), but for the *optimized* cover h^2 overcorrects
        # (crushes the tori); the right exponent is being calibrated, so it is a
        # parameter. 0.0 (off, the default) reproduces the historical behavior.
        self._dirichlet_scale = 1.0
        if density_normalize_power:
            h = graph.bandwidth() if hasattr(graph, "bandwidth") else None
            if h is None or h <= 0:
                raise ValueError(
                    "density_normalize_power requires a graph with a positive "
                    "bandwidth; build it with graph_from_pointcloud."
                )
            self._dirichlet_scale = 1.0 / (h ** density_normalize_power)

        number_of_losses = 4
        if weights is None:
            weights = [1.0, 1.0, 1.0, 1.0]
        if probabilities is None:
            probabilities = [1.0, 1.0, 1.0, 1.0]

        assert len(weights) == number_of_losses
        assert len(probabilities) == number_of_losses

        self._loss_weights = torch.tensor(weights, dtype=float, requires_grad=False)
        self._probabilities = probabilities
        self._log = log
        if self._log:
            self._historical_losses = [[] for _ in range(number_of_losses + 1)]

        coboundary_matrix_scipy, edge_weights_numpy = (
            graph.coboundary_matrix_and_edge_weights()
        )
        self._coboundary_matrix = scipy_sparse_matrix_to_torch_sparse(
            coboundary_matrix_scipy
        ).to(torch.float32)
        self._edge_weights = torch.tensor(edge_weights_numpy, requires_grad=False).to(
            torch.float32
        )
        self._total_edge_weight = np.sum(edge_weights_numpy)
        self._adjacency_list = graph.efficient_adjacency_list()

        # Optional curvature redistribution of the regularity (Dirichlet) loss:
        # multiply each edge's smoothness penalty by a per-edge weight, the mean of
        # its endpoints' curvature, normalized so the *total* edge weight is
        # unchanged. This keeps the regularity budget fixed but reallocates it
        # toward curved regions (strengthening smoothness there to resist the
        # over-sharpening that breaks inhomogeneous manifolds like the R^3 donut),
        # rather than adding regularity globally (which over-corrects). ``None``
        # (default) is exactly the historical, uniform regularity.
        self._regularity_edge_weight = None
        if regularity_point_weights is not None:
            point_weights = np.asarray(regularity_point_weights, dtype=float)
            # (c_i + c_j) per edge via |coboundary| @ c (two endpoints per row)
            endpoint_sum = np.abs(coboundary_matrix_scipy) @ point_weights
            edge_curvature = endpoint_sum / 2.0
            # normalize so sum_e edge_weight_e * w_e == total_edge_weight (budget
            # preserved; only the distribution across edges changes)
            denom = float(np.sum(edge_weights_numpy * edge_curvature))
            if denom > 0:
                edge_curvature = edge_curvature * (self._total_edge_weight / denom)
            self._regularity_edge_weight = torch.tensor(
                edge_curvature, requires_grad=False
            ).to(torch.float32)

        self._initialized_losses = [
            self._measure_loss,
            self._geometry_loss,
            self._topology_loss,
            self._regularization_loss,
        ]

        self.loss_names = [
            "measure",
            "geometry",
            "topology",
            "regularization",
            "total",
        ]

    def __call__(self, pou, iteration_number=0):

        total_loss = torch.tensor(0.0)
        for i, (weight, loss_function) in enumerate(
            zip(self._loss_weights, self._initialized_losses)
        ):
            # stochastic part
            if self._loss_weights[i] == 0 or (
                self._probabilities[i] != 1.0
                and self._rng.random() > self._probabilities[i]
            ):
                continue
            weighted_numerical_loss = weight * loss_function(pou)
            total_loss += weighted_numerical_loss
            if self._log:
                self._historical_losses[i].append(
                    [iteration_number, weighted_numerical_loss.detach().numpy()]
                )

        if self._log:
            self._historical_losses[-1].append(
                [iteration_number, total_loss.detach().numpy()]
            )

        return total_loss

    def _measure_loss(self, pou):
        n_pou_functions = pou.shape[0]
        n_points = pou.shape[1]
        if self._measure_point_weights is None:
            element_mass = torch.sum(pou, axis=1)
        else:
            # curvature-weighted mass: mass_a = sum_x w(x) phi_a(x). w has mean 1,
            # so the n_points**2 normalization keeps the loss on the same scale.
            element_mass = pou @ self._measure_point_weights
        return torch.sum(torch.pow(element_mass, 2)) / (
            n_pou_functions * n_points**2
        )

    def _geometry_loss(self, pou):
        n_pou_functions = pou.shape[0]
        return self._dirichlet_scale * torch.sum(
            torch.pow(
                torch.norm(
                    (pou @ self._coboundary_matrix.T) * self._edge_weights, p=1, dim=1
                ),
                2,
            )
        ) / (self._total_edge_weight**2 * n_pou_functions)

    def _topology_loss(self, pou, persistence_threshold=0.1):
        n_pou_functions = pou.shape[0]
        n_points = pou.shape[1]
        loss_conn = torch.tensor(0.0)

        for j in range(n_pou_functions):
            clusters_to_shrink, cluster_deaths = persistence_based_flattening(
                self._adjacency_list,
                pou[j].detach().numpy(),
                threshold=persistence_threshold,
            )

            this_loss = torch.tensor(0.0)
            for cluster, death in zip(clusters_to_shrink, cluster_deaths):
                this_loss += torch.sum(torch.pow(pou[j][cluster] - death, 2))

            loss_conn += this_loss / n_points

        return loss_conn / n_pou_functions

    def _regularization_loss(self, pou):
        n_pou_functions = pou.shape[0]
        edge_penalty = torch.pow(pou @ self._coboundary_matrix.T, 2) * self._edge_weights
        if self._regularity_edge_weight is not None:
            # curvature redistribution: reweight each edge (budget preserved)
            edge_penalty = edge_penalty * self._regularity_edge_weight
        return self._dirichlet_scale * torch.sum(edge_penalty) / (
            self._total_edge_weight * n_pou_functions
        )


def simplex_to_psimplex_numpy(functions, p=2):
    return functions / np.linalg.norm(functions, ord=p, axis=0)


# The nerve construction is two numba kernels at module level so they compile
# once (a kernel closed over ``functions`` recompiles on every call) and so the
# heavy births pass can run in parallel.


@nb.njit
def _enumerate_simplices_of_dimension(n, r, simplices):
    """Fill ``simplices`` row by row with the ``C(n, r)`` increasing ``r``-subsets
    of ``range(n)`` in lexicographic order.

    Sequential by construction (each subset is the successor of the previous),
    so this cannot be parallelized; the births pass below can.
    """
    n_simplices = simplices.shape[0]
    for k in range(n_simplices):
        if k == 0:
            for i in range(r):
                simplices[0, i] = i
            continue
        # advance to the next subset in lexicographic order
        i = r - 1
        while i >= 0 and simplices[k - 1, i] == i + n - r:
            i -= 1
        for j in range(r):
            simplices[k, j] = simplices[k - 1, j]
        simplices[k, i] += 1
        for j in range(i + 1, r):
            simplices[k, j] = simplices[k, j - 1] + 1


@nb.njit(parallel=True)
def _simplex_births(functions, simplices, births):
    """Birth value of each simplex: the largest, over all points, of the minimum
    membership across the simplex's vertices (0 when their supports never meet).

    Parallel over simplices; the inner min/max are exact comparisons, so the
    result is independent of the thread count (no floating-point reordering).
    """
    r = simplices.shape[1]
    n_points = functions.shape[1]
    for k in nb.prange(simplices.shape[0]):
        best = 0.0
        for x_index in range(n_points):
            # minimum membership over the simplex's vertices at this point
            m = functions[simplices[k, 0], x_index]
            for vi in range(1, r):
                v = functions[simplices[k, vi], x_index]
                if v < m:
                    m = v
            if m > best:
                best = m
        births[k] = best


def fuzzy_cover_to_filtered_complex(functions, max_dimension=1):
    """Build the nerve of a fuzzy cover as a filtered simplicial complex.

    ``functions`` is the cover in the internal ``(n_cover_elements, n_points)``
    orientation. A ``d``-simplex (a ``(d+1)``-subset of cover elements) is kept
    only when its members share a point of positive membership (a nonempty
    intersection); its birth value is the largest such common membership.
    Simplices with an empty intersection (birth 0) are dropped.

    Note: a partition of unity with full support (e.g. a softmax cover) makes
    every intersection nonempty, so the nerve is the full simplex on the cover
    elements and nothing is dropped. Genuine sparsity requires a compact-support
    cover (see the cover-sparsification direction in the project notes).
    """
    functions = np.ascontiguousarray(functions)
    n_cover_elements = functions.shape[0]

    births = []
    simplices = []

    for dimension in range(max_dimension + 1):
        r = dimension + 1
        n_simplices = comb(n_cover_elements, r, exact=True)
        simplices_of_dimension = np.zeros((n_simplices, r), dtype=np.int64)
        births_of_dimension = np.zeros(n_simplices, dtype=np.float64)

        _enumerate_simplices_of_dimension(
            n_cover_elements, r, simplices_of_dimension
        )
        _simplex_births(functions, simplices_of_dimension, births_of_dimension)

        # keep only simplices whose cover elements actually intersect
        nonempty = births_of_dimension > 0
        simplices.append(simplices_of_dimension[nonempty])
        births.append(births_of_dimension[nonempty])

    return FilteredComplex(simplices, births)


def fuzzy_cover_from_kmeans(pointcloud, n_clusters, random_state=None):
    clusterer = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state)
    clustering_labels = np.array(clusterer.fit_predict(pointcloud)).reshape(-1, 1)

    encoder = OneHotEncoder(sparse_output=False)
    clustering_as_function_to_simplex = encoder.fit_transform(clustering_labels).T

    return clustering_as_function_to_simplex


def fuzzy_cover_from_fuzzycmeans(pointcloud, n_clusters, m=2.0, random_state=None):
    """Fuzzy c-means memberships, L-infinity-normalized to a fuzzy cover.

    ``m`` is the fuzzifier (fuzziness exponent) of fuzzy c-means: ``m -> 1`` is
    hard clustering, larger ``m`` gives softer memberships. The default ``2.0``
    preserves the historical behavior used by the spectral-fuzzy initialization.
    """
    try:
        import skfuzzy
    except ImportError as e:
        raise ImportError(
            "scikit-fuzzy is required for the 'spectral_fuzzy_clustering' "
            "initialization. Install it with `pip install scikit-fuzzy` "
            "(it is part of this package's optional 'extras')."
        ) from e
    _, fuzzy_clustering, _, _, _, _, _ = skfuzzy.cluster.cmeans(
        pointcloud.T, n_clusters, m, error=0.005, maxiter=1000, init=None,
        seed=random_state,
    )
    return simplex_to_psimplex_numpy(fuzzy_clustering,p=float("inf"))


def standard_intersection(phi1, phi2):
    """Max-min fuzzy intersection of two membership functions.

    An alternative ``weighing_function`` for ``fuzzy_cover_to_weighted_edges``;
    the default there is ``volume_intersection``.
    """
    return np.max(np.minimum(phi1, phi2))


def volume_intersection(phi1, phi2):
    """L1 (volume) fuzzy intersection of two membership functions.

    The default ``weighing_function`` for ``fuzzy_cover_to_weighted_edges``.
    """
    return np.linalg.norm(np.minimum(phi1, phi2), ord=1)


def fuzzy_cover_to_weighted_edges(
    functions,
    weighing_function=volume_intersection,
    min_weight=0,
):
    n_cover_elements = functions.shape[0]

    all_possible_edges = np.array(
        [
            [i, j]
            for i in range(n_cover_elements)
            for j in range(i + 1, n_cover_elements)
        ],
        dtype=int,
    )
    weights = np.array(
        [weighing_function(functions[i], functions[j]) for i, j in all_possible_edges]
    )
    edges = all_possible_edges[weights > min_weight]
    weights = weights[weights > min_weight]

    return edges, weights


def threshold_fuzzy_cover(functions, threshold):
    thresholded = np.where(functions < threshold, 0, 1)
    non_zero_indices = np.any(thresholded, axis=1)
    return thresholded[non_zero_indices], non_zero_indices


def fuzzy_cover_with_labels_to_fractions(functions, labels):
    encoder = OneHotEncoder(sparse_output=False)
    encoded_labels = encoder.fit_transform(np.array(labels).reshape(-1, 1))
    return functions @ encoded_labels / np.sum(functions, axis=1)[:, None]
