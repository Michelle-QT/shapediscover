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
    ):
        # instance-local RNG for the stochastic loss sampling (no global side effect)
        self._rng = np.random.default_rng(random_state)

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
        return torch.sum(torch.pow(torch.sum(pou, axis=1), 2)) / (
            n_pou_functions * n_points**2
        )

    def _geometry_loss(self, pou):
        n_pou_functions = pou.shape[0]
        return torch.sum(
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
        return torch.sum(
            torch.pow(pou @ self._coboundary_matrix.T, 2) * self._edge_weights
        ) / (self._total_edge_weight * n_pou_functions)


def simplex_to_psimplex_numpy(functions, p=2):
    return functions / np.linalg.norm(functions, ord=p, axis=0)


def fuzzy_cover_to_filtered_complex(functions, max_dimension=1):

    @nb.jit(nopython=True)
    def _fuzzy_cover_to_filtered_complex_main_loop(
        n_points, n, dimension, simplices_of_dimension, births_of_dimension
    ):
        r = dimension + 1
        n_simplices = births_of_dimension.shape[0]
        for k in range(n_simplices):

            # build simplex
            if k == 0:
                simplex = np.arange(r)
            else:
                for i in range(r - 1, -1, -1):
                    if simplex[i] != i + n - r:
                        break
                simplex[i] += 1
                for j in range(i + 1, r):
                    simplex[j] = simplex[j - 1] + 1

            simplices_of_dimension[k] = simplex

            for x_index in range(n_points):
                birth_according_to_x = min(functions[simplex, x_index])
                births_of_dimension[k] = max(
                    birth_according_to_x, births_of_dimension[k]
                )

    n_cover_elements = functions.shape[0]
    n_points = functions.shape[1]

    births = []
    simplices = []

    for dimension in range(max_dimension + 1):
        n_simplices = comb(n_cover_elements, dimension + 1, exact=True)
        simplices_of_dimension = np.zeros((n_simplices, dimension + 1), dtype=int)
        births_of_dimension = np.full(n_simplices, -1, dtype=float)

        _fuzzy_cover_to_filtered_complex_main_loop(
            n_points,
            n_cover_elements,
            dimension,
            simplices_of_dimension,
            births_of_dimension,
        )

        # filter out simplices that never appeared
        simplices_of_dimension = simplices_of_dimension[births_of_dimension != -1]
        births_of_dimension = births_of_dimension[births_of_dimension != -1]

        simplices.append(simplices_of_dimension)
        births.append(births_of_dimension)

    return FilteredComplex(simplices, births)


def fuzzy_cover_from_kmeans(pointcloud, n_clusters, random_state=None):
    clusterer = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state)
    clustering_labels = np.array(clusterer.fit_predict(pointcloud)).reshape(-1, 1)

    encoder = OneHotEncoder(sparse_output=False)
    clustering_as_function_to_simplex = encoder.fit_transform(clustering_labels).T

    return clustering_as_function_to_simplex


def fuzzy_cover_from_fuzzycmeans(pointcloud, n_clusters, random_state=None):
    try:
        import skfuzzy
    except ImportError as e:
        raise ImportError(
            "scikit-fuzzy is required for the 'spectral_fuzzy_clustering' "
            "initialization. Install it with `pip install scikit-fuzzy` "
            "(it is part of this package's optional 'extras')."
        ) from e
    _, fuzzy_clustering, _, _, _, _, _ = skfuzzy.cluster.cmeans(
        pointcloud.T, n_clusters, 2, error=0.005, maxiter=1000, init=None,
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
