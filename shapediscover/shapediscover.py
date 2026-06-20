import torch
import numpy as np
import time

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .parametric_fuzzy_cover import (
    PointCloudFunction,
    SetFunction,
    GraphFunction,
    PartitionOfUnity,
)
from .fuzzy_cover import (
    FuzzyCoverLossFunction,
    fuzzy_cover_from_kmeans,
    fuzzy_cover_from_fuzzycmeans,
    fuzzy_cover_to_filtered_complex,
)
from .weighted_graph import graph_from_pointcloud
from .shapediscover_plot import plot_losses


def _check_2d_finite(X: np.ndarray, shape_desc: str) -> np.ndarray:
    """Coerce ``X`` to a 2D finite numpy array, raising ``ValueError`` otherwise.

    ``shape_desc`` describes the expected array for the error message (e.g.
    ``"array of shape (n_samples, n_features)"``).
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X must be a 2D {shape_desc}; got ndim={X.ndim}.")
    if not np.all(np.isfinite(X)):
        raise ValueError("X must not contain NaN or infinite values.")
    return X


def _validate_pointcloud(X: np.ndarray, n_cover: int, knn: int) -> np.ndarray:
    """Validate and coerce a point cloud ``X`` before fitting.

    Returns ``X`` as a 2D ``numpy`` array, raising ``ValueError`` on an empty,
    non-2D, or non-finite input, or when ``n_cover`` / ``knn`` are inconsistent
    with the number of points.
    """
    X = _check_2d_finite(X, "array of shape (n_samples, n_features)")
    n_samples = X.shape[0]
    if n_samples == 0:
        raise ValueError("X must contain at least one point.")
    if n_cover < 1:
        raise ValueError(f"n_cover must be a positive integer; got {n_cover}.")
    if knn < 1:
        raise ValueError(f"knn must be a positive integer; got {knn}.")
    if n_samples < n_cover:
        raise ValueError(
            f"n_cover ({n_cover}) cannot exceed the number of points ({n_samples})."
        )
    if n_samples <= knn:
        raise ValueError(
            f"knn ({knn}) must be smaller than the number of points ({n_samples})."
        )
    return X


def _cover_to_simplex_tree(cover, max_dimension, clique_complex, log_normalization):
    """Build the gudhi simplex tree of the nerve of a fuzzy ``cover``.

    ``cover`` is in the internal orientation ``(n_cover_elements, n_points)``.
    """
    if clique_complex:
        simplex_tree = fuzzy_cover_to_filtered_complex(
            cover, max_dimension=1
        ).to_simplex_tree(log_normalization=log_normalization)
        simplex_tree.expansion(max_dimension + 1)
    else:
        simplex_tree = fuzzy_cover_to_filtered_complex(
            cover, max_dimension=max_dimension + 1
        ).to_simplex_tree(log_normalization=log_normalization)
    return simplex_tree


class ShapeDiscover(TransformerMixin, BaseEstimator):
    """Learn a fuzzy cover of a point cloud by geometric optimization.

    This is the full, lower-level interface; see ``ShapeDiscoverLite`` for the
    recommended high-level one. It follows the scikit-learn estimator API: call
    ``fit`` to learn the cover, ``transform`` (or ``fit_transform``) to retrieve
    it as an array of shape ``(n_points, n_cover)``, and ``fit_persistence`` to
    compute the persistent homology of its nerve.

    Parameters
    ----------
    n_cover : int
        Number of cover elements to learn.
    knn : int
        Number of nearest neighbors for the knn graph.
    loss_weights : list of float or None
        Weights of the [measure, geometry, topology, regularization] losses.
        ``None`` uses ``[1, 10, 1, 10]``.
    graph_algorithm : str
        Algorithm used to build the neighborhood graph (e.g. "umap").
    initialization_algorithm : str
        One of "random", "kmeans", "spectral_clustering", or
        "spectral_fuzzy_clustering".
    model : str
        One of "set_function", "pointcloud_nn", or "graph_nn".
    simplex_p : int
        Exponent of the p-simplex parametrization.
    inner_layer_widths : list of int or None
        Hidden-layer widths for the neural-network models.
    n_eigenfunctions : int or None
        Number of Laplacian eigenfunctions for spectral init / graph features
        (defaults to ``n_cover``).
    learning_rate : float
        Optimizer learning rate.
    n_max_iter : int
        Maximum number of optimization iterations.
    early_stop : bool
        Whether to stop early once the gradient norm is small.
    early_stop_tolerance : float
        Gradient-norm tolerance for early stopping.
    n_saved_iterations : int
        Number of intermediate covers to snapshot into ``historical_outputs_``
        during fitting (0 disables snapshotting).
    verbose : bool
        Whether to print timing information while fitting.
    plot_loss_curve : bool
        Whether to plot the loss curves while fitting.
    random_state : int, RandomState instance or None
        Controls the randomness of the graph construction, initialization, and
        optimization. Pass an int for reproducible runs; ``None`` (the default)
        draws fresh randomness each run.

    Attributes
    ----------
    cover_ : ndarray of shape (n_points, n_cover)
        The learned fuzzy cover (one column per cover element), available after
        ``fit``.
    precover_ : ndarray of shape (n_points, n_cover)
        The learned cover before the final p=inf normalization.
    initialization_precover_ : ndarray of shape (n_points, n_cover)
        The clustering used to initialize the optimization.
    """

    def __init__(
        self,
        n_cover: int = 10,
        knn: int = 15,
        loss_weights: list[float] | None = None,
        graph_algorithm: str = "umap",
        # connectivity threshold for the "cknn" graph (Berry-Sauer); ignored otherwise
        cknn_delta: float = 1.0,
        # weight "cknn" edges by a self-tuning Gaussian kernel (vs unweighted 0/1)
        cknn_weighted: bool = False,
        # either random, kmeans, spectral_clustering, or spectral_fuzzy_clustering
        initialization_algorithm: str = "spectral_clustering",
        # either set_function, pointcloud_nn, or graph_nn
        model: str = "set_function",
        simplex_p: int = 5,
        # divide the geometry/regularity (Dirichlet) losses by the squared graph
        # bandwidth so they are density-invariant (consistent across sample sizes)
        density_normalize_losses: bool = False,
        # base map onto the simplex: "softmax" (dense nerve) or "sparsemax"
        # (compact-support cover, sparse nerve)
        partition_of_unity_map: str = "softmax",
        # sparsity knob for "sparsemax": larger keeps more cover elements per point
        partition_of_unity_temperature: float = 1.0,
        inner_layer_widths: list[int] | None = None,
        n_eigenfunctions: int | None = None,
        learning_rate: float = 1e-1,
        n_max_iter: int = 250,
        early_stop: bool = True,
        early_stop_tolerance: float = 1e-4,
        n_saved_iterations: int = 0,
        verbose: bool = True,
        plot_loss_curve: bool = True,
        random_state=None,
    ):
        # scikit-learn convention: __init__ only stores the constructor
        # arguments verbatim (no validation or transformation), so that
        # get_params / set_params / clone work. Validation and the derivation
        # of defaults happen in fit.
        self.n_cover = n_cover
        self.knn = knn
        self.loss_weights = loss_weights
        self.graph_algorithm = graph_algorithm
        self.cknn_delta = cknn_delta
        self.cknn_weighted = cknn_weighted
        self.initialization_algorithm = initialization_algorithm
        self.model = model
        self.simplex_p = simplex_p
        self.density_normalize_losses = density_normalize_losses
        self.partition_of_unity_map = partition_of_unity_map
        self.partition_of_unity_temperature = partition_of_unity_temperature
        self.inner_layer_widths = inner_layer_widths
        self.n_eigenfunctions = n_eigenfunctions
        self.learning_rate = learning_rate
        self.n_max_iter = n_max_iter
        self.early_stop = early_stop
        self.early_stop_tolerance = early_stop_tolerance
        self.n_saved_iterations = n_saved_iterations
        self.verbose = verbose
        self.plot_loss_curve = plot_loss_curve
        self.random_state = random_state
        # Fitted attributes (graph_, cover_, precover_, ...) are intentionally
        # not set here: scikit-learn convention is that __init__ stores only the
        # constructor arguments, and that trailing-underscore attributes appear
        # only after fit (so check_is_fitted works). They are populated by fit /
        # fit_persistence.

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        # default random_state is None, so repeated fits need not be identical
        tags.non_deterministic = True
        return tags

    def fit(self, X: np.ndarray, y=None) -> "ShapeDiscover":
        """Learn the fuzzy cover of the point cloud ``X``.

        After fitting, the learned cover is available as ``self.cover_`` (shape
        ``(n_points, n_cover)``), with the intermediate ``precover_``,
        ``graph_``, ... attributes also populated. Returns ``self``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Point cloud in Euclidean space.
        y : Ignored
            Present for API consistency.
        """
        if self.initialization_algorithm not in [
            "random",
            "kmeans",
            "spectral_clustering",
            "spectral_fuzzy_clustering",
        ]:
            raise ValueError(
                "Initialization method not recognized: "
                f"{self.initialization_algorithm!r}."
            )
        if self.model not in ["set_function", "pointcloud_nn", "graph_nn"]:
            raise ValueError(f"Model not recognized: {self.model!r}.")

        X = _validate_pointcloud(X, self.n_cover, self.knn)

        (
            loss_weights,
            loss_probabilities,
            n_eigenfunctions,
            inner_layer_widths,
            optimization_algorithm,
        ) = self._resolve_fit_parameters()

        # a single master RNG drives every stochastic step (graph construction,
        # clustering init, torch model init, stochastic loss sampling); a fixed
        # random_state therefore gives reproducible runs, and nothing mutates
        # global numpy / torch random state beyond the local torch seeding below
        rng = np.random.default_rng(self.random_state)

        def next_seed():
            return int(rng.integers(0, np.iinfo(np.int32).max))

        torch.manual_seed(next_seed())
        if self.verbose:
            print("pointcloud shape:", X.shape)

        graph = self._build_neighborhood_graph(X, next_seed())
        self.graph_ = graph
        n_points = graph.n_vertices()

        clustering, laplacian_eigenmaps = self._initialize_precover(
            X, graph, n_eigenfunctions, next_seed
        )

        partition_of_unity = self._build_model(
            X,
            graph,
            n_points,
            clustering,
            laplacian_eigenmaps,
            n_eigenfunctions,
            inner_layer_widths,
            rng,
            next_seed,
        )
        self.model_ = partition_of_unity

        self._pretrain_on_initialization(
            partition_of_unity, clustering, n_points, optimization_algorithm
        )

        self._optimize(
            partition_of_unity,
            graph,
            loss_weights,
            loss_probabilities,
            optimization_algorithm,
            next_seed(),
        )

        # the final cover is an output, not part of any backward pass, so do not
        # build an autograd graph for it
        with torch.no_grad():
            last_pfuzzy_cover = simplex_to_psimplex(
                partition_of_unity(), p=self.simplex_p
            )
            output_cover = simplex_to_psimplex(last_pfuzzy_cover, p=float("inf"))
        # stored in the public (n_points, n_cover) orientation
        self.precover_ = last_pfuzzy_cover.numpy().T
        self.cover_ = output_cover.numpy().T

        return self

    def _resolve_fit_parameters(self):
        """Resolve parameter defaults that depend on other parameters.

        Kept out of ``__init__`` so the constructor stores its arguments
        verbatim (scikit-learn convention).
        """
        loss_weights = self.loss_weights
        if loss_weights is None:
            loss_weights = [1, 10, 1, 10]
        loss_probabilities = [1, 1, 1, 1]
        n_eigenfunctions = (
            self.n_eigenfunctions if self.n_eigenfunctions is not None else self.n_cover
        )
        inner_layer_widths = self.inner_layer_widths
        optimization_algorithm = "adam"
        return (
            loss_weights,
            loss_probabilities,
            n_eigenfunctions,
            inner_layer_widths,
            optimization_algorithm,
        )

    def _build_neighborhood_graph(self, X, seed):
        """Build the neighborhood graph of the point cloud ``X``."""
        time_start = time.time()
        graph = graph_from_pointcloud(
            X,
            n_neighbors=self.knn,
            algorithm=self.graph_algorithm,
            random_state=seed,
            delta=self.cknn_delta,
            cknn_weighted=self.cknn_weighted,
        )
        if self.verbose:
            print("time create graph", time.time() - time_start)
        return graph

    def _initialize_precover(self, X, graph, n_eigenfunctions, next_seed):
        """Compute the clustering that initializes the optimization.

        Returns ``(clustering, laplacian_eigenmaps)``, both ``None`` for the
        "random" initialization; ``clustering`` is ``(n_cover, n_points)``.
        """
        if self.initialization_algorithm == "random":
            return None, None

        time_start = time.time()
        laplacian_eigenmaps = None
        if self.initialization_algorithm == "kmeans":
            clustering = fuzzy_cover_from_kmeans(
                X, n_clusters=self.n_cover, random_state=next_seed()
            )
        elif self.initialization_algorithm == "spectral_clustering":
            laplacian_eigenmaps = graph.laplacian_eigenfunctions(
                n_eigenfunctions, random_state=next_seed()
            )
            clustering = fuzzy_cover_from_kmeans(
                laplacian_eigenmaps,
                n_clusters=self.n_cover,
                random_state=next_seed(),
            )
        elif self.initialization_algorithm == "spectral_fuzzy_clustering":
            laplacian_eigenmaps = graph.laplacian_eigenfunctions(
                n_eigenfunctions, random_state=next_seed()
            )
            clustering = fuzzy_cover_from_fuzzycmeans(
                laplacian_eigenmaps,
                n_clusters=self.n_cover,
                random_state=next_seed(),
            )

        # stored in the public (n_points, n_cover) orientation
        self.initialization_precover_ = clustering.T
        if self.verbose:
            print("time clustering", time.time() - time_start)
        return clustering, laplacian_eigenmaps

    def _build_model(
        self,
        X,
        graph,
        n_points,
        clustering,
        laplacian_eigenmaps,
        n_eigenfunctions,
        inner_layer_widths,
        rng,
        next_seed,
    ):
        """Construct the optimizable partition of unity for the chosen model."""
        if self.model == "set_function":
            if self.initialization_algorithm == "random":
                vector_valued_function = SetFunction(
                    n_points,
                    self.n_cover,
                    initialization=rng.random((self.n_cover, n_points)),
                )
            else:
                vector_valued_function = SetFunction(
                    n_points, self.n_cover, initialization=clustering
                )
        elif self.model == "pointcloud_nn":
            if not inner_layer_widths:
                inner_layer_widths = [self.n_cover]
            vector_valued_function = PointCloudFunction(
                X, self.n_cover, inner_layer_widths=inner_layer_widths
            )
        elif self.model == "graph_nn":
            if not inner_layer_widths:
                n_inner_layers = 2
                inner_layer_widths = [self.n_cover for _ in range(n_inner_layers)]
            if laplacian_eigenmaps is None:
                laplacian_eigenmaps = graph.laplacian_eigenfunctions(
                    n_eigenfunctions, random_state=next_seed()
                )
            vector_valued_function = GraphFunction(
                graph,
                laplacian_eigenmaps,
                self.n_cover,
                inner_layer_widths=inner_layer_widths,
            )
        return PartitionOfUnity(
            vector_valued_function,
            map_to_simplex=self.partition_of_unity_map,
            temperature=self.partition_of_unity_temperature,
        )

    def _make_optimizer(self, partition_of_unity, optimization_algorithm):
        if optimization_algorithm == "adam":
            return torch.optim.Adam(
                partition_of_unity.parameters(), lr=self.learning_rate
            )
        return torch.optim.SGD(partition_of_unity.parameters(), lr=self.learning_rate)

    def _pretrain_on_initialization(
        self, partition_of_unity, clustering, n_points, optimization_algorithm
    ):
        """Pre-train a neural-network model to match the clustering init.

        No-op for the ``set_function`` model (initialized directly) and for the
        "random" initialization.
        """
        if self.initialization_algorithm == "random" or self.model == "set_function":
            return

        time_start = time.time()
        optimizer_initialization = self._make_optimizer(
            partition_of_unity, optimization_algorithm
        )
        if self.early_stop:
            early_stopper = GradientEarlyStopper(
                partition_of_unity, self.early_stop_tolerance
            )

        initialization_losses = []
        initialization_target = torch.tensor(
            clustering, dtype=torch.float32, requires_grad=False
        )

        for iteration_number in range(self.n_max_iter):
            loss = torch.sum((partition_of_unity() - initialization_target) ** 2) / (
                self.n_cover * n_points
            )
            initialization_losses.append([iteration_number, loss.detach().numpy()])
            optimizer_initialization.zero_grad()
            loss.backward()
            optimizer_initialization.step()

            if self.early_stop and early_stopper.early_stop():
                break

        self.initialization_losses_ = np.array(initialization_losses)
        if self.verbose:
            print("time initialization", time.time() - time_start)
        if self.plot_loss_curve:
            plot_losses([self.initialization_losses_], ["initialization loss"])

    def _optimize(
        self,
        partition_of_unity,
        graph,
        loss_weights,
        loss_probabilities,
        optimization_algorithm,
        loss_seed,
    ):
        """Run the main optimization minimizing the fuzzy-cover loss."""
        optimizer = self._make_optimizer(partition_of_unity, optimization_algorithm)
        loss_function = FuzzyCoverLossFunction(
            graph, loss_weights, loss_probabilities, log=True, random_state=loss_seed,
            density_normalize=self.density_normalize_losses,
        )
        if self.early_stop:
            early_stopper = GradientEarlyStopper(
                partition_of_unity, self.early_stop_tolerance
            )

        save_output_at_iterations = []
        if self.n_saved_iterations > 0:
            save_output_at_iterations = list(
                range(0, self.n_max_iter, int(self.n_max_iter / self.n_saved_iterations))
            )

        historical_outputs = []
        self.historical_outputs_ = historical_outputs

        time_start = time.time()
        for iteration_number in range(self.n_max_iter):
            current_pfuzzy_cover = simplex_to_psimplex(
                partition_of_unity(), p=self.simplex_p
            )
            loss = loss_function(current_pfuzzy_cover, iteration_number)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if self.n_saved_iterations > 0:
                if iteration_number in save_output_at_iterations:
                    historical_outputs.append(current_pfuzzy_cover.detach().numpy())

            if self.early_stop and early_stopper.early_stop():
                break

        self.main_optimization_losses_ = loss_function._historical_losses
        self.loss_names_ = loss_function.loss_names
        if self.verbose:
            print("time optimization", time.time() - time_start)
        if self.plot_loss_curve:
            plot_losses(
                self.main_optimization_losses_, self.loss_names_, from_onwards=0
            )

    def transform(self, X=None, y=None) -> np.ndarray:
        """Return the learned fuzzy cover, of shape ``(n_points, n_cover)``.

        The cover is tied to the point cloud passed to ``fit`` (the default
        ``set_function`` model has no out-of-sample mapping), so ``X`` is
        ignored and present only for scikit-learn API consistency.
        """
        check_is_fitted(self, "cover_")
        return self.cover_

    def fit_persistence(
        self,
        max_dimension: int = 1,
        clique_complex: bool = False,
        verbose: bool = True,
    ) -> None:
        """Compute the persistent homology of the learned cover's nerve.

        Must be called after ``fit``. Populates ``self.simplex_tree_``,
        ``self.persistence_diagram_`` (a list of birth/death arrays, one per
        homological dimension), and ``self.gudhi_persistence_diagram_``.

        Parameters
        ----------
        max_dimension : int
            Highest homological dimension to compute.
        clique_complex : bool
            If True, build the nerve as a clique (flag) complex.
        """
        if max_dimension < 0:
            raise ValueError(f"max_dimension must be non-negative; got {max_dimension}.")
        check_is_fitted(self, "cover_")

        time_start = time.time()
        # cover_ is (n_points, n_cover); the nerve construction expects the
        # internal (n_cover, n_points) orientation.
        simplex_tree = _cover_to_simplex_tree(
            self.cover_.T, max_dimension, clique_complex, log_normalization=True
        )
        time_end = time.time()

        self.simplex_tree_ = simplex_tree

        if verbose:
            print("time create simplicial complex", time_end - time_start)

        time_start = time.time()
        self.gudhi_persistence_diagram_ = simplex_tree.persistence()
        self.persistence_diagram_ = [
            np.array(simplex_tree.persistence_intervals_in_dimension(i))
            for i in range(max_dimension + 1)
        ]
        time_end = time.time()
        if verbose:
            print("time compute persistence", time_end - time_start)


class GradientEarlyStopper:
    # https://stackoverflow.com/a/73704579
    def __init__(self, model, tolerance, patience=5):
        self._patience = patience
        self._tolerance = tolerance
        self._model = model
        self._counter = 0

    def early_stop(self):
        gradient_norm = model_gradient_norm(self._model)

        if gradient_norm < self._tolerance:
            self._counter += 1
            if self._counter >= self._patience:
                return True
        else:
            self._counter = 0
        return False


def model_gradient_norm(model):
    parameter_gradients = [
        parameter.grad.detach().flatten()
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    return torch.cat(parameter_gradients).norm()


def simplex_to_psimplex(functions, p=2):
    return functions / torch.norm(functions, p=p, dim=0)


class ShapeDiscoverLite(TransformerMixin, BaseEstimator):
    """Build a fuzzy cover of a point cloud X using geometric optimization.

    This is the recommended high-level interface. It follows the scikit-learn
    estimator API: ``fit`` learns the cover, ``transform`` / ``fit_transform``
    return it as an array of shape ``(n_points, n_cover)``.

    Parameters
    ----------
    n_cover : int, optional
        Number of cover elements to use (default is 10).
    knn : int, optional
        Number of nearest neighbors for knn graph (default is 15).
    regularization : float, optional
        Geometric regularization strength for the loss function (default is 10).
        Usual range is (1,100)
    optimization : bool, optional
        Whether to perform optimization (default is True).
        Should be kept as is unless you know what you are doing.
    n_max_iter : int, optional
        Maximum number of optimization iterations (default is 500).
        Should be kept as is unless you know what you are doing.
    early_stop_tolerance : float, optional
        Tolerance for early stopping during optimization (default is 1e-5).
        Should be kept as is unless you know what you are doing.
    fuzzy_clustering : bool, optional
        Whether to use fuzzy clustering initialization (default is False).
        Should be kept as is unless you know what you are doing.
    random_state : int, RandomState instance or None
        Controls the randomness of the pipeline. Pass an int for reproducible
        runs; ``None`` (the default) draws fresh randomness each run.

    Attributes
    ----------
    cover_ : ndarray of shape (n_points, n_cover)
        The learned fuzzy cover, available after ``fit``.

    Methods
    -------
    fit_transform(X, y=None)
        Fits the model to the data X and returns the fuzzy cover.
    """

    def __init__(
        self,
        n_cover: int = 10,
        knn: int = 15,
        regularization: float = 10,
        optimization: bool = True,
        n_max_iter: int = 500,
        early_stop_tolerance: float = 1e-5,
        fuzzy_clustering: bool = False,
        random_state=None,
    ):
        self.n_cover = n_cover
        self.knn = knn
        self.regularization = regularization
        self.optimization = optimization
        self.n_max_iter = n_max_iter
        self.early_stop_tolerance = early_stop_tolerance
        self.fuzzy_clustering = fuzzy_clustering
        self.random_state = random_state

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        # default random_state is None, so repeated fits need not be identical
        tags.non_deterministic = True
        return tags

    def fit(self, X: np.ndarray, y=None) -> "ShapeDiscoverLite":
        """Fit the model to the input data ``X``. Returns ``self``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Array of points in Euclidean space.
        y : Ignored, optional
            Not used, present for API consistency by convention.
        """
        if self.regularization < 0:
            raise ValueError(
                f"regularization must be non-negative; got {self.regularization}."
            )
        n_max_iter = self.n_max_iter if self.optimization else 0
        initialization_algorithm = (
            "spectral_clustering"
            if not self.fuzzy_clustering
            else "spectral_fuzzy_clustering"
        )
        self._discover = ShapeDiscover(
            n_cover=self.n_cover,
            knn=self.knn,
            loss_weights=[1, 0, 0, self.regularization],
            initialization_algorithm=initialization_algorithm,
            n_max_iter=n_max_iter,
            early_stop_tolerance=self.early_stop_tolerance,
            verbose=False,
            plot_loss_curve=False,
            random_state=self.random_state,
        )
        self._discover.fit(X)
        self.cover_ = self._discover.cover_
        return self

    def transform(self, X=None, y=None) -> np.ndarray:
        """Return the learned fuzzy cover, of shape ``(n_points, n_cover)``.

        The cover is tied to the point cloud passed to ``fit``, so ``X`` is
        ignored and present only for scikit-learn API consistency.
        """
        check_is_fitted(self, "cover_")
        return self.cover_


class FuzzyCoverPersistence(TransformerMixin, BaseEstimator):
    """Persistent homology of the nerve of a fuzzy cover.

    Transforms a fuzzy cover (as returned by ``ShapeDiscoverLite.fit_transform``
    or ``ShapeDiscover.transform``) into the persistence diagram of its nerve.

    Parameters
    ----------
    max_dimension : int
        Highest homological dimension to compute.
    log_rescaling : bool
        If True, rescale the filtration logarithmically.
    clique_complex : bool
        If True, build the nerve as a clique (flag) complex.
    verbose : bool
        Unused placeholder kept for API consistency.
    """

    def __init__(
        self,
        max_dimension: int = 1,
        log_rescaling: bool = False,
        clique_complex: bool = False,
        verbose: bool = False,
    ):
        self.max_dimension = max_dimension
        self.log_rescaling = log_rescaling
        self.clique_complex = clique_complex
        self.verbose = verbose

    def fit(self, X: np.ndarray, y=None) -> "FuzzyCoverPersistence":
        """No-op fit, present for scikit-learn API consistency. Returns ``self``."""
        return self

    def transform(self, X: np.ndarray, y=None) -> list:
        """Compute the persistence diagram of the nerve of the fuzzy cover ``X``.

        Parameters
        ----------
        X : ndarray of shape (n_points, n_cover_elements)
            A fuzzy cover (each column a cover-membership function over the
            points).
        y : Ignored
            Present for API consistency.

        Returns
        -------
        persistence : list of (int, (float, float))
            The gudhi persistence diagram: ``(dimension, (birth, death))`` pairs.
        """
        if self.max_dimension < 0:
            raise ValueError(
                f"max_dimension must be non-negative; got {self.max_dimension}."
            )
        X = _check_2d_finite(X, "fuzzy cover of shape (n_points, n_cover_elements)")
        if X.shape[1] == 0:
            raise ValueError("X must have at least one cover element.")

        # X is (n_points, n_cover); the nerve construction expects the internal
        # (n_cover, n_points) orientation.
        simplex_tree = _cover_to_simplex_tree(
            X.T,
            self.max_dimension,
            self.clique_complex,
            log_normalization=self.log_rescaling,
        )

        return simplex_tree.persistence()
