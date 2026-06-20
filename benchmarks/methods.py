"""Method adapters behind a common interface.

Every method exposes whichever of the three capability outputs it can produce:

- ``labels()``           -> ndarray (n_points,)        for the clustering axis.
- ``embedding()``        -> ndarray (n_points, 2)      for the embedding axis.
- ``persistence(d)``     -> (intervals, complex_size)  for the topology axis.

A method's ``provides`` tuple declares which axes it supports, so the runner only
asks for what is available. ShapeDiscover is the first (and currently only)
method; external baselines (UMAP, KMeans, HDBSCAN, Rips/Alpha/Witness, ...) are
deferred and slot in here later behind the same interface (see benchmarks/README).
"""

from __future__ import annotations

import numpy as np


class Method:
    """Common interface. Subclasses set ``name`` and ``provides`` and override
    the capability methods they support."""

    name: str = "method"
    provides: tuple[str, ...] = ()

    def fit(self, X: np.ndarray) -> "Method":
        raise NotImplementedError

    def labels(self) -> np.ndarray:
        raise NotImplementedError(f"{self.name} does not provide a clustering")

    def embedding(self) -> np.ndarray:
        raise NotImplementedError(f"{self.name} does not provide an embedding")

    def persistence(self, max_dim: int) -> tuple[list[np.ndarray], int]:
        raise NotImplementedError(f"{self.name} does not provide persistence")

    def params(self) -> dict:
        """Parameters recorded with every result row."""
        return {}


class ShapeDiscoverMethod(Method):
    """Adapter over ``ShapeDiscoverLite`` / ``ShapeDiscover``.

    From the single learned fuzzy cover it derives all three capability outputs:

    - clustering: connected components of the thresholded nerve (``label_mode``
      ``"components"``, the distinctive "intrinsic number of clusters" route) or a
      plain ``"argmax"`` over cover elements;
    - embedding: each point placed at the membership-weighted barycenter of the
      2D nerve layout of its cover elements (a per-point pseudo-embedding);
    - topology: persistence of the nerve (filtered simplicial complex) plus the
      complex size.
    """

    name = "shapediscover"
    provides = ("clustering", "embedding", "topology")

    def __init__(
        self,
        n_cover: int = 15,
        knn: int = 15,
        regularization: float = 10.0,
        threshold: float = 0.5,
        label_mode: str = "components",
        lite: bool = False,  # full ShapeDiscover by default; Lite drops geometry/topology losses
        random_state: int | None = 0,
        layout_seed: int = 0,
        extra: dict | None = None,
    ):
        self.n_cover = n_cover
        self.knn = knn
        self.regularization = regularization
        self.threshold = threshold
        self.label_mode = label_mode
        self.lite = lite
        self.random_state = random_state
        self.layout_seed = layout_seed
        self.extra = dict(extra or {})

    # -- fit ---------------------------------------------------------------- #

    def fit(self, X: np.ndarray) -> "ShapeDiscoverMethod":
        from shapediscover import ShapeDiscover, ShapeDiscoverLite

        if self.lite:
            est = ShapeDiscoverLite(
                n_cover=self.n_cover,
                knn=self.knn,
                regularization=self.regularization,
                random_state=self.random_state,
                **self.extra,
            )
        else:
            est = ShapeDiscover(
                n_cover=self.n_cover,
                knn=self.knn,
                random_state=self.random_state,
                verbose=False,
                plot_loss_curve=False,
                **self.extra,
            )
        self.cover_ = np.asarray(est.fit_transform(X))  # (n_points, n_cover)
        self.estimator_ = est
        # the underlying ShapeDiscover that exposes fit_persistence / simplex_tree_
        # (Lite wraps one at est._discover); used for the topology axis.
        self.discover_ = est if not self.lite else est._discover
        return self

    # -- internal helpers --------------------------------------------------- #

    def _internal_cover(self) -> np.ndarray:
        """Cover in the library-internal ``(n_cover, n_points)`` orientation."""
        return self.cover_.T

    def _survivors(self) -> np.ndarray:
        """Boolean mask over cover elements that survive thresholding."""
        from shapediscover.fuzzy_cover import threshold_fuzzy_cover

        _, mask = threshold_fuzzy_cover(self._internal_cover(), self.threshold)
        return mask

    def _nerve_components(self, survivors: np.ndarray) -> np.ndarray:
        """Connected-component label per surviving cover element."""
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        from shapediscover.fuzzy_cover import fuzzy_cover_to_weighted_edges

        thresholded = np.where(
            self._internal_cover()[survivors] < self.threshold, 0, 1
        )
        n = thresholded.shape[0]
        edges, _ = fuzzy_cover_to_weighted_edges(thresholded, min_weight=0)
        if len(edges) == 0:
            adj = coo_matrix((n, n))
        else:
            rows = np.concatenate([edges[:, 0], edges[:, 1]])
            cols = np.concatenate([edges[:, 1], edges[:, 0]])
            adj = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
        _, comp = connected_components(adj, directed=False)
        return comp

    # -- capability outputs ------------------------------------------------- #

    def labels(self) -> np.ndarray:
        if self.label_mode == "argmax":
            return np.argmax(self.cover_, axis=1)
        if self.label_mode == "components":
            survivors = self._survivors()
            if survivors.sum() == 0:
                return np.zeros(self.cover_.shape[0], dtype=int)
            comp = self._nerve_components(survivors)
            # each point -> component of its argmax surviving element
            surv_cover = self.cover_[:, survivors]  # (n_points, n_survivors)
            point_to_surv = np.argmax(surv_cover, axis=1)
            return comp[point_to_surv]
        raise ValueError(f"unknown label_mode {self.label_mode!r}")

    def embedding(self) -> np.ndarray:
        from shapediscover.nerve_layout import nerve_layout

        survivors = self._survivors()
        if survivors.sum() < 2:
            return np.zeros((self.cover_.shape[0], 2))
        positions, _ = nerve_layout(
            self._internal_cover(), self.threshold, seed=self.layout_seed
        )  # (n_survivors, 2), aligned with surviving elements in order
        weights = self.cover_[:, survivors]  # raw memberships as barycentric weights
        denom = weights.sum(axis=1, keepdims=True)
        denom[denom == 0] = 1.0
        return (weights @ positions) / denom

    def persistence(self, max_dim: int) -> tuple[list[np.ndarray], int]:
        # Delegate to the maintained ShapeDiscover.fit_persistence: it builds the
        # nerve from the public cover_, transposing to the internal orientation
        # itself (the same self.cover_.T boundary as PR5), with the log
        # normalization used in the paper's homology-recovery experiments.
        self.discover_.fit_persistence(max_dim, verbose=False)
        intervals = [
            np.asarray(pd).reshape(-1, 2) for pd in self.discover_.persistence_diagram_
        ]
        return intervals, int(self.discover_.simplex_tree_.num_simplices())

    # -- bookkeeping -------------------------------------------------------- #

    def n_active_cover(self) -> int:
        return int(self._survivors().sum())

    def params(self) -> dict:
        # the loss weights actually used, for reproducibility: Lite is
        # [1,0,0,reg]; full uses ShapeDiscover's default [1,10,1,10] unless
        # overridden via extra["loss_weights"].
        effective_loss_weights = (
            [1, 0, 0, self.regularization]
            if self.lite
            else self.extra.get("loss_weights", [1, 10, 1, 10])
        )
        return {
            "n_cover": self.n_cover,
            "knn": self.knn,
            "regularization": self.regularization,
            "threshold": self.threshold,
            "label_mode": self.label_mode,
            "lite": self.lite,
            "loss_weights": effective_loss_weights,
            **{f"extra.{k}": v for k, v in self.extra.items()},
        }
