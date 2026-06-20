"""Classical TDA topology-axis baselines: Alpha, Vietoris-Rips, Witness.

Each computes the persistent homology of a standard geometric complex on the
point cloud and returns ``(intervals_per_dim, complex_size)`` in the same format
as the cover methods, so the topology metrics (recovery quotient, bar dominance,
complex size) apply on identical footing. These are the references ShapeDiscover
is compared against: Alpha is the parameter-free low-dimensional gold standard
(used in the notes as the torus ground truth), Witness is the few-landmark
small-complex baseline most comparable to the nerve's size, and Rips is the
classical (expensive) flag complex.

All three deterministically farthest-point subsample to a landmark budget (via
gudhi), which keeps them tractable on large point clouds and controls the
complex size. The runner catches per-axis failures, so a baseline that is
impractical on a given dataset (e.g. Alpha in high ambient dimension) is recorded
as an error and skipped rather than sinking the run.
"""

from __future__ import annotations

import numpy as np

from .methods import Method


def _farthest_point_subsample(X: np.ndarray, n_landmarks: int | None) -> np.ndarray:
    """Deterministic farthest-point subsample to ``n_landmarks`` points."""
    X = np.asarray(X, dtype=float)
    if n_landmarks is None or X.shape[0] <= n_landmarks:
        return X
    import gudhi

    pts = gudhi.subsampling.choose_n_farthest_points(
        points=X, nb_points=n_landmarks, starting_point=0
    )
    return np.asarray(pts, dtype=float)


def _intervals(simplex_tree, max_dim: int) -> list[np.ndarray]:
    """Per-dimension birth/death arrays from a persistence-computed simplex tree."""
    return [
        np.asarray(simplex_tree.persistence_intervals_in_dimension(d)).reshape(-1, 2)
        for d in range(max_dim + 1)
    ]


class AlphaComplexMethod(Method):
    """Alpha complex (Delaunay) persistence. Parameter-free, exact, and small in
    low ambient dimension; impractical above ``max_ambient_dim`` (skipped there).
    """

    name = "alpha"
    provides = ("topology",)

    def __init__(self, n_landmarks: int | None = 400, max_ambient_dim: int = 10,
                 random_state: int | None = 0, **_):
        self.n_landmarks = n_landmarks
        self.max_ambient_dim = max_ambient_dim
        self.random_state = random_state

    def fit(self, X: np.ndarray) -> "AlphaComplexMethod":
        if X.shape[1] > self.max_ambient_dim:
            raise ValueError(
                f"AlphaComplex is impractical in ambient dimension {X.shape[1]} "
                f"(> {self.max_ambient_dim})"
            )
        self.X_ = _farthest_point_subsample(X, self.n_landmarks)
        return self

    def persistence(self, max_dim: int) -> tuple[list[np.ndarray], int]:
        import gudhi

        st = gudhi.AlphaComplex(points=self.X_).create_simplex_tree()
        st.persistence()
        return _intervals(st, max_dim), int(st.num_simplices())

    def params(self) -> dict:
        return {"n_landmarks": self.n_landmarks}


class RipsComplexMethod(Method):
    """Vietoris-Rips flag complex persistence on a farthest-point subsample.

    ``max_edge_length`` is set to the ``edge_quantile`` quantile of the
    subsample's pairwise distances (data-scaled), which bounds the complex size;
    the flag complex is expanded to ``max_dim + 1``.
    """

    name = "rips"
    provides = ("topology",)

    def __init__(self, n_landmarks: int | None = 150, edge_quantile: float = 0.7,
                 random_state: int | None = 0, **_):
        self.n_landmarks = n_landmarks
        self.edge_quantile = edge_quantile
        self.random_state = random_state

    def fit(self, X: np.ndarray) -> "RipsComplexMethod":
        self.X_ = _farthest_point_subsample(X, self.n_landmarks)
        return self

    def persistence(self, max_dim: int) -> tuple[list[np.ndarray], int]:
        import gudhi
        from scipy.spatial.distance import pdist

        max_edge = float(np.quantile(pdist(self.X_), self.edge_quantile))
        st = gudhi.RipsComplex(points=self.X_, max_edge_length=max_edge).create_simplex_tree(
            max_dimension=max_dim + 1
        )
        st.persistence()
        return _intervals(st, max_dim), int(st.num_simplices())

    def params(self) -> dict:
        return {"n_landmarks": self.n_landmarks, "edge_quantile": self.edge_quantile}


class WitnessComplexMethod(Method):
    """Euclidean (weak) witness complex persistence (de Silva-Carlsson).

    Few landmarks (``n_landmarks``), all points as witnesses. ``max_alpha_square``
    defaults to ``(alpha_scale * median nearest-landmark distance)^2`` so the
    relaxation scales with the landmark spacing rather than the absolute units.

    Caveat (verified): the witness complex is strongly scale-fragile under a fixed
    ``alpha_scale`` -- no single value recovers the circle, 2-sphere, and torus
    together (e.g. ``1.5`` recovers the torus but not the circle/2-sphere). Fair
    witness numbers need per-dataset ``alpha_scale`` selection or the paper's
    binary-search-to-target-recovery mode (a planned follow-up); the default here
    is only a starting point, not a tuned baseline.
    """

    name = "witness"
    provides = ("topology",)

    def __init__(self, n_landmarks: int = 100, alpha_scale: float = 1.5,
                 random_state: int | None = 0, **_):
        self.n_landmarks = n_landmarks
        self.alpha_scale = alpha_scale
        self.random_state = random_state

    def fit(self, X: np.ndarray) -> "WitnessComplexMethod":
        self.X_ = np.asarray(X, dtype=float)
        self.landmarks_ = _farthest_point_subsample(X, self.n_landmarks)
        return self

    def persistence(self, max_dim: int) -> tuple[list[np.ndarray], int]:
        import gudhi
        from scipy.spatial.distance import cdist

        # landmark spacing: median nearest-other-landmark distance
        d = cdist(self.landmarks_, self.landmarks_)
        np.fill_diagonal(d, np.inf)
        spacing = float(np.median(d.min(axis=1)))
        max_alpha_square = (self.alpha_scale * spacing) ** 2

        wc = gudhi.EuclideanWitnessComplex(
            landmarks=self.landmarks_, witnesses=self.X_
        )
        st = wc.create_simplex_tree(
            max_alpha_square=max_alpha_square, limit_dimension=max_dim + 1
        )
        st.persistence()
        return _intervals(st, max_dim), int(st.num_simplices())

    def params(self) -> dict:
        return {"n_landmarks": self.n_landmarks, "alpha_scale": self.alpha_scale}


TDA_METHODS = {
    "alpha": AlphaComplexMethod,
    "rips": RipsComplexMethod,
    "witness": WitnessComplexMethod,
}
