"""Dimensionality-reduction embedding-axis baselines: UMAP and PCA.

Each produces a 2D per-point embedding scored by the same embedding metrics
(trustworthiness, continuity, kNN recall, global distance correlation) as
ShapeDiscover's barycentric pseudo-embedding, so the visualization/DR axis has a
reference. UMAP is the headline local-structure method; PCA is a cheap global,
linear floor. More DR baselines (t-SNE, PaCMAP, spectral, IsUMAP) slot in here
behind the same interface later.
"""

from __future__ import annotations

import numpy as np

from .methods import Method


class UMAPMethod(Method):
    """UMAP 2D embedding (the headline DR baseline)."""

    name = "umap"
    provides = ("embedding",)

    def __init__(self, knn: int = 15, random_state: int | None = 0, **_):
        self.knn = knn
        self.random_state = random_state

    def fit(self, X: np.ndarray) -> "UMAPMethod":
        import umap

        reducer = umap.UMAP(
            n_components=2, n_neighbors=self.knn, random_state=self.random_state
        )
        self.embedding_ = np.asarray(reducer.fit_transform(X))
        return self

    def embedding(self) -> np.ndarray:
        return self.embedding_

    def params(self) -> dict:
        return {"knn": self.knn}


class PCAMethod(Method):
    """PCA 2D embedding (a cheap linear, global-structure floor)."""

    name = "pca"
    provides = ("embedding",)

    def __init__(self, random_state: int | None = 0, **_):
        self.random_state = random_state

    def fit(self, X: np.ndarray) -> "PCAMethod":
        from sklearn.decomposition import PCA

        self.embedding_ = np.asarray(
            PCA(n_components=2, random_state=self.random_state).fit_transform(X)
        )
        return self

    def embedding(self) -> np.ndarray:
        return self.embedding_

    def params(self) -> dict:
        return {}


DR_METHODS = {
    "umap": UMAPMethod,
    "pca": PCAMethod,
}
