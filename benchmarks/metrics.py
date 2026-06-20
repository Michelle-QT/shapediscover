"""Per-axis benchmark metrics.

Each function returns a flat ``{metric_name: value}`` dict so the runner can emit
one tidy row per metric. Three axes:

- ``topology_metrics``   homology recovery quotient + complex size.
- ``clustering_metrics`` ARI / AMI / NMI / homogeneity / completeness + cluster counts.
- ``embedding_metrics``  trustworthiness / continuity / kNN recall / global distance
  correlation (the per-point structure-preservation view of visualization / DR).
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Topology
# --------------------------------------------------------------------------- #

def homology_recovery_quotient(intervals, target_betti, n_bins: int = 1000) -> float:
    """Fraction of the filtration whose Betti numbers equal ``target_betti``.

    The paper's homology-recovery metric (see ``examples/experiment_functions``
    and ``tests/helpers``): discretize the filtration into ``n_bins`` values and
    report the fraction at which the Betti numbers match ``target_betti`` exactly
    in every dimension. ``intervals[d]`` is an ``(n_d, 2)`` birth/death array.
    """
    target_betti = np.asarray(target_betti)
    if len(intervals) != len(target_betti):
        raise ValueError("intervals and target_betti must have the same length")

    finite = lambda a: a[a < np.inf]
    nonempty = [pd for pd in intervals if len(pd) > 0]
    if not nonempty:
        return 0.0
    min_value = min(np.min(pd) for pd in nonempty)
    finite_vals = [finite(pd) for pd in nonempty if len(finite(pd)) > 0]
    if not finite_vals:
        return 0.0
    max_value = max(np.max(fv) for fv in finite_vals)

    grid = np.linspace(min_value, max_value, n_bins)
    betti = np.zeros((n_bins, len(target_betti)), dtype=int)
    for dim, pd in enumerate(intervals):
        for birth, death in pd:
            start = np.searchsorted(grid, birth)
            end = np.searchsorted(grid, death)
            betti[start:end, dim] += 1
    return float(np.mean(np.all(betti == target_betti, axis=1)))


def topology_metrics(intervals, complex_size, target_betti, n_bins: int = 1000) -> dict:
    return {
        "recovery_quotient": homology_recovery_quotient(intervals, target_betti, n_bins),
        "complex_size": int(complex_size),
    }


# --------------------------------------------------------------------------- #
# Clustering
# --------------------------------------------------------------------------- #

def clustering_metrics(labels_true, labels_pred) -> dict:
    from sklearn.metrics import (
        adjusted_mutual_info_score,
        adjusted_rand_score,
        completeness_score,
        homogeneity_score,
        normalized_mutual_info_score,
    )

    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)
    return {
        "ari": float(adjusted_rand_score(labels_true, labels_pred)),
        "ami": float(adjusted_mutual_info_score(labels_true, labels_pred)),
        "nmi": float(normalized_mutual_info_score(labels_true, labels_pred)),
        "homogeneity": float(homogeneity_score(labels_true, labels_pred)),
        "completeness": float(completeness_score(labels_true, labels_pred)),
        "n_clusters_pred": int(len(np.unique(labels_pred))),
        "n_clusters_true": int(len(np.unique(labels_true))),
    }


# --------------------------------------------------------------------------- #
# Embedding (per-point structure preservation)
# --------------------------------------------------------------------------- #

def _knn_indices(D: np.ndarray, k: int) -> np.ndarray:
    """Indices of the ``k`` nearest neighbours (excluding self) per row."""
    order = np.argsort(D, axis=1)
    return order[:, 1 : k + 1]


def knn_recall(X_high: np.ndarray, X_emb: np.ndarray, k: int = 15) -> float:
    """Mean fraction of each point's high-dim ``k``-NN preserved in the embedding."""
    from sklearn.metrics import pairwise_distances

    n = X_high.shape[0]
    k = min(k, n - 1)
    hi = _knn_indices(pairwise_distances(X_high), k)
    lo = _knn_indices(pairwise_distances(X_emb), k)
    overlap = [len(set(hi[i]) & set(lo[i])) for i in range(n)]
    return float(np.mean(overlap) / k)


def global_distance_correlation(
    X_high: np.ndarray, X_emb: np.ndarray, max_points: int = 1000, seed: int = 0
) -> float:
    """Spearman correlation of pairwise distances (global structure preservation).

    Subsamples to ``max_points`` for tractability; deterministic given ``seed``.
    """
    from scipy.spatial.distance import pdist
    from scipy.stats import spearmanr

    n = X_high.shape[0]
    rng = np.random.default_rng(seed)
    if n > max_points:
        idx = rng.choice(n, size=max_points, replace=False)
        X_high, X_emb = X_high[idx], X_emb[idx]
    dh = pdist(X_high)
    de = pdist(X_emb)
    rho, _ = spearmanr(dh, de)
    return float(rho)


def embedding_metrics(
    X_high: np.ndarray, X_emb: np.ndarray, k: int = 15, seed: int = 0
) -> dict:
    """Standard DR quality metrics, directly comparable across methods.

    ``trustworthiness`` (no false neighbours introduced) and ``continuity`` (no
    true neighbours lost) are sklearn's local-structure measures; ``knn_recall``
    is the kNN-overlap view of the same; ``global_corr`` captures large-scale
    geometry. NaNs are returned for degenerate embeddings rather than raising.
    """
    from sklearn.manifold import trustworthiness

    n = X_high.shape[0]
    k = min(k, n - 1)
    out: dict = {}
    try:
        out["trustworthiness"] = float(trustworthiness(X_high, X_emb, n_neighbors=k))
        # continuity = trustworthiness with the roles of the two spaces swapped
        out["continuity"] = float(trustworthiness(X_emb, X_high, n_neighbors=k))
        out["knn_recall"] = knn_recall(X_high, X_emb, k)
        out["global_corr"] = global_distance_correlation(X_high, X_emb, seed=seed)
    except Exception as exc:  # degenerate embedding (e.g. collapsed points)
        for key in ("trustworthiness", "continuity", "knn_recall", "global_corr"):
            out.setdefault(key, float("nan"))
        out["embedding_error"] = type(exc).__name__
    return out
