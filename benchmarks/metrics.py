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


# Cap for "no competing bar", so dominance values stay finite and printable.
_DOMINANCE_CAP = 1e3


def bar_dominance(intervals, target_betti):
    """Bar-based reading of how clearly the barcode shows ``target_betti``.

    Complements the recovery quotient, which is harsh on short noise bars (it can
    read near zero even when the right bars dominate). For each dimension ``d``
    with target Betti number ``b_d``, sort that dimension's bar lengths
    (persistences) decreasing, counting an infinite essential bar as longest. The
    per-dimension dominance is the ratio of the ``b_d``-th longest bar to the
    ``(b_d+1)``-th longest: large when the ``b_d`` expected features separate
    cleanly from the rest, 0 when a required feature is missing. For ``b_d == 0``
    it is instead ``(real-feature scale) / (longest bar in d)`` (large when the
    dimension is empty of long bars as it should be). Values are capped at
    ``_DOMINANCE_CAP`` (``inf`` -> cap). Returns ``(per_dim_list, min_over_dims)``;
    the min is the weakest-recovered dimension, and ``>> 1`` means the whole
    barcode reads as ``target_betti``.

    v1 caveat: a dimension whose ``b_d`` longest bars include the essential
    infinite bar (typically H0) is trivially capped and does not constrain the
    min; penalizing spurious *finite* H0 components is a future refinement.
    """
    persistences = []
    for pd in intervals:
        persistences.append(np.sort(pd[:, 1] - pd[:, 0])[::-1] if len(pd) else np.array([]))

    # reference scale: smallest finite "real" feature across dims with b_d >= 1.
    real = [s[b - 1] for b, s in zip(target_betti, persistences)
            if b >= 1 and len(s) >= b and np.isfinite(s[b - 1])]
    ref = min(real) if real else float("inf")

    per_dim = []
    for b, s in zip(target_betti, persistences):
        if b == 0:
            if len(s) == 0:
                dom = _DOMINANCE_CAP
            elif not np.isfinite(s[0]):
                dom = 0.0  # a spurious essential class in a should-be-empty dimension
            elif s[0] <= 0 or not np.isfinite(ref):
                dom = _DOMINANCE_CAP
            else:
                dom = ref / s[0]
        elif len(s) < b:
            dom = 0.0  # a required feature is missing
        elif len(s) == b:
            dom = _DOMINANCE_CAP  # exactly b bars, nothing competing
        else:
            nxt = s[b]
            dom = _DOMINANCE_CAP if nxt <= 0 else s[b - 1] / nxt
        per_dim.append(min(float(dom), _DOMINANCE_CAP))

    overall = min(per_dim) if per_dim else _DOMINANCE_CAP
    return per_dim, overall


def topology_metrics(intervals, complex_size, target_betti, n_bins: int = 1000) -> dict:
    per_dim, overall = bar_dominance(intervals, target_betti)
    out = {
        "recovery_quotient": homology_recovery_quotient(intervals, target_betti, n_bins),
        "complex_size": int(complex_size),
        "bar_dominance_min": round(float(overall), 3),
    }
    for d, v in enumerate(per_dim):
        out[f"bar_dominance_h{d}"] = round(float(v), 3)
    return out


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
