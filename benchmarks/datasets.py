"""Dataset registry for the benchmark harness.

A single source of truth for benchmark data. Each entry is a factory
``(seed, **kwargs) -> BenchmarkDataset`` registered in :data:`DATASETS`. A
:class:`BenchmarkDataset` carries the point cloud, optional ground-truth labels
(for the clustering axis), optional target Betti numbers (for the topology axis),
and the set of capability ``axes`` it supports.

Synthetic generators are deterministic given ``seed`` (they use a local
``numpy.random.Generator``, never the global stream). Real datasets are loaded
from ``DATA_DIR`` (the paper repo's ``examples/datasets`` by default); registry
entries that need a missing file raise a clear error when built, so the rest of
the suite still runs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

# Capability axes a dataset can be evaluated on.
TOPOLOGY = "topology"
CLUSTERING = "clustering"
EMBEDDING = "embedding"

# Where to find the real (non-synthetic) datasets. Override with the
# SHAPEDISCOVER_DATA_DIR environment variable. Defaults to this repo's own
# examples/datasets (the superset), then the sibling paper repo's copy.
def _resolve_data_dir() -> Path:
    env = os.environ.get("SHAPEDISCOVER_DATA_DIR")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    candidates = [
        here.parents[1] / "examples" / "datasets",  # shapediscover-repo (superset)
        here.parents[2] / "paper-repo" / "examples" / "datasets",  # paper-repo fallback
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return candidates[0]


DATA_DIR = _resolve_data_dir()


@dataclass
class BenchmarkDataset:
    """A materialized benchmark dataset.

    Attributes
    ----------
    name : str
        Registry key.
    X : ndarray of shape (n_points, n_features)
        The point cloud.
    labels : ndarray of shape (n_points,) or None
        Ground-truth labels for the clustering axis, if any.
    target_betti : list[int] or None
        Expected Betti numbers ``[b0, b1, ...]`` for the topology axis, if known.
    axes : tuple[str, ...]
        Which capability axes this dataset supports.
    preprocessing : str
        The declared feature-preprocessing policy applied in :func:`load`
        (``"none"`` / ``"standardize"`` / ``"unit_norm"``); see :func:`_preprocess`.
    metadata : dict
        Free-form provenance / parameters (kept in the results table).
    """

    name: str
    X: np.ndarray
    labels: np.ndarray | None = None
    target_betti: list[int] | None = None
    axes: tuple[str, ...] = ()
    preprocessing: str = "none"
    metadata: dict = field(default_factory=dict)

    @property
    def n_points(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])


# --------------------------------------------------------------------------- #
# Synthetic generators (seeded, no global random state).
# --------------------------------------------------------------------------- #

def _sphere(n: int, d: int, rng: np.random.Generator, noise: float = 0.0) -> np.ndarray:
    """``n`` points on the unit ``d``-sphere in R^{d+1} (Gaussian, normalized)."""
    pts = rng.standard_normal((n, d + 1))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    if noise:
        pts = pts + rng.normal(scale=np.sqrt(noise), size=pts.shape)
    return pts


def _torus(n: int, rng: np.random.Generator, r1: float = 1.0, r2: float = 0.5,
           noise: float = 0.0, sampling: str = "angle") -> np.ndarray:
    """``n`` points on a torus in R^3, tube radius ``r2`` and center radius ``r1``.

    ``sampling`` controls the tube-angle distribution, which sets the *density*
    while the embedding (hence the *curvature*) is fixed by ``r1``/``r2``:

    - ``"angle"`` uniform in both angles (the historical default); denser on the
      inner rim, where the tube circle is shorter.
    - ``"area"`` uniform with respect to the Riemannian area, by rejection
      sampling the tube angle with acceptance ``(r1 + r2 cos t2) / (r1 + r2)``.

    Used to separate density from curvature (the donut diagnosis: area-uniform
    resampling does *not* fix recovery, so the obstacle is curvature variation,
    not density). The Gaussian curvature is ``cos t2 / (r2 (r1 + r2 cos t2))``,
    so its range grows as the tube thins or fattens and is 0 only in the flat
    (Clifford) torus; see ``clifford_torus``.
    """
    t1 = rng.random(n) * 2 * np.pi
    if sampling == "area":
        accepted = []
        while len(accepted) < n:
            cand = rng.random(n) * 2 * np.pi
            keep = rng.random(n) < (r1 + r2 * np.cos(cand)) / (r1 + r2)
            accepted.extend(cand[keep].tolist())
        t2 = np.array(accepted[:n])
    elif sampling == "angle":
        t2 = rng.random(n) * 2 * np.pi
    else:
        raise ValueError(f"sampling must be 'angle' or 'area'; got {sampling!r}")
    x = (r1 + r2 * np.cos(t2)) * np.cos(t1)
    y = (r1 + r2 * np.cos(t2)) * np.sin(t1)
    z = r2 * np.sin(t2)
    pts = np.stack([x, y, z], axis=1)
    if noise:
        pts = pts + rng.normal(scale=np.sqrt(noise), size=pts.shape)
    return pts


def _disk(n: int, d: int, rng: np.random.Generator, noise: float = 0.0) -> np.ndarray:
    """``n`` points uniformly in the unit ``d``-disk (contractible)."""
    y = _sphere(n, d - 1, rng)
    radii = np.power(rng.random(n), 1.0 / d)
    pts = radii.reshape(-1, 1) * y
    if noise:
        pts = pts + rng.normal(scale=np.sqrt(noise), size=pts.shape)
    return pts


# Higher-dimensional / product manifolds with known homology, so the topology
# axis is not over-fit to the single R^3 2-torus. Homology of products is the
# tensor product of the factors' (Kuenneth); for products of spheres and circles
# the Betti numbers are field-independent (orientable, torsion-free), so they
# match gudhi's field-coefficient persistence. References: Hatcher, *Algebraic
# Topology* (Kuenneth / products); the d-sphere / k-torus families follow the
# tadasets (scikit-tda) and high-dimensional-PH-benchmark conventions.

def _flat_torus(n: int, k: int, rng: np.random.Generator,
                noise: float = 0.0) -> np.ndarray:
    """``n`` points on the flat ``k``-torus ``(S^1)^k`` in ``R^{2k}``.

    Each circle contributes ``(cos, sin)/sqrt(k)`` for a fixed radius; the angles
    are uniform, which (unlike the curved R^3 donut) is genuinely uniform on the
    flat torus. ``k=2`` is the Clifford torus. Betti numbers are the binomials
    ``C(k, j)`` (``T^2``: ``[1,2,1]``; ``T^3``: ``[1,3,3,1]``).
    """
    angles = rng.random((n, k)) * 2 * np.pi
    coords = []
    scale = 1.0 / np.sqrt(k)
    for j in range(k):
        coords.append(scale * np.cos(angles[:, j]))
        coords.append(scale * np.sin(angles[:, j]))
    pts = np.stack(coords, axis=1)
    if noise:
        pts = pts + rng.normal(scale=np.sqrt(noise), size=pts.shape)
    return pts


def _sphere_times_circle(n: int, d: int, rng: np.random.Generator,
                         noise: float = 0.0) -> np.ndarray:
    """``n`` points on ``S^d x S^1`` in ``R^{d+3}`` (product of uniform samples).

    Poincare polynomial ``(1+t^d)(1+t)``: for ``d=2`` the Betti numbers are
    ``[1,1,1,1]`` (a connected component, one loop from the circle, one void from
    the sphere, one 3-cycle from their product).
    """
    s = _sphere(n, d, rng)                      # (n, d+1)
    theta = rng.random(n) * 2 * np.pi
    c = np.stack([np.cos(theta), np.sin(theta)], axis=1)  # (n, 2)
    pts = np.concatenate([s, c], axis=1)
    if noise:
        pts = pts + rng.normal(scale=np.sqrt(noise), size=pts.shape)
    return pts


def _embed_ambient(X: np.ndarray, ambient: int | None,
                   rng: np.random.Generator) -> np.ndarray:
    """Isometrically embed ``X`` (n, d) into ``R^ambient`` (d < ambient).

    Zero-pads to ``ambient`` then applies a random rotation (QR of a Gaussian),
    preserving all pairwise distances. Tests robustness to a high *ambient*
    dimension while the intrinsic dimension and homology are unchanged (the
    high-dimensional-PH benchmark convention).
    """
    d = X.shape[1]
    if ambient is None or ambient <= d:
        return X
    Z = np.zeros((X.shape[0], ambient))
    Z[:, :d] = X
    Q, _ = np.linalg.qr(rng.standard_normal((ambient, ambient)))
    return Z @ Q.T


# --------------------------------------------------------------------------- #
# Registry. Each factory takes (seed, **kwargs) and returns a BenchmarkDataset.
# --------------------------------------------------------------------------- #

DATASETS: dict[str, Callable[..., BenchmarkDataset]] = {}


def register(name: str) -> Callable[[Callable[..., BenchmarkDataset]], Callable[..., BenchmarkDataset]]:
    def deco(fn: Callable[..., BenchmarkDataset]) -> Callable[..., BenchmarkDataset]:
        DATASETS[name] = fn
        return fn
    return deco


# ---- topology (known Betti numbers) --------------------------------------- #

@register("circle")
def _ds_circle(seed: int = 0, n: int = 400, noise: float = 0.0) -> BenchmarkDataset:
    rng = np.random.default_rng(seed)
    return BenchmarkDataset(
        "circle", _sphere(n, 1, rng, noise), target_betti=[1, 1],
        axes=(TOPOLOGY, EMBEDDING), metadata={"n": n, "noise": noise},
    )


@register("sphere2")
def _ds_sphere2(seed: int = 0, n: int = 600, noise: float = 0.0) -> BenchmarkDataset:
    rng = np.random.default_rng(seed)
    return BenchmarkDataset(
        "sphere2", _sphere(n, 2, rng, noise), target_betti=[1, 0, 1],
        axes=(TOPOLOGY, EMBEDDING), metadata={"n": n, "noise": noise},
    )


@register("sphere3")
def _ds_sphere3(seed: int = 0, n: int = 1200, noise: float = 0.0) -> BenchmarkDataset:
    rng = np.random.default_rng(seed)
    return BenchmarkDataset(
        "sphere3", _sphere(n, 3, rng, noise), target_betti=[1, 0, 0, 1],
        axes=(TOPOLOGY,), metadata={"n": n, "noise": noise},
    )


@register("torus")
def _ds_torus(seed: int = 0, n: int = 3000, noise: float = 0.0,
              r1: float = 1.0, r2: float = 0.5,
              sampling: str = "angle") -> BenchmarkDataset:
    # n=3000 (was 1500): the torus needs ~55+ points per cover element to
    # resolve [1,2,1]; at n_cover=52 the old 1500 (~29 pts/elt) was too sparse
    # to recover (see the synthetic-torus resolution in the project notes).
    # r1/r2 (center/tube radius) and sampling (angle/area) are exposed so the
    # curvature / density diagnosis is reproducible from load(): e.g.
    # load("torus", r2=0.65) or load("torus", sampling="area"). The R^3 donut's
    # hardness is curvature variation, not density (see the project notes); the
    # flat counterpart is clifford_torus.
    rng = np.random.default_rng(seed)
    return BenchmarkDataset(
        "torus", _torus(n, rng, r1=r1, r2=r2, noise=noise, sampling=sampling),
        target_betti=[1, 2, 1], axes=(TOPOLOGY, EMBEDDING),
        metadata={"n": n, "noise": noise, "r1": r1, "r2": r2, "sampling": sampling},
    )


@register("two_circles")
def _ds_two_circles(seed: int = 0, n: int = 400, noise: float = 0.0) -> BenchmarkDataset:
    """Two disjoint circles: b0 = 2, b1 = 2. Doubles as a 2-cluster dataset."""
    rng = np.random.default_rng(seed)
    a = _sphere(n, 1, rng, noise)
    b = _sphere(n, 1, rng, noise) + np.array([3.0, 0.0])
    X = np.vstack([a, b])
    labels = np.concatenate([np.zeros(n, int), np.ones(n, int)])
    return BenchmarkDataset(
        "two_circles", X, labels=labels, target_betti=[2, 2],
        axes=(TOPOLOGY, CLUSTERING, EMBEDDING), metadata={"n": n, "noise": noise},
    )


# ---- higher-dimensional / product manifolds (topology axis, harder) -------- #

@register("clifford_torus")
def _ds_clifford_torus(seed: int = 0, n: int = 3000, noise: float = 0.0) -> BenchmarkDataset:
    """Flat 2-torus (S^1)^2 in R^4, Betti [1,2,1].

    The curvature-free, uniformly-sampled counterpart of the R^3 donut ``torus``:
    same topology, no inner/outer sampling-density distortion, so it isolates the
    topology from the donut's geometry.
    """
    rng = np.random.default_rng(seed)
    return BenchmarkDataset(
        "clifford_torus", _flat_torus(n, 2, rng, noise), target_betti=[1, 2, 1],
        axes=(TOPOLOGY,), metadata={"n": n, "noise": noise, "intrinsic_dim": 2},
    )


@register("anisotropic_torus")
def _ds_anisotropic_torus(seed: int = 0, n: int = 3000, ratio: float = 0.4,
                          noise: float = 0.0) -> BenchmarkDataset:
    """Flat torus (S^1 x S^1) in R^4 with UNEQUAL cycle radii (1 and ``ratio``).

    Zero Gaussian curvature, but the two H1 cycles live at different geometric
    scales. This isolates *cycle-scale anisotropy* from curvature: recovery
    collapses as ``ratio`` shrinks (e.g. ~0 at ratio 0.35) on raw features, but is
    fully restored by scale-normalization (``preprocessing="standardize"`` or
    ``"whiten"``), which is why this dataset declares ``whiten``. The driver of the
    R^3 ``torus`` (donut) difficulty is largely this anisotropy, not curvature
    (see the project notes); the donut additionally has a nonlinear/local part
    that global whitening does not fix.
    """
    rng = np.random.default_rng(seed)
    a = rng.random(n) * 2 * np.pi
    b = rng.random(n) * 2 * np.pi
    X = np.stack([np.cos(a), np.sin(a), ratio * np.cos(b), ratio * np.sin(b)], axis=1)
    if noise:
        X = X + rng.normal(scale=np.sqrt(noise), size=X.shape)
    return BenchmarkDataset(
        "anisotropic_torus", X, target_betti=[1, 2, 1], axes=(TOPOLOGY,),
        preprocessing="whiten",
        metadata={"n": n, "ratio": ratio, "noise": noise, "intrinsic_dim": 2},
    )


@register("torus3")
def _ds_torus3(seed: int = 0, n: int = 4000, noise: float = 0.0) -> BenchmarkDataset:
    """Flat 3-torus (S^1)^3 in R^6, Betti [1,3,3,1] (binomials C(3,j)).

    Higher intrinsic dimension and higher Betti numbers than the 2-torus; the
    natural stress test for whether the high-overlap regime generalizes.
    """
    rng = np.random.default_rng(seed)
    return BenchmarkDataset(
        "torus3", _flat_torus(n, 3, rng, noise), target_betti=[1, 3, 3, 1],
        axes=(TOPOLOGY,), metadata={"n": n, "noise": noise, "intrinsic_dim": 3},
    )


@register("s2_times_s1")
def _ds_s2_times_s1(seed: int = 0, n: int = 3000, noise: float = 0.0) -> BenchmarkDataset:
    """S^2 x S^1 in R^5, Betti [1,1,1,1] (Kuenneth of [1,0,1] and [1,1]).

    A product with a loop, a void, and a 3-cycle: a different higher-dimensional
    topology from the tori (mixed-degree homology).
    """
    rng = np.random.default_rng(seed)
    return BenchmarkDataset(
        "s2_times_s1", _sphere_times_circle(n, 2, rng, noise), target_betti=[1, 1, 1, 1],
        axes=(TOPOLOGY,), metadata={"n": n, "noise": noise, "intrinsic_dim": 3},
    )


@register("sphere4")
def _ds_sphere4(seed: int = 0, n: int = 2000, noise: float = 0.0) -> BenchmarkDataset:
    """S^4 in R^5, Betti [1,0,0,0,1] (a single 4-void)."""
    rng = np.random.default_rng(seed)
    return BenchmarkDataset(
        "sphere4", _sphere(n, 4, rng, noise), target_betti=[1, 0, 0, 0, 1],
        axes=(TOPOLOGY,), metadata={"n": n, "noise": noise, "intrinsic_dim": 4},
    )


@register("clifford_torus_amb50")
def _ds_clifford_torus_amb50(seed: int = 0, n: int = 3000, noise: float = 0.0,
                             ambient: int = 50) -> BenchmarkDataset:
    """Flat 2-torus isometrically embedded in R^50, Betti [1,2,1].

    Same intrinsic torus, high ambient dimension: tests robustness to ambient
    dimension / distance concentration (the high-dimensional-PH benchmark setup)
    without changing the homology.
    """
    rng = np.random.default_rng(seed)
    X = _embed_ambient(_flat_torus(n, 2, rng, noise), ambient, rng)
    return BenchmarkDataset(
        "clifford_torus_amb50", X, target_betti=[1, 2, 1],
        axes=(TOPOLOGY,), metadata={"n": n, "noise": noise, "ambient": ambient,
                                    "intrinsic_dim": 2},
    )


# Manifolds reimplemented (clean-room) from the collaborator's manifold library
# (code-snippets/manifold_benchmarks.py, untested -> used as the mathematical
# spec only). Betti numbers verified by Rips on a farthest-point subsample.

@register("cp2")
def _ds_cp2(seed: int = 0, n: int = 3000, noise: float = 0.0) -> BenchmarkDataset:
    """Complex projective plane CP^2 in R^9, Betti [1,0,1,0,1] (field-independent).

    Sample S^5 in C^3 and apply the rank-1 Hermitian projector map (U(1)-invariant,
    so it descends to CP^2): the 9 real coordinates of z z^* upper triangle. An
    orientable 4-manifold with H2 and H4 -- a high-dimensional-homology test.
    """
    rng = np.random.default_rng(seed)
    s5 = _sphere(n, 5, rng)                      # (n, 6) unit vectors in R^6 = C^3
    z = s5[:, :3] + 1j * s5[:, 3:]
    z0, z1, z2 = z[:, 0], z[:, 1], z[:, 2]
    X = np.column_stack([
        np.abs(z0) ** 2, np.abs(z1) ** 2, np.abs(z2) ** 2,
        (np.conj(z0) * z1).real, (np.conj(z0) * z1).imag,
        (np.conj(z0) * z2).real, (np.conj(z0) * z2).imag,
        (np.conj(z1) * z2).real, (np.conj(z1) * z2).imag,
    ])
    if noise:
        X = X + rng.normal(scale=np.sqrt(noise), size=X.shape)
    return BenchmarkDataset(
        "cp2", X, target_betti=[1, 0, 1, 0, 1], axes=(TOPOLOGY,),
        metadata={"n": n, "noise": noise, "intrinsic_dim": 4},
    )


@register("figure_eight")
def _ds_figure_eight(seed: int = 0, n: int = 800, noise: float = 0.01) -> BenchmarkDataset:
    """Figure-eight (lemniscate of Gerono) in R^2, Betti [1,2].

    A single closed curve crossing itself once: a wedge of two circles
    (b1 = 2). The self-crossing is a *bottleneck* (the two branches pass close),
    so it is a clean threshold / inhomogeneous-geometry stress in 1D.
    """
    rng = np.random.default_rng(seed)
    t = rng.random(n) * 2 * np.pi
    X = np.column_stack([np.cos(t), np.sin(t) * np.cos(t)])
    if noise:
        X = X + rng.normal(scale=noise, size=X.shape)
    return BenchmarkDataset(
        "figure_eight", X, target_betti=[1, 2], axes=(TOPOLOGY,),
        metadata={"n": n, "noise": noise, "intrinsic_dim": 1},
    )


@register("linked_circles")
def _ds_linked_circles(seed: int = 0, n: int = 800, noise: float = 0.0) -> BenchmarkDataset:
    """Two linked circles (Hopf link) in R^3, Betti [2,2].

    Homologically identical to ``two_circles`` (two components, two loops);
    ordinary homology does not see the *linking*, so this stresses the embedding /
    geometry rather than the topology axis (a useful negative-distinction set).
    """
    rng = np.random.default_rng(seed)
    half = n // 2
    t = rng.random(half) * 2 * np.pi
    s = rng.random(n - half) * 2 * np.pi
    a = np.column_stack([np.cos(t), np.sin(t), np.zeros(half)])
    b = np.column_stack([1 + np.cos(s), np.zeros(n - half), np.sin(s)])
    X = np.vstack([a, b])
    labels = np.concatenate([np.zeros(half, int), np.ones(n - half, int)])
    if noise:
        X = X + rng.normal(scale=np.sqrt(noise), size=X.shape)
    return BenchmarkDataset(
        "linked_circles", X, labels=labels, target_betti=[2, 2],
        axes=(TOPOLOGY, CLUSTERING), metadata={"n": n, "noise": noise},
    )


@register("noise")
def _ds_noise(seed: int = 0, n: int = 1000, dim: int = 10) -> BenchmarkDataset:
    """Isotropic Gaussian point cloud in R^dim: a topological NEGATIVE CONTROL.

    Structureless noise has no topology, so the target is ``[1, 0, 0]`` (one
    component, no loops, no voids): a method that reports loops or voids here is
    hallucinating features. Every topology-recovery number should be read against
    this control. The high-dimensional setup follows the high-dimensional-PH
    benchmark's noise control (NeurIPS 2024, arXiv:2311.03087).
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, dim))
    return BenchmarkDataset(
        "noise", X, target_betti=[1, 0, 0], axes=(TOPOLOGY,),
        metadata={"n": n, "dim": dim, "negative_control": True},
    )


# ---- synthetic manifolds (embedding / DR; contractible, no interesting H) -- #

@register("swiss_roll")
def _ds_swiss_roll(seed: int = 0, n: int = 1500, noise: float = 0.05) -> BenchmarkDataset:
    from sklearn.datasets import make_swiss_roll
    X, t = make_swiss_roll(n_samples=n, noise=noise, random_state=seed)
    return BenchmarkDataset(
        "swiss_roll", X.astype(float), axes=(EMBEDDING,),
        metadata={"n": n, "noise": noise, "manifold_param": t.tolist()},
    )


@register("s_curve")
def _ds_s_curve(seed: int = 0, n: int = 1500, noise: float = 0.05) -> BenchmarkDataset:
    from sklearn.datasets import make_s_curve
    X, t = make_s_curve(n_samples=n, noise=noise, random_state=seed)
    return BenchmarkDataset(
        "s_curve", X.astype(float), axes=(EMBEDDING,),
        metadata={"n": n, "noise": noise, "manifold_param": t.tolist()},
    )


# ---- clustering / embedding (labeled) ------------------------------------- #

@register("digits")
def _ds_digits(seed: int = 0) -> BenchmarkDataset:
    from sklearn.datasets import load_digits
    d = load_digits()
    return BenchmarkDataset(
        "digits", d.data.astype(float), labels=d.target.astype(int),
        axes=(CLUSTERING, EMBEDDING),
        metadata={"n": int(d.data.shape[0]), "n_classes": 10},
    )


@register("blobs")
def _ds_blobs(seed: int = 0, n: int = 900, centers: int = 5, dim: int = 10) -> BenchmarkDataset:
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples=n, centers=centers, n_features=dim, random_state=seed)
    return BenchmarkDataset(
        "blobs", X.astype(float), labels=y.astype(int),
        axes=(CLUSTERING, EMBEDDING),
        metadata={"n": n, "centers": centers, "dim": dim},
    )


@register("moons")
def _ds_moons(seed: int = 0, n: int = 800, noise: float = 0.08) -> BenchmarkDataset:
    from sklearn.datasets import make_moons
    X, y = make_moons(n_samples=n, noise=noise, random_state=seed)
    return BenchmarkDataset(
        "moons", X.astype(float), labels=y.astype(int),
        axes=(CLUSTERING, EMBEDDING), metadata={"n": n, "noise": noise},
    )


@register("nested_circles")
def _ds_nested_circles(seed: int = 0, n: int = 800, noise: float = 0.05,
                       factor: float = 0.5) -> BenchmarkDataset:
    from sklearn.datasets import make_circles
    X, y = make_circles(n_samples=n, noise=noise, factor=factor, random_state=seed)
    return BenchmarkDataset(
        "nested_circles", X.astype(float), labels=y.astype(int),
        axes=(CLUSTERING, EMBEDDING), metadata={"n": n, "noise": noise},
    )


@register("iris")
def _ds_iris(seed: int = 0) -> BenchmarkDataset:
    from sklearn.datasets import load_iris
    d = load_iris()
    # Raw: a single unit (cm) with only mild (~4x) scale spread, so standardizing
    # is unwarranted (and empirically slightly hurts); see the preprocessing policy.
    return BenchmarkDataset(
        "iris", d.data.astype(float), labels=d.target.astype(int),
        axes=(CLUSTERING, EMBEDDING),
        metadata={"n": int(d.data.shape[0]), "n_classes": 3},
    )


@register("wine")
def _ds_wine(seed: int = 0) -> BenchmarkDataset:
    # Heterogeneous feature scales (e.g. proline ~1000 vs others ~1): standardized
    # by the registry's preprocessing policy (see _preprocess).
    from sklearn.datasets import load_wine
    d = load_wine()
    return BenchmarkDataset(
        "wine", d.data.astype(float), labels=d.target.astype(int),
        axes=(CLUSTERING, EMBEDDING), preprocessing="standardize",
        metadata={"n": int(d.data.shape[0]), "n_classes": 3},
    )


# ---- real datasets (loaded from DATA_DIR) --------------------------------- #

@register("diabetes")
def _ds_diabetes(seed: int = 0) -> BenchmarkDataset:
    """Reaven-Miller diabetes data (original Mapper paper); 3 clinical groups."""
    import pandas as pd
    path = DATA_DIR / "diabetes.csv"
    if not path.exists():
        raise FileNotFoundError(f"diabetes.csv not found at {path}; set SHAPEDISCOVER_DATA_DIR")
    df = pd.read_csv(path)
    y = df["group"].astype("category").cat.codes.to_numpy()
    X = df.drop(columns=["group"]).to_numpy(dtype=float)
    return BenchmarkDataset(
        "diabetes", X, labels=y, axes=(CLUSTERING, EMBEDDING),
        preprocessing="standardize",
        metadata={"n": int(X.shape[0]), "n_classes": int(len(set(y))), "source": str(path)},
    )


@register("mice_protein")
def _ds_mice_protein(seed: int = 0) -> BenchmarkDataset:
    """Mice protein expression (UCI); 8 classes (genotype x treatment x behavior)."""
    import pandas as pd
    path = DATA_DIR / "mice_protein_no_NaN.csv"
    if not path.exists():
        raise FileNotFoundError(f"mice_protein_no_NaN.csv not found at {path}")
    df = pd.read_csv(path)
    y = df["class"].astype("category").cat.codes.to_numpy()
    X = df.select_dtypes("number").to_numpy(dtype=float)
    return BenchmarkDataset(
        "mice_protein", X, labels=y, axes=(CLUSTERING, EMBEDDING),
        preprocessing="standardize",
        metadata={"n": int(X.shape[0]), "n_classes": int(len(set(y))), "source": str(path)},
    )


@register("dynamical_system")
def _ds_dynamical(seed: int = 0) -> BenchmarkDataset:
    """Lederman-Talmon dynamical-system video embedding; topologically a 2-torus."""
    path = DATA_DIR / "dynamical_system_video.npy"
    if not path.exists():
        raise FileNotFoundError(f"dynamical_system_video.npy not found at {path}")
    X = np.load(path).astype(float)
    return BenchmarkDataset(
        "dynamical_system", X, target_betti=[1, 2, 1],
        axes=(TOPOLOGY, EMBEDDING), metadata={"n": int(X.shape[0]), "source": str(path)},
    )


@register("octopus")
def _ds_octopus(seed: int = 0) -> BenchmarkDataset:
    """Octopus surface mesh sample (visualization; large-scale structure)."""
    path = DATA_DIR / "octopus.txt"
    if not path.exists():
        raise FileNotFoundError(f"octopus.txt not found at {path}")
    X = np.loadtxt(path)
    return BenchmarkDataset(
        "octopus", X, axes=(EMBEDDING,), metadata={"n": int(X.shape[0]), "source": str(path)},
    )


@register("human")
def _ds_human(seed: int = 0) -> BenchmarkDataset:
    """Human surface mesh sample; topologically a 2-sphere."""
    path = DATA_DIR / "human.txt"
    if not path.exists():
        raise FileNotFoundError(f"human.txt not found at {path}")
    X = np.loadtxt(path)
    return BenchmarkDataset(
        "human", X, target_betti=[1, 0, 1], axes=(TOPOLOGY, EMBEDDING),
        metadata={"n": int(X.shape[0]), "source": str(path)},
    )


# ---- downloaded image benchmarks (pulled by download_image_datasets.py) ---- #

def _load_image(name: str, n: int | None, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Load a downloaded image dataset's flattened pixels + labels (subsampled).

    ``n`` is the subsample size (``None`` = the full dataset; the loaders default
    to a suite-friendly subsample but the full data is on disk, so pass ``n=None``
    or a larger ``n`` to use more). Missing files raise a clear error.
    """
    xp = DATA_DIR / f"{name}_X.npy"
    yp = DATA_DIR / f"{name}_y.npy"
    if not xp.exists():
        raise FileNotFoundError(
            f"{name}_X.npy not found at {xp}; run "
            "`python -m benchmarks.download_image_datasets` (needs network)."
        )
    X = np.load(xp).astype(float)
    y = np.load(yp).astype(int)
    if n is not None and n < len(X):
        idx = np.random.default_rng(seed).choice(len(X), size=n, replace=False)
        X, y = X[idx], y[idx]
    return X, y


@register("mnist")
def _ds_mnist(seed: int = 0, n: int = 3000) -> BenchmarkDataset:
    """MNIST handwritten digits (28x28 = 784-dim pixels), 10 classes.

    Subsampled to ``n`` for the suite; the full 60k is on disk (pass ``n=None``).
    Pixels are common-scale [0,1], so left raw.
    """
    X, y = _load_image("mnist", n, seed)
    return BenchmarkDataset(
        "mnist", X, labels=y, axes=(CLUSTERING, EMBEDDING),
        metadata={"n": int(X.shape[0]), "n_classes": 10, "subsample": n},
    )


@register("fashion_mnist")
def _ds_fashion_mnist(seed: int = 0, n: int = 3000) -> BenchmarkDataset:
    """Fashion-MNIST (28x28 = 784-dim), 10 clothing classes. Subsampled to ``n``."""
    X, y = _load_image("fashion_mnist", n, seed)
    return BenchmarkDataset(
        "fashion_mnist", X, labels=y, axes=(CLUSTERING, EMBEDDING),
        metadata={"n": int(X.shape[0]), "n_classes": 10, "subsample": n},
    )


@register("cifar10")
def _ds_cifar10(seed: int = 0, n: int = 3000) -> BenchmarkDataset:
    """CIFAR-10 (32x32x3 = 3072-dim raw pixels), 10 classes. Subsampled to ``n``.

    Raw pixels are a hard, high-dimensional clustering/embedding stress (no
    learned features); standardized per the policy is not applied (common-scale).
    """
    X, y = _load_image("cifar10", n, seed)
    return BenchmarkDataset(
        "cifar10", X, labels=y, axes=(CLUSTERING, EMBEDDING),
        metadata={"n": int(X.shape[0]), "n_classes": 10, "subsample": n},
    )


# ---- single-cell datasets ------------------------------------------------- #

@register("celegans")
def _ds_celegans(seed: int = 0, n: int = 3000) -> BenchmarkDataset:
    """C. elegans embryo single-cell (Packer et al. 2019), 50 PCs, cell-type labels.

    The collaborator's recipe: take the 50 principal components and normalize each
    cell to unit 2-norm (the ``unit_norm`` policy). Labels are the annotated cell
    types (cells with no annotation are dropped). Subsampled to ``n``.
    """
    import pandas as pd
    base = DATA_DIR / "celegans_embryo_data"
    xp = base / "CESub_Xa.csv"
    if not xp.exists():
        raise FileNotFoundError(f"celegans data not found at {base}")
    df = pd.read_csv(xp, index_col=0)
    pcs = [f"PC{i+1}" for i in range(50)]
    X = df[pcs].to_numpy(dtype=float)
    ct = pd.read_csv(base / "celltype.csv", index_col=0).iloc[:, 0]
    keep = ct.notna().to_numpy()  # drop unannotated cells (NA cell type)
    labels_raw = ct[keep].astype(str).to_numpy()
    X = X[keep]
    _, y = np.unique(labels_raw, return_inverse=True)
    if n is not None and n < len(X):
        idx = np.random.default_rng(seed).choice(len(X), size=n, replace=False)
        X, y = X[idx], y[idx]
    return BenchmarkDataset(
        "celegans", X, labels=y.astype(int), axes=(CLUSTERING, EMBEDDING),
        preprocessing="unit_norm",
        metadata={"n": int(X.shape[0]), "n_classes": int(len(set(y))), "source": str(xp)},
    )


@register("seurat")
def _ds_seurat(seed: int = 0, n: int = 3000) -> BenchmarkDataset:
    """Seurat-normalized single-cell expression (no ground-truth labels).

    A wide single-cell matrix (cells x normalized features); used on the embedding
    axis (and clustering by intrinsic structure). Subsampled to ``n``; unit-2-norm
    per the single-cell policy.
    """
    import pandas as pd
    path = DATA_DIR / "seurat_normalized.csv"
    if not path.exists():
        raise FileNotFoundError(f"seurat_normalized.csv not found at {path}")
    df = pd.read_csv(path, index_col=0)
    X = df.to_numpy(dtype=float)
    if n is not None and n < len(X):
        idx = np.random.default_rng(seed).choice(len(X), size=n, replace=False)
        X = X[idx]
    return BenchmarkDataset(
        "seurat", X, axes=(EMBEDDING,), preprocessing="unit_norm",
        metadata={"n": int(X.shape[0]), "n_features": int(X.shape[1]), "source": str(path)},
    )


def _preprocess(X: np.ndarray, kind: str) -> np.ndarray:
    """Apply the declared feature preprocessing (the harness-wide policy).

    Applied centrally in :func:`load` so every method and every metric sees the
    same feature space (preprocessing affects all methods equally):

    - ``"none"``        raw features (synthetic manifolds, meshes, embeddings,
      and common-scale sets like the digit pixels).
    - ``"standardize"`` zero-mean unit-variance per feature, for heterogeneous-
      scale tabular sets (wine, mice protein, diabetes, iris); zero-variance
      features are left untouched by ``StandardScaler``.
    - ``"unit_norm"``   project each point onto the unit 2-sphere (the single-cell
      recipe), guarding zero-norm rows.
    """
    if kind == "none":
        return X
    if kind == "standardize":
        from sklearn.preprocessing import StandardScaler

        return StandardScaler().fit_transform(X)
    if kind == "whiten":
        # PCA-sphering: center, rotate to the covariance eigenbasis, and scale each
        # component to unit variance. Removes global *linear* anisotropy (unlike
        # per-feature standardize, it is rotation-invariant), which fully restores
        # recovery on anisotropically-scaled embeddings (see anisotropic_torus).
        Xc = X - X.mean(axis=0)
        cov = np.cov(Xc, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        scale = 1.0 / np.sqrt(np.maximum(eigvals, 1e-9))
        return Xc @ (eigvecs * scale)
    if kind == "unit_norm":
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return X / norms
    raise ValueError(f"unknown preprocessing {kind!r}")


def load(name: str, seed: int = 0, preprocess: bool = True, **kwargs) -> BenchmarkDataset:
    """Build a registered dataset by name, applying its declared preprocessing.

    Preprocessing (``BenchmarkDataset.preprocessing``) is applied here, the single
    point every consumer (runner, sweeps, ad hoc) goes through, so it is applied
    identically to every method and metric. Pass ``preprocess=False`` for the raw
    features. The effective policy is recorded in ``metadata["preprocessing"]``.
    """
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; available: {sorted(DATASETS)}")
    ds = DATASETS[name](seed=seed, **kwargs)
    if preprocess and ds.preprocessing != "none":
        ds.X = _preprocess(ds.X, ds.preprocessing)
        ds.metadata["preprocessing"] = ds.preprocessing
    else:
        ds.metadata["preprocessing"] = "none"
    return ds


def available(axis: str | None = None) -> list[str]:
    """Registered dataset names, optionally filtered to those supporting ``axis``.

    Real datasets whose backing file is missing are silently skipped when an
    ``axis`` filter is given (so a default suite only lists runnable datasets).
    """
    names = sorted(DATASETS)
    if axis is None:
        return names
    out = []
    for nm in names:
        try:
            ds = load(nm)
        except Exception:
            continue
        if axis in ds.axes:
            out.append(nm)
    return out
