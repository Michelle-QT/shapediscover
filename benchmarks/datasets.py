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
    metadata : dict
        Free-form provenance / parameters (kept in the results table).
    """

    name: str
    X: np.ndarray
    labels: np.ndarray | None = None
    target_betti: list[int] | None = None
    axes: tuple[str, ...] = ()
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
           noise: float = 0.0) -> np.ndarray:
    """``n`` points on a torus in R^3 (uniform in the two angles)."""
    t1 = rng.random(n) * 2 * np.pi
    t2 = rng.random(n) * 2 * np.pi
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
def _ds_torus(seed: int = 0, n: int = 3000, noise: float = 0.0) -> BenchmarkDataset:
    # n=3000 (was 1500): the torus needs ~55+ points per cover element to
    # resolve [1,2,1]; at n_cover=52 the old 1500 (~29 pts/elt) was too sparse
    # to recover (see the synthetic-torus resolution in the project notes).
    rng = np.random.default_rng(seed)
    return BenchmarkDataset(
        "torus", _torus(n, rng, noise=noise), target_betti=[1, 2, 1],
        axes=(TOPOLOGY, EMBEDDING), metadata={"n": n, "noise": noise},
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
    return BenchmarkDataset(
        "iris", d.data.astype(float), labels=d.target.astype(int),
        axes=(CLUSTERING, EMBEDDING), metadata={"n": int(d.data.shape[0]), "n_classes": 3},
    )


@register("wine")
def _ds_wine(seed: int = 0) -> BenchmarkDataset:
    # NB: raw features (very different scales); see the standardization-policy TODO.
    from sklearn.datasets import load_wine
    d = load_wine()
    return BenchmarkDataset(
        "wine", d.data.astype(float), labels=d.target.astype(int),
        axes=(CLUSTERING, EMBEDDING), metadata={"n": int(d.data.shape[0]), "n_classes": 3},
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


def load(name: str, seed: int = 0, **kwargs) -> BenchmarkDataset:
    """Build a registered dataset by name."""
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; available: {sorted(DATASETS)}")
    return DATASETS[name](seed=seed, **kwargs)


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
