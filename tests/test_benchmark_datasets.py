"""Guarded smoke tests for the higher-dimensional benchmark manifolds.

The benchmark dataset registry is development-only (not in the shipped wheel), so
this adds the repo root to ``sys.path`` and skips if it cannot import. It pins
the new ground-truth manifolds: that they build, have the declared ambient
dimension and target Betti, and that the isometric ambient embedding preserves
pairwise distances (so its homology is unchanged).
"""

import os
import sys

import numpy as np
import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

datasets = pytest.importorskip("benchmarks.datasets")

# name -> (expected target Betti, expected ambient dimension)
EXPECTED = {
    "clifford_torus": ([1, 2, 1], 4),
    "torus3": ([1, 3, 3, 1], 6),
    "s2_times_s1": ([1, 1, 1, 1], 5),
    "sphere4": ([1, 0, 0, 0, 1], 5),
    "clifford_torus_amb50": ([1, 2, 1], 50),
    "cp2": ([1, 0, 1, 0, 1], 9),
    "figure_eight": ([1, 2], 2),
    "linked_circles": ([2, 2], 3),
}


@pytest.mark.parametrize("name", list(EXPECTED))
def test_manifold_shape_and_betti(name):
    betti, ambient = EXPECTED[name]
    ds = datasets.load(name, seed=0)
    assert ds.target_betti == betti
    assert ds.X.shape[1] == ambient
    assert ds.X.shape[0] == ds.n_points
    assert np.isfinite(ds.X).all()


def test_anisotropic_torus_and_whiten():
    # raw is anisotropic (unequal cycle scales); whitening makes it isotropic.
    raw = datasets.load("anisotropic_torus", seed=0, preprocess=False)
    assert raw.target_betti == [1, 2, 1]
    assert raw.X.shape[1] == 4
    # the two cycle "planes" have very different scales in the raw embedding
    raw_std = raw.X.std(axis=0)
    assert raw_std[:2].mean() > 2 * raw_std[2:].mean()
    # whitened (the declared preprocessing): all components ~unit variance
    white = datasets.load("anisotropic_torus", seed=0)
    assert white.metadata["preprocessing"] == "whiten"
    assert np.allclose(white.X.std(axis=0), white.X.std(axis=0)[0], rtol=0.1)


def test_noise_negative_control():
    ds = datasets.load("noise", seed=0)
    assert ds.target_betti == [1, 0, 0]
    assert ds.metadata["negative_control"] is True
    assert ds.X.shape == (1000, 10)
    assert np.isfinite(ds.X).all()


def test_torus_curvature_density_knobs():
    # the r1/r2/sampling knobs (for the curvature/density diagnosis) must be
    # reproducible from load() and leave the default torus unchanged.
    default = datasets.load("torus", seed=0)
    assert default.metadata["sampling"] == "angle"
    assert default.metadata["r2"] == 0.5

    fat = datasets.load("torus", seed=0, r2=0.65)
    assert fat.metadata["r2"] == 0.65
    assert fat.X.shape == default.X.shape

    area = datasets.load("torus", seed=0, sampling="area")
    assert area.metadata["sampling"] == "area"
    assert area.X.shape[0] == default.X.shape[0]
    # area-uniform concentrates fewer points on the inner rim than angle-uniform;
    # the two samples must differ (not a no-op relabel).
    assert not np.allclose(np.sort(area.X[:, 2]), np.sort(default.X[:, 2]))


def test_ambient_embedding_is_isometric():
    # the R^50 clifford torus must have the same pairwise distances as the R^4 one
    # (a random rotation of a zero-padding), so its homology is unchanged.
    from scipy.spatial.distance import pdist

    base = datasets.load("clifford_torus", seed=0).X
    amb = datasets.load("clifford_torus_amb50", seed=0).X
    # same seed -> same underlying angles -> distances must match
    assert np.allclose(np.sort(pdist(base)), np.sort(pdist(amb)), atol=1e-6)
