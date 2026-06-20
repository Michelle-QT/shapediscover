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
}


@pytest.mark.parametrize("name", list(EXPECTED))
def test_manifold_shape_and_betti(name):
    betti, ambient = EXPECTED[name]
    ds = datasets.load(name, seed=0)
    assert ds.target_betti == betti
    assert ds.X.shape[1] == ambient
    assert ds.X.shape[0] == ds.n_points
    assert np.isfinite(ds.X).all()


def test_ambient_embedding_is_isometric():
    # the R^50 clifford torus must have the same pairwise distances as the R^4 one
    # (a random rotation of a zero-padding), so its homology is unchanged.
    from scipy.spatial.distance import pdist

    base = datasets.load("clifford_torus", seed=0).X
    amb = datasets.load("clifford_torus_amb50", seed=0).X
    # same seed -> same underlying angles -> distances must match
    assert np.allclose(np.sort(pdist(base)), np.sort(pdist(amb)), atol=1e-6)
