"""Guarded smoke tests for the wired real / image datasets.

These load from local data files that are not committed (large / external), so
each test skips cleanly when its file is absent (e.g. in CI). When present, it
checks the loader returns a finite matrix of the expected width and, where
applicable, aligned labels.
"""

import os
import sys

import numpy as np
import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

datasets = pytest.importorskip("benchmarks.datasets")

# name -> expected feature width (None = don't check)
REAL = {
    "mnist": 784,
    "fashion_mnist": 784,
    "cifar10": 3072,
    "celegans": 50,
    "seurat": None,
}


@pytest.mark.parametrize("name", list(REAL))
def test_real_loader(name):
    try:
        ds = datasets.load(name, seed=0, n=200)
    except FileNotFoundError:
        pytest.skip(f"{name} data file not present")
    width = REAL[name]
    if width is not None:
        assert ds.X.shape[1] == width
    assert ds.X.shape[0] <= 200
    assert np.isfinite(ds.X).all()
    if ds.labels is not None:
        assert len(ds.labels) == ds.X.shape[0]
