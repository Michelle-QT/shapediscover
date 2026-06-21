"""Tests for the curvature-adaptive measure weighting (off-by-default knob).

Pins the design contracts: the weights have mean 1 (the measure scale is
preserved), ``strength=0`` is exactly uniform, and the proxy is a near-no-op on a
homogeneous manifold while it actually varies on a curved one (so the weighting
only acts where curvature varies). Also checks the estimator wiring: the default
``measure_weighting="uniform"`` is bit-identical to not setting it.
"""

import os
import sys

import numpy as np
import pytest

_EXAMPLES_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, "examples")
)
if _EXAMPLES_DIR not in sys.path:
    sys.path.insert(0, _EXAMPLES_DIR)

from shapediscover.weighted_graph import (  # noqa: E402
    curvature_measure_weights,
    graph_from_pointcloud,
)


def _flat_torus(n, seed):
    rng = np.random.default_rng(seed)
    a = rng.random(n) * 2 * np.pi
    b = rng.random(n) * 2 * np.pi
    s = 1 / np.sqrt(2)
    return np.stack([s * np.cos(a), s * np.sin(a), s * np.cos(b), s * np.sin(b)], 1)


def _donut(n, seed, r1=1.0, r2=0.5):
    rng = np.random.default_rng(seed)
    t1 = rng.random(n) * 2 * np.pi
    t2 = rng.random(n) * 2 * np.pi
    x = (r1 + r2 * np.cos(t2)) * np.cos(t1)
    y = (r1 + r2 * np.cos(t2)) * np.sin(t1)
    z = r2 * np.sin(t2)
    return np.stack([x, y, z], 1)


def _weights(X, **kw):
    g = graph_from_pointcloud(X, n_neighbors=15, algorithm="knn")
    return curvature_measure_weights(X, g.adjacency_matrix(), bandwidth=g.bandwidth(), **kw)


def test_weights_have_mean_one():
    w = _weights(_donut(1500, 0), strength=1.0)
    assert w.mean() == pytest.approx(1.0, abs=1e-6)
    assert (w > 0).all()


def test_strength_zero_is_uniform():
    w = _weights(_donut(1500, 0), strength=0.0)
    assert np.allclose(w, 1.0)


def test_near_noop_on_homogeneous_but_varies_on_curved():
    # the curvature proxy spread is much smaller on the flat (homogeneous) torus
    # than on the curved donut, so the weighting is ~a no-op on the former.
    flat_cv = _weights(_flat_torus(2000, 0), strength=1.0).std()
    donut_cv = _weights(_donut(2000, 0), strength=1.0).std()
    assert flat_cv < donut_cv
    assert flat_cv < 0.5  # tight spread on the homogeneous manifold


def test_inverse_is_reciprocal_shaped():
    X = _donut(1500, 0)
    hi = _weights(X, strength=1.0, invert=False)
    lo = _weights(X, strength=1.0, invert=True)
    # where the forward weight is large (curved), the inverse weight is small
    assert np.corrcoef(hi, lo)[0, 1] < 0
    assert lo.mean() == pytest.approx(1.0, abs=1e-6)


def test_estimator_uniform_default_bit_identical():
    pytest.importorskip("torch")
    import synthetic_data as sd  # noqa: E402
    from shapediscover import ShapeDiscover

    X = sd.sphere(300, 2)
    kw = dict(n_cover=12, knn=15, n_max_iter=20, random_state=0,
              verbose=False, plot_loss_curve=False)
    a = ShapeDiscover(**kw).fit_transform(X)
    b = ShapeDiscover(measure_weighting="uniform", **kw).fit_transform(X)
    assert np.array_equal(a, b)
