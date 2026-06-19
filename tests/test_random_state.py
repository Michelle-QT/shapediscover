"""Reproducibility tests for the random_state parameter.

A fixed ``random_state`` makes the whole pipeline reproducible, and fitting must
not mutate the global numpy random stream (the loss function and clustering use
an instance-local RNG rather than ``np.random.seed`` / the global singleton).
"""

import numpy as np

import synthetic_data as sd
from shapediscover import ShapeDiscover, ShapeDiscoverLite


def test_same_random_state_is_reproducible():
    X = sd.sphere(150, 2)
    a = ShapeDiscoverLite(n_cover=6, n_max_iter=10, random_state=42).fit_transform(X)
    b = ShapeDiscoverLite(n_cover=6, n_max_iter=10, random_state=42).fit_transform(X)
    assert np.array_equal(a, b)


def test_shapediscover_same_random_state_is_reproducible():
    X = sd.sphere(150, 2)
    common = dict(n_cover=6, n_max_iter=10, verbose=False, plot_loss_curve=False)
    a = ShapeDiscover(random_state=7, **common).fit(X).cover_
    b = ShapeDiscover(random_state=7, **common).fit(X).cover_
    assert np.array_equal(a, b)


def test_fit_does_not_touch_global_numpy_state():
    X = sd.sphere(150, 2)
    np.random.seed(123)
    before = np.random.get_state()[1].copy()
    ShapeDiscoverLite(n_cover=6, n_max_iter=10, random_state=0).fit_transform(X)
    after = np.random.get_state()[1]
    np.testing.assert_array_equal(before, after)
