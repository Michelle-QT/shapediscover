"""Characterization tests for the ``ShapeDiscoverLite.fit_transform`` API."""

import numpy as np
import pytest

import synthetic_data as sd
from shapediscover import ShapeDiscoverLite


@pytest.fixture(scope="module")
def points():
    return sd.sphere(300, 2)


def test_fit_transform_shape(points):
    n_cover = 8
    cover = ShapeDiscoverLite(n_cover=n_cover).fit_transform(points)
    assert isinstance(cover, np.ndarray)
    assert np.issubdtype(cover.dtype, np.floating)
    # current convention: one row per cover element, one column per data point
    assert cover.shape == (n_cover, points.shape[0])


def test_fit_transform_is_a_valid_fuzzy_cover(points):
    cover = ShapeDiscoverLite(n_cover=8).fit_transform(points)
    assert np.all(np.isfinite(cover))
    assert np.all(cover >= 0.0)
    # the cover functions are normalized so each one has maximum value 1 (p=inf)
    assert np.allclose(cover.max(axis=1), 1.0)


def test_fit_transform_is_deterministic(points):
    first = ShapeDiscoverLite(n_cover=8).fit_transform(points)
    second = ShapeDiscoverLite(n_cover=8).fit_transform(points)
    assert np.array_equal(first, second)


def test_fit_transform_without_optimization(points):
    # the optimization=False path (initialization only) still returns a cover
    cover = ShapeDiscoverLite(n_cover=8, optimization=False).fit_transform(points)
    assert cover.shape == (8, points.shape[0])
    assert np.all(np.isfinite(cover))
