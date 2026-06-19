"""Tests for the scikit-learn estimator API.

``ShapeDiscover`` and ``ShapeDiscoverLite`` are scikit-learn transformers
(``BaseEstimator`` + ``TransformerMixin``): they expose ``get_params`` /
``set_params``, are ``clone``-able, validate in ``fit`` (not ``__init__``),
``fit`` returns ``self``, and ``transform`` / ``fit_transform`` return the cover
in the ``(n_points, n_cover)`` orientation.
"""

import numpy as np
import pytest
from sklearn.base import clone

import synthetic_data as sd
from shapediscover import FuzzyCoverPersistence, ShapeDiscover, ShapeDiscoverLite


@pytest.fixture(scope="module")
def points():
    return sd.sphere(150, 2)


def test_get_set_params_roundtrip():
    est = ShapeDiscoverLite(n_cover=7, regularization=20)
    params = est.get_params()
    assert params["n_cover"] == 7
    assert params["regularization"] == 20
    est.set_params(n_cover=9)
    assert est.get_params()["n_cover"] == 9


def test_clone_preserves_params():
    est = ShapeDiscover(
        n_cover=6, knn=10, n_max_iter=5, verbose=False, plot_loss_curve=False
    )
    cloned = clone(est)
    assert cloned is not est
    assert cloned.get_params() == est.get_params()


def test_fuzzy_cover_persistence_get_params():
    fp = FuzzyCoverPersistence(max_dimension=2, log_rescaling=True)
    params = fp.get_params()
    assert params["max_dimension"] == 2
    assert params["log_rescaling"] is True


def test_fit_returns_self(points):
    est = ShapeDiscoverLite(n_cover=6, n_max_iter=5)
    assert est.fit(points) is est


def test_transform_matches_fit_transform(points):
    est = ShapeDiscoverLite(n_cover=6, n_max_iter=5)
    cover_ft = est.fit_transform(points)
    cover_t = est.transform(points)  # X is ignored; the cover is tied to fit's data
    assert cover_ft.shape == (points.shape[0], 6)
    np.testing.assert_array_equal(cover_ft, cover_t)


def test_shapediscover_has_fit_transform(points):
    # ShapeDiscover gets fit_transform for free from TransformerMixin
    cover = ShapeDiscover(
        n_cover=6, n_max_iter=5, verbose=False, plot_loss_curve=False
    ).fit_transform(points)
    assert cover.shape == (points.shape[0], 6)


def test_transform_before_fit_raises():
    with pytest.raises(Exception):
        ShapeDiscoverLite(n_cover=6).transform(np.zeros((10, 2)))


def test_validation_is_deferred_to_fit(points):
    # invalid arguments must not raise at construction (scikit-learn convention)
    est = ShapeDiscover(
        initialization_algorithm="bogus", n_max_iter=5, verbose=False,
        plot_loss_curve=False,
    )
    # ... they raise when fitting
    with pytest.raises(ValueError):
        est.fit(points)

    lite = ShapeDiscoverLite(regularization=-1.0)
    with pytest.raises(ValueError):
        lite.fit(points)


def test_set_params_flows_into_fit(points):
    est = ShapeDiscoverLite(n_max_iter=5).set_params(n_cover=5)
    cover = est.fit_transform(points)
    assert cover.shape == (points.shape[0], 5)
