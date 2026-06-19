"""Targeted scikit-learn conformance tests.

Full ``check_estimator`` compliance is intentionally not pursued (see
notes/LOG.md, PR9): ShapeDiscover is transductive (``transform`` returns the
cover learned for the *training* points; the default ``set_function`` model has
no out-of-sample map) and needs ``n_points > knn``, so sklearn's tiny-data and
row-invariance checks do not apply. These tests pin the conformance properties
that *are* meaningful and that PR9 fixed: a clean ``__init__`` (no fitted
attributes), ``NotFittedError`` before fit, and the ``non_deterministic`` tag.
"""

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

import synthetic_data as sd
from shapediscover import ShapeDiscover, ShapeDiscoverLite


@pytest.mark.parametrize("cls", [ShapeDiscover, ShapeDiscoverLite])
def test_init_sets_no_fitted_attributes(cls):
    # scikit-learn: __init__ stores only constructor args; no trailing-underscore
    # (fitted) attributes should exist before fit
    est = cls()
    assert not any(k.endswith("_") and not k.endswith("__") for k in vars(est))


@pytest.mark.parametrize("cls", [ShapeDiscover, ShapeDiscoverLite])
def test_transform_before_fit_raises_not_fitted(cls):
    with pytest.raises(NotFittedError):
        cls().transform(np.zeros((20, 2)))


def test_check_is_fitted_after_fit():
    X = sd.sphere(150, 2)
    est = ShapeDiscoverLite(n_cover=6, n_max_iter=10, random_state=0).fit(X)
    # does not raise once fitted
    check_is_fitted(est)


@pytest.mark.parametrize("cls", [ShapeDiscover, ShapeDiscoverLite])
def test_non_deterministic_tag(cls):
    assert cls().__sklearn_tags__().non_deterministic is True
