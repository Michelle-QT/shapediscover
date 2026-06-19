"""End-to-end tests for the neural-network model paths.

``model="pointcloud_nn"`` and ``model="graph_nn"`` go through the pre-training
path (a network is trained to match the clustering initialization, then the main
optimization runs). The ``graph_nn`` path additionally needs ``torch_geometric``
(the ``nn`` extra) and previously crashed before PR5/PR6; these tests pin that it
now fits, produces a valid ``(n_points, n_cover)`` cover, supports persistence,
and is reproducible under a fixed ``random_state``.
"""

import numpy as np
import pytest

import synthetic_data as sd
from shapediscover import ShapeDiscover


def _assert_valid_cover(cover, n_points, n_cover):
    assert cover.shape == (n_points, n_cover)
    assert np.all(np.isfinite(cover))
    assert np.all(cover >= 0.0)
    # p=inf normalization: each point's max membership across cover elements is 1
    assert np.allclose(cover.max(axis=1), 1.0)


def test_pointcloud_nn_fits_and_is_reproducible():
    X = sd.sphere(150, 2)
    common = dict(
        n_cover=6, model="pointcloud_nn", n_max_iter=20,
        verbose=False, plot_loss_curve=False,
    )
    a = ShapeDiscover(random_state=0, **common).fit(X)
    b = ShapeDiscover(random_state=0, **common).fit(X)
    _assert_valid_cover(a.cover_, X.shape[0], 6)
    assert np.array_equal(a.cover_, b.cover_)


def test_graph_nn_fits_persists_and_is_reproducible():
    pytest.importorskip("torch_geometric")
    X = sd.sphere(150, 2)
    common = dict(
        n_cover=6, model="graph_nn", n_max_iter=20,
        verbose=False, plot_loss_curve=False,
    )
    a = ShapeDiscover(random_state=0, **common).fit(X)
    _assert_valid_cover(a.cover_, X.shape[0], 6)

    a.fit_persistence(max_dimension=2, verbose=False)
    assert len(a.persistence_diagram_) == 3  # one interval array per dimension 0..2

    b = ShapeDiscover(random_state=0, **common).fit(X)
    assert np.array_equal(a.cover_, b.cover_)


def test_graph_nn_random_init_runs():
    pytest.importorskip("torch_geometric")
    X = sd.sphere(150, 2)
    cover = ShapeDiscover(
        n_cover=6, model="graph_nn", initialization_algorithm="random",
        n_max_iter=20, verbose=False, plot_loss_curve=False, random_state=0,
    ).fit(X).cover_
    _assert_valid_cover(cover, X.shape[0], 6)
