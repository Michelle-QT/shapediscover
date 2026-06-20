"""Behavior tests: ShapeDiscover recovers the homology of standard shapes.

These mirror the topological-inference experiments in the paper: run the
recommended ``ShapeDiscoverLite`` + ``FuzzyCoverPersistence`` pipeline and check
that the recovered nerve has the right Betti numbers over a non-trivial range of
the filtration (the ``homology_recovery_quotient`` metric).

The 2-sphere (H2) and the circle (H1) recover robustly with the recommended
``ShapeDiscoverLite``. The torus (H1=2, H2=1) also recovers, but only with the
full ``ShapeDiscover`` at adequate sampling density: it needs ~55+ points per
cover element (n=3000 at n_cover=52), and ``ShapeDiscoverLite`` fails on it
because its tighter early-stop tolerance (1e-5 vs 1e-4) over-optimizes the cover
(see the synthetic-torus resolution in the project notes).
"""

from helpers import diagram_to_intervals, homology_recovery_quotient

import synthetic_data as sd
from shapediscover import FuzzyCoverPersistence, ShapeDiscover, ShapeDiscoverLite


def _recovery_quotient(X, n_cover, target_betti, estimator=None):
    max_dimension = len(target_betti) - 1
    if estimator is None:
        estimator = ShapeDiscoverLite(n_cover=n_cover, random_state=0)
    cover = estimator.fit_transform(X)
    diagram = FuzzyCoverPersistence(
        max_dimension=max_dimension, log_rescaling=True
    ).fit_transform(cover)
    intervals = diagram_to_intervals(diagram, max_dimension)
    return homology_recovery_quotient(intervals, target_betti)


def test_circle_h1_recovery():
    # circle in the plane: b0 = 1, b1 = 1
    quotient = _recovery_quotient(sd.sphere(400, 1), n_cover=12, target_betti=[1, 1])
    assert quotient > 0.4  # measured ~0.87


def test_sphere_h2_recovery():
    # 2-sphere: b0 = 1, b1 = 0, b2 = 1
    quotient = _recovery_quotient(
        sd.sphere(500, 2), n_cover=12, target_betti=[1, 0, 1]
    )
    assert quotient > 0.2  # measured ~0.50


def test_torus_h1_h2_recovery():
    # torus: b0 = 1, b1 = 2, b2 = 1. Recovers with the full ShapeDiscover at
    # adequate density (n=3000 / n_cover=52, ~58 points per cover element);
    # ShapeDiscoverLite over-optimizes and fails (see the module docstring).
    quotient = _recovery_quotient(
        sd.torus(3000, seed=0),
        n_cover=52,
        target_betti=[1, 2, 1],
        estimator=ShapeDiscover(
            n_cover=52, knn=15, random_state=0, verbose=False, plot_loss_curve=False
        ),
    )
    assert quotient > 0.2  # measured ~0.55
