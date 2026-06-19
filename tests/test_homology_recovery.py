"""Behavior tests: ShapeDiscover recovers the homology of standard shapes.

These mirror the topological-inference experiments in the paper: run the
recommended ``ShapeDiscoverLite`` + ``FuzzyCoverPersistence`` pipeline and check
that the recovered nerve has the right Betti numbers over a non-trivial range of
the filtration (the ``homology_recovery_quotient`` metric).

The torus is intentionally excluded: with the current interface it does not
recover ``[1, 2, 1]`` reliably (see notes/TODO.md, datasets/benchmarking). The
2-sphere (H2) and the circle (H1) recover robustly, matching the paper's results.
"""

from helpers import diagram_to_intervals, homology_recovery_quotient

import synthetic_data as sd
from shapediscover import FuzzyCoverPersistence, ShapeDiscoverLite


def _recovery_quotient(X, n_cover, target_betti):
    max_dimension = len(target_betti) - 1
    cover = ShapeDiscoverLite(n_cover=n_cover, random_state=0).fit_transform(X)
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
