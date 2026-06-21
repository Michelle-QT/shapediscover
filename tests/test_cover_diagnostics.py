"""Guarded tests for the cover-diagnostics tooling (``benchmarks/cover_diagnostics``).

The diagnostics live in the development-only ``benchmarks`` package (not the
shipped wheel), so this module adds the repo root to ``sys.path`` and skips
cleanly if the benchmark dependencies (numba / gudhi / the benchmarks package)
are unavailable. The load-bearing test pins the reimplemented nerve persistence
to the maintained ``ShapeDiscover.fit_persistence``: the birth-filtration
recovery and complex size must match exactly, so the diagnostics measure the real
pipeline and not a drifting copy.
"""

import os
import sys

import numpy as np
import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

pytest.importorskip("numba")
pytest.importorskip("gudhi")
cd = pytest.importorskip("benchmarks.cover_diagnostics")

import synthetic_data as sd  # noqa: E402
from shapediscover import ShapeDiscover  # noqa: E402

TARGET_BETTI = [1, 0, 1]  # 2-sphere
MAX_DIM = 2


@pytest.fixture(scope="module")
def fitted():
    X = sd.sphere(250, 2)
    est = ShapeDiscover(
        n_cover=10, n_max_iter=40, verbose=False, plot_loss_curve=False,
        random_state=0,
    ).fit(X)
    cover = np.asarray(est.cover_).T  # internal (n_cover, n_points)
    return X, est, cover


def test_filtration_recovery_matches_fit_persistence(fitted):
    # The load-bearing pin: the reimplemented birth-filtration nerve persistence
    # must reproduce the maintained fit_persistence exactly (recovery + size).
    from benchmarks.metrics import homology_recovery_quotient

    _, est, cover = fitted
    est.fit_persistence(MAX_DIM, verbose=False)
    lib_intervals = [np.asarray(pd).reshape(-1, 2) for pd in est.persistence_diagram_]
    lib_recovery = homology_recovery_quotient(lib_intervals, TARGET_BETTI)
    lib_size = est.simplex_tree_.num_simplices()

    rec = cd.filtration_recovery(cover, TARGET_BETTI, filtration="birth",
                                 max_dimension=MAX_DIM)
    assert rec["complex_size"] == lib_size
    assert rec["recovery_quotient"] == pytest.approx(lib_recovery, abs=1e-9)


def test_simplex_table_volume_is_monotone(fitted):
    # Validity of the prune-by-volume sweep: a coface's mass never exceeds any of
    # its faces' masses, so a single global mass cut keeps a valid complex.
    _, _, cover = fitted
    edges = cd.simplex_table(cover, 1)
    triangles = cd.simplex_table(cover, 2)
    edge_mass = {tuple(s): m for s, m in zip(edges["simplices"], edges["mass"])}
    # every triangle's mass <= the mass of each of its three edges
    checked = 0
    for tri, m in zip(triangles["simplices"][:200], triangles["mass"][:200]):
        a, b, c = tri
        for e in [(a, b), (a, c), (b, c)]:
            if e in edge_mass:
                assert m <= edge_mass[e] + 1e-9
                checked += 1
    assert checked > 0


def test_surviving_structure_shape(fitted):
    _, _, cover = fitted
    struct = cd.surviving_structure(cover, max_dimension=MAX_DIM)
    assert 1 <= struct["n_active"] <= struct["n_cover"]
    assert len(struct["per_dimension"]) == MAX_DIM + 1
    # softmax cover has full support: every intersection is nonempty
    for pd in struct["per_dimension"]:
        assert pd["fill_ratio"] == pytest.approx(1.0)


def test_betti_trajectory_window(fitted):
    _, _, cover = fitted
    rec = cd.filtration_recovery(cover, TARGET_BETTI, filtration="birth",
                                 max_dimension=MAX_DIM)
    traj = rec["trajectory"]
    assert traj["betti"].shape[1] == len(TARGET_BETTI)
    # recovery_quotient is the fraction of the grid that matches the target
    assert 0.0 <= traj["recovery_quotient"] <= 1.0
    assert traj["recovery_quotient"] == pytest.approx(rec["recovery_quotient"])


def test_prune_sweep_shrinks_complex(fitted):
    _, _, cover = fitted
    sw = cd.prune_sweep(cover, TARGET_BETTI, prune_key="mass",
                        quantiles=(0.0, 0.5, 0.9))
    sizes = [r["complex_size"] for r in sw["rows"]]
    # higher prune cut -> fewer simplices kept (monotone non-increasing)
    assert sizes[0] >= sizes[1] >= sizes[2]
    assert all(0.0 <= r["recovery_quotient"] <= 1.0 for r in sw["rows"])


def test_finalize_cover_is_pinf_normalized(fitted):
    _, _, cover = fitted
    finalized = cd.finalize_cover_internal(cover)
    # p=inf normalization: each point's max membership over cover elements is 1
    assert np.allclose(finalized.max(axis=0), 1.0)


def test_aggregate_evolution_classifies_shapes():
    # pure-function test (no fitting): one over-optimizing trajectory (interior
    # peak then large decline) and one monotone-improving one.
    def ev(recoveries, actives):
        return {"rows": [{"iteration": 10 * i, "recovery_quotient": r, "n_active": a,
                          "bar_dominance_min": 1.0, "best_slice_betti": [1],
                          "per_dimension": []}
                         for i, (r, a) in enumerate(zip(recoveries, actives))]}

    over = ev([0.0, 0.5, 0.6, 0.3, 0.1], [10, 10, 9, 8, 7])      # peaks at idx 2
    mono = ev([0.0, 0.2, 0.4, 0.5, 0.6], [10, 10, 10, 10, 10])   # peaks at the end
    agg = cd.aggregate_evolution([over, mono])

    assert agg["n_seeds"] == 2
    assert agg["n_over_optimizing"] == 1
    assert agg["per_seed"][0]["over_optimizes"] is True
    assert agg["per_seed"][0]["peak_iteration"] == 20
    assert agg["per_seed"][1]["over_optimizes"] is False
    # per-iteration median is computed over both seeds
    assert agg["per_iteration"][0]["recovery_median"] == pytest.approx(0.0)
    assert agg["per_iteration"][2]["recovery_median"] == pytest.approx(0.5)


def _row(it, rq, active, pr, strong_edges=0, strong_tri=0):
    return {"iteration": it, "recovery_quotient": rq, "n_active": active,
            "overlap_pr": pr,
            "per_dimension": [{"dimension": 0, "n_strong": 0},
                              {"dimension": 1, "n_strong": strong_edges},
                              {"dimension": 2, "n_strong": strong_tri}]}


def test_evolution_severity_definitions():
    # interior peak at idx 2 (recovery 0.6), then decline to 0.1 at convergence;
    # after the peak the cover sheds elements (10->7), sharpens (PR 5->3), and
    # gains strong triangles (8->14) -- the candidate mechanism deltas.
    rows = [_row(0, 0.0, 10, 8.0, 20, 4), _row(10, 0.5, 10, 6.0, 22, 6),
            _row(20, 0.6, 9, 5.0, 24, 8), _row(30, 0.3, 8, 4.0, 20, 12),
            _row(40, 0.1, 7, 3.0, 16, 14)]
    rec = cd.evolution_severity(rows)
    assert rec["peak_iteration"] == 20
    assert rec["peak_recovery"] == pytest.approx(0.6)
    assert rec["final_recovery"] == pytest.approx(0.1)
    assert rec["abs_severity"] == pytest.approx(0.5)
    assert rec["rel_severity"] == pytest.approx(0.5 / 0.6)
    assert rec["over_optimizes"] is True
    # peak -> final mechanism deltas
    assert rec["d_active"] == 7 - 9
    assert rec["d_pr"] == pytest.approx(3.0 - 5.0)
    assert rec["d_strong_tri"] == 14 - 8     # over-filling (positive)
    assert rec["d_strong_edge"] == 16 - 24

    # a monotone trajectory has zero severity and is not flagged
    mono = [_row(0, 0.0, 10, 8.0), _row(10, 0.3, 10, 7.0), _row(20, 0.6, 10, 6.0)]
    mrec = cd.evolution_severity(mono)
    assert mrec["abs_severity"] == pytest.approx(0.0)
    assert mrec["over_optimizes"] is False

    # never-recovers: peak below the floor -> rel_severity is nan (undefined)
    flat = [_row(0, 0.0, 10, 8.0), _row(10, 0.0, 10, 7.0)]
    assert np.isnan(cd.evolution_severity(flat)["rel_severity"])


def test_summarize_severity_drops_nan():
    recs = [cd.evolution_severity(r) for r in (
        # interior peak at idx 1 (0.6), declines to 0.2 -> over-optimizes
        [_row(0, 0.0, 10, 8.0), _row(10, 0.6, 9, 5.0), _row(20, 0.4, 8, 4.0),
         _row(30, 0.2, 7, 3.0)],
        [_row(0, 0.0, 10, 8.0), _row(10, 0.0, 10, 7.0)],  # nan rel_severity
    )]
    summ = cd.summarize_severity(recs)
    assert summ["n_seeds"] == 2
    assert summ["n_over_optimizing"] == 1
    # the nan rel_severity row is dropped from the median, leaving the one value
    assert summ["rel_severity_median"] == pytest.approx(0.4 / 0.6)
