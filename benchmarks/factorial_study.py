"""Factorial parameter study for homology recovery.

The torus diagnosis (see the project notes) reproduced the paper's recovery but
left the cause confounded: going from the failing config to the working one
changed ``n_cover`` (15 -> 52) *and* the ``ShapeDiscoverLite`` -> full switch at
once, and the latter bundles three things together (the loss weights, plus
``n_max_iter`` 500 vs 250 and ``early_stop_tolerance`` 1e-5 vs 1e-4). ``knn`` was
never varied.

This module runs a full factorial over those factors *independently*, on the
full ``ShapeDiscover`` estimator, so main effects and interactions can be read
off directly. The topology axis is the response (recovery quotient and the
bar-dominance metric); ``fit`` and ``complex_size`` come along on the cost axis.

Run::

    python -m benchmarks.factorial_study --tag factorial_v0
    python -m benchmarks.factorial_study --quick      # tiny grid, smoke test

Output is a long-format CSV (one row per metric) augmented with one column per
swept factor, written to ``benchmarks/results/<tag>.csv``.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import pandas as pd

from .datasets import TOPOLOGY
from .methods import ShapeDiscoverMethod
from .runner import run_suite

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# The two tori are the confounded case; sphere2 (H2) and circle (H1) are
# controls that recover robustly, to check whether any conclusion generalizes.
DATASETS = ["torus", "dynamical_system", "sphere2", "circle"]

# Factor levels. loss_weights are [measure, geometry, topology, regularization].
GRID = {
    "n_cover": [15, 30, 52],
    "knn": [15, 30],
    "loss_weights": [[1, 10, 1, 10], [1, 0, 0, 10]],  # all losses vs measure+reg only
    "opt": [(250, 1e-4), (500, 1e-5)],                # (n_max_iter, early_stop_tolerance)
}

QUICK_GRID = {
    "n_cover": [15, 52],
    "knn": [15],
    "loss_weights": [[1, 10, 1, 10]],
    "opt": [(250, 1e-4)],
}

# Follow-up variants, each varying one (or two) factors against the v0
# "cheap good" backdrop (knn=15, measure+reg losses, the 250/1e-4 budget).
# Each is {"grid": <factor levels>, "datasets": <subset>}. Reproducible via
# ``--variant <name>``. Chosen adaptively from v0; see the project LOG.
TORI = ["torus", "dynamical_system"]
VARIANTS = {
    # does pushing n_cover past 52 help the tori, and at what cost?
    "ncover_high": {
        "grid": {"n_cover": [52, 65, 80, 100], "knn": [15],
                 "loss_weights": [[1, 0, 0, 10]], "opt": [(250, 1e-4)]},
        "datasets": TORI + ["sphere2"],
    },
    # isolate geometry vs topology individually (v0 only did all-vs-neither)
    "loss_decomp": {
        "grid": {"n_cover": [52], "knn": [15],
                 "loss_weights": [[1, 0, 0, 10], [1, 10, 0, 10],
                                  [1, 0, 1, 10], [1, 10, 1, 10]],
                 "opt": [(250, 1e-4)]},
        "datasets": TORI + ["sphere2"],
    },
    # knn sweet spot per dataset (knn mattered and interacted in v0)
    "knn_sweep": {
        "grid": {"n_cover": [52], "knn": [8, 15, 30, 50, 80],
                 "loss_weights": [[1, 0, 0, 10]], "opt": [(250, 1e-4)]},
        "datasets": TORI + ["sphere2", "circle"],
    },
    # finer optimization budget (more training hurt the paper torus in v0)
    "opt_budget": {
        "grid": {"n_cover": [52], "knn": [15], "loss_weights": [[1, 0, 0, 10]],
                 "opt": [(100, 1e-3), (250, 1e-4), (500, 1e-5), (1000, 1e-6)]},
        "datasets": TORI + ["sphere2", "circle"],
    },
    # regularization strength (the dominant weight in the measure+reg config)
    "reg_sweep": {
        "grid": {"n_cover": [52], "knn": [15],
                 "loss_weights": [[1, 0, 0, 1], [1, 0, 0, 3], [1, 0, 0, 10],
                                  [1, 0, 0, 30], [1, 0, 0, 100]],
                 "opt": [(250, 1e-4)]},
        "datasets": TORI + ["sphere2"],
    },
}


def run_study(grid: dict, datasets, seeds, verbose: bool = True) -> pd.DataFrame:
    """Run every combination in ``grid`` and return one augmented long table."""
    frames: list[pd.DataFrame] = []
    combos = list(itertools.product(*grid.values()))
    keys = list(grid.keys())
    for i, combo in enumerate(combos, 1):
        cfg = dict(zip(keys, combo))
        n_max_iter, early_stop_tolerance = cfg["opt"]
        base_params = dict(
            n_cover=cfg["n_cover"],
            knn=cfg["knn"],
            lite=False,
            extra=dict(
                loss_weights=cfg["loss_weights"],
                n_max_iter=n_max_iter,
                early_stop_tolerance=early_stop_tolerance,
            ),
        )
        if verbose:
            print(f"\n[config {i}/{len(combos)}] {cfg}")
        df = run_suite(
            dataset_names=datasets,
            method_class=ShapeDiscoverMethod,
            base_params=base_params,
            seeds=tuple(seeds),
            axes=(TOPOLOGY,),
            verbose=verbose,
        )
        if not len(df):
            continue
        # explicit factor columns for easy groupby (also present in `params` JSON)
        df["nc"] = cfg["n_cover"]
        df["knn_"] = cfg["knn"]
        df["loss_weights"] = json.dumps(cfg["loss_weights"])
        df["n_max_iter"] = n_max_iter
        df["early_stop_tol"] = early_stop_tolerance
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", default=None, help="output filename stem (default: the variant/grid name)")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--variant", choices=sorted(VARIANTS), help="a named follow-up variant")
    p.add_argument("--quick", action="store_true", help="tiny grid for a smoke test")
    args = p.parse_args(argv)

    if args.variant:
        grid = VARIANTS[args.variant]["grid"]
        datasets = args.datasets or VARIANTS[args.variant]["datasets"]
        tag = args.tag or f"factorial_{args.variant}"
    else:
        grid = QUICK_GRID if args.quick else GRID
        datasets = args.datasets or DATASETS
        tag = args.tag or ("factorial_quick" if args.quick else "factorial_v0")
    args.tag = tag
    df = run_study(grid, datasets, range(args.seeds))
    if not len(df):
        print("no results produced")
        return 1

    out_csv = RESULTS_DIR / f"{args.tag}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nwrote {len(df)} rows -> {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
