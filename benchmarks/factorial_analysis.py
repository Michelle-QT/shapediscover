"""Read a factorial-study CSV and print main effects, interactions, and cost.

Auto-adapts to whichever factors actually vary in the file, so it works for the
full v0 grid and for every single-factor follow-up variant
(see :mod:`benchmarks.factorial_study`). For the recovery quotient, the
bar-dominance metric, and the cost metrics it prints, per varying factor, the
level means by dataset (averaged over the other factors and seeds). When both
``n_cover`` and ``loss_weights`` vary it also prints their interaction (the
original torus confound).

Run::

    python -m benchmarks.factorial_analysis --tag factorial_v0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESPONSES = ["recovery_quotient", "bar_dominance_min"]
COST = ["fit_s", "persistence_s", "complex_size"]
# candidate factor columns (added by factorial_study); only the varying ones are used
FACTORS = ["nc", "knn_", "loss_weights", "n_max_iter", "early_stop_tol"]


def _load(tag: str) -> pd.DataFrame:
    df = pd.read_csv(RESULTS_DIR / f"{tag}.csv")
    sub = df[df["axis"].isin(["topology", "cost"])].copy()
    sub["value"] = pd.to_numeric(sub["value"], errors="coerce")
    return sub


def _means_by_factor(sub: pd.DataFrame, factor: str, fmt: int) -> None:
    tbl = sub.pivot_table(index=factor, columns="dataset", values="value", aggfunc="mean")
    tbl["ALL"] = sub.groupby(factor)["value"].mean()
    print(f"\n[{factor}]")
    print(tbl.round(fmt).to_string())


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", default="factorial_v0")
    args = p.parse_args(argv)

    data = _load(args.tag)
    pd.set_option("display.width", 160, "display.max_columns", 40)

    varying = [f for f in FACTORS if f in data.columns and data[f].nunique() > 1]
    n_cfg = data[[f for f in FACTORS if f in data.columns]].drop_duplicates().shape[0]
    print(f"loaded tag={args.tag!r}: {n_cfg} configs x {data['seed'].nunique()} seeds "
          f"x {data['dataset'].nunique()} datasets; varying factors: {varying or '(none)'}")

    for metric, fmt in [(m, 3) for m in RESPONSES] + \
                       [("fit_s", 2), ("persistence_s", 3), ("complex_size", 0)]:
        sub = data[data["metric"] == metric]
        if not len(sub):
            continue
        kind = "RESPONSE" if metric in RESPONSES else "COST"
        print(f"\n{'='*78}\n{kind}: {metric}\n{'='*78}")
        for factor in (varying or FACTORS):
            if factor in sub.columns:
                _means_by_factor(sub, factor, fmt)
        if "nc" in varying and "loss_weights" in varying:
            inter = sub.pivot_table(index="nc", columns="loss_weights",
                                    values="value", aggfunc="mean")
            print("\n-- interaction: n_cover x loss_weights --")
            print(inter.round(fmt).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
