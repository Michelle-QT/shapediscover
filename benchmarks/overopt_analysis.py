"""Analyze the over-optimization characterization CSVs (factors + mechanism).

Reads the tidy per-seed severity tables written by ``diagnose.py
--over-opt-classify`` (the discrete master table) and ``--severity-sweep`` /
``--factor-sweep`` (the single-factor controlled sweeps), and answers the two
deliverable questions *empirically*, with sample sizes and caveats:

1. **Which factors predict over-optimization severity?** A per-manifold severity
   table, the spread of severity across the controlled sweeps (each isolates one
   factor while holding the rest fixed), and bivariate / grouped comparisons on
   the discrete master table. Small N is flagged loudly; no causation is asserted.

2. **Is the candidate cover-level mechanism real?** ``element death -> coarsening
   -> over-filling`` predicts that severity should rise with element death
   (``d_active`` more negative), strong-triangle over-filling (``d_strong_tri``
   more positive), and overlap sharpening (``d_pr`` more negative). We pool the
   per-seed rows and correlate severity with each delta, and check the known
   counterexample (sphere2 sheds more elements yet does not over-optimize)
   explicitly.

Run::

    python -m benchmarks.overopt_analysis
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def load_all(results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    """Concatenate every ``overopt_*.csv`` in ``results_dir`` into one frame."""
    frames = []
    for path in sorted(glob.glob(str(results_dir / "overopt_*.csv"))):
        df = pd.read_csv(path)
        df["source"] = Path(path).stem
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"no overopt_*.csv under {results_dir}")
    return pd.concat(frames, ignore_index=True)


def _corr(x, y):
    """Spearman + Pearson with shared finite mask; returns (rho, r, n)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3 or np.unique(x[m]).size < 2 or np.unique(y[m]).size < 2:
        return float("nan"), float("nan"), int(m.sum())
    rho = stats.spearmanr(x[m], y[m]).statistic
    r = stats.pearsonr(x[m], y[m]).statistic
    return float(rho), float(r), int(m.sum())


def master_table(df: pd.DataFrame, out=print) -> pd.DataFrame:
    """Per-manifold severity summary from the discrete master classification."""
    master = df[df["source"].str.contains("master")]
    if master.empty:
        out("(no master table found)")
        return master
    g = master.groupby("manifold")
    summ = g.agg(
        d=("intrinsic_dim", "first"),
        betti=("betti_sum", "first"),
        aniso=("aniso", "first"),
        curv=("curv", "first"),
        n_cover=("n_cover", "first"),
        n_pts=("n_points", "first"),
        peak=("peak_recovery", "median"),
        final=("final_recovery", "median"),
        abs_sev=("abs_severity", "median"),
        rel_sev=("rel_severity", "median"),
        n_overopt=("over_optimizes", "sum"),
        n_seeds=("seed", "count"),
    ).sort_values("abs_sev", ascending=False)
    out("\n=== MASTER: per-manifold over-optimization severity "
        "(median over seeds) ===")
    out(summ.to_string(float_format=lambda v: f"{v:.3f}"))
    return summ


def factor_analysis(summ: pd.DataFrame, df: pd.DataFrame, out=print):
    """Bivariate + grouped factor effects on the discrete master severity.

    With ~11 manifolds these are descriptive, not inferential: flagged as such.
    """
    out("\n=== FACTORS vs severity (discrete master; N = "
        f"{len(summ)} manifolds, descriptive only) ===")
    # only manifolds with a non-trivial peak carry interpretable severity
    rec = summ[summ["peak"] > 0.10]
    out(f"(restricting to the {len(rec)} manifolds with median peak recovery "
        "> 0.10; the rest never recover so severity is undefined)")
    for factor in ("d", "betti", "n_cover", "n_pts"):
        rho, r, n = _corr(rec[factor], rec["abs_sev"])
        out(f"  abs_severity vs {factor:8s}: Spearman {rho:+.2f}  "
            f"Pearson {r:+.2f}  (n={n})")
    out("  grouped medians (abs_severity):")
    for flag in ("aniso", "curv"):
        for val, sub in rec.groupby(flag):
            out(f"    {flag}={val}: {sub['abs_sev'].median():.3f} "
                f"(manifolds: {', '.join(sub.index)})")

    # pooled OLS on the per-seed master rows, all factors at once (small N caveat)
    master = df[df["source"].str.contains("master")]
    master = master[master["peak_recovery"] > 0.10]
    try:
        import statsmodels.api as sm  # noqa
        have_sm = True
    except Exception:
        have_sm = False
    if have_sm and len(master) > 10:
        import statsmodels.api as sm
        X = master[["intrinsic_dim", "betti_sum", "aniso", "curv"]].astype(float)
        X = sm.add_constant(X)
        model = sm.OLS(master["abs_severity"].astype(float), X).fit()
        out("\n  pooled per-seed OLS abs_severity ~ dim + betti + aniso + curv "
            f"(n={len(master)} seed-rows; standard errors ignore seed clustering):")
        out("    " + model.params.to_string().replace("\n", "\n    "))
        out(f"    R^2 = {model.rsquared:.3f}")
    else:
        # sklearn fallback: standardized linear coefficients
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import StandardScaler
        cols = ["intrinsic_dim", "betti_sum", "aniso", "curv"]
        Xc = master[cols].astype(float)
        keep = [c for c in cols if Xc[c].nunique() > 1]
        if keep and len(master) > 6:
            Xs = StandardScaler().fit_transform(master[keep].astype(float))
            reg = LinearRegression().fit(Xs, master["abs_severity"].astype(float))
            out(f"\n  standardized linear coefficients (n={len(master)} "
                "seed-rows, sklearn; descriptive):")
            for c, w in zip(keep, reg.coef_):
                out(f"    {c:14s}: {w:+.3f}")
            out(f"    R^2 = {reg.score(Xs, master['abs_severity'].astype(float)):.3f}")


def sweep_analysis(df: pd.DataFrame, out=print):
    """Severity vs the swept knob within each controlled single-factor sweep."""
    sweeps = df[df["source"].str.contains("sweep") | df["source"].str.contains("naxis")]
    if sweeps.empty:
        out("\n(no sweep CSVs found)")
        return
    out("\n=== CONTROLLED SWEEPS: severity vs the isolated knob ===")
    for src, sub in sweeps.groupby("source"):
        knob = sub["knob"].iloc[0] if "knob" in sub else "?"
        # knob_value may be numeric (ratio/r2/n) or categorical (sampling)
        try:
            kv = sub["knob_value"].astype(float)
            numeric = True
        except (ValueError, TypeError):
            numeric = False
        out(f"\n  {src}  (knob = {knob})")
        med = sub.groupby("knob_value").agg(
            peak=("peak_recovery", "median"),
            final=("final_recovery", "median"),
            abs_sev=("abs_severity", "median"),
            rel_sev=("rel_severity", "median"),
            n_overopt=("over_optimizes", "sum"),
            n=("seed", "count"),
        )
        out(med.to_string(float_format=lambda v: f"{v:.3f}"))
        if numeric:
            rho, r, n = _corr(kv, sub["abs_severity"])
            rho_p, _, _ = _corr(kv, sub["peak_recovery"])
            out(f"    Spearman(knob, abs_severity) = {rho:+.2f} (n={n}); "
                f"Spearman(knob, peak_recovery) = {rho_p:+.2f}")


def mechanism_analysis(df: pd.DataFrame, out=print):
    """Test the death -> coarsening -> over-filling story on the pooled rows.

    The mechanism predicts severity should rise with element death (d_active more
    negative -> positive correlation with -d_active), over-filling (d_strong_tri
    up), and sharpening (d_pr down). Restrict to rows with a non-trivial peak
    (mechanism deltas are meaningless when nothing recovered), then correlate.
    """
    out("\n=== MECHANISM: does death/coarsening/over-filling predict severity? ===")
    pool = df[df["peak_recovery"] > 0.10].copy()
    out(f"(pooled per-seed rows with peak recovery > 0.10: n={len(pool)} "
        f"of {len(df)} total)")
    for col, sign, story in [
        ("d_active", -1, "element death (d_active<0)"),
        ("d_strong_tri", +1, "over-filling (d_strong_tri>0)"),
        ("d_pr", -1, "overlap sharpening (d_pr<0)"),
    ]:
        rho, r, n = _corr(pool[col], pool["abs_severity"])
        direction = "supports" if (np.isfinite(rho) and rho * sign > 0.2) else (
            "WEAK/contradicts" if np.isfinite(rho) else "n/a")
        out(f"  abs_severity vs {col:13s}: Spearman {rho:+.2f} Pearson {r:+.2f} "
            f"(n={n})  [{story}: {direction}]")

    # explicit counterexample: a manifold that sheds elements yet does not over-opt
    master = df[df["source"].str.contains("master")]
    if not master.empty:
        out("\n  per-manifold element death vs over-optimization "
            "(the 'death is not sufficient' check):")
        g = master.groupby("manifold").agg(
            peak=("peak_recovery", "median"),
            d_active=("d_active", "median"),
            abs_sev=("abs_severity", "median"),
            n_overopt=("over_optimizes", "sum"),
            n=("seed", "count"))
        g = g[g["peak"] > 0.10].sort_values("d_active")
        out(g.to_string(float_format=lambda v: f"{v:.3f}"))


def main():
    df = load_all()
    print(f"loaded {len(df)} per-seed rows from "
          f"{df['source'].nunique()} CSV(s): {sorted(df['source'].unique())}")
    summ = master_table(df)
    if not summ.empty:
        factor_analysis(summ, df)
    sweep_analysis(df)
    mechanism_analysis(df)


if __name__ == "__main__":
    main()
