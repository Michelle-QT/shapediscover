"""Generate a self-contained static results page from the committed result CSVs.

Builds a single ``index.html`` (inline CSS, no external assets) summarizing the
benchmark results, with every table computed directly from
``benchmarks/results/*.csv`` so no number is hand-transcribed. Intended for a
GitHub Pages deploy.

Run::

    python -m benchmarks.make_results_site --out site/index.html
"""

from __future__ import annotations

import argparse
import datetime as _dt
import html
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RESULTS = Path(__file__).resolve().parent / "results"


# --------------------------------------------------------------------------- #
# data helpers (exact, from CSV)
# --------------------------------------------------------------------------- #

def _load(name):
    return pd.read_csv(RESULTS / name)


def _med(df, col):
    v = df[col].to_numpy(float)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else float("nan")


def master_rows():
    df = _load("overopt_master_v0.csv")
    out = []
    for name, g in df.groupby("manifold"):
        out.append(dict(
            manifold=name, d=int(g["intrinsic_dim"].iloc[0]),
            n_cover=int(g["n_cover"].iloc[0]),
            peak=_med(g, "peak_recovery"), final=_med(g, "final_recovery"),
            rel=_med(g, "rel_severity"), abs=_med(g, "abs_severity"),
            n_over=int(g["over_optimizes"].sum()), n_seeds=int(g["seed"].nunique()),
        ))
    return sorted(out, key=lambda r: r["abs"], reverse=True)


def sweep_points(name, knob="knob_value"):
    """knob_value -> dict of medians + over-opt count, ordered by knob."""
    df = _load(name)
    rows = {}
    for kv, g in df.groupby(knob):
        rows[kv] = dict(rel=_med(g, "rel_severity"), absv=_med(g, "abs_severity"),
                        peak=_med(g, "peak_recovery"), final=_med(g, "final_recovery"),
                        n_over=int(g["over_optimizes"].sum()),
                        n=int(g["seed"].nunique()))
    return rows


def mechanism():
    import glob
    frames = [pd.read_csv(f) for f in glob.glob(str(RESULTS / "overopt_*.csv"))]
    df = pd.concat(frames, ignore_index=True)
    pool = df[df["peak_recovery"] > 0.10]
    out = {}
    for col in ("d_active", "d_pr", "d_strong_tri"):
        x = pool[col].to_numpy(float)
        y = pool["abs_severity"].to_numpy(float)
        m = np.isfinite(x) & np.isfinite(y)
        out[col] = (float(stats.spearmanr(x[m], y[m]).statistic), int(m.sum()))
    return out, int(len(pool))


def external_topo():
    df = _load("baseline_external_v0.csv")
    r = df[df.metric == "recovery_quotient"]
    piv = r.pivot_table(index="dataset", columns="method", values="value",
                        aggfunc="mean")
    order = ["circle", "two_circles", "sphere2", "torus"]
    return piv.loc[[d for d in order if d in piv.index],
                   ["shapediscover", "alpha", "rips"]]


def highdim_boundary():
    df = _load("baseline_track2_v0.csv")
    r = df[df.metric == "recovery_quotient"].groupby("dataset")["value"].mean()
    order = ["torus3", "s2_times_s1", "sphere4", "cp2", "noise"]
    return [(d, float(r[d])) for d in order if d in r.index]


# --------------------------------------------------------------------------- #
# html
# --------------------------------------------------------------------------- #

def _f(x, n=3):
    return "n/a" if (x is None or (isinstance(x, float) and np.isnan(x))) else f"{x:.{n}f}"


CSS = """
:root{--ink:#1b1f24;--mut:#5b6470;--line:#e2e6ea;--accent:#2c6e8f;
--drive:#b4452f;--confound:#7a828c;--remove:#2e7d52;--bg:#fbfcfd}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:16px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
-webkit-font-smoothing:antialiased}
main{max-width:880px;margin:0 auto;padding:42px 22px 80px}
h1{font-size:25px;margin:0 0 4px;letter-spacing:-.2px}
h2{font-size:17px;margin:38px 0 8px;padding-top:14px;border-top:1px solid var(--line)}
.sub{color:var(--mut);margin:0 0 6px}
p{margin:6px 0}
.cap{color:var(--mut);font-size:13px;margin:2px 0 10px}
table{border-collapse:collapse;width:100%;font-size:14px;margin:6px 0 2px}
th,td{padding:6px 10px;border-bottom:1px solid var(--line);text-align:right;
font-variant-numeric:tabular-nums}
th:first-child,td:first-child{text-align:left}
thead th{border-bottom:2px solid #cfd6dc;font-weight:600;color:#2a2f36}
tbody tr:hover{background:#f1f5f8}
code{background:#eef2f5;padding:1px 5px;border-radius:4px;font-size:13px}
.tag{display:inline-block;padding:1px 7px;border-radius:10px;font-size:12px;
font-weight:600;color:#fff}
.t-drive{background:var(--drive)}.t-confound{background:var(--confound)}
.t-remove{background:var(--remove)}.t-note{background:var(--accent)}
ul{margin:6px 0;padding-left:20px}li{margin:3px 0}
footer{margin-top:46px;color:var(--mut);font-size:12px;border-top:1px solid var(--line);
padding-top:12px}
a{color:var(--accent)}
"""


def _table(headers, rows, cap=None):
    h = "".join(f"<th>{html.escape(str(x))}</th>" for x in headers)
    body = ""
    for r in rows:
        body += "<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>"
    cap_html = f'<div class="cap">{cap}</div>' if cap else ""
    return f"<table><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table>{cap_html}"


def build():
    today = _dt.date.today().isoformat()

    # T1 spectrum
    m = master_rows()
    t1 = _table(
        ["manifold", "dim", "n_cover", "peak", "final", "rel sev.", "over-opt"],
        [[r["manifold"], r["d"], r["n_cover"], _f(r["peak"]), _f(r["final"]),
          _f(r["rel"], 2), f"{r['n_over']}/{r['n_seeds']}"] for r in m],
        cap="peak / final = median homology-recovery quotient at the best interior "
            "iterate and at the fixed-budget convergence point (250 iters, early stop "
            "off); rel sev. = (peak−final)/peak; over-opt = seeds with an interior "
            "peak and (peak−final) &gt; 0.05.")

    # T2 factor verdicts (numbers pulled from the sweep CSVs)
    ar = sweep_points("overopt_sweep_aniso_ratio_v0.csv")
    aw = sweep_points("overopt_sweep_aniso_ratio_white_v0.csv")
    cr = sweep_points("overopt_sweep_clifford_reg_v0.csv")
    dm = sweep_points("overopt_sweep_donut_measure_v0.csv")
    cn = sweep_points("overopt_naxis_clifford_v0.csv")
    dn = sweep_points("overopt_sweep_donut_ncover_v0.csv")
    dsamp = sweep_points("overopt_sweep_donut_sampling_v0.csv")

    def tag(kind, label):
        return f'<span class="tag t-{kind}">{label}</span>'

    t2 = _table(
        ["factor (controlled)", "knob range", "rel sev.", "verdict"],
        [
            ["anisotropy, flat torus raw", "ratio 1.0 → 0.5",
             f'{_f(ar[1.0]["rel"],2)} → {_f(ar[0.5]["rel"],2)}', tag("drive", "driver")],
            ["…same, whitened", "ratio 1.0 → 0.2",
             f'{_f(aw[1.0]["rel"],2)} → {_f(aw[0.2]["rel"],2)}', tag("remove", "whitening removes")],
            ["loss balance: regularity, isotropic torus", "reg 5 → 40",
             f'{_f(cr[5]["rel"],2)} → {_f(cr[40]["rel"],2)}', tag("drive", "driver (universal)")],
            ["loss balance: measure, donut", "measure 0.5 → 4",
             f'{_f(dm[0.5]["rel"],2)} → {_f(dm[4.0]["rel"],2)}', tag("remove", "measure=4 removes")],
            ["under-sampling, isotropic torus", "n 1500 → 6000",
             f'{_f(cn[1500]["rel"],2)} → {_f(cn[6000]["rel"],2)}', tag("drive", "driver")],
            ["n_cover, donut", "35 / 52 / 64",
             f'over-opt {dn[35]["n_over"]}/{dn[35]["n"]}, {dn[52]["n_over"]}/{dn[52]["n"]}, {dn[64]["n_over"]}/{dn[64]["n"]}',
             tag("confound", "confound")],
            ["sampling density, donut", "angle vs area",
             f'{_f(dsamp["angle"]["rel"],2)} vs {_f(dsamp["area"]["rel"],2)}',
             tag("note", "density not the variable")],
        ],
        cap="Each row varies one knob with the others fixed; rel sev. is the median "
            "over 4 seeds. Anisotropy / loss balance / under-sampling change severity; "
            "n_cover does not (the donut over-optimizes at every n_cover that recovers).")

    # T3 mechanism
    mech, npool = mechanism()
    t3 = _table(
        ["peak→final change", "Spearman vs severity", "reading"],
        [["element death (Δ active elements)", _f(mech["d_active"][0], 2),
          "more death → more severity"],
         ["overlap sharpening (Δ participation ratio)", _f(mech["d_pr"][0], 2),
          "more sharpening → more severity"],
         ["over-filling (Δ strong triangles)", _f(mech["d_strong_tri"][0], 2),
          "no relation (refuted)"]],
        cap=f"Pooled over {npool} seed-rows with peak recovery &gt; 0.10. Element death "
            "and overlap sharpening track severity and are causally moved by the loss "
            "balance (the regularity term coarsens / kills elements, the measure term "
            "keeps them alive); the earlier &ldquo;over-filling&rdquo; sub-story is not "
            "supported.")

    # T4 earlier: external method comparison + high-dim boundary
    ext = external_topo()
    t4a = _table(
        ["dataset", "ShapeDiscover", "Alpha", "Rips"],
        [[d, _f(ext.loc[d, "shapediscover"]), _f(ext.loc[d, "alpha"]),
          _f(ext.loc[d, "rips"])] for d in ext.index],
        cap="Recovery quotient, mean over 3 seeds. ShapeDiscover at a fixed "
            "<code>n_cover=52</code>, default early stop (not tuned per manifold); "
            "Alpha / Rips parameter-free. Alpha recovers the donut (<code>torus</code>) "
            "cleanly, so the donut gap is method-specific, not a data problem.")
    hd = highdim_boundary()
    t4b = _table(
        ["dataset", "recovery"],
        [[d, _f(v)] for d, v in hd],
        cap="Recovery quotient, mean over 2 seeds, <code>n_cover=15</code>. The high "
            "intrinsic-dimensional manifolds do not recover (dense-nerve "
            "<code>C(n_cover, d+2)</code> blow-up); <code>noise</code> is the negative "
            "control (target b=[1,0,0], not hallucinated).")

    body = f"""<main>
<h1>ShapeDiscover: cover-learning benchmark results</h1>
<p class="sub">Homology recovery and the over-optimization study. Generated {today} from committed result CSVs.</p>
<p>ShapeDiscover learns a fuzzy cover whose <em>nerve</em> should recover the topology of a point cloud. The headline metric is the homology-recovery quotient (fraction of the nerve filtration with the exactly-correct Betti vector). <strong>Over-optimization</strong>: at fixed data and <code>n_cover</code>, with early stopping off, recovery rises to an interior-iteration peak then declines toward convergence. All tables are computed directly from the CSVs in <code>benchmarks/results/</code>.</p>

<h2>1. Over-optimization severity spectrum</h2>
<p class="cap">Master classification, 5 seeds, default loss weights <code>[1,10,1,10]</code>, each manifold at its natural <code>n_cover</code>. Recovering synthetic set only.</p>
{t1}

<h2>2. What drives severity (single-factor sweeps)</h2>
{t2}

<h2>3. Cover-level mechanism</h2>
{t3}

<h2>4. Earlier benchmarking</h2>
<p class="cap">External method comparison (topology recovery):</p>
{t4a}
<p class="cap">High intrinsic-dimensional boundary (separate failure mode, excluded from the over-optimization study):</p>
{t4b}

<h2>5. Future directions</h2>
<ul>
<li><strong>Anti-element-death / loss-balance lever</strong> (best-grounded candidate): a per-element measure floor or a higher default measure weight. Evidence: <code>measure=4</code> removes the donut's over-optimization. <em>Untested</em>: generalization to Klein bottle / RP² / raw anisotropic torus, cost on the simple manifolds, sampling-invariance. No change to the default algorithm has been made.</li>
<li><strong>Whitening as preprocessing</strong> for linearly anisotropic data: validated on the flat torus (removes the anisotropy-induced over-optimization); does not fix the donut's nonlinear/local anisotropy.</li>
<li><strong>High intrinsic-dimensional homology</strong>: gated by the dense-nerve blow-up; needs genuinely sparse covers before <code>n_cover</code> can be pushed higher.</li>
<li><strong>Persistence-based clustering</strong>: the current connected-components route under-segments real high-dimensional data to a single cluster.</li>
</ul>

<footer>
Method: ShapeDiscover, cover learning for large-scale topology representation (ICML 2025).
Code and data: <a href="https://github.com/Michelle-QT/shapediscover">github.com/Michelle-QT/shapediscover</a>.
Tables regenerated by <code>benchmarks/make_results_site.py</code> from <code>results/*.csv</code>.
No numbers are hand-entered; medians/means and seed counts are as stated per table.
</footer>
</main>"""

    return (f"<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            f"<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            f"<title>ShapeDiscover benchmark results</title><style>{CSS}</style>"
            f"</head><body>{body}</body></html>")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("site/index.html"))
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build())
    print(f"wrote {args.out} ({args.out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
