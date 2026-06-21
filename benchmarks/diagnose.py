"""CLI driver: apply the cover diagnostics to a target dataset and controls.

Runs the :mod:`benchmarks.cover_diagnostics` tooling on the torus (the target,
which breaks under every adaptive lever) with the circle and 2-sphere as
controls, and writes a markdown report plus layout-free figures (barcode, Betti
trajectory, birth-vs-volume scatter, point cloud colored by membership /
nerve-component). The figures are regenerable scratch output, written under
``results/diagnostics/`` (gitignored); the findings are summarized to stdout and
into the report for the project log.

Examples
--------
Full diagnostic on the default target + controls::

    python -m benchmarks.diagnose

Just the per-dataset structure + prune sweep, no figures, plus the torus-only
optimization-evolution and n_cover sweep::

    python -m benchmarks.diagnose --no-figures --evolution --n-cover-sweep
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np

from . import datasets as ds_mod
from . import cover_diagnostics as cd

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "diagnostics"


def _fit_cover(X, n_cover, knn, seed, extra=None):
    from shapediscover import ShapeDiscover

    est = ShapeDiscover(n_cover=n_cover, knn=knn, random_state=seed,
                        verbose=False, plot_loss_curve=False, **(extra or {}))
    cover = np.asarray(est.fit_transform(X)).T  # internal (n_cover, n_points)
    return cover, est


def _fmt_window(w):
    return "none" if w is None else f"[{w[0]:.3f}, {w[1]:.3f}]"


def diagnose_dataset(name, *, n_cover, knn, seed, figures, out_dir, lines):
    ds = ds_mod.load(name, seed=seed)
    target = ds.target_betti
    max_dim = len(target) - 1
    cover, est = _fit_cover(ds.X, n_cover, knn, seed)

    lines.append(f"## {name}  (target Betti {target}, n={ds.n_points}, "
                 f"n_cover={n_cover}, knn={knn}, seed={seed})\n")

    # 1. surviving structure
    struct = cd.surviving_structure(cover, max_dimension=max_dim)
    lines.append(f"- active cover elements: {struct['n_active']} / {struct['n_cover']}")
    for pd in struct["per_dimension"]:
        lines.append(
            f"  - dim {pd['dimension']}: nonempty {pd['n_nonempty']}/{pd['n_possible']} "
            f"(fill {pd['fill_ratio']:.3f}), strong(>0.5) {pd['n_strong']}, "
            f"birth median {pd['birth_median']:.3f} / p90 {pd['birth_p90']:.3f}")
    if struct["strong_degree"]:
        d = struct["strong_degree"]
        lines.append(f"  - strong-edge degree: mean {d['degree_mean']:.1f}, "
                     f"median {d['degree_median']:.0f}, range [{d['degree_min']}, {d['degree_max']}]")

    # 2. membership-filtration recovery + Betti trajectory
    rec = cd.filtration_recovery(cover, target, filtration="birth", max_dimension=max_dim)
    tr = rec["trajectory"]
    lines.append(f"- birth filtration: recovery {rec['recovery_quotient']:.3f}, "
                 f"bar-dominance {rec['bar_dominance_min']:.2f}, "
                 f"complex_size {rec['complex_size']}, "
                 f"best-slice Betti {tr['best_slice_betti']}, window {_fmt_window(tr['window'])}")
    for s in tr["spurious"]:
        if s["dimension"] >= 1:
            lines.append(
                f"  - H{s['dimension']}: betti@best {s['betti_at_best']} "
                f"(target {s['target']}), total persistence of extra bars "
                f"{s['spurious_persistence_sum']:.2f}")

    # 3. alternative filtrations (volume / support as the order)
    for filt in ("mass", "support"):
        r = cd.filtration_recovery(cover, target, filtration=filt, max_dimension=max_dim)
        lines.append(f"- {filt} filtration: recovery {r['recovery_quotient']:.3f}, "
                     f"bar-dominance {r['bar_dominance_min']:.2f}, "
                     f"best-slice Betti {r['trajectory']['best_slice_betti']}")

    # 4. prune-by-volume sweep (birth filtration, drop low-mass simplices)
    sw = cd.prune_sweep(cover, target, prune_key="mass")
    lines.append("- prune-by-mass sweep (birth filtration; drop low-volume simplices):")
    for r in sw["rows"]:
        lines.append(
            f"  - q{r['quantile']:.2f}: recovery {r['recovery_quotient']:.3f}, "
            f"bar-dominance {r['bar_dominance_min']:.2f}, complex_size {r['complex_size']}, "
            f"best-slice Betti {r['best_slice_betti']}")
    lines.append("")

    if figures:
        _figures_for_dataset(name, ds, cover, target, max_dim, rec, tr, out_dir)
    return rec


def _figures_for_dataset(name, ds, cover, target, max_dim, rec, tr, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # barcode + Betti trajectory
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    cd.plot_barcode(rec["intervals"], ax=axes[0], title=f"{name}: barcode (birth)")
    cd.plot_betti_trajectory(tr, target, ax=axes[1], title=f"{name}: Betti trajectory")
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}_barcode_betti.png", dpi=110)
    plt.close(fig)

    # birth-vs-volume scatter for the top dimension (the void-creating simplices)
    tbl = cd.simplex_table(cover, max_dim)
    fig, ax = plt.subplots(figsize=(6, 4))
    cd.plot_birth_volume(tbl, ax=ax,
                         title=f"{name}: dim-{max_dim} simplices (birth vs volume)")
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}_birth_volume.png", dpi=110)
    plt.close(fig)

    # point cloud colored by argmax cover element and by nerve component
    argmax_elt = np.argmax(cover, axis=0)
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121, projection="3d" if ds.X.shape[1] == 3 else None)
    cd.plot_pointcloud_colored(ds.X, argmax_elt, ax=ax1, title=f"{name}: argmax element")
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}_pointcloud.png", dpi=110)
    plt.close(fig)


def diagnose_evolution(name, *, n_cover, knn, seed, out_dir, figures, lines):
    ds = ds_mod.load(name, seed=seed)
    target = ds.target_betti
    lines.append(f"## {name}: evolution across optimization "
                 f"(n_cover={n_cover}, early stop off)\n")
    ev = cd.optimization_evolution(ds.X, target, n_cover=n_cover, knn=knn,
                                   random_state=seed, n_snapshots=12)
    lines.append("| iter | n_active | recovery | bar-dom | best Betti |")
    lines.append("|---|---|---|---|---|")
    for r in ev["rows"]:
        lines.append(f"| {r['iteration']} | {r['n_active']} | "
                     f"{r['recovery_quotient']:.3f} | {r['bar_dominance_min']:.2f} | "
                     f"{r['best_slice_betti']} |")
    lines.append("")

    if figures:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        iters = [r["iteration"] for r in ev["rows"]]
        rq = [r["recovery_quotient"] for r in ev["rows"]]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(iters, rq, "o-")
        ax.set_xlabel("iteration")
        ax.set_ylabel("recovery quotient")
        ax.set_title(f"{name}: recovery vs optimization iteration")
        fig.tight_layout()
        fig.savefig(out_dir / f"{name}_evolution.png", dpi=110)
        plt.close(fig)
    return ev


# Each dataset's natural cover size (validated in the diagnostics): the torus
# needs the fine n_cover=52, the simple manifolds far fewer. Used by the
# multi-seed evolution so the monotone-vs-non-monotone contrast is not confounded
# by running the controls at the torus's (over-fine) cover size. The 2-manifolds
# (donut, clifford/anisotropic torus, klein, rp2) need ~52; the 1-manifolds far
# fewer. anisotropic_torus is run RAW (preprocess off) in the severity study so
# its anisotropy is present (the registry default whitens it away).
NATURAL_N_COVER = {"torus": 52, "circle": 12, "sphere2": 24, "sphere3": 24,
                   "two_circles": 24, "clifford_torus": 52, "sphere4": 32,
                   "torus3": 64, "s2_times_s1": 40, "anisotropic_torus": 52,
                   "figure_eight": 16, "linked_circles": 24, "rp2": 52,
                   "klein_bottle": 52}

# The RECOVERING synthetic topology set (the over-optimization study backbone).
# Excludes torus3 / s2_times_s1 / sphere4 / cp2 (never recover; nerve-blowup
# gated, a separate failure). Each value gives the two per-manifold factor flags
# the severity regression needs beyond (intrinsic dim, Betti sum), which are
# derived from target_betti: aniso = the two H1 cycles at different geometric
# scales; curv = non-constant Gaussian curvature (only the R^3 donut here). Spheres
# have constant (non-varying) curvature, so curv=0; the Clifford / anisotropic flat
# tori are flat. ``preprocess`` defaults True; anisotropic_torus runs raw.
RECOVERING_FACTORS = {
    "circle":            dict(aniso=0, curv=0),
    "figure_eight":      dict(aniso=0, curv=0),
    "two_circles":       dict(aniso=0, curv=0),
    "linked_circles":    dict(aniso=0, curv=0),
    "sphere2":           dict(aniso=0, curv=0),
    "sphere3":           dict(aniso=0, curv=0),
    "clifford_torus":    dict(aniso=0, curv=0),
    "anisotropic_torus": dict(aniso=1, curv=0, preprocess=False),
    "torus":             dict(aniso=1, curv=1),
    "rp2":               dict(aniso=0, curv=0),
    "klein_bottle":      dict(aniso=0, curv=0),
}


def _run_severity(name, *, n_cover, knn, seeds, load_kwargs=None, preprocess=True,
                  extra=None, n_max_iter=250, n_snapshots=12):
    """Run multi-seed optimization-evolution for one config; return per-seed
    severity records + the manifold's target Betti / intrinsic dim.

    Each seed is an independent point-cloud draw AND fit (data seed = model
    ``random_state``), so the spread is the realistic sample-to-sample +
    optimization variance. Early stop is off inside ``optimization_evolution``, so
    this measures the *iteration* axis (over-training), never the early-stop /
    n_points (over/under-sampling) axis.
    """
    records = []
    target = None
    n_points = None
    for s in seeds:
        ds = ds_mod.load(name, seed=s, preprocess=preprocess, **(load_kwargs or {}))
        target = ds.target_betti
        n_points = ds.n_points
        ev = cd.optimization_evolution(
            ds.X, target, n_cover=n_cover, knn=knn, random_state=s,
            n_snapshots=n_snapshots, n_max_iter=n_max_iter, extra=extra,
            field=ds.homology_field)
        records.append(cd.evolution_severity(ev["rows"]))
    return records, target, n_points


def _severity_csv_row(name, *, dim, betti_sum, aniso, curv, n_cover, n_points,
                      measure_w, reg_w, seed, rec, extra_cols=None):
    """Flatten one per-seed severity record into a tidy CSV row dict."""
    row = {
        "manifold": name, "intrinsic_dim": dim, "betti_sum": betti_sum,
        "aniso": aniso, "curv": curv, "n_cover": n_cover, "n_points": n_points,
        "measure_w": measure_w, "reg_w": reg_w, "seed": seed,
        "peak_iteration": rec["peak_iteration"],
        "peak_recovery": rec["peak_recovery"],
        "final_recovery": rec["final_recovery"],
        "abs_severity": rec["abs_severity"],
        "rel_severity": rec["rel_severity"],
        "over_optimizes": int(rec["over_optimizes"]),
        "n_active_peak": rec["n_active_peak"],
        "n_active_final": rec["n_active_final"],
        "d_active": rec["d_active"],
        "d_pr": rec["d_pr"],
        "d_strong_tri": rec["d_strong_tri"],
        "d_strong_edge": rec["d_strong_edge"],
        "strong_tri_peak": rec["strong_tri_peak"],
        "strong_tri_final": rec["strong_tri_final"],
    }
    if extra_cols:
        row.update(extra_cols)
    return row


def _write_csv(path, rows):
    import csv

    if not rows:
        return
    keys = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def diagnose_multiseed_evolution(name, *, knn, seeds, n_cover=None, n_snapshots=12,
                                 n_max_iter=250, out_dir, figures, lines):
    """Confirm the optimization-evolution shape across independent seeds.

    Each seed draws an independent point cloud (data seed) and an independent fit
    (model ``random_state``), so the band spans the realistic sample-to-sample +
    optimization variance. Reports per-seed peak/decline and the median trajectory
    with a min/max band.
    """
    ncov = n_cover or NATURAL_N_COVER.get(name, 52)
    evolutions = []
    for s in seeds:
        ds = ds_mod.load(name, seed=s)
        ev = cd.optimization_evolution(ds.X, ds.target_betti, n_cover=ncov, knn=knn,
                                       random_state=s, n_snapshots=n_snapshots,
                                       n_max_iter=n_max_iter)
        evolutions.append(ev)
    agg = cd.aggregate_evolution(evolutions)

    lines.append(f"## {name}: multi-seed optimization evolution "
                 f"(n_cover={ncov}, seeds={list(seeds)}, early stop off)\n")
    lines.append(f"- over-optimizing seeds (interior recovery peak, then decline "
                 f">0.05): {agg['n_over_optimizing']} / {agg['n_seeds']}")
    lines.append("- per seed: peak_iter / peak_recovery / final_recovery / "
                 "n_active start->end")
    for i, s in zip(seeds, agg["per_seed"]):
        lines.append(f"  - seed {i}: peak@{s['peak_iteration']} "
                     f"{s['peak_recovery']:.3f} -> final {s['final_recovery']:.3f} "
                     f"(decline {s['decline_from_peak']:+.3f}), "
                     f"active {s['n_active_start']}->{s['n_active_end']}"
                     f"{'  [OVER-OPT]' if s['over_optimizes'] else ''}")
    lines.append("\n| iter | recovery median [min, max] | n_active median |")
    lines.append("|---|---|---|")
    for r in agg["per_iteration"]:
        lines.append(f"| {r['iteration']} | {r['recovery_median']:.3f} "
                     f"[{r['recovery_min']:.3f}, {r['recovery_max']:.3f}] | "
                     f"{r['n_active_median']:.0f} |")
    lines.append("")

    if figures:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        it = [r["iteration"] for r in agg["per_iteration"]]
        med = [r["recovery_median"] for r in agg["per_iteration"]]
        lo = [r["recovery_min"] for r in agg["per_iteration"]]
        hi = [r["recovery_max"] for r in agg["per_iteration"]]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.fill_between(it, lo, hi, alpha=0.25, label="min-max band")
        ax.plot(it, med, "o-", label="median")
        ax.set_xlabel("iteration")
        ax.set_ylabel("recovery quotient")
        ax.set_title(f"{name}: recovery vs iteration "
                     f"({agg['n_seeds']} seeds, n_cover={ncov})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"{name}_multiseed_evolution.png", dpi=110)
        plt.close(fig)
    return agg


def diagnose_weight_sweep(name, *, knn, seed, n_cover=None, weight_grid=None,
                          n_snapshots=10, n_max_iter=250, lines):
    """Experiment #1: does rebalancing measure vs regularity fix over-optimization?

    For each ``(measure, regularity)`` weight pair (geometry and topology held at
    0 to isolate the two drivers), replay the fit with early stop off and read the
    evolution *shape*: whether recovery has an interior peak that then declines
    (over-optimization, the good cover is a transient) or improves to convergence
    (the good cover is a near-minimizer). Also reports the convergent overlap
    (participation ratio) and the filtration-window width (the
    disconnected->connected span over which the target topology holds, the user's
    framing). If raising regularity removes the interior peak and lands the
    convergent overlap at the recovery optimum, reweighting is a fix; if the peak
    persists at every balance, the objective itself needs a topology-aware term.
    """
    ncov = n_cover or NATURAL_N_COVER.get(name, 52)
    grid = weight_grid or [(1, 10), (1, 20), (1, 40), (1, 80), (0.5, 10), (2, 10)]
    ds = ds_mod.load(name, seed=seed)
    target = ds.target_betti

    lines.append(f"## {name}: measure/regularity sweep "
                 f"(n_cover={ncov}, seed={seed}, geometry=topology=0, early stop off)\n")
    lines.append("| measure | reg | peak rec @iter | final rec | peak PR | final PR "
                 "| final window | shape |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for measure, reg in grid:
        ev = cd.optimization_evolution(
            ds.X, target, n_cover=ncov, knn=knn, random_state=seed,
            n_snapshots=n_snapshots, n_max_iter=n_max_iter,
            extra={"loss_weights": [measure, 0, 0, reg]},
        )
        agg = cd.aggregate_evolution([ev])
        s = agg["per_seed"][0]
        rows = ev["rows"]
        final_pr = rows[-1]["overlap_pr"]
        peak_idx = int(np.argmax([r["recovery_quotient"] for r in rows]))
        peak_pr = rows[peak_idx]["overlap_pr"]
        final_window = rows[-1]["window_width"]
        shape = "OVER-OPT (transient)" if s["over_optimizes"] else "monotone-ish"
        lines.append(
            f"| {measure} | {reg} | {s['peak_recovery']:.3f} @{s['peak_iteration']} "
            f"| {s['final_recovery']:.3f} | {peak_pr:.2f} | {final_pr:.2f} "
            f"| {final_window:.2f} | {shape} |")
    lines.append("")


# Frontier A: homogeneous (constant / zero curvature) manifolds only, so the
# overlap<->dimension law is not confounded by curvature (the donut is excluded
# on purpose). Each entry is (name, [n_cover grid], intrinsic_dim); the n_cover
# grids are capped so the top-dimension nerve C(n_cover, d+2) stays tractable
# (the dense-nerve blow-up is itself part of the high-dim finding).
OVERLAP_LAW_MANIFOLDS = [
    ("circle", [8, 12, 20], 1),
    ("sphere2", [15, 24, 40], 2),
    ("clifford_torus", [24, 40, 52], 2),
    ("sphere3", [24, 40, 52], 3),       # C(52,5)=2.6M caps the dim-4 nerve build
    ("s2_times_s1", [40, 52], 3),
    ("torus3", [40, 52], 3),
    ("sphere4", [24, 32], 4),           # C(32,6)=0.9M caps the dim-5 nerve build
]


# The recovering homogeneous subset where the overlap law holds (fast: excludes
# the high intrinsic-dim manifolds that never recover and dominate the runtime).
RECOVERING_OVERLAP_LAW = [
    ("circle", [8, 12, 20], 1),
    ("sphere2", [15, 24, 40], 2),
    ("clifford_torus", [24, 40, 52], 2),
    ("sphere3", [24, 40, 52], 3),
]


def diagnose_overlap_law(manifolds=None, *, knn, seeds, n_max_iter=200,
                         n_snapshots=8, out_dir, figures, lines, csv_path=None):
    """Frontier A: optimal overlap vs intrinsic dimension and topological complexity.

    Recovery is non-monotone in the mean per-point overlap (participation ratio
    PR): too much (complete nerve) and too little (over-sharp) both fail, with an
    optimal band between. For each homogeneous manifold this finds, over an
    ``n_cover`` sweep and the optimization trajectory (early stop off), the
    snapshot of peak recovery and records the overlap there (``PR@peak``, the
    "optimal overlap"), plus the convergent overlap (``PR@conv``). Two questions:

    - does optimal overlap scale with intrinsic dimension ``d`` and with
      topological complexity (sum of Betti numbers)? -> a predictive law for
      auto-setting ``n_cover`` / the measure-regularity balance;
    - does the cover *overshoot* the optimal overlap at convergence
      (``PR@conv < PR@peak``, i.e. over-optimization), and is that overshoot the
      thing that distinguishes the manifolds the method handles from those it
      does not?
    """
    manifolds = manifolds or OVERLAP_LAW_MANIFOLDS
    lines.append(f"## Overlap law: optimal overlap vs dimension / complexity "
                 f"(knn={knn}, seeds={list(seeds)}, early stop off)\n")
    lines.append("| manifold | d | betti | best n_cover | peak rec | PR@peak | "
                 "PR@conv | overshoot PR@conv/PR@peak | conv rec |")
    lines.append("|---|---|---|---|---|---|---|---|---|")

    plot_rows = []
    timings = []
    csv_rows = []
    for name, ncovers, dim in manifolds:
        t_manifold = time.perf_counter()
        ds0 = ds_mod.load(name, seed=seeds[0])
        target = ds0.target_betti
        betti_sum = int(sum(target))
        per_seed = []
        for s in seeds:
            X = ds_mod.load(name, seed=s).X
            best = None
            for nc in ncovers:
                try:
                    ev = cd.optimization_evolution(
                        X, target, n_cover=nc, knn=knn, random_state=s,
                        n_snapshots=n_snapshots, n_max_iter=n_max_iter)
                except (MemoryError, ValueError):
                    continue  # nerve blow-up at this (n_cover, dim); skip the config
                rows = ev["rows"]
                recs = [r["recovery_quotient"] for r in rows]
                pk = int(np.argmax(recs))
                cand = {
                    "peak_rec": recs[pk], "pr_peak": rows[pk]["overlap_pr"],
                    "pr_conv": rows[-1]["overlap_pr"], "conv_rec": recs[-1],
                    "n_cover": nc, "peak_iter": rows[pk]["iteration"],
                }
                if best is None or cand["peak_rec"] > best["peak_rec"]:
                    best = cand
            if best is not None:
                per_seed.append(best)
                csv_rows.append({"manifold": name, "intrinsic_dim": dim,
                                 "betti_sum": betti_sum, "seed": s,
                                 "best_n_cover": best["n_cover"],
                                 "peak_recovery": best["peak_rec"],
                                 "pr_peak": best["pr_peak"], "pr_conv": best["pr_conv"],
                                 "conv_recovery": best["conv_rec"]})
        if not per_seed:
            lines.append(f"| {name} | {dim} | {target} | - | (all configs skipped) "
                         "| - | - | - | - |")
            continue

        def mean(key):
            return float(np.mean([b[key] for b in per_seed]))

        def std(key):
            return float(np.std([b[key] for b in per_seed]))

        pr_peak, pr_conv = mean("pr_peak"), mean("pr_conv")
        overshoot = pr_conv / pr_peak if pr_peak else float("nan")
        # the n_cover most often selected as best
        ncs = [b["n_cover"] for b in per_seed]
        best_nc = max(set(ncs), key=ncs.count)
        lines.append(
            f"| {name} | {dim} | {target} | {best_nc} | "
            f"{mean('peak_rec'):.3f} | {pr_peak:.2f}±{std('pr_peak'):.2f} | "
            f"{pr_conv:.2f} | {overshoot:.2f} | {mean('conv_rec'):.3f} |")
        plot_rows.append({"name": name, "dim": dim, "betti_sum": betti_sum,
                          "pr_peak": pr_peak, "pr_peak_std": std("pr_peak"),
                          "peak_rec": mean("peak_rec"), "overshoot": overshoot})
        timings.append((name, dim, time.perf_counter() - t_manifold))
    lines.append("")
    # efficiency note: this experiment is in the large-n_cover / high-dim regime
    # where the dense nerve C(n_cover, d+2) dominates (see the efficiency TODO).
    lines.append("Wall time per manifold (this is the nerve-bound regime): "
                 + ", ".join(f"{n}(d{d}) {t:.0f}s" for n, d, t in timings)
                 + f"; total {sum(t for _, _, t in timings):.0f}s.")
    lines.append("")
    lines.append("PR@peak = mean overlap (effective elements/point) at the "
                 "peak-recovery iterate (the *optimal* overlap); PR@conv = overlap "
                 "at convergence; overshoot < 1 means the cover sharpens past the "
                 "optimum (over-optimization).")
    lines.append("")

    if csv_path is not None:
        _write_csv(csv_path, csv_rows)
        lines.append(f"[tidy per-seed rows -> {csv_path}]\n")
    if figures and plot_rows:
        _plot_overlap_law(plot_rows, out_dir)
    return plot_rows


def _plot_overlap_law(plot_rows, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    dims = [r["dim"] for r in plot_rows]
    prs = [r["pr_peak"] for r in plot_rows]
    errs = [r["pr_peak_std"] for r in plot_rows]
    sizes = [40 + 30 * r["betti_sum"] for r in plot_rows]
    sc = ax.scatter(dims, prs, s=sizes, c=[r["peak_rec"] for r in plot_rows],
                    cmap="viridis", vmin=0, vmax=1, zorder=3)
    ax.errorbar(dims, prs, yerr=errs, fmt="none", ecolor="grey", alpha=0.5, zorder=2)
    for r in plot_rows:
        ax.annotate(f"{r['name']}\nb={r['betti_sum']}", (r["dim"], r["pr_peak"]),
                    fontsize=7, xytext=(6, 0), textcoords="offset points", va="center")
    # reference lines d+1 and d+2 (the naive local-nerve overlap for a d-patch)
    xs = np.array(sorted(set(dims)))
    ax.plot(xs, xs + 1, "--", color="C1", alpha=0.6, label="d+1")
    ax.plot(xs, xs + 2, ":", color="C3", alpha=0.6, label="d+2")
    ax.set_xlabel("intrinsic dimension d")
    ax.set_ylabel("optimal overlap PR@peak (effective elements / point)")
    ax.set_title("Overlap law: optimal overlap vs dimension\n"
                 "(marker size ~ sum of Betti; color = peak recovery)")
    ax.legend()
    plt.colorbar(sc, ax=ax, label="peak recovery")
    fig.tight_layout()
    fig.savefig(out_dir / "overlap_law.png", dpi=120)
    plt.close(fig)


# Homogeneous manifolds at a few n_cover each, capped to keep the dense nerve
# tractable. The question: is the convergent overlap flat in n_cover?
PR_INVARIANCE_MANIFOLDS = [
    ("circle", [8, 12, 20, 32], 1),
    ("sphere2", [12, 24, 40, 60], 2),
    ("clifford_torus", [24, 40, 52, 64], 2),
    ("sphere3", [24, 40, 52], 3),
]


def diagnose_pr_invariance(manifolds=None, *, knn, seeds, n_max_iter=200,
                           out_dir, figures, lines):
    """Is the convergent overlap (PR) set by the loss balance + geometry, hence
    ~invariant to n_cover and emergently dimension-adaptive?

    For each homogeneous manifold, at the *fixed default* loss balance
    (``[1,10,1,10]``) and early stop off, fit at several ``n_cover`` and record
    the convergent overlap (participation ratio) and recovery. If PR is ~flat in
    ``n_cover`` (and sits near ``d+1``), the optimization self-tunes overlap to
    the intrinsic dimension without anyone estimating it -> dimension estimation
    is sidestepped and ``n_cover`` is only a resolution knob. If PR drifts with
    ``n_cover``, the overlap is not an emergent geometric quantity and the
    sidestep fails.
    """
    from shapediscover import ShapeDiscover

    manifolds = manifolds or PR_INVARIANCE_MANIFOLDS
    lines.append(f"## PR invariance: convergent overlap vs n_cover at fixed balance "
                 f"[1,10,1,10] (knn={knn}, seeds={list(seeds)}, early stop off)\n")
    lines.append("| manifold | d | n_cover | conv PR | recovery | n_active |")
    lines.append("|---|---|---|---|---|---|")

    plot_data = {}
    for name, ncovers, dim in manifolds:
        target = ds_mod.load(name, seed=seeds[0]).target_betti
        max_dim = len(target) - 1
        pr_by_nc = []
        for nc in ncovers:
            prs, recs, actives = [], [], []
            for s in seeds:
                X = ds_mod.load(name, seed=s).X
                est = ShapeDiscover(
                    n_cover=nc, knn=knn, random_state=s, n_max_iter=n_max_iter,
                    early_stop=False, verbose=False, plot_loss_curve=False)
                cover = np.asarray(est.fit_transform(X)).T
                prs.append(cd.mean_participation_ratio(cover))
                rec = cd.filtration_recovery(cover, target, filtration="birth",
                                             max_dimension=max_dim)
                recs.append(rec["recovery_quotient"])
                actives.append(int(np.unique(np.argmax(cover, axis=0)).size))
            pr_m, pr_s = float(np.mean(prs)), float(np.std(prs))
            lines.append(f"| {name} | {dim} | {nc} | {pr_m:.2f}±{pr_s:.2f} | "
                         f"{np.mean(recs):.3f} | {np.mean(actives):.0f} |")
            pr_by_nc.append((nc, pr_m, pr_s))
        # spread of convergent PR across the n_cover range (the invariance test)
        prs_only = [p for _, p, _ in pr_by_nc]
        spread = max(prs_only) - min(prs_only)
        rel = spread / np.mean(prs_only) if np.mean(prs_only) else float("nan")
        lines.append(f"|   | | **PR range** | **{spread:.2f} "
                     f"({100*rel:.0f}% of mean)** | | |")
        plot_data[name] = {"dim": dim, "points": pr_by_nc}
    lines.append("")
    lines.append("If conv PR is ~flat in n_cover (small PR range) and sits near "
                 "d+1, overlap is emergent/dimension-adaptive (dimension estimation "
                 "is sidestepped). If it drifts with n_cover, it is not.")
    lines.append("")

    if figures and plot_data:
        _plot_pr_invariance(plot_data, out_dir)
    return plot_data


def _plot_pr_invariance(plot_data, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (name, d) in enumerate(plot_data.items()):
        ncs = [p[0] for p in d["points"]]
        prs = [p[1] for p in d["points"]]
        errs = [p[2] for p in d["points"]]
        dim = d["dim"]
        ax.errorbar(ncs, prs, yerr=errs, marker="o", capsize=3,
                    label=f"{name} (d={dim})", color=f"C{i}")
        ax.axhline(dim + 1, ls="--", color=f"C{i}", alpha=0.35)
    ax.set_xlabel("n_cover")
    ax.set_ylabel("convergent overlap PR")
    ax.set_title("PR invariance: convergent overlap vs n_cover\n"
                 "(dashed = d+1; flat lines => overlap is emergent, dimension auto-handled)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "pr_invariance.png", dpi=120)
    plt.close(fig)


def diagnose_n_cover(name, *, knn, seed, n_covers, lines):
    ds = ds_mod.load(name, seed=seed)
    target = ds.target_betti
    lines.append(f"## {name}: n_cover sweep (knn={knn}, seed={seed})\n")
    sw = cd.n_cover_sweep(ds.X, target, n_covers=n_covers, knn=knn, random_state=seed)
    lines.append("| n_cover | n_active | recovery | bar-dom | complex_size |")
    lines.append("|---|---|---|---|---|")
    for r in sw["rows"]:
        lines.append(f"| {r['n_cover']} | {r['n_active']} | "
                     f"{r['recovery_quotient']:.3f} | {r['bar_dominance_min']:.2f} | "
                     f"{r['complex_size']} |")
    lines.append("")
    return sw


def diagnose_over_opt_classify(manifolds=None, *, knn, seeds, n_max_iter=250,
                               n_snapshots=12, loss_weights=None, csv_path=None,
                               lines):
    """Master multi-seed over-optimization severity table over the recovering set.

    For each recovering manifold (at its natural ``n_cover``) run one independent
    evolution per seed (early stop off, the iteration axis) and report the median
    peak / final recovery, absolute and relative severity, the over-optimizing-seed
    count, and the candidate cover-level mechanism as peak->final deltas (element
    death ``d_active``, strong-triangle over-filling ``d_strong_tri``, overlap
    sharpening ``d_pr``). Writes a tidy per-(manifold, seed) CSV for the regression.

    ``loss_weights`` defaults to the shipped algorithm default ``[1,10,1,10]`` (NOT
    the earlier ``[1,0,0,10]`` classification), so the table characterizes the
    actual default pipeline; pass a 4-list to override.
    """
    manifolds = manifolds or list(RECOVERING_FACTORS)
    extra = {"loss_weights": loss_weights} if loss_weights else None
    measure_w = loss_weights[0] if loss_weights else 1
    reg_w = loss_weights[3] if loss_weights else 10
    lines.append(f"## Over-optimization severity classification "
                 f"(recovering set, seeds={list(seeds)}, natural n_cover, "
                 f"loss_weights={loss_weights or '[1,10,1,10] (default)'}, "
                 f"early stop off)\n")
    lines.append("severity = peak_recovery - final_recovery (recovery lost to "
                 "over-training); rel = that / peak; #over-opt = seeds with an "
                 "interior peak and abs severity > 0.05.\n")
    lines.append("| manifold | d | betti | aniso | curv | n_cover | peak rec | "
                 "final rec | abs sev | rel sev | #over-opt | d_active | "
                 "d_strong_tri | d_pr |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    csv_rows = []
    for name in manifolds:
        f = RECOVERING_FACTORS.get(name, {})
        aniso, curv = f.get("aniso", 0), f.get("curv", 0)
        preprocess = f.get("preprocess", True)
        ncov = NATURAL_N_COVER.get(name, 52)
        records, target, n_points = _run_severity(
            name, n_cover=ncov, knn=knn, seeds=seeds, preprocess=preprocess,
            extra=extra, n_max_iter=n_max_iter, n_snapshots=n_snapshots)
        dim, betti_sum = len(target) - 1, int(sum(target))
        agg = cd.summarize_severity(records)
        lines.append(
            f"| {name} | {dim} | {target} | {aniso} | {curv} | {ncov} | "
            f"{agg['peak_recovery_median']:.3f} | {agg['final_recovery_median']:.3f} | "
            f"{agg['abs_severity_median']:.3f} | {agg['rel_severity_median']:.2f} | "
            f"{agg['n_over_optimizing']}/{agg['n_seeds']} | "
            f"{agg['d_active_median']:+.0f} | {agg['d_strong_tri_median']:+.0f} | "
            f"{agg['d_pr_median']:+.2f} |")
        for s, rec in zip(seeds, records):
            csv_rows.append(_severity_csv_row(
                name, dim=dim, betti_sum=betti_sum, aniso=aniso, curv=curv,
                n_cover=ncov, n_points=n_points, measure_w=measure_w, reg_w=reg_w,
                seed=s, rec=rec))
    lines.append("")
    lines.append("Mechanism columns are medians of the peak->final delta: "
                 "d_active < 0 = element death; d_strong_tri > 0 = over-filling "
                 "(extra strong triangles); d_pr < 0 = overlap sharpening. They are "
                 "only interpretable where peak recovery is non-trivial.")
    lines.append("")
    if csv_path is not None:
        _write_csv(csv_path, csv_rows)
        lines.append(f"[tidy per-seed rows -> {csv_path}]\n")
    return csv_rows


# Single-factor severity sweeps: each isolates one knob while holding the rest
# fixed (the controlled experiments the discrete table cannot give). Each spec is
# (manifold, knob-label, list of (value, load_kwargs, preprocess, extra, n_cover)).
def _sweep_specs():
    donut_r2 = [(r2, {"r2": r2}, True, None, 52)
                for r2 in (0.25, 0.40, 0.50, 0.65, 0.80)]
    donut_sampling = [(s, {"sampling": s}, True, None, 52)
                      for s in ("angle", "area")]
    aniso_ratio = [(r, {"ratio": r}, False, None, 52)   # RAW (preprocess off)
                   for r in (1.0, 0.7, 0.5, 0.35, 0.2)]
    aniso_ratio_white = [(r, {"ratio": r}, True, None, 52)  # whitened
                         for r in (1.0, 0.7, 0.5, 0.35, 0.2)]
    # loss-weight sweeps (the measure vs regularity balance), geometry/topology
    # held at 0 to isolate the two active drivers (matches the earlier weight sweep)
    donut_reg = [(r, {}, True, {"loss_weights": [1, 0, 0, r]}, 52)
                 for r in (5, 10, 20, 40)]
    donut_measure = [(m, {}, True, {"loss_weights": [m, 0, 0, 10]}, 52)
                     for m in (0.5, 1.0, 2.0, 4.0)]
    clifford_reg = [(r, {}, True, {"loss_weights": [1, 0, 0, r]}, 52)
                    for r in (5, 10, 20, 40)]
    # negative-lever reassessments (off-by-default knobs, run at default loss
    # weights). density-normalization of the Dirichlet energies (the ÷h^power arc)
    # and curvature-adaptive weighting (the Track-1 lever). Each compares the knob
    # against the unmodified default on the donut.
    density_donut = [(p, {}, True, {"density_normalize_power": p}, 52)
                     for p in (0.0, 1.0, 2.0)]
    density_circle = [(p, {}, True, {"density_normalize_power": p}, 12)
                      for p in (0.0, 1.0, 2.0)]
    curv_donut = [
        ("uniform", {}, True, None, 52),
        ("measure_curv", {}, True,
         {"measure_weighting": "curvature", "measure_weighting_strength": 1.0}, 52),
        ("measure_curv_inv", {}, True,
         {"measure_weighting": "curvature_inverse", "measure_weighting_strength": 1.0}, 52),
        ("reg_curv", {}, True,
         {"regularity_weighting": "curvature", "regularity_weighting_strength": 1.0}, 52),
    ]
    return {
        "donut_r2": ("torus", "r2", donut_r2),
        "donut_sampling": ("torus", "sampling", donut_sampling),
        "aniso_ratio": ("anisotropic_torus", "ratio", aniso_ratio),
        "aniso_ratio_white": ("anisotropic_torus", "ratio", aniso_ratio_white),
        "donut_reg": ("torus", "reg_weight", donut_reg),
        "donut_measure": ("torus", "measure_weight", donut_measure),
        "clifford_reg": ("clifford_torus", "reg_weight", clifford_reg),
        "density_donut": ("torus", "density_power", density_donut),
        "density_circle": ("circle", "density_power", density_circle),
        "curv_donut": ("torus", "curv_weighting", curv_donut),
    }


def diagnose_severity_sweep(spec_name, *, knn, seeds, n_max_iter=250,
                            n_snapshots=12, csv_path=None, lines):
    """Single-factor severity sweep (one of :func:`_sweep_specs`).

    Sweeps one knob (donut tube radius / sampling, anisotropy ratio raw or
    whitened) and reports severity vs the knob, multi-seed. Isolates the factor the
    discrete table confounds.
    """
    specs = _sweep_specs()
    if spec_name not in specs:
        raise ValueError(f"unknown sweep {spec_name!r}; have {sorted(specs)}")
    name, knob, rows_spec = specs[spec_name]
    lines.append(f"## Severity sweep: {spec_name}  "
                 f"({name}, vary {knob}, seeds={list(seeds)}, early stop off)\n")
    lines.append(f"| {knob} | n_cover | peak rec | final rec | abs sev | rel sev | "
                 "#over-opt | d_active | d_strong_tri | d_pr |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    csv_rows = []
    for value, load_kwargs, preprocess, extra, ncov in rows_spec:
        records, target, n_points = _run_severity(
            name, n_cover=ncov, knn=knn, seeds=seeds, load_kwargs=load_kwargs,
            preprocess=preprocess, extra=extra, n_max_iter=n_max_iter,
            n_snapshots=n_snapshots)
        dim, betti_sum = len(target) - 1, int(sum(target))
        agg = cd.summarize_severity(records)
        lines.append(
            f"| {value} | {ncov} | {agg['peak_recovery_median']:.3f} | "
            f"{agg['final_recovery_median']:.3f} | {agg['abs_severity_median']:.3f} | "
            f"{agg['rel_severity_median']:.2f} | "
            f"{agg['n_over_optimizing']}/{agg['n_seeds']} | "
            f"{agg['d_active_median']:+.0f} | {agg['d_strong_tri_median']:+.0f} | "
            f"{agg['d_pr_median']:+.2f} |")
        ff = RECOVERING_FACTORS.get(name, {})
        # record the actual loss weights when the sweep varies them
        mw = extra["loss_weights"][0] if extra and "loss_weights" in extra else 1
        rw = extra["loss_weights"][3] if extra and "loss_weights" in extra else 10
        for s, rec in zip(seeds, records):
            csv_rows.append(_severity_csv_row(
                name, dim=dim, betti_sum=betti_sum,
                aniso=ff.get("aniso", 0), curv=ff.get("curv", 0),
                n_cover=ncov, n_points=n_points, measure_w=mw, reg_w=rw,
                seed=s, rec=rec, extra_cols={"sweep": spec_name, "knob": knob,
                                             "knob_value": value}))
    lines.append("")
    if csv_path is not None:
        _write_csv(csv_path, csv_rows)
        lines.append(f"[tidy per-seed rows -> {csv_path}]\n")
    return csv_rows


def diagnose_factor_sweep(name, knob, values, *, knn, seeds, n_max_iter=250,
                          n_snapshots=12, csv_path=None, lines):
    """Severity vs an integer config knob: ``n_cover`` or ``n_points`` (``n``).

    The two knobs the discrete table holds at one value. ``n_cover`` is the cover
    resolution; ``n_points`` (passed to the dataset as ``n``) is the SAMPLING axis.
    Both are run with early stop OFF, so they measure how the *iteration-axis*
    over-optimization severity itself changes with resolution / sample size, which
    is distinct from the early-stop-coupled "denser sample under-trains" effect.
    """
    f = RECOVERING_FACTORS.get(name, {})
    aniso, curv = f.get("aniso", 0), f.get("curv", 0)
    preprocess = f.get("preprocess", True)
    base_ncov = NATURAL_N_COVER.get(name, 52)
    lines.append(f"## Factor sweep: {name}, vary {knob}  "
                 f"(seeds={list(seeds)}, early stop off)\n")
    lines.append(f"| {knob} | n_cover | n_points | peak rec | final rec | abs sev | "
                 "rel sev | #over-opt | d_active | d_pr |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    csv_rows = []
    for v in values:
        ncov = v if knob == "n_cover" else base_ncov
        load_kwargs = {"n": v} if knob == "n_points" else None
        records, target, n_points = _run_severity(
            name, n_cover=ncov, knn=knn, seeds=seeds, load_kwargs=load_kwargs,
            preprocess=preprocess, n_max_iter=n_max_iter, n_snapshots=n_snapshots)
        dim, betti_sum = len(target) - 1, int(sum(target))
        agg = cd.summarize_severity(records)
        lines.append(
            f"| {v} | {ncov} | {n_points} | {agg['peak_recovery_median']:.3f} | "
            f"{agg['final_recovery_median']:.3f} | {agg['abs_severity_median']:.3f} | "
            f"{agg['rel_severity_median']:.2f} | "
            f"{agg['n_over_optimizing']}/{agg['n_seeds']} | "
            f"{agg['d_active_median']:+.0f} | {agg['d_pr_median']:+.2f} |")
        for s, rec in zip(seeds, records):
            csv_rows.append(_severity_csv_row(
                name, dim=dim, betti_sum=betti_sum, aniso=aniso, curv=curv,
                n_cover=ncov, n_points=n_points, measure_w=1, reg_w=10,
                seed=s, rec=rec, extra_cols={"sweep": f"{name}_{knob}",
                                             "knob": knob, "knob_value": v}))
    lines.append("")
    if csv_path is not None:
        _write_csv(csv_path, csv_rows)
        lines.append(f"[tidy per-seed rows -> {csv_path}]\n")
    return csv_rows


def diagnose_n_axis(name, n_values, *, knn, seeds, n_max_iter=250, n_snapshots=12,
                    csv_path=None, lines):
    """Contrast the ITERATION axis with the n_points (over/under-sampling) axis.

    The two are easy to conflate but may move in opposite directions. For each
    sample size ``n`` and seed this records both:

    - **iteration axis** (early stop OFF, full budget): the over-optimization curve
      -> ``peak`` / ``final`` recovery and ``abs_severity`` (over-training as the
      optimizer runs past the peak);
    - **default-algorithm view** (early stop ON, the shipped ``gradient``
      criterion): the recovery at the iterate the early stop actually lands on, and
      *which* iterate that is (``stop_iter``). The hypothesis from the notes is that
      a denser sample trips the gradient early stop SOONER (fewer iters ->
      *under*-training), the opposite direction from over-optimization.

    So the question is whether more points helps (the data/peak gets better) while
    the default stop simultaneously moves, decoupling "best achievable" from "what
    the shipped pipeline returns". Both fits share the seed / config; early-stop ON
    is the default ``ShapeDiscover`` gradient criterion.
    """
    from shapediscover import ShapeDiscover

    f = RECOVERING_FACTORS.get(name, {})
    aniso, curv = f.get("aniso", 0), f.get("curv", 0)
    preprocess = f.get("preprocess", True)
    ncov = NATURAL_N_COVER.get(name, 52)
    lines.append(f"## n-axis: iteration vs sampling for {name}  "
                 f"(n_cover={ncov}, seeds={list(seeds)})\n")
    lines.append("early-stop OFF columns = the iteration axis (over-training); "
                 "early-stop ON columns = what the shipped default returns.\n")
    lines.append("| n_points | peak rec (off) | final rec (off) | abs sev (off) | "
                 "#over-opt | stop rec (on) | stop_iter (on) |")
    lines.append("|---|---|---|---|---|---|---|")
    csv_rows = []
    for n in n_values:
        off_records, target, _ = _run_severity(
            name, n_cover=ncov, knn=knn, seeds=seeds, load_kwargs={"n": n},
            preprocess=preprocess, n_max_iter=n_max_iter, n_snapshots=n_snapshots)
        dim, betti_sum = len(target) - 1, int(sum(target))
        agg = cd.summarize_severity(off_records)
        # early-stop ON (default gradient criterion): the shipped-pipeline view
        stop_recs, stop_iters = [], []
        for s in seeds:
            ds = ds_mod.load(name, seed=s, preprocess=preprocess, n=n)
            est = ShapeDiscover(n_cover=ncov, knn=knn, random_state=s,
                                n_max_iter=n_max_iter, early_stop=True,
                                verbose=False, plot_loss_curve=False)
            cover = np.asarray(est.fit_transform(ds.X)).T
            rec = cd.filtration_recovery(cover, target, filtration="birth",
                                         max_dimension=dim, field=ds.homology_field)
            stop_recs.append(rec["recovery_quotient"])
            stop_iters.append(len(est.main_optimization_losses_[-1]))
        stop_rec_med = float(np.median(stop_recs))
        stop_iter_med = float(np.median(stop_iters))
        lines.append(
            f"| {n} | {agg['peak_recovery_median']:.3f} | "
            f"{agg['final_recovery_median']:.3f} | {agg['abs_severity_median']:.3f} | "
            f"{agg['n_over_optimizing']}/{agg['n_seeds']} | {stop_rec_med:.3f} | "
            f"{stop_iter_med:.0f} |")
        for s, rec, sr, si in zip(seeds, off_records, stop_recs, stop_iters):
            csv_rows.append(_severity_csv_row(
                name, dim=dim, betti_sum=betti_sum, aniso=aniso, curv=curv,
                n_cover=ncov, n_points=n, measure_w=1, reg_w=10, seed=s, rec=rec,
                extra_cols={"sweep": f"{name}_n_axis", "knob": "n_points",
                            "knob_value": n, "stop_recovery": sr, "stop_iter": si}))
    lines.append("")
    if csv_path is not None:
        _write_csv(csv_path, csv_rows)
        lines.append(f"[tidy per-seed rows -> {csv_path}]\n")
    return csv_rows


def diagnose_preprocess_compare(manifolds, *, knn, seeds, kinds=("none", "standardize",
                                "whiten"), csv_path=None, lines):
    """Recovery under each feature-preprocessing kind (the normalization policy).

    For each manifold, load the raw features and fit the shipped default pipeline
    (early stop ON) on the raw / standardized / whitened features, recording the
    homology-recovery quotient (multi-seed). Tests the policy claim that
    normalization is not free: standardizing already-isotropic data can hurt, while
    whitening rescues linearly anisotropic data.
    """
    from shapediscover import ShapeDiscover

    lines.append(f"## Preprocessing comparison (recovery, seeds={list(seeds)}, "
                 "default pipeline)\n")
    lines.append("| manifold | n_cover | " + " | ".join(kinds) + " |")
    lines.append("|---|---|" + "---|" * len(kinds))
    csv_rows = []
    for name in manifolds:
        ncov = NATURAL_N_COVER.get(name, 52)
        meds = {}
        for kind in kinds:
            recs = []
            for s in seeds:
                ds = ds_mod.load(name, seed=s, preprocess=False)
                X = ds_mod._preprocess(ds.X, kind) if kind != "none" else ds.X
                est = ShapeDiscover(n_cover=ncov, knn=knn, random_state=s,
                                    verbose=False, plot_loss_curve=False)
                cover = np.asarray(est.fit_transform(X)).T
                rec = cd.filtration_recovery(cover, ds.target_betti, filtration="birth",
                                             max_dimension=len(ds.target_betti) - 1,
                                             field=ds.homology_field)
                recs.append(rec["recovery_quotient"])
                csv_rows.append({"manifold": name, "n_cover": ncov, "preprocess": kind,
                                 "seed": s, "recovery": rec["recovery_quotient"]})
            meds[kind] = float(np.median(recs))
        lines.append(f"| {name} | {ncov} | "
                     + " | ".join(f"{meds[k]:.3f}" for k in kinds) + " |")
    lines.append("")
    if csv_path is not None:
        _write_csv(csv_path, csv_rows)
        lines.append(f"[tidy per-seed rows -> {csv_path}]\n")
    return csv_rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=["torus", "circle", "sphere2"],
                        help="datasets to diagnose (first is treated as the target)")
    parser.add_argument("--n-cover", type=int, default=52)
    parser.add_argument("--knn", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-figures", action="store_true", help="skip figure output")
    parser.add_argument("--evolution", action="store_true",
                        help="also run the optimization-evolution diagnostic on the target")
    parser.add_argument("--n-cover-sweep", action="store_true",
                        help="also run the n_cover sweep on the target")
    parser.add_argument("--n-covers", nargs="+", type=int, default=[20, 35, 52, 75])
    parser.add_argument("--multiseed", type=int, default=0, metavar="N",
                        help="run the optimization-evolution across N independent "
                             "seeds for every dataset (each at its natural n_cover)")
    parser.add_argument("--weight-sweep", action="store_true",
                        help="run the measure/regularity weight sweep (experiment #1) "
                             "on every dataset at its natural n_cover")
    parser.add_argument("--overlap-law", action="store_true",
                        help="Frontier A: measure optimal overlap (PR) vs intrinsic "
                             "dimension / topological complexity across the "
                             "homogeneous manifolds (n_cover sweep, multi-seed)")
    parser.add_argument("--pr-invariance", action="store_true",
                        help="test whether convergent overlap (PR) is invariant to "
                             "n_cover at fixed balance (the dimension-sidestep check)")
    parser.add_argument("--over-opt-classify", action="store_true",
                        help="master multi-seed over-optimization severity table over "
                             "the recovering manifold set (writes a tidy CSV)")
    parser.add_argument("--severity-sweep", metavar="SPEC",
                        help="single-factor severity sweep; SPEC in "
                             "{donut_r2, donut_sampling, aniso_ratio, aniso_ratio_white}")
    parser.add_argument("--factor-sweep", nargs="+", metavar="ARG",
                        help="severity vs an integer knob: "
                             "MANIFOLD n_cover|n_points V1 V2 ... "
                             "(e.g. --factor-sweep torus n_points 1500 3000 6000)")
    parser.add_argument("--n-axis", nargs="+", metavar="ARG",
                        help="contrast the iteration axis (early stop off) with the "
                             "n_points / early-stop axis: MANIFOLD N1 N2 ... "
                             "(e.g. --n-axis circle 400 800 1600 3200)")
    parser.add_argument("--recovering-only", action="store_true",
                        help="(--overlap-law) restrict to the recovering homogeneous "
                             "subset (circle/sphere2/clifford_torus/sphere3), fast")
    parser.add_argument("--preprocess-compare", nargs="+", metavar="MANIFOLD",
                        help="recovery under none/standardize/whiten preprocessing "
                             "for each MANIFOLD (the per-dataset normalization policy)")
    parser.add_argument("--loss-weights", nargs=4, type=float, default=None,
                        metavar=("M", "G", "T", "R"),
                        help="(--over-opt-classify) override loss_weights "
                             "[measure geometry topology regularity]")
    parser.add_argument("--csv", type=Path, default=None,
                        help="path for the tidy per-seed CSV (severity modes)")
    parser.add_argument("--seeds", type=int, default=3, metavar="N",
                        help="number of seeds 0..N-1 (overlap-law / severity modes)")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)

    # bar_dominance divides bar lengths and can hit inf/inf on essential bars;
    # the value is consumed defensively, so silence the benign warning.
    warnings.filterwarnings("ignore", message="invalid value encountered")

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    figures = not args.no_figures

    lines = ["# Cover diagnostics report\n",
             f"target + controls: {', '.join(args.datasets)}\n"]

    if args.multiseed:
        # multi-seed evolution is the standalone confirmation mode: each dataset at
        # its natural cover size, across N independent seeds.
        seeds = tuple(range(args.multiseed))
        for name in args.datasets:
            diagnose_multiseed_evolution(name, knn=args.knn, seeds=seeds,
                                         out_dir=out_dir, figures=figures, lines=lines)
        report = "\n".join(lines)
        (out_dir / "report_multiseed.md").write_text(report)
        print(report)
        print(f"\n[diagnostics written to {out_dir}]")
        return

    if args.weight_sweep:
        for name in args.datasets:
            diagnose_weight_sweep(name, knn=args.knn, seed=args.seed, lines=lines)
        report = "\n".join(lines)
        (out_dir / "report_weight_sweep.md").write_text(report)
        print(report)
        print(f"\n[diagnostics written to {out_dir}]")
        return

    if args.overlap_law:
        csv_path = args.csv or (out_dir / "overlap_law.csv")
        manifolds = RECOVERING_OVERLAP_LAW if args.recovering_only else None
        diagnose_overlap_law(manifolds=manifolds, knn=args.knn,
                             seeds=tuple(range(args.seeds)), out_dir=out_dir,
                             figures=figures, lines=lines, csv_path=csv_path)
        report = "\n".join(lines)
        (out_dir / "report_overlap_law.md").write_text(report)
        print(report)
        print(f"\n[diagnostics written to {out_dir}]")
        return

    if args.pr_invariance:
        diagnose_pr_invariance(knn=args.knn, seeds=tuple(range(args.seeds)),
                               out_dir=out_dir, figures=figures, lines=lines)
        report = "\n".join(lines)
        (out_dir / "report_pr_invariance.md").write_text(report)
        print(report)
        print(f"\n[diagnostics written to {out_dir}]")
        return

    if args.over_opt_classify:
        csv_path = args.csv or (out_dir / "over_opt_severity.csv")
        diagnose_over_opt_classify(
            knn=args.knn, seeds=tuple(range(args.seeds)),
            loss_weights=args.loss_weights, csv_path=csv_path, lines=lines)
        report = "\n".join(lines)
        (out_dir / "report_over_opt_classify.md").write_text(report)
        print(report)
        print(f"\n[diagnostics written to {out_dir}]")
        return

    if args.severity_sweep:
        csv_path = args.csv or (out_dir / f"severity_sweep_{args.severity_sweep}.csv")
        diagnose_severity_sweep(args.severity_sweep, knn=args.knn,
                                seeds=tuple(range(args.seeds)), csv_path=csv_path,
                                lines=lines)
        report = "\n".join(lines)
        (out_dir / f"report_severity_sweep_{args.severity_sweep}.md").write_text(report)
        print(report)
        print(f"\n[diagnostics written to {out_dir}]")
        return

    if args.factor_sweep:
        fname, knob, *vals = args.factor_sweep
        if knob not in ("n_cover", "n_points"):
            parser.error("--factor-sweep knob must be n_cover or n_points")
        values = [int(v) for v in vals]
        csv_path = args.csv or (out_dir / f"factor_sweep_{fname}_{knob}.csv")
        diagnose_factor_sweep(fname, knob, values, knn=args.knn,
                              seeds=tuple(range(args.seeds)), csv_path=csv_path,
                              lines=lines)
        report = "\n".join(lines)
        (out_dir / f"report_factor_sweep_{fname}_{knob}.md").write_text(report)
        print(report)
        print(f"\n[diagnostics written to {out_dir}]")
        return

    if args.preprocess_compare:
        csv_path = args.csv or (out_dir / "preprocess_compare.csv")
        diagnose_preprocess_compare(args.preprocess_compare, knn=args.knn,
                                    seeds=tuple(range(args.seeds)), csv_path=csv_path,
                                    lines=lines)
        report = "\n".join(lines)
        (out_dir / "report_preprocess_compare.md").write_text(report)
        print(report)
        print(f"\n[diagnostics written to {out_dir}]")
        return

    if args.n_axis:
        nname, *vals = args.n_axis
        values = [int(v) for v in vals]
        csv_path = args.csv or (out_dir / f"n_axis_{nname}.csv")
        diagnose_n_axis(nname, values, knn=args.knn,
                        seeds=tuple(range(args.seeds)), csv_path=csv_path, lines=lines)
        report = "\n".join(lines)
        (out_dir / f"report_n_axis_{nname}.md").write_text(report)
        print(report)
        print(f"\n[diagnostics written to {out_dir}]")
        return

    for name in args.datasets:
        diagnose_dataset(name, n_cover=args.n_cover, knn=args.knn, seed=args.seed,
                         figures=figures, out_dir=out_dir, lines=lines)

    target_name = args.datasets[0]
    if args.evolution:
        diagnose_evolution(target_name, n_cover=args.n_cover, knn=args.knn,
                           seed=args.seed, out_dir=out_dir, figures=figures, lines=lines)
    if args.n_cover_sweep:
        diagnose_n_cover(target_name, knn=args.knn, seed=args.seed,
                         n_covers=args.n_covers, lines=lines)

    report = "\n".join(lines)
    (out_dir / "report.md").write_text(report)
    print(report)
    print(f"\n[diagnostics written to {out_dir}]")


if __name__ == "__main__":
    main()
