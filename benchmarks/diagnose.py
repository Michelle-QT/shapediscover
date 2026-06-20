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
# by running the controls at the torus's (over-fine) cover size.
NATURAL_N_COVER = {"torus": 52, "circle": 12, "sphere2": 24, "sphere3": 24,
                   "two_circles": 24, "clifford_torus": 52, "sphere4": 32,
                   "torus3": 64, "s2_times_s1": 40}


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


def diagnose_overlap_law(manifolds=None, *, knn, seeds, n_max_iter=200,
                         n_snapshots=8, out_dir, figures, lines):
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
    parser.add_argument("--seeds", type=int, default=3, metavar="N",
                        help="(--overlap-law) number of seeds 0..N-1")
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
        diagnose_overlap_law(knn=args.knn, seeds=tuple(range(args.seeds)),
                             out_dir=out_dir, figures=figures, lines=lines)
        report = "\n".join(lines)
        (out_dir / "report_overlap_law.md").write_text(report)
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
