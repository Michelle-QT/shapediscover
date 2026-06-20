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
