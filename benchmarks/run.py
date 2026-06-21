"""CLI entry point for the benchmark harness.

Examples
--------
Run the default ShapeDiscover suite over 3 seeds and print a seed-averaged summary::

    python -m benchmarks.run --seeds 3

Restrict to specific datasets / axes and tweak the cover size::

    python -m benchmarks.run --datasets circle sphere2 torus --axes topology --n-cover 20

Results are written to ``benchmarks/results/<tag>.csv`` (long format).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import datasets as ds_mod
from .datasets import CLUSTERING, EMBEDDING, TOPOLOGY
from .methods import FuzzyCoverMethod, ShapeDiscoverMethod
from .tda_baselines import TDA_METHODS
from .dr_baselines import DR_METHODS
from .runner import run_suite, summarize

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Default suite spanning all three axes. Synthetic + sklearn-built-in sets are
# always available; the small real sets (diabetes, mice_protein, dynamical_system)
# load if their data files are present and are skipped gracefully otherwise.
DEFAULT_DATASETS = [
    # topology (known Betti)
    "circle", "sphere2", "sphere3", "torus", "two_circles",
    # higher-dimensional / product manifolds (known Betti)
    "clifford_torus", "torus3", "s2_times_s1", "sphere4", "clifford_torus_amb50",
    # more manifolds: anisotropy control, CP^2, figure-eight, linked circles
    "anisotropic_torus", "cp2", "figure_eight", "linked_circles",
    # non-orientable (read over Z/2): RP^2, Klein bottle
    "rp2", "klein_bottle",
    # topological negative control (no topology; must report [1,0,0])
    "noise",
    # synthetic manifolds (embedding / DR)
    "swiss_roll", "s_curve",
    # classical clustering
    "blobs", "moons", "nested_circles", "iris", "wine", "digits",
    # small real datasets
    "diabetes", "mice_protein", "dynamical_system",
    # image + single-cell real data (loaders skip if the data file is absent)
    "mnist", "fashion_mnist", "cifar10", "celegans", "seurat", "rat_brain",
]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="ShapeDiscover benchmark harness")
    p.add_argument("--datasets", nargs="+", default=None,
                   help="dataset names (default: a small built-in suite)")
    p.add_argument("--axes", nargs="+", default=[TOPOLOGY, CLUSTERING, EMBEDDING],
                   choices=[TOPOLOGY, CLUSTERING, EMBEDDING])
    p.add_argument("--seeds", type=int, default=3, help="number of seeds (0..N-1)")
    p.add_argument("--n-cover", type=int, default=15)
    p.add_argument("--knn", type=int, default=15)
    p.add_argument("--regularization", type=float, default=10.0,
                   help="(Lite only) regularization weight; ignored by the full ShapeDiscover default")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--label-mode", default="components", choices=["components", "argmax"])
    p.add_argument("--lite", action="store_true",
                   help="use ShapeDiscoverLite (drops geometry/topology losses, faster) "
                        "instead of the full ShapeDiscover default")
    p.add_argument("--method", default="shapediscover",
                   choices=["shapediscover", "fuzzy_cover", "alpha", "rips", "witness",
                            "umap", "pca"],
                   help="cover method (shapediscover / fuzzy_cover) or a baseline "
                        "(alpha/rips/witness topology, umap/pca embedding)")
    p.add_argument("--space", default="spectral", choices=["spectral", "euclidean"],
                   help="(fuzzy_cover) feature space for fuzzy c-means: spectral "
                        "(Laplacian eigenmaps, = ShapeDiscover's init) or euclidean (raw features)")
    p.add_argument("--fuzzifier", type=float, default=2.0,
                   help="(fuzzy_cover) fuzzy c-means exponent m (2.0 standard; ->1 hard, larger softer)")
    p.add_argument("--n-landmarks", type=int, default=None,
                   help="(alpha/rips/witness) farthest-point landmark budget (method default if unset)")
    p.add_argument("--tag", default="run", help="output filename stem")
    p.add_argument("--profile-memory", action="store_true",
                   help="record peak Python memory per phase (tracemalloc; adds overhead)")
    p.add_argument("--list", action="store_true", help="list available datasets and exit")
    args = p.parse_args(argv)

    if args.list:
        print("available datasets:")
        for nm in ds_mod.available():
            try:
                ds = ds_mod.load(nm)
                print(f"  {nm:18s} axes={ds.axes} betti={ds.target_betti} n={ds.n_points}")
            except Exception as exc:
                print(f"  {nm:18s} (unavailable: {type(exc).__name__})")
        return 0

    datasets = args.datasets or DEFAULT_DATASETS
    if args.method == "fuzzy_cover":
        method_class = FuzzyCoverMethod
        base_params = dict(
            n_cover=args.n_cover,
            knn=args.knn,
            space=args.space,
            fuzzifier=args.fuzzifier,
            threshold=args.threshold,
            label_mode=args.label_mode,
        )
    elif args.method in TDA_METHODS:
        method_class = TDA_METHODS[args.method]
        base_params = {} if args.n_landmarks is None else dict(n_landmarks=args.n_landmarks)
    elif args.method in DR_METHODS:
        method_class = DR_METHODS[args.method]
        base_params = dict(knn=args.knn) if args.method == "umap" else {}
    else:
        method_class = ShapeDiscoverMethod
        base_params = dict(
            n_cover=args.n_cover,
            knn=args.knn,
            regularization=args.regularization,
            threshold=args.threshold,
            label_mode=args.label_mode,
            lite=args.lite,
        )
    out_csv = RESULTS_DIR / f"{args.tag}.csv"

    df = run_suite(
        dataset_names=datasets,
        method_class=method_class,
        base_params=base_params,
        seeds=tuple(range(args.seeds)),
        axes=tuple(args.axes),
        out_csv=out_csv,
        profile_memory=args.profile_memory,
    )

    if len(df):
        summary = summarize(df)
        pd.set_option("display.max_rows", None, "display.width", 140)
        print("\n=== seed-averaged summary (mean +/- std) ===")
        for axis in list(args.axes) + ["cost"]:
            sub = summary[summary["axis"] == axis]
            if len(sub):
                print(f"\n[{axis}]")
                view = sub.assign(
                    value=sub.apply(
                        lambda r: f"{r['mean']:.3f} +/- {r['std']:.3f}"
                        if pd.notna(r["std"]) else f"{r['mean']:.3f}",
                        axis=1,
                    )
                ).pivot_table(index="dataset", columns="metric", values="value", aggfunc="first")
                print(view.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
