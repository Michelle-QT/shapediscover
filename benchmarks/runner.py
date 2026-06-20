"""Run methods over datasets and emit a tidy long results table.

One fit per ``(dataset, seed)``; every applicable capability axis is then scored
off that single fit. Output is long format (one row per metric) so results are
trivial to group, pivot, and diff across R&D iterations.

Columns: ``dataset, method, axis, metric, value, seed, n_points, n_features,
fit_runtime_s, n_active_cover, params, sd_version, timestamp``.

A ``cost`` axis carries per-phase timings (``fit_s`` / ``persistence_s`` /
``clustering_s`` / ``embedding_s``), the topology cost driver ``complex_size``,
and (with ``profile_memory``) ``*_peak_mb`` peak-memory metrics, so computational
bottlenecks fall out of the same table as the quality metrics.
"""

from __future__ import annotations

import json
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import datasets as ds_mod
from . import metrics as metrics_mod
from .datasets import CLUSTERING, EMBEDDING, TOPOLOGY, BenchmarkDataset
from .methods import Method


def _sd_version() -> str:
    try:
        import shapediscover

        return getattr(shapediscover, "__version__", "unknown")
    except Exception:
        return "unknown"


def evaluate_fit(
    method: Method, ds: BenchmarkDataset, axes, seed: int = 0, profile_memory: bool = False
) -> dict:
    """Score every requested-and-supported axis off an already-fitted ``method``,
    timing each phase (and optionally its peak Python memory).

    Returns ``{axis: {metric: value}}`` including a ``"cost"`` axis carrying the
    per-phase compute seconds (``persistence_s`` / ``clustering_s`` /
    ``embedding_s``), ``complex_size`` as the topology cost driver, and (when
    ``profile_memory``) ``*_peak_mb``. Per-axis failures are caught and recorded
    as an ``{"error": ...}`` entry so one broken axis does not sink the run.
    """
    out: dict = {}
    cost: dict = {}
    runnable = set(axes) & set(ds.axes) & set(method.provides)

    def _peak_mb() -> float:
        return round(tracemalloc.get_traced_memory()[1] / 1e6, 1)

    if TOPOLOGY in runnable and ds.target_betti is not None:
        try:
            max_dim = len(ds.target_betti) - 1
            if profile_memory:
                tracemalloc.reset_peak()
            t = time.perf_counter()
            intervals, size = method.persistence(max_dim)
            cost["persistence_s"] = round(time.perf_counter() - t, 4)
            if profile_memory:
                cost["persistence_peak_mb"] = _peak_mb()
            topo = metrics_mod.topology_metrics(intervals, size, ds.target_betti)
            cost["complex_size"] = topo["complex_size"]
            out[TOPOLOGY] = topo
        except Exception as exc:
            out[TOPOLOGY] = {"error": type(exc).__name__, "error_msg": str(exc)[:200]}

    if CLUSTERING in runnable and ds.labels is not None:
        try:
            if profile_memory:
                tracemalloc.reset_peak()
            t = time.perf_counter()
            pred = method.labels()
            cost["clustering_s"] = round(time.perf_counter() - t, 4)
            if profile_memory:
                cost["clustering_peak_mb"] = _peak_mb()
            out[CLUSTERING] = metrics_mod.clustering_metrics(ds.labels, pred)
        except Exception as exc:
            out[CLUSTERING] = {"error": type(exc).__name__, "error_msg": str(exc)[:200]}

    if EMBEDDING in runnable:
        try:
            if profile_memory:
                tracemalloc.reset_peak()
            t = time.perf_counter()
            emb = method.embedding()
            cost["embedding_s"] = round(time.perf_counter() - t, 4)
            if profile_memory:
                cost["embedding_peak_mb"] = _peak_mb()
            out[EMBEDDING] = metrics_mod.embedding_metrics(ds.X, emb, seed=seed)
        except Exception as exc:
            out[EMBEDDING] = {"error": type(exc).__name__, "error_msg": str(exc)[:200]}

    out["cost"] = cost
    return out


def run_suite(
    dataset_names,
    method_class,
    base_params: dict | None = None,
    seeds=(0, 1, 2),
    axes=(TOPOLOGY, CLUSTERING, EMBEDDING),
    out_csv: str | Path | None = None,
    dataset_kwargs: dict | None = None,
    profile_memory: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Fit ``method_class`` on each dataset across ``seeds`` and score all axes.

    Parameters
    ----------
    dataset_names : list[str]
        Registry keys (see ``datasets.available``).
    method_class : type[Method]
        e.g. :class:`benchmarks.methods.ShapeDiscoverMethod`. Built per seed as
        ``method_class(random_state=seed, **base_params)``.
    base_params : dict
        Method constructor kwargs (besides ``random_state``).
    seeds : iterable[int]
        Both the dataset seed and the method ``random_state`` (so data resampling
        and method stochasticity are varied together; the robustness view).
    axes : iterable[str]
        Axes to attempt (intersected per dataset/method capabilities).
    out_csv : path or None
        If given, the long table is written there.
    """
    base_params = dict(base_params or {})
    dataset_kwargs = dict(dataset_kwargs or {})
    records: list[dict] = []
    sd_version = _sd_version()
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

    for name in dataset_names:
        for seed in seeds:
            try:
                ds = ds_mod.load(name, seed=seed, **dataset_kwargs.get(name, {}))
            except Exception as exc:
                if verbose:
                    print(f"[skip] dataset {name!r} (seed {seed}): {type(exc).__name__}: {exc}")
                continue

            method = method_class(random_state=seed, **base_params)
            if profile_memory:
                tracemalloc.start()
                tracemalloc.reset_peak()
            t0 = time.perf_counter()
            try:
                method.fit(ds.X)
            except Exception as exc:
                if profile_memory:
                    tracemalloc.stop()
                if verbose:
                    print(f"[fail] fit {name!r} (seed {seed}): {type(exc).__name__}: {exc}")
                continue
            fit_dt = time.perf_counter() - t0
            fit_peak_mb = (
                round(tracemalloc.get_traced_memory()[1] / 1e6, 1) if profile_memory else None
            )

            n_active = method.n_active_cover() if hasattr(method, "n_active_cover") else None
            results = evaluate_fit(method, ds, axes, seed=seed, profile_memory=profile_memory)
            if profile_memory:
                tracemalloc.stop()
            cost = results.setdefault("cost", {})
            cost["fit_s"] = round(fit_dt, 3)
            if fit_peak_mb is not None:
                cost["fit_peak_mb"] = fit_peak_mb
            params_json = json.dumps(method.params(), sort_keys=True, default=str)

            for axis, metric_dict in results.items():
                for metric, value in metric_dict.items():
                    records.append(
                        {
                            "dataset": name,
                            "method": method.name,
                            "axis": axis,
                            "metric": metric,
                            "value": value,
                            "seed": seed,
                            "n_points": ds.n_points,
                            "n_features": ds.n_features,
                            "fit_runtime_s": round(fit_dt, 3),
                            "n_active_cover": n_active,
                            "params": params_json,
                            "sd_version": sd_version,
                            "timestamp": stamp,
                        }
                    )
            if verbose:
                axsum = ", ".join(sorted(results)) or "none"
                print(f"[ok]   {name:16s} seed {seed}  fit {fit_dt:5.1f}s  axes: {axsum}")

    df = pd.DataFrame.from_records(records)
    if out_csv is not None and len(df):
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        if verbose:
            print(f"\nwrote {len(df)} rows -> {out_csv}")
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Mean +/- std of each numeric metric over seeds, per dataset/method/axis."""
    numeric = df[pd.to_numeric(df["value"], errors="coerce").notna()].copy()
    numeric["value"] = pd.to_numeric(numeric["value"])
    g = numeric.groupby(["dataset", "method", "axis", "metric"])["value"]
    summary = g.agg(["mean", "std", "count"]).reset_index()
    return summary
