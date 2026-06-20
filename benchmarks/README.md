# benchmarks

Reproducible R&D harness for ShapeDiscover: assess cover learning across three
axes (topology, clustering, embedding) from one fitted cover. Development tool;
not shipped in the `shapediscover` wheel.

## Layout

- `datasets.py` registry of named `BenchmarkDataset` (point cloud + optional
  labels + optional target Betti + supported `axes`).
- `methods.py` the `Method` interface (`labels` / `embedding` / `persistence`)
  and `ShapeDiscoverMethod`. External baselines slot in here later.
- `metrics.py` per-axis metrics (topology, clustering, embedding).
- `runner.py` one fit per `(dataset, seed)`, scores every applicable axis, emits a
  tidy long table.
- `run.py` CLI.
- `cover_diagnostics.py` look *inside* a learned cover / its nerve (surviving
  structure, filtered-Betti trajectory, per-simplex birth vs volume, volume-vs-birth
  filtration, optimization-evolution replay, `n_cover` sweep, layout-free plots).
- `diagnose.py` CLI driving the diagnostics on a target dataset + controls.
- `results/` output CSVs (`results/diagnostics/` holds regenerable diagnostic output).

## Usage

Run the default suite over 3 seeds and print a seed-averaged summary:

```
python -m benchmarks.run --seeds 3
```

List datasets; restrict datasets/axes; tweak the cover:

```
python -m benchmarks.run --list
python -m benchmarks.run --datasets circle sphere2 torus --axes topology --n-cover 20
```

From Python:

```python
from benchmarks.methods import ShapeDiscoverMethod
from benchmarks.runner import run_suite, summarize

df = run_suite(["sphere2", "digits"], ShapeDiscoverMethod,
               base_params={"n_cover": 15}, seeds=(0, 1, 2))
print(summarize(df))
```

Diagnose what a cover / nerve actually looks like (target + controls, figures and a
markdown report under `results/diagnostics/`):

```
python -m benchmarks.diagnose --evolution --n-cover-sweep
```

Results are long format (one row per metric): `dataset, method, axis, metric,
value, seed, n_points, n_features, fit_runtime_s, n_active_cover, params,
sd_version, timestamp`.

## Notes

- Run from the repo root so `benchmarks` and `shapediscover` both import.
- Needs the package plus `pandas`, and `networkx` (the `viz` extra) for the
  embedding axis.
- Real datasets load from `SHAPEDISCOVER_DATA_DIR` (default: `examples/datasets`,
  then the sibling paper repo); missing files are skipped, and the large data
  files there are not committed.
- Curated milestone baselines are committed as `results/baseline_*.csv`; other
  `--tag` runs are gitignored.
