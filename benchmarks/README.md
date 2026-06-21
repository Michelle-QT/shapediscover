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
- `diagnose.py` CLI driving the diagnostics on a target dataset + controls, plus
  the over-optimization characterization modes (severity classification + sweeps).
- `overopt_analysis.py` reads the over-optimization study CSVs (`results/overopt_*.csv`)
  and reports which factors predict severity + the cover-level mechanism.
- `make_results_site.py` builds a self-contained `index.html` (tables computed
  directly from `results/*.csv`) for a static results page.
- `results/` output CSVs (`results/diagnostics/` holds regenerable diagnostic output;
  `results/overopt_*.csv` is the committed over-optimization study dataset).

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

Characterize over-optimization (recovery rising to an interior-iteration peak then
declining toward convergence) as a controlled multi-seed dataset, then analyze it:

```
# master severity table over the recovering manifold set (writes a tidy CSV)
python -m benchmarks.diagnose --over-opt-classify --seeds 5 --csv results/overopt_master_v0.csv
# single-factor sweeps (anisotropy ratio, donut tube/sampling, loss balance)
python -m benchmarks.diagnose --severity-sweep aniso_ratio --seeds 4 --csv results/overopt_sweep_aniso_ratio_v0.csv
# resolution / sampling knobs, and the iteration-vs-sampling-axis contrast
python -m benchmarks.diagnose --factor-sweep torus n_cover 20 35 52 64 --seeds 4
python -m benchmarks.diagnose --n-axis clifford_torus 1500 3000 6000 --seeds 4
# which factors predict severity + confirm/refute the cover-level mechanism
python -m benchmarks.overopt_analysis
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
