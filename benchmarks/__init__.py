"""ShapeDiscover benchmark harness.

A reproducible R&D instrument for assessing cover learning across three
capability axes from a single fitted cover:

- ``topology``   homology recovery from the nerve (TDA-style topological inference).
- ``clustering`` hard labels from the cover, scored against ground truth.
- ``embedding``  per-point and nerve-level structure preservation (DR / visualization).

plus *robustness* (variation across seeds, noise levels, and parameters), which is
the first-class goal: make all three axes work reliably with default parameters
across many datasets (the "robust like UMAP" target).

The pieces:

- ``datasets``  a registry of named :class:`~benchmarks.datasets.BenchmarkDataset`.
- ``methods``   adapters exposing ``labels`` / ``embedding`` / ``persistence`` behind a
  common :class:`~benchmarks.methods.Method` interface (ShapeDiscover first; external
  baselines are wired in later, see ``benchmarks/README.md``).
- ``metrics``   per-axis metric functions.
- ``runner``    runs ``(dataset, method, seed, axis)`` combinations into a tidy long
  results table (one row per metric), written to ``benchmarks/results/``.

This package is a development tool; it is not part of the shipped ``shapediscover``
wheel (see ``pyproject``'s package discovery).
"""

from . import datasets, methods, metrics, runner

__all__ = ["datasets", "methods", "metrics", "runner"]
