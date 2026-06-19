"""Shared test helpers for reading homology out of a persistence diagram."""

import numpy as np


def diagram_to_intervals(gudhi_diagram, max_dimension):
    """Group a gudhi persistence diagram into per-dimension interval arrays.

    Parameters
    ----------
    gudhi_diagram : list of (dim, (birth, death))
        Output of ``FuzzyCoverPersistence.fit_transform`` (i.e. gudhi's
        ``SimplexTree.persistence``).
    max_dimension : int
        Highest homological dimension to keep.

    Returns
    -------
    list of ndarray
        ``intervals[d]`` has shape ``(n_d, 2)`` with the birth/death pairs in
        dimension ``d`` (an empty ``(0, 2)`` array when there are none).
    """
    per_dimension = [[] for _ in range(max_dimension + 1)]
    for dimension, (birth, death) in gudhi_diagram:
        if dimension <= max_dimension:
            per_dimension[dimension].append([birth, death])
    return [
        np.array(rows) if rows else np.empty((0, 2)) for rows in per_dimension
    ]


def homology_recovery_quotient(intervals, target_betti, n_bins=1000):
    """Fraction of the filtration whose Betti numbers equal ``target_betti``.

    This is the homology-recovery metric used in the paper experiments (see
    ``examples/experiment_functions.correct_homology_quotient``): discretize the
    filtration into ``n_bins`` values and report the fraction of values at which
    the Betti numbers match ``target_betti`` exactly in every dimension.
    """
    target_betti = np.asarray(target_betti)
    if len(intervals) != len(target_betti):
        raise ValueError("intervals and target_betti must have the same length")

    finite = lambda intervals_array: intervals_array[intervals_array < np.inf]
    nonempty = [pd for pd in intervals if len(pd) > 0]
    min_value = min(np.min(pd) for pd in nonempty)
    max_value = max(np.max(finite(pd)) for pd in nonempty if len(finite(pd)) > 0)

    grid = np.linspace(min_value, max_value, n_bins)
    betti = np.zeros((n_bins, len(target_betti)), dtype=int)
    for dimension, pd in enumerate(intervals):
        for birth, death in pd:
            start = np.searchsorted(grid, birth)
            end = np.searchsorted(grid, death)
            betti[start:end, dimension] += 1

    return float(np.mean(np.all(betti == target_betti, axis=1)))
