"""Cover / nerve diagnostics.

Reusable tooling to look *inside* a learned fuzzy cover and its nerve, rather
than only at the end-to-end homology-recovery number. Built to answer "what
exactly is wrong with the output cover/nerve" before attempting any more fixes
(three blind algorithm levers all hit the same wall: every adaptive change helps
the simple manifolds and breaks the torus).

Everything here works from a fuzzy cover in the library-internal
``(n_cover, n_points)`` orientation (the public ``cover_`` transposed), so it is
agnostic to how the cover was produced. The functions are pure and return plain
dicts / numpy arrays; plotting is kept separate at the bottom and is optional.

The diagnostics, matching the project plan:

- :func:`surviving_structure`     active cover elements + nerve degree/intersection
                                  distribution (sparse vs complete-graph dense).
- :func:`betti_trajectory`        Betti vector at every filtration value; where (if
                                  anywhere) the target Betti holds; the spurious
                                  features and in which dimension.
- :func:`simplex_table`           per-simplex birth, L1 volume (mass), and support
                                  count, the input to the prune-by-volume test.
- :func:`filtration_recovery`     recovery quotient + Betti trajectory under a chosen
                                  *filtration* of the nerve (membership birth vs
                                  volume vs support), the operational volume-vs-birth
                                  separability test.
- :func:`optimization_evolution`  replay a fit (``historical_outputs_``) and track how
                                  the nerve structure / Betti trajectory evolve.
- :func:`n_cover_sweep`           how the nerve structure changes with cover size.
"""

from __future__ import annotations

import numba as nb
import numpy as np
from scipy.special import comb

from shapediscover.fuzzy_cover import (
    _enumerate_simplices_of_dimension,
    simplex_to_psimplex_numpy,
)
from .metrics import bar_dominance, homology_recovery_quotient


# --------------------------------------------------------------------------- #
# Cover finalization
# --------------------------------------------------------------------------- #

def finalize_cover_internal(precover_internal: np.ndarray) -> np.ndarray:
    """Apply the final ``p=inf`` normalization to an internal-orientation cover.

    ``historical_outputs_`` snapshots are ``p=simplex_p`` normalized (the live
    optimization variable), whereas the public ``cover_`` carries an additional
    ``p=inf`` normalization (max membership 1 per point). Replaying a snapshot
    through the nerve must apply the same final step to be comparable.
    """
    return simplex_to_psimplex_numpy(np.asarray(precover_internal, dtype=float),
                                     p=float("inf"))


# --------------------------------------------------------------------------- #
# Per-simplex birth / volume / support (the prune-by-volume input)
# --------------------------------------------------------------------------- #

@nb.njit(parallel=True)
def _simplex_birth_volume(functions, simplices, births, mass, support):
    """For each simplex record three quantities over the point cloud.

    - ``births[k]``  = max over points of the min membership across vertices
      (the current nerve filtration value = peak overlap strength).
    - ``mass[k]``    = sum over points of that per-point min (the L1 "volume" of
      the fuzzy intersection).
    - ``support[k]`` = number of points where the min is strictly positive (how
      many points actually lie in the intersection).

    Parallel over simplices; the per-point min and the max/sum reductions are
    exact, so the result is independent of the thread count.
    """
    r = simplices.shape[1]
    n_points = functions.shape[1]
    for k in nb.prange(simplices.shape[0]):
        best = 0.0
        total = 0.0
        count = 0
        for x_index in range(n_points):
            m = functions[simplices[k, 0], x_index]
            for vi in range(1, r):
                v = functions[simplices[k, vi], x_index]
                if v < m:
                    m = v
            if m > 0.0:
                total += m
                count += 1
                if m > best:
                    best = m
        births[k] = best
        mass[k] = total
        support[k] = count


def simplex_table(cover_internal: np.ndarray, dimension: int) -> dict:
    """Birth / volume (mass) / support of every ``dimension``-simplex of the nerve.

    Returns a dict with ``simplices`` (``(n_simplices, dimension+1)`` int array of
    cover-element indices), ``births``, ``mass``, ``support`` (each
    ``(n_simplices,)``), filtered to the nonempty simplices (birth > 0). The
    arrays are the joint ``(birth, volume)`` data the prune-by-volume hypothesis
    is tested on.
    """
    functions = np.ascontiguousarray(np.asarray(cover_internal, dtype=np.float64))
    n_cover = functions.shape[0]
    r = dimension + 1
    n_simplices = comb(n_cover, r, exact=True)

    simplices = np.zeros((n_simplices, r), dtype=np.int64)
    _enumerate_simplices_of_dimension(n_cover, r, simplices)

    births = np.zeros(n_simplices, dtype=np.float64)
    mass = np.zeros(n_simplices, dtype=np.float64)
    support = np.zeros(n_simplices, dtype=np.int64)
    _simplex_birth_volume(functions, simplices, births, mass, support)

    nonempty = births > 0
    return {
        "dimension": dimension,
        "simplices": simplices[nonempty],
        "births": births[nonempty],
        "mass": mass[nonempty],
        "support": support[nonempty],
        "n_possible": int(n_simplices),
        "n_nonempty": int(nonempty.sum()),
    }


# --------------------------------------------------------------------------- #
# Surviving structure: active elements + nerve degree distribution
# --------------------------------------------------------------------------- #

def surviving_structure(cover_internal: np.ndarray, max_dimension: int = 2,
                        eps: float = 1e-12) -> dict:
    """Nerve density summary: active cover elements + per-dimension simplex counts.

    For each dimension ``d`` up to ``max_dimension`` report how many of the
    ``C(n_cover, d+1)`` possible ``d``-simplices are nonempty (intersection
    fill ratio), plus the *strong* count (those whose birth exceeds ``0.5``, a
    genuine overlap rather than a partition-of-unity tail). A softmax cover has
    full support, so the raw fill ratio is ~1 everywhere (every intersection has a
    witness); the strong counts and the volume distribution are what distinguish a
    torus-faithful nerve from a complete-graph-dense one.

    ``n_active`` is the number of cover elements that are some point's argmax (the
    elements that actually carry the partition); ``degree`` summarizes the
    strong-edge degree per active element.
    """
    functions = np.asarray(cover_internal, dtype=float)
    n_cover, n_points = functions.shape

    argmax_element = np.argmax(functions, axis=0)
    active = np.unique(argmax_element)
    n_active = int(active.size)

    per_dim = []
    edge_tables = None
    for d in range(max_dimension + 1):
        tbl = simplex_table(functions, d)
        births = tbl["births"]
        per_dim.append({
            "dimension": d,
            "n_possible": tbl["n_possible"],
            "n_nonempty": tbl["n_nonempty"],
            "fill_ratio": tbl["n_nonempty"] / tbl["n_possible"] if tbl["n_possible"] else 0.0,
            "n_strong": int((births > 0.5).sum()),
            "birth_median": float(np.median(births)) if births.size else 0.0,
            "birth_p90": float(np.quantile(births, 0.9)) if births.size else 0.0,
        })
        if d == 1:
            edge_tables = tbl

    # strong-edge degree distribution over cover elements
    degree = {}
    if edge_tables is not None:
        strong = edge_tables["births"] > 0.5
        strong_edges = edge_tables["simplices"][strong]
        deg = np.zeros(n_cover, dtype=int)
        for a, b in strong_edges:
            deg[a] += 1
            deg[b] += 1
        deg_active = deg[active] if n_active else deg
        degree = {
            "strong_edges": int(strong_edges.shape[0]),
            "degree_mean": float(np.mean(deg_active)) if deg_active.size else 0.0,
            "degree_median": float(np.median(deg_active)) if deg_active.size else 0.0,
            "degree_max": int(np.max(deg_active)) if deg_active.size else 0,
            "degree_min": int(np.min(deg_active)) if deg_active.size else 0,
        }

    return {
        "n_cover": int(n_cover),
        "n_points": int(n_points),
        "n_active": n_active,
        "per_dimension": per_dim,
        "strong_degree": degree,
    }


# --------------------------------------------------------------------------- #
# Nerve construction under a chosen filtration (membership / volume / support)
# --------------------------------------------------------------------------- #

def _simplex_tree_from_tables(tables: list[dict], values_key: str, transform,
                              prune_key: str | None = None, prune_cut: float = 0.0):
    """Build a gudhi SimplexTree from per-dimension simplex tables.

    ``transform`` maps the chosen per-simplex quantity (``values_key``:
    ``"births"``, ``"mass"`` or ``"support"``) to a gudhi filtration value
    (lower = earlier). For all three quantities a *larger* value means a stronger
    / earlier simplex, and the quantity is monotone non-increasing from a face to
    its cofaces (a coface intersects no more than its faces), so any decreasing
    ``transform`` yields a valid (non-decreasing) filtration.

    If ``prune_key`` is given, simplices with ``tbl[prune_key] < prune_cut`` are
    dropped before insertion. Because that quantity is also monotone (a face has
    at least its coface's value), a single global ``prune_cut`` keeps the result a
    valid simplicial complex (closed under faces): any kept simplex's faces have a
    value at least as large, so they are kept too.
    """
    import gudhi

    st = gudhi.SimplexTree()
    for tbl in tables:
        simplices = tbl["simplices"]
        if simplices.shape[0] == 0:
            continue
        keep = np.ones(simplices.shape[0], dtype=bool)
        if prune_key is not None:
            keep = tbl[prune_key].astype(float) >= prune_cut
        if not keep.any():
            continue
        values = transform(tbl[values_key].astype(float)[keep])
        st.insert_batch(simplices[keep].T, values)
    st.make_filtration_non_decreasing()
    return st


_FILTRATIONS = {
    # membership birth, the current nerve convention: f = -log(birth) over [0, inf)
    "birth": ("births", lambda v: -np.log(v)),
    # L1 volume of the fuzzy intersection: f = -log(mass)
    "mass": ("mass", lambda v: -np.log(np.maximum(v, 1e-300))),
    # intersection support count: f = -log(support)
    "support": ("support", lambda v: -np.log(np.maximum(v, 1.0))),
}


def filtration_recovery(cover_internal: np.ndarray, target_betti, *,
                        filtration: str = "birth", max_dimension: int | None = None,
                        n_bins: int = 1000, prune_key: str | None = None,
                        prune_cut: float = 0.0, tables=None) -> dict:
    """Recovery quotient + Betti trajectory under a chosen nerve filtration.

    ``filtration`` selects which per-simplex quantity orders the nerve:

    - ``"birth"``   the membership convention used everywhere else (the baseline);
    - ``"mass"``    the L1 volume of the fuzzy intersection (the prune-by-volume
      hypothesis: a broad real overlap can have small birth yet large volume, so a
      volume order may separate real from spurious where the birth order cannot);
    - ``"support"`` the number of points in the intersection (volume's cruder
      cousin).

    Comparing ``"birth"`` against ``"mass"`` is the operational separability test:
    if the volume order recovers ``target_betti`` in a window where the birth order
    does not (higher quotient / wider window), prune-by-volume is a candidate fix;
    if not, it is cleanly ruled out.
    """
    if max_dimension is None:
        max_dimension = len(target_betti) - 1
    # build one dimension higher than read: the (max_dimension+1)-simplices kill
    # the max_dimension-cycles, so the top-dimension persistence is correct. This
    # mirrors fit_persistence, which builds fuzzy_cover_to_filtered_complex at
    # max_dimension + 1.
    if tables is None:
        tables = [simplex_table(cover_internal, d) for d in range(max_dimension + 2)]
    values_key, transform = _FILTRATIONS[filtration]
    st = _simplex_tree_from_tables(tables, values_key, transform,
                                   prune_key=prune_key, prune_cut=prune_cut)
    st.persistence()
    intervals = [
        np.asarray(st.persistence_intervals_in_dimension(d)).reshape(-1, 2)
        for d in range(max_dimension + 1)
    ]
    traj = betti_trajectory(intervals, target_betti, n_bins=n_bins)
    per_dim, overall = bar_dominance(intervals, target_betti)
    return {
        "filtration": filtration,
        "intervals": intervals,
        "complex_size": int(st.num_simplices()),
        "recovery_quotient": homology_recovery_quotient(intervals, target_betti, n_bins),
        "bar_dominance_min": float(overall),
        "bar_dominance_per_dim": [float(x) for x in per_dim],
        "trajectory": traj,
    }


def prune_sweep(cover_internal: np.ndarray, target_betti, *, prune_key: str = "mass",
                quantiles=(0.0, 0.25, 0.5, 0.75, 0.9), max_dimension: int | None = None,
                n_bins: int = 1000) -> dict:
    """Birth-filtration recovery while pruning low-``prune_key`` simplices.

    The faithful test of the prune-by-volume hypothesis: keep the membership
    (``birth``) filtration that works, but first drop simplices whose volume
    (``"mass"``) or ``"support"`` falls below a global cut, sweeping the cut over
    quantiles of the top-dimension simplices' values. If pruning low-volume
    simplices *raises* recovery or *widens* the target window, volume is
    separating real overlap from spurious thin intersections; if recovery only
    falls, volume-pruning is cleanly ruled out.
    """
    if max_dimension is None:
        max_dimension = len(target_betti) - 1
    tables = [simplex_table(cover_internal, d) for d in range(max_dimension + 2)]
    # cut candidates from the top-dimension simplices' value distribution
    top_vals = tables[max_dimension][prune_key].astype(float)
    rows = []
    for q in quantiles:
        cut = float(np.quantile(top_vals, q)) if top_vals.size else 0.0
        rec = filtration_recovery(cover_internal, target_betti, filtration="birth",
                                  max_dimension=max_dimension, n_bins=n_bins,
                                  prune_key=prune_key, prune_cut=cut, tables=tables)
        rows.append({
            "quantile": q,
            "cut": cut,
            "recovery_quotient": rec["recovery_quotient"],
            "bar_dominance_min": rec["bar_dominance_min"],
            "complex_size": rec["complex_size"],
            "window": rec["trajectory"]["window"],
            "best_slice_betti": rec["trajectory"]["best_slice_betti"],
        })
    return {"prune_key": prune_key, "rows": rows}


# --------------------------------------------------------------------------- #
# Filtered-Betti trajectory
# --------------------------------------------------------------------------- #

def betti_trajectory(intervals, target_betti, n_bins: int = 1000) -> dict:
    """Betti vector at every filtration value, and where the target holds.

    Returns the filtration ``grid``, the ``betti`` array ``(n_bins, n_dim)``, the
    boolean ``match`` mask (where the Betti vector equals ``target_betti``), the
    matched filtration ``window`` (min/max grid value, or ``None``), the
    ``recovery_quotient`` (fraction of the grid that matches), and a per-dimension
    ``spurious`` summary at the *most-matched* slice: how many extra bars each
    dimension carries beyond ``target_betti`` and their total persistence. This
    surfaces the over-counted-H2 question (does the torus read ``[1,2,1]`` with
    extra spurious voids?) at the cover level.
    """
    target_betti = np.asarray(target_betti)
    n_dim = len(target_betti)

    finite = lambda a: a[a < np.inf]
    nonempty = [pd for pd in intervals if len(pd) > 0]
    if not nonempty:
        return {"grid": np.array([]), "betti": np.zeros((0, n_dim), int),
                "match": np.zeros(0, bool), "window": None, "recovery_quotient": 0.0,
                "spurious": []}
    min_value = min(np.min(pd) for pd in nonempty)
    finite_vals = [finite(pd) for pd in nonempty if len(finite(pd)) > 0]
    if not finite_vals:
        return {"grid": np.array([]), "betti": np.zeros((0, n_dim), int),
                "match": np.zeros(0, bool), "window": None, "recovery_quotient": 0.0,
                "spurious": []}
    max_value = max(np.max(fv) for fv in finite_vals)

    grid = np.linspace(min_value, max_value, n_bins)
    betti = np.zeros((n_bins, n_dim), dtype=int)
    for dim, pd in enumerate(intervals):
        if dim >= n_dim:
            break
        for birth, death in pd:
            start = np.searchsorted(grid, birth)
            end = np.searchsorted(grid, death)
            betti[start:end, dim] += 1

    match = np.all(betti == target_betti, axis=1)
    window = None
    if match.any():
        idx = np.flatnonzero(match)
        window = (float(grid[idx[0]]), float(grid[idx[-1]]))

    # spurious features: at the slice closest to the target (by L1 distance over
    # dimensions), how many extra bars and how much persistence beyond target.
    l1 = np.abs(betti - target_betti).sum(axis=1)
    best_slice = int(np.argmin(l1))
    spurious = []
    for dim, pd in enumerate(intervals):
        if dim >= n_dim:
            break
        n_here = int(betti[best_slice, dim])
        extra = max(0, n_here - int(target_betti[dim]))
        persist = np.sort(pd[:, 1] - pd[:, 0])[::-1] if len(pd) else np.array([])
        spurious.append({
            "dimension": dim,
            "betti_at_best": n_here,
            "target": int(target_betti[dim]),
            "extra": extra,
            "spurious_persistence_sum": float(persist[int(target_betti[dim]):].sum())
            if len(persist) > target_betti[dim] else 0.0,
        })

    return {
        "grid": grid,
        "betti": betti,
        "match": match,
        "window": window,
        "best_slice_betti": [int(x) for x in betti[best_slice]],
        "recovery_quotient": float(np.mean(match)),
        "spurious": spurious,
    }


# --------------------------------------------------------------------------- #
# Evolution across optimization
# --------------------------------------------------------------------------- #

def optimization_evolution(X, target_betti, *, n_cover=52, knn=15, n_snapshots=12,
                           n_max_iter=250, random_state=0, max_dimension=None,
                           extra=None) -> dict:
    """Replay one fit and track the nerve structure / Betti trajectory per snapshot.

    Fits ``ShapeDiscover`` with early stopping off and ``n_saved_iterations`` set,
    then for each snapshot finalizes the cover (``p=inf``) and records the
    surviving structure plus the membership-filtration recovery. Exposes the
    over-optimization mechanism at the cover level: whether the nerve starts clean
    after the spectral init and accumulates spurious voids as elements shrink.
    """
    from shapediscover import ShapeDiscover

    if max_dimension is None:
        max_dimension = len(target_betti) - 1

    est = ShapeDiscover(
        n_cover=n_cover, knn=knn, random_state=random_state,
        n_max_iter=n_max_iter, early_stop=False,
        n_saved_iterations=n_snapshots, verbose=False, plot_loss_curve=False,
        **(extra or {}),
    )
    est.fit(X)
    snapshots = est.historical_outputs_ + [est.precover_.T]  # internal orientation
    step = max(1, int(n_max_iter / n_snapshots))

    rows = []
    for i, precover in enumerate(snapshots):
        cover = finalize_cover_internal(precover)
        struct = surviving_structure(cover, max_dimension=max_dimension)
        rec = filtration_recovery(cover, target_betti, filtration="birth",
                                  max_dimension=max_dimension)
        iteration = i * step if i < len(est.historical_outputs_) else n_max_iter
        rows.append({
            "iteration": int(iteration),
            "n_active": struct["n_active"],
            "recovery_quotient": rec["recovery_quotient"],
            "bar_dominance_min": rec["bar_dominance_min"],
            "best_slice_betti": rec["trajectory"].get("best_slice_betti"),
            "per_dimension": struct["per_dimension"],
        })
    return {"rows": rows, "n_max_iter": n_max_iter}


def aggregate_evolution(evolutions: list[dict]) -> dict:
    """Aggregate several :func:`optimization_evolution` runs (same iteration grid).

    Each run is one independent seed (data + model). Returns per-iteration median
    and min/max/IQR of the recovery quotient and active-element count, plus a
    per-seed shape classification: each trajectory's peak iteration, peak vs final
    recovery, whether the peak is interior (over-optimization) or at convergence
    (monotone improvement), and the active-element count at start vs end. The
    headline summary counts how many seeds over-optimize (an interior peak with a
    meaningful decline) vs improve to convergence.
    """
    rows_per_seed = [ev["rows"] for ev in evolutions]
    n_iters = min(len(r) for r in rows_per_seed)
    per_iteration = []
    for i in range(n_iters):
        rq = np.array([rows[i]["recovery_quotient"] for rows in rows_per_seed])
        na = np.array([rows[i]["n_active"] for rows in rows_per_seed])
        per_iteration.append({
            "iteration": int(rows_per_seed[0][i]["iteration"]),
            "recovery_median": float(np.median(rq)),
            "recovery_min": float(rq.min()),
            "recovery_max": float(rq.max()),
            "recovery_q25": float(np.quantile(rq, 0.25)),
            "recovery_q75": float(np.quantile(rq, 0.75)),
            "n_active_median": float(np.median(na)),
            "n_active_min": int(na.min()),
            "n_active_max": int(na.max()),
        })

    per_seed = []
    for rows in rows_per_seed:
        rq = np.array([r["recovery_quotient"] for r in rows])
        na = np.array([r["n_active"] for r in rows])
        peak_idx = int(np.argmax(rq))
        last_idx = len(rq) - 1
        decline = float(rq[peak_idx] - rq[-1])
        # over-optimization = a clearly interior peak that then loses recovery;
        # the 0.05 margin guards against flat / noise-level wiggles.
        interior_peak = bool(peak_idx < last_idx - 1)
        per_seed.append({
            "peak_iteration": int(rows[peak_idx]["iteration"]),
            "peak_recovery": float(rq[peak_idx]),
            "final_recovery": float(rq[-1]),
            "decline_from_peak": decline,
            "over_optimizes": bool(interior_peak and decline > 0.05),
            "n_active_start": int(na[0]),
            "n_active_end": int(na[-1]),
        })

    n_over = sum(s["over_optimizes"] for s in per_seed)
    return {
        "per_iteration": per_iteration,
        "per_seed": per_seed,
        "n_seeds": len(per_seed),
        "n_over_optimizing": int(n_over),
    }


# --------------------------------------------------------------------------- #
# n_cover sweep
# --------------------------------------------------------------------------- #

def n_cover_sweep(X, target_betti, *, n_covers=(20, 35, 52, 75), knn=15,
                  random_state=0, max_dimension=None, extra=None) -> dict:
    """How the nerve structure and recovery change with cover size ``n_cover``."""
    from shapediscover import ShapeDiscover

    if max_dimension is None:
        max_dimension = len(target_betti) - 1

    rows = []
    for n_cover in n_covers:
        est = ShapeDiscover(
            n_cover=n_cover, knn=knn, random_state=random_state,
            verbose=False, plot_loss_curve=False, **(extra or {}),
        )
        cover = np.asarray(est.fit_transform(X)).T  # internal orientation
        struct = surviving_structure(cover, max_dimension=max_dimension)
        rec = filtration_recovery(cover, target_betti, filtration="birth",
                                  max_dimension=max_dimension)
        rows.append({
            "n_cover": int(n_cover),
            "n_active": struct["n_active"],
            "recovery_quotient": rec["recovery_quotient"],
            "bar_dominance_min": rec["bar_dominance_min"],
            "complex_size": rec["complex_size"],
            "per_dimension": struct["per_dimension"],
        })
    return {"rows": rows}


# --------------------------------------------------------------------------- #
# Plotting (layout-free, optional). Imported lazily; matplotlib only.
# --------------------------------------------------------------------------- #

def plot_barcode(intervals, ax=None, title=None):
    """Layout-free barcode of a persistence diagram (one row per bar, by dim)."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    colors = ["C0", "C1", "C2", "C3", "C4"]
    finite_max = 0.0
    for pd in intervals:
        fin = pd[np.isfinite(pd[:, 1])]
        if len(fin):
            finite_max = max(finite_max, float(fin[:, 1].max()))
    cap = finite_max * 1.05 if finite_max > 0 else 1.0
    y = 0
    yticks, ylabels = [], []
    for dim, pd in enumerate(intervals):
        order = np.argsort(pd[:, 0]) if len(pd) else []
        start_y = y
        for k in order:
            birth, death = pd[k]
            death = cap if not np.isfinite(death) else death
            ax.plot([birth, death], [y, y], color=colors[dim % len(colors)], lw=1.2)
            y += 1
        if y > start_y:
            yticks.append((start_y + y) / 2)
            ylabels.append(f"H{dim}")
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_xlabel("filtration")
    if title:
        ax.set_title(title)
    return ax


def plot_birth_volume(table: dict, ax=None, title=None, color_by="support"):
    """Scatter every simplex by (birth, volume/mass); the prune-by-volume picture.

    ``color_by`` colors points by ``"support"`` (or any key in ``table``). A
    low-volume tail at non-trivial birth is the signature the prune-by-volume
    hypothesis predicts for the spurious simplices.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    c = table.get(color_by)
    sc = ax.scatter(table["births"], table["mass"], c=c, s=6, alpha=0.4,
                    cmap="viridis")
    ax.set_xlabel("birth (peak overlap)")
    ax.set_ylabel("volume (L1 mass)")
    ax.set_yscale("log")
    if c is not None:
        plt.colorbar(sc, ax=ax, label=color_by)
    if title:
        ax.set_title(title)
    return ax


def plot_betti_trajectory(traj: dict, target_betti, ax=None, title=None):
    """Betti numbers vs filtration value, with the target-match window shaded."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    grid, betti = traj["grid"], traj["betti"]
    colors = ["C0", "C1", "C2", "C3", "C4"]
    for d in range(betti.shape[1]):
        ax.step(grid, betti[:, d], where="post", color=colors[d % len(colors)],
                label=f"b{d} (target {target_betti[d]})")
    if traj.get("window"):
        ax.axvspan(traj["window"][0], traj["window"][1], color="grey", alpha=0.2)
    ax.set_xlabel("filtration")
    ax.set_ylabel("Betti number")
    ax.legend(fontsize=8)
    if title:
        ax.set_title(title)
    return ax


def plot_pointcloud_colored(X, color, ax=None, title=None, cmap="tab20"):
    """Color the original point cloud (its own faithful embedding) by ``color``.

    For 3D point clouds (torus/sphere) a 3D scatter; for >3D the first two PCA
    axes. Layout-free in the sense that it uses the data's real geometry, not a
    nerve layout (a 2-torus cannot embed faithfully in a 2D nerve layout).
    """
    import matplotlib.pyplot as plt

    X = np.asarray(X)
    if X.shape[1] > 3:
        from sklearn.decomposition import PCA
        X = PCA(n_components=3).fit_transform(X)
    if X.shape[1] == 3:
        if ax is None:
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(111, projection="3d")
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, s=6, cmap=cmap)
    else:
        if ax is None:
            _, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(X[:, 0], X[:, 1], c=color, s=6, cmap=cmap)
        ax.set_aspect("equal")
    if title:
        ax.set_title(title)
    return ax
