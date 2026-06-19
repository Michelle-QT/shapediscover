"""Smoke + structural tests for the plotting module (the PR10 safety net).

Guarded with ``importorskip`` on the viz extras (matplotlib / networkx / pyvis /
glasbey) so the core suite stays installable without them, and uses the Agg
backend (headless). These pin that the plot entry points run and have the
expected coarse structure; they do not check visual fidelity (that still needs a
human eyeball or image baselines).
"""

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import synthetic_data as sd  # noqa: E402
from shapediscover import ShapeDiscover  # noqa: E402

N_COVER = 10


@pytest.fixture(scope="module")
def fitted():
    X = sd.sphere(200, 2)
    est = ShapeDiscover(
        n_cover=N_COVER, n_max_iter=25, verbose=False, plot_loss_curve=False,
        random_state=0,
    ).fit(X)
    est.fit_persistence(max_dimension=2, verbose=False)
    return X, est


def test_plot_pointcloud_with_function_one_panel_per_cover(fitted):
    from shapediscover.shapediscover_plot import plot_pointcloud_with_function

    X, est = fitted
    fig = plot_pointcloud_with_function(X[:, :2], est.cover_)
    assert fig is not None
    assert len(fig.axes) >= est.cover_.shape[1]  # one panel per cover element
    plt.close("all")


def test_plot_losses_returns_figure(fitted):
    from shapediscover.shapediscover_plot import plot_losses

    _, est = fitted
    fig = plot_losses(est.main_optimization_losses_, est.loss_names_)
    assert fig is not None
    plt.close("all")


def test_plot_persistence_barcode_returns_figure(fitted):
    from shapediscover.shapediscover_plot import plot_persistence_barcode

    _, est = fitted
    fig = plot_persistence_barcode(est.gudhi_persistence_diagram_)
    assert fig is not None
    plt.close("all")


def test_plot_nerve_matplotlib_returns_figure(fitted):
    pytest.importorskip("networkx")
    from shapediscover import plot_nerve

    _, est = fitted
    fig = plot_nerve(est.cover_, threshold=0.3, interactive=False)
    assert fig is not None
    assert len(fig.axes) >= 1
    plt.close("all")


def test_plot_nerve_interactive_writes_html(fitted, tmp_path, monkeypatch):
    pytest.importorskip("networkx")
    pytest.importorskip("pyvis")
    pytest.importorskip("glasbey")
    import webbrowser

    monkeypatch.setattr(webbrowser, "open", lambda *a, **k: True, raising=False)
    monkeypatch.setattr(webbrowser, "open_new_tab", lambda *a, **k: True, raising=False)
    from shapediscover import plot_nerve

    X, est = fitted
    monkeypatch.chdir(tmp_path)
    (tmp_path / "visualizations").mkdir()
    labels = np.zeros(X.shape[0], dtype=int)
    labels[: X.shape[0] // 2] = 1
    plot_nerve(est.cover_, threshold=0.3, interactive=True, labels=labels)
    assert (tmp_path / "visualizations" / "pyvis_test.html").exists()
    plt.close("all")


def test_shapediscover_plot_runs_matplotlib_branches(fitted):
    pytest.importorskip("networkx")
    from shapediscover.shapediscover_plot import shapediscover_plot

    X, est = fitted
    shapediscover_plot(est, X[:, :2], cover_threshold=0.3, interactive=False)
    plt.close("all")


def test_compute_nerve_plot_data_structure(fitted):
    pytest.importorskip("networkx")
    from shapediscover.fuzzy_cover import threshold_fuzzy_cover
    from shapediscover.shapediscover_plot import _compute_nerve_plot_data

    X, est = fitted
    labels = np.zeros(X.shape[0], dtype=int)
    labels[: X.shape[0] // 2] = 1
    data = _compute_nerve_plot_data(
        est.cover_, threshold=0.3, max_dimension=2, labels=labels,
        dummy_legend=False, seed=0,
    )
    n = data["n_vertices"]
    assert n >= 1
    assert data["positions"].shape == (n, 2)
    assert len(data["normalized_radii"]) == n
    assert np.isclose(data["normalized_radii"].max(), 1.0)
    # one fraction row per vertex, one column per class (2 classes here)
    assert data["fractions"].shape == (n, 2)
    # the vertex count equals the cover elements surviving the threshold
    surviving = threshold_fuzzy_cover(est.cover_.T, 0.3)[0].shape[0]
    assert n == surviving


def test_compute_nerve_plot_data_without_labels(fitted):
    pytest.importorskip("networkx")
    from shapediscover.shapediscover_plot import _compute_nerve_plot_data

    _, est = fitted
    data = _compute_nerve_plot_data(
        est.cover_, threshold=0.3, max_dimension=2, labels=None,
        dummy_legend=False, seed=0,
    )
    assert data["fractions"] is None
    assert data["classes"] is None
