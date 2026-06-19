"""The package imports out of the box and exposes its public API."""

import os


def test_import_sets_numba_threading_layer():
    import shapediscover  # noqa: F401

    # Importing must leave a numba threading layer set (defaults to "workqueue")
    # to avoid the numba/torch OpenMP clash that segfaults fit_transform on macOS.
    assert os.environ.get("NUMBA_THREADING_LAYER")


def test_public_api_is_importable():
    from shapediscover import (  # noqa: F401
        FuzzyCoverPersistence,
        ShapeDiscover,
        ShapeDiscoverLite,
        plot_nerve,
        plot_pointcloud_with_function,
        shapediscover_plot,
    )

    for cls in (ShapeDiscover, ShapeDiscoverLite, FuzzyCoverPersistence):
        assert isinstance(cls, type)


def test_has_version():
    import shapediscover

    assert isinstance(shapediscover.__version__, str)
    assert shapediscover.__version__
