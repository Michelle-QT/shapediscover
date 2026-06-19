import os

# Set before numba initializes (i.e. before importing any numba-using module):
# the default threading layer can clash with torch's OpenMP runtime and segfault
# on macOS. "workqueue" avoids it. setdefault keeps any user-provided value.
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")

from .shapediscover import ShapeDiscover, ShapeDiscoverLite, FuzzyCoverPersistence
from .shapediscover_plot import shapediscover_plot, plot_nerve
from .shapediscover_plot import plot_nerve, plot_pointcloud_with_function


from ._version import __version__
