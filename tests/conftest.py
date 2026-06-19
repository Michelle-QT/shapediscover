"""Pytest configuration for the shapediscover test suite.

The synthetic point-cloud generators (``sphere``, ``torus``, ...) live in
``examples/synthetic_data.py`` rather than in the installed package, so the tests
add that directory to ``sys.path`` to reuse the canonical generators.
"""

import os
import sys

_EXAMPLES_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, "examples")
)
if _EXAMPLES_DIR not in sys.path:
    sys.path.insert(0, _EXAMPLES_DIR)
