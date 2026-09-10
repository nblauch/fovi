"""Compatibility import for :mod:`fovi.models.knnvit`."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("fovi.models.knnvit")
