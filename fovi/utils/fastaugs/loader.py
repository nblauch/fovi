"""Compatibility import for :mod:`fovi.training.loader`."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("fovi.training.loader")
