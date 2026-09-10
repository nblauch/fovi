"""Compatibility import for :mod:`fovi.models.dinov3`."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("fovi.models.dinov3")
