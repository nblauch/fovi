"""Compatibility import for :mod:`fovi.training.utils.knnprobe`."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("fovi.training.utils.knnprobe")
