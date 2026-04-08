"""
Fast augmentation modules for KNNConv.

This module contains fast image augmentation operations optimized for
foveated vision processing.
"""

from .functional import *
from .functional_tensor import *
from .transforms import *
try:
    from .loader import *
except:
    # non-ffcv ops only
    pass