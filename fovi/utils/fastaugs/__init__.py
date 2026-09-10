"""Image transforms shared by sensing and training; loaders are opt-in."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fovi.training.loader import FlashLoader

from .functional import *
from .functional_tensor import *
from .transforms import *


def __getattr__(name: str) -> type[FlashLoader]:
    if name == "FlashLoader":
        from fovi.training.loader import FlashLoader

        return FlashLoader
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
