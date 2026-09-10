"""KNN layers and shared neural-network primitives.

Complete architectures live in :mod:`fovi.models`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import nn


def __getattr__(name: str) -> dict[str, Callable[..., nn.Module]]:
    if name == "ARCHITECTURE_REGISTRY":
        from fovi.models.architectures import ARCHITECTURE_REGISTRY

        return ARCHITECTURE_REGISTRY
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
