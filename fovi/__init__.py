"""Foveated sensing and KNN primitives, with optional models and training.

Historical root trainer exports are resolved only when requested.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import TYPE_CHECKING

from ._version import __version__

if TYPE_CHECKING:
    from .training.loading import get_trainer_from_base_fn
    from .training.trainer import Trainer

__all__ = [
    "__version__",
    "Trainer",
    "get_trainer_from_base_fn",
]


def __getattr__(
    name: str,
) -> type[Trainer] | Callable[..., Trainer]:
    modules = {
        "Trainer": "fovi.training.trainer",
        "get_trainer_from_base_fn": "fovi.training.loading",
    }
    if name not in modules:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(modules[name]), name)
    globals()[name] = value
    return value
