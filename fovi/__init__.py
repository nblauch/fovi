"""Foveated sensing and KNN primitives, with optional models and training.

Historical root model/trainer exports are resolved only when requested.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import TYPE_CHECKING

from ._version import __version__

if TYPE_CHECKING:
    from .models.fovinet import FoviNet
    from .models.loading import (
        ConfigCheckpoint,
        find_config,
        get_model_from_base_fn,
        load_config,
    )
    from .training.loading import get_trainer_from_base_fn
    from .training.trainer import Trainer

__all__ = [
    "__version__",
    "FoviNet",
    "Trainer",
    "find_config",
    "get_model_from_base_fn",
    "get_trainer_from_base_fn",
    "load_config",
]


def __getattr__(
    name: str,
) -> type[FoviNet | Trainer] | Callable[..., FoviNet | Trainer | ConfigCheckpoint]:
    modules = {
        "FoviNet": "fovi.models.fovinet",
        "Trainer": "fovi.training.trainer",
        "find_config": "fovi.models.loading",
        "load_config": "fovi.models.loading",
        "get_model_from_base_fn": "fovi.models.loading",
        "get_trainer_from_base_fn": "fovi.training.loading",
    }
    if name not in modules:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(modules[name]), name)
    globals()[name] = value
    return value
