"""Compatibility imports; inference configuration loading needs no trainer."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models.loading import ConfigCheckpoint, find_config
    from .training.trainer import Trainer

__all__ = ["Trainer", "find_config"]


def __getattr__(
    name: str,
) -> type[Trainer] | Callable[..., ConfigCheckpoint | None | bool | int | str]:
    if name.startswith("__"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if name in {"find_config", "load_config", "load_sharded_state_dict"}:
        module = import_module("fovi.models.loading")
    else:
        module = import_module("fovi.training.trainer")
    return getattr(module, name)
