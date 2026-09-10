"""Optional model architectures and inference loading (``fovi[models]``)."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import TYPE_CHECKING

from .._optional import require_dependencies

if TYPE_CHECKING:
    from torch import nn

    from .architectures import ARCHITECTURE_REGISTRY
    from .fovinet import FoviNet
    from .loading import (
        ConfigCheckpoint,
        find_config,
        get_model_from_base_fn,
        load_config,
    )

require_dependencies(
    "models",
    ("accelerate", "huggingface_hub", "hydra", "omegaconf", "timm", "transformers"),
)

__all__ = [
    "ARCHITECTURE_REGISTRY",
    "FoviNet",
    "find_config",
    "get_model_from_base_fn",
    "load_config",
]


def __getattr__(
    name: str,
) -> (
    type[FoviNet]
    | dict[str, Callable[..., nn.Module]]
    | Callable[..., nn.Module | ConfigCheckpoint]
):
    modules = {
        "FoviNet": "fovi.models.fovinet",
        "ARCHITECTURE_REGISTRY": "fovi.models.architectures",
        "find_config": "fovi.models.loading",
        "load_config": "fovi.models.loading",
        "get_model_from_base_fn": "fovi.models.loading",
    }
    if name not in modules:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(modules[name]), name)
    globals()[name] = value
    return value
