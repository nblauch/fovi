"""Configuration and checkpoint loading without dataset or trainer imports."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Union

import hydra
import torch
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf, open_dict
from torch import nn

from ..utils import HiddenPrints

Checkpoint = dict[str, dict[str, torch.Tensor]]
# Unlike annotations, this alias is evaluated on supported Python 3.9 runtimes.
ConfigCheckpoint = tuple[DictConfig, Union[Checkpoint, None], Union[str, None]]


def default_model_dirs() -> list[str]:
    """Include research log directories only when explicitly configured."""
    folders = ["../models"]
    for name in ("FOVI_SAVE_DIR", "FOVI_SLOW_DIR"):
        if name in os.environ:
            folder = str(Path(os.environ[name]) / "logs")
            if folder not in folders:
                folders.append(folder)
    return folders


def load_sharded_state_dict(
    model_dir: str | Path,
    base_name: str = "state_dict",
    device: str | torch.device = "cuda",
) -> Checkpoint:
    """Load weights from the shard files named by a checkpoint index."""
    model_dir = Path(model_dir)
    with (model_dir / f"{base_name}.index.json").open() as stream:
        index = json.load(stream)
    weights = {}
    for shard_file in sorted(set(index["weight_map"].values())):
        weights.update(
            torch.load(model_dir / shard_file, map_location=device, weights_only=True)
        )
    return {"state_dict": weights}


def load_config(
    base_fn: str,
    load: bool,
    folder: str | Path,
    device: str | torch.device = "cuda",
) -> ConfigCheckpoint:
    """Load a local configuration and optionally its checkpoint.

    Args:
        base_fn: Model directory name or standalone Hydra config stem.
        load: Whether to load weights.
        folder: Parent directory containing the model/configuration.
        device: Device receiving checkpoint tensors.

    Returns:
        Configuration, checkpoint (or None), and model state key (or None).

    Raises:
        FileNotFoundError: No configuration exists at the requested location.
        ValueError: A requested checkpoint is absent.
        TypeError: Configuration is not a mapping.
    """
    base_dir = Path(folder) / base_fn
    if (base_dir / "resolved_config.yaml").is_file():
        cfg = OmegaConf.load(base_dir / "resolved_config.yaml")
    elif (base_dir / "hydra/config.yaml").is_file():
        cfg = OmegaConf.load(base_dir / "hydra/config.yaml")
    elif (base_dir / "config.yaml").is_file():
        cfg = OmegaConf.load(base_dir / "config.yaml")
    elif Path(f"{base_dir}.yaml").is_file():
        # Model defaults resolve against their own directory, even inside a
        # Hydra application. Preserve its live singleton, including on failure.
        application_hydra = GlobalHydra.instance()
        application_state = application_hydra.hydra
        application_hydra.clear()
        try:
            with hydra.initialize_config_dir(
                version_base=None, config_dir=str(Path(folder).resolve())
            ):
                cfg = hydra.compose(config_name=f"{base_fn}.yaml")
        finally:
            application_hydra.clear()
            if application_state is not None:
                application_hydra.initialize(application_state)
            GlobalHydra.set_instance(application_hydra)
    else:
        with (base_dir / "params.json").open() as stream:
            cfg = OmegaConf.create(json.load(stream))
    if not isinstance(cfg, DictConfig):
        raise TypeError(f"Model configuration for {base_fn!r} must be a mapping")
    if not load:
        return cfg, None, None
    if (base_dir / "state_dict.index.json").is_file():
        return cfg, load_sharded_state_dict(base_dir, device=device), "state_dict"
    for filename, model_key in (
        ("state_dict.pth", "state_dict"),
        ("final_weights.pth", "state_dict"),
        ("model.pth", "model"),
    ):
        path = base_dir / filename
        if path.is_file():
            return (
                cfg,
                torch.load(path, map_location=device, weights_only=True),
                model_key,
            )
    raise ValueError(f"Model {base_fn!r} state_dict not found in {base_dir}")


def find_config(
    base_fn: str,
    load: bool,
    model_dirs: Sequence[str | Path] | None = None,
    device: str | torch.device = "cuda",
) -> ConfigCheckpoint:
    """Load local/Hub models or an explicitly selected ``wandb://`` snapshot.

    An existing local model with malformed configuration or missing weights raises
    immediately. It must not silently select a different checkpoint from the Hub.
    """
    model_path = resolve_model_path(base_fn, model_dirs)
    return load_config(model_path.name, load, model_path.parent, device=device)


def resolve_model_path(
    base_fn: str,
    model_dirs: Sequence[str | Path] | None = None,
    *,
    expected_checkpoint_md5: str | None = None,
) -> Path:
    """Resolve a local model, Hub identifier, or ``wandb://`` run to a local path.

    W&B runs may be selected by exact display name or run ID, with an optional
    ``#filename.pth`` (default ``model.pth``). Their checkpoint and embedded config
    form an immutable cached snapshot. Resolve once and reuse the returned path
    when configuring a sensor and loading its perception model together.
    ``expected_checkpoint_md5`` pins a W&B URI to a previous snapshot's checksum
    (from ``source.json``); it is not used for local paths or Hub identifiers.
    """
    if base_fn.startswith("wandb:"):
        from .wandb import download_wandb_model

        return download_wandb_model(
            base_fn, expected_checkpoint_md5=expected_checkpoint_md5
        )
    if expected_checkpoint_md5 is not None:
        raise ValueError("expected_checkpoint_md5 requires a wandb:// model URI")
    folders = default_model_dirs() if model_dirs is None else model_dirs
    for folder in folders:
        base_dir = Path(folder) / base_fn
        if base_dir.exists() or Path(f"{base_dir}.yaml").exists():
            return base_dir.resolve()
    from .hub import download_model

    return Path(download_model(base_fn)).resolve()


def get_model_from_base_fn(
    base_fn: str,
    load: bool = True,
    load_strict: bool = True,
    quiet: bool = False,
    device: str | torch.device = "cuda",
    model_dirs: Sequence[str | Path] = ("../models",),
    fovinet_cls: type[nn.Module] | None = None,
    **kwargs: str | float | bool | None,
) -> nn.Module:
    """Construct a model and restore weights without creating a Trainer.

    Args:
        base_fn: Local model name, HuggingFace identifier, or ``wandb://`` run URI.
        load: Whether to restore weights.
        load_strict: Passed to the model's state-dict loader.
        quiet: Suppress model construction output.
        device: Device for model construction and checkpoint tensors.
        model_dirs: Local search locations, in priority order.
        fovinet_cls: Model constructor; defaults to FoviNet.
        **kwargs: Dotted configuration overrides.

    Returns:
        Constructed model with optional checkpoint weights.
    """
    if fovinet_cls is None:
        from .fovinet import FoviNet

        fovinet_cls = FoviNet
    with HiddenPrints(quiet):
        cfg, state_dict, model_key = find_config(
            base_fn, load, model_dirs, device=device
        )
        if "logging" in cfg:
            cfg.logging.use_wandb = 0
        load_head = True
        for key, value in kwargs.items():
            if key == "data.num_classes" and cfg.data.num_classes != value:
                load_head = False
            with open_dict(cfg):
                OmegaConf.update(cfg, key, value)
        if "logging.base_fn" in kwargs and "logging.folder" not in kwargs:
            if "FOVI_SAVE_DIR" not in os.environ:
                raise ValueError(
                    "Overriding logging.base_fn requires an explicit logging.folder "
                    "override or the FOVI_SAVE_DIR environment variable."
                )
            cfg.logging.folder = str(
                Path(os.environ["FOVI_SAVE_DIR"]) / "logs" / cfg.logging.base_fn
            )
        model = fovinet_cls(cfg, device=str(device))
        if load:
            assert state_dict is not None and model_key is not None
            weights = {}
            for key, value in state_dict[model_key].items():
                if "head." in key and not load_head:
                    continue
                weights[key.removeprefix("module.")] = value
            model.load_state_dict(weights, strict=load_strict)
    return model
