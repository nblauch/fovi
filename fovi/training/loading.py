"""Restore training state from an existing experiment."""

from __future__ import annotations

from collections.abc import Sequence

from numba.core.config import NUMBA_NUM_THREADS
from omegaconf import OmegaConf, open_dict

from ..models.loading import find_config
from ..paths import SAVE_DIR, SLOW_DIR
from ..utils import HiddenPrints
from .trainer import Trainer


def get_trainer_from_base_fn(
    base_fn: str,
    load: bool = True,
    load_strict: bool = True,
    quiet: bool = False,
    allow_distributed: bool = False,
    gpu: int | None = 0,
    model_dirs: Sequence[str] = ("../models", SAVE_DIR + "/logs", SLOW_DIR + "/logs"),
    **kwargs: str | float | bool | None,
) -> Trainer:
    """
    Get a Trainer instance based on a base filename and optional parameters.

    This function loads a model configuration and optionally its weights from a specified directory,
    creates a Trainer instance with the loaded configuration, and returns it.

    Args:
        base_fn (str): The base filename to look for in the logs directory.
        load (bool, optional): Whether to load the model weights. Defaults to True.
        load_strict (bool, optional): Whether to strictly enforce matching keys when loading weights. Defaults to True.
        quiet (bool, optional): Whether to suppress print statements. Defaults to False.
        allow_distributed (bool, optional): Whether to allow distributed training configuration. Defaults to False.
        **kwargs: Additional keyword arguments to override or add to the configuration.

    Returns:
        Trainer: An instance of Trainer with the specified configuration and optionally loaded weights.

    Note:
        The function searches for the model in both SLOW_DIR and SAVE_DIR.
        It prioritizes loading final weights over non-final weights if available.
    """
    with HiddenPrints(quiet):
        cfg, state_dict, model_key = find_config(
            base_fn, load=load, model_dirs=model_dirs
        )

        cfg.logging.use_wandb = 0

        # we changed from specifying foveal diameter "fovea" ($2a$) to specifying $a$ directly
        if not hasattr(cfg.saccades, "cmf_a"):
            cfg.saccades.cmf_a = cfg.saccades.fovea / 2

        load_head = True
        for k, v in kwargs.items():
            if k == "data.num_classes" and cfg.data.num_classes != v:
                load_head = False  # changing # of classes, this will mess up state dict loading, so we don't load the head
            with open_dict(cfg):
                OmegaConf.update(cfg, k, v)

        if not allow_distributed and cfg.training.distributed:
            prev_num_workers = cfg.data.num_workers
            prev_world_size = cfg.dist.world_size
            if prev_num_workers is None:
                cfg.data.num_workers = NUMBA_NUM_THREADS - 2
            else:
                cfg.data.num_workers = prev_num_workers // prev_world_size
            cfg.dist.world_size = 1
            cfg.dist.ngpus = 1
            cfg.dist.nodes = 1
            cfg.training.distributed = 0

        if "logging.base_fn" in kwargs and "logging.folder" not in kwargs:
            # update logging folder if updating base_fn
            cfg.logging.folder = f"{SAVE_DIR}/logs/{cfg.logging.base_fn}"

        trainer = Trainer(gpu, cfg, load_checkpoint=False)

        if load:
            if not allow_distributed:
                # check for any module. keys:
                keys = [model_key]
                if "probes" in state_dict:
                    keys.append("probes")
                for this_key in keys:
                    new_state_dict = {}
                    for k, v in state_dict[this_key].items():
                        if "head." in k and not load_head:
                            continue
                        if k.startswith("module."):
                            new_state_dict[k[7:]] = v
                        else:
                            new_state_dict[k] = v
                    state_dict[this_key] = new_state_dict

            trainer.model.load_state_dict(state_dict[model_key], strict=load_strict)
            if load_head and "probes" in state_dict:
                trainer.probes.load_state_dict(state_dict["probes"], strict=load_strict)

    return trainer
