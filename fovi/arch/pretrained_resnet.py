"""Pretrained torchvision ResNet weight transfer and LoRA preparation."""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.distributed as dist
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from ..utils.lora import apply_lora
from .knn import KNNConvLayer
from .knnresnet import KNNResNet


def flattened_basic_blocks(model: nn.Module) -> list[nn.Module]:
    """Return ResNet BasicBlocks in forward order."""
    return [block for stage in (model.layer1, model.layer2, model.layer3, model.layer4) for block in stage]


def load_torchvision_resnet18(weights: str = "IMAGENET1K_V1") -> nn.Module:
    """Load a torchvision ResNet-18, serializing a first-time DDP download."""
    try:
        weights_enum = ResNet18_Weights[weights]
    except KeyError as exc:
        choices = ", ".join(ResNet18_Weights.__members__)
        raise ValueError(f"Unknown ResNet-18 weights {weights!r}; expected one of: {choices}") from exc

    distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if distributed else 0

    source = None
    if rank == 0:
        source = resnet18(weights=weights_enum)
    if distributed:
        dist.barrier()
    if rank != 0:
        source = resnet18(weights=weights_enum)

    return source


def _copy_batch_norm(target: nn.modules.batchnorm._BatchNorm, source: nn.modules.batchnorm._BatchNorm) -> None:
    if target.num_features != source.num_features:
        raise ValueError(
            f"BatchNorm feature mismatch: target={target.num_features}, source={source.num_features}"
        )
    target.load_state_dict(source.state_dict())


@torch.no_grad()
def load_torchvision_resnet_backbone(
    target: KNNResNet,
    source: nn.Module,
    preserve_kernel_norm: bool = False,
) -> KNNResNet:
    """Transfer a torchvision ResNet-18 backbone into a FOVI KNN ResNet-18."""
    if not isinstance(target.conv1, KNNConvLayer):
        raise TypeError(f"Expected a KNNConvLayer stem, got {type(target.conv1).__name__}")

    target.conv1.load_conv2d_weights(
        source.conv1, preserve_kernel_norm=preserve_kernel_norm
    )
    _copy_batch_norm(target.bn1, source.bn1)

    target_blocks = flattened_basic_blocks(target)
    source_blocks = flattened_basic_blocks(source)
    if len(target_blocks) != len(source_blocks):
        raise ValueError(
            f"ResNet block count mismatch: target={len(target_blocks)}, source={len(source_blocks)}"
        )

    for block_index, (target_block, source_block) in enumerate(zip(target_blocks, source_blocks)):
        for conv_name in ("conv1", "conv2"):
            target_conv = getattr(target_block, conv_name)
            source_conv = getattr(source_block, conv_name)
            if not isinstance(target_conv, KNNConvLayer):
                raise TypeError(
                    f"Expected KNNConvLayer at block {block_index}.{conv_name}, "
                    f"got {type(target_conv).__name__}"
                )
            target_conv.load_conv2d_weights(
                source_conv, preserve_kernel_norm=preserve_kernel_norm
            )

        _copy_batch_norm(target_block.norm1, source_block.bn1)
        _copy_batch_norm(target_block.norm2, source_block.bn2)

        target_downsample = target_block.downsample
        source_downsample = source_block.downsample
        if (target_downsample is None) != (source_downsample is None):
            raise ValueError(f"Downsample mismatch in flattened block {block_index}")
        if target_downsample is not None:
            target_downsample[0].load_conv2d_weights(
                source_downsample[0], preserve_kernel_norm=preserve_kernel_norm
            )
            _copy_batch_norm(target_downsample[1], source_downsample[1])

    return target


def _resolve_lora_sublayers(block: nn.Module, paths: Iterable[str]) -> Iterable[nn.Module]:
    for path in paths:
        if path.startswith("downsample.") and block.downsample is None:
            continue
        try:
            module = block.get_submodule(path)
        except AttributeError as exc:
            raise ValueError(f"Unknown ResNet LoRA sublayer {path!r} in {type(block).__name__}") from exc
        if not isinstance(module, KNNConvLayer):
            raise TypeError(f"ResNet LoRA target {path!r} is not a KNNConvLayer")
        yield module


def prep_fovi_resnet_finetuning(
    model: KNNResNet,
    cfg,
    device: str = "cuda",
    key: str = "pretrained_model",
) -> KNNResNet:
    """Freeze a pretrained FOVI ResNet and enable configured LoRA adapters."""
    finetune_cfg = cfg.get(key)

    if finetune_cfg.freeze_backbone:
        model.requires_grad_(False)

    if finetune_cfg.get("unfreeze_norm", False):
        model.layer4[-1].norm2.requires_grad_(True)

    if finetune_cfg.get("unfreeze_all_norms", False):
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.requires_grad_(True)

    unfreeze_layers = finetune_cfg.get("unfreeze_layers")
    if unfreeze_layers is not None:
        if isinstance(unfreeze_layers, int):
            unfreeze_layers = [unfreeze_layers]
        blocks = flattened_basic_blocks(model)
        for layer_index in unfreeze_layers:
            if layer_index == -1:
                model.conv1.requires_grad_(True)
                model.bn1.requires_grad_(True)
            else:
                if layer_index < 0 or layer_index >= len(blocks):
                    raise ValueError(
                        f"ResNet fine-tune layer index {layer_index} is out of range "
                        f"for {len(blocks)} blocks"
                    )
                blocks[layer_index].requires_grad_(True)

    lora_cfg = finetune_cfg.get("lora")
    if lora_cfg is None or lora_cfg.layers is None:
        return model

    blocks = flattened_basic_blocks(model)
    for layer_index in lora_cfg.layers:
        if layer_index == -1:
            modules = (model.conv1,)
        else:
            if layer_index < 0 or layer_index >= len(blocks):
                raise ValueError(
                    f"ResNet LoRA layer index {layer_index} is out of range for {len(blocks)} blocks"
                )
            modules = _resolve_lora_sublayers(blocks[layer_index], lora_cfg.sublayers)

        for module in modules:
            apply_lora(module, r=lora_cfg.r, alpha=lora_cfg.alpha, device=device)

    return model
