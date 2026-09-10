"""Checkpoint layout migration must preserve LoRA tensors and strict loading."""

from __future__ import annotations

from collections import OrderedDict

import pytest
import torch
from torch import nn

from fovi.models import dinov3
from fovi.utils.lora import apply_lora

LAYOUTS = ("layer", "model.layer", "encoder.layer")


def make_model(layout: str, *, compatibility: bool) -> nn.Module:
    """Build a nested backbone with nonzero LoRA updates and ordinary parameters."""
    backbone = nn.Module()
    block = nn.Linear(4, 4)
    lora = apply_lora(block, r=2, alpha=2, device="cpu")
    with torch.no_grad():
        lora.A.normal_()
        lora.B.normal_()
    layers = nn.ModuleList([block])
    if layout == "layer":
        backbone.layer = layers
    else:
        container, child = layout.split(".")
        parent = nn.Module()
        parent.add_module(child, layers)
        backbone.add_module(container, parent)
    backbone.norm = nn.LayerNorm(4)
    if compatibility:
        backbone.register_load_state_dict_pre_hook(dinov3._remap_dinov3_layer_keys)
    model = nn.Module()
    model.backbone = backbone
    model.head = nn.Linear(4, 2)
    return model


@pytest.mark.parametrize("source_layout", LAYOUTS)
@pytest.mark.parametrize("target_layout", LAYOUTS)
def test_strict_restore_preserves_lora_and_outputs(
    source_layout: str,
    target_layout: str,
) -> None:
    source = make_model(source_layout, compatibility=False)
    target = make_model(target_layout, compatibility=True)
    checkpoint = source.state_dict()
    keys_before = tuple(checkpoint)
    target.load_state_dict(checkpoint, strict=True)
    assert tuple(checkpoint) == keys_before
    source_block = source.get_submodule(f"backbone.{source_layout}.0")
    target_block = target.get_submodule(f"backbone.{target_layout}.0")
    for name, value in source_block.state_dict().items():
        torch.testing.assert_close(
            target_block.state_dict()[name], value, rtol=0, atol=0
        )
    inputs = torch.randn(3, 4)
    torch.testing.assert_close(
        target.get_submodule("head")(
            target.get_submodule("backbone.norm")(target_block(inputs))
        ),
        source.get_submodule("head")(
            source.get_submodule("backbone.norm")(source_block(inputs))
        ),
        rtol=0,
        atol=0,
    )


def test_ambiguous_layouts_raise_even_without_strict_loading() -> None:
    model = make_model("model.layer", compatibility=True)
    checkpoint = model.state_dict()
    checkpoint["backbone.layer.0.bias"] = checkpoint[
        "backbone.model.layer.0.bias"
    ].clone()
    with pytest.raises(RuntimeError, match="multiple DINOv3 layer layouts"):
        model.load_state_dict(checkpoint, strict=False)


@pytest.mark.parametrize("corruption", ("missing_lora", "shape", "unknown"))
def test_migration_does_not_hide_invalid_checkpoints(corruption: str) -> None:
    source = make_model("layer", compatibility=False)
    target = make_model("model.layer", compatibility=True)
    checkpoint = OrderedDict(source.state_dict())
    lora_key = "backbone.layer.0.parametrizations.weight.0.A"
    if corruption == "missing_lora":
        del checkpoint[lora_key]
    elif corruption == "shape":
        checkpoint[lora_key] = torch.zeros(1)
    elif corruption == "unknown":
        checkpoint["backbone.layer.0.unknown"] = torch.zeros(1)
    else:
        raise ValueError(corruption)
    with pytest.raises(RuntimeError):
        target.load_state_dict(checkpoint, strict=True)
