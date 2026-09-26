"""Dense sensors use image positions; vector sensors use Cartesian patch positions."""

import copy

import pytest
import torch
from fovi.models import dinov3
from fovi.models.knnvit import FoviDinoV3RoPE
from omegaconf import OmegaConf
from transformers import DINOv3ViTConfig, DINOv3ViTModel
from transformers.models.dinov3_vit.modeling_dinov3_vit import (
    DINOv3ViTRopePositionEmbedding,
)
from fovi.sensing.coords import SamplingCoords


@pytest.mark.parametrize(
    "style",
    [
        "uniform_as_grid",
        "warped_cartesian_as_grid",
        "logpolar_as_grid",
        "isotropic",
        "warped_cartesian",
        "logpolar",
    ],
)
def test_builder_selects_positions_from_sensor_layout(
    style: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    native = DINOv3ViTModel(
        DINOv3ViTConfig(
            image_size=32,
            patch_size=8,
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=64,
            num_register_tokens=0,
        )
    ).eval()

    def load_local_model(
        path: str, device: str, pretrained: bool
    ) -> tuple[DINOv3ViTModel, None]:
        return copy.deepcopy(native).to(device), None

    monkeypatch.setattr(dinov3, "load_dinov3", load_local_model)
    cfg = OmegaConf.create(
        {
            "saccades": {
                "mode": style,
                "resize_size": 32,
                "fov": 16.0,
                "cmf_a": 0.5,
                "sample_cortex": False,
                "field_geometry": "planar",
            },
            "model": {
                "output_mode": "pooled",
                "vit": {
                    "patch_size": 8,
                    # Fixed KNN patches isolate RoPE selection from partition coverage.
                    "partitioning_patches": False,
                    "patch_overlap_factor": 1,
                    "new_parameterization": False,
                    "force_patches_less_than_matched": False,
                },
            },
            "pretrained_model": {
                "path": "local-test-model",
                "patch_size": 8,
                "use_patch_weights": True,
                "freeze_backbone": True,
                "unfreeze_layers": None,
                "unfreeze_all_norms": False,
                "lora": None,
            },
        }
    )
    model = dinov3.build_fovi_dinov3(cfg, device="cpu").eval()
    if style.endswith("_as_grid"):
        assert isinstance(model.rope_embeddings, DINOv3ViTRopePositionEmbedding)
        inputs = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            for actual, expected in zip(
                model.rope_embeddings(inputs),
                native.rope_embeddings(inputs),
                strict=True,
            ):
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            torch.testing.assert_close(
                model(inputs), native(inputs).pooler_output.unsqueeze(1), atol=0, rtol=0
            )
    else:
        assert isinstance(model.rope_embeddings, FoviDinoV3RoPE)
        patch_coords = model.embeddings.patch_embeddings.out_coords
        torch.testing.assert_close(
            model.rope_embeddings.coords,
            torch.stack(
                (-patch_coords.cartesian[:, 1], patch_coords.cartesian[:, 0]), -1
            ),
            atol=0,
            rtol=0,
        )
        assert model.rope_embeddings.coords.shape[-1] == 2


@pytest.mark.parametrize(
    "style,resolution", [("uniform_as_grid", 111), ("warped_cartesian_as_grid", 200)]
)
def test_rectangular_dense_rope_matches_patch_grid(
    style: str, resolution: int
) -> None:
    coords = SamplingCoords((60.0, 80.0), 1.0, resolution, style=style)
    height, width = coords.grid_shape
    model = DINOv3ViTModel(
        DINOv3ViTConfig(
            image_size=max(height, width),
            patch_size=16,
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=64,
            num_register_tokens=0,
        )
    ).eval()
    dinov3.configure_dinov3_positions(
        model, sensor_coords=coords, patch_size=16,
        position_coordinate_space="cartesian",
    )
    assert isinstance(model.rope_embeddings, FoviDinoV3RoPE)
    assert model.rope_embeddings.coords.shape[0] == (height // 16) * (width // 16)
    with torch.no_grad():
        output = model(torch.randn(1, 3, height, width))
    assert output.last_hidden_state.shape[1] == (height // 16) * (width // 16) + 1
