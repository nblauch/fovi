"""Mixed-level geometry, CUDA derivatives, and complete image-to-logit routing."""

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from omegaconf import open_dict
from transformers import DINOv3ViTConfig, DINOv3ViTModel

from fovi.models import dinov3
from fovi.models.fovinet import FoviNet
from fovi.models.knnvit import KNNPartitioningPatchEmbedding
from fovi.models.loading import load_config
from fovi.models.omni_isotropic import OmniIsotropicPatchEmbedding
from fovi.sensing.coords import SamplingCoords, SamplingCoordsBank
from fovi.sensing.projection import CameraModel

LEVELS = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)


@pytest.mark.parametrize("resolution", [8, 128])
def test_exact_deterministic_count_and_planar_scale_invariance(resolution: int) -> None:
    bank = SamplingCoordsBank(16, LEVELS, resolution, style="isotropic_fixn_calibrated")
    assert bank.cartesian.shape == (7, resolution**2, 2)
    assert bank.valid_mask.all()
    for level, a in enumerate(LEVELS):
        scaled = SamplingCoords(
            160, a * 10, resolution, style="isotropic_fixn_calibrated"
        )
        torch.testing.assert_close(
            bank.cartesian[level], scaled.cartesian, atol=1e-7, rtol=1e-6
        )
        assert torch.unique(bank.cartesian[level], dim=0).shape[0] == resolution**2
        assert torch.isfinite(scaled.cortical).all()
        torch.testing.assert_close(
            bank.cartesian[level], bank.for_level(level).cartesian, atol=0, rtol=0
        )
    with pytest.raises(ValueError, match="circular"):
        SamplingCoordsBank(
            160,
            LEVELS,
            resolution,
            style="isotropic_fixn_calibrated",
            fov_type="square",
        )


@pytest.fixture(scope="module")
def patch_bank() -> OmniIsotropicPatchEmbedding:
    return OmniIsotropicPatchEmbedding(
        fov=16,
        cmf_a=LEVELS,
        resolution=32,
        patch_size=8,
        reference_side=8,
        in_channels=3,
        embed_dim=32,
        sample_cortex=True,
        device="cpu",
    )


def test_mixed_patch_outputs_match_independent_scalar_layers(
    patch_bank: OmniIsotropicPatchEmbedding,
) -> None:
    x = torch.randn(7, 3, 1024)
    actual = patch_bank(x, torch.arange(7))
    for index, value in enumerate(LEVELS):
        scalar = KNNPartitioningPatchEmbedding(
            in_channels=3,
            embed_dim=32,
            in_res=32,
            fov=16,
            cmf_a=value,
            in_coords=SamplingCoords(16, value, 32, style="isotropic_fixn_calibrated"),
            out_coords=SamplingCoords(16, value, 4, style="isotropic_fixn_calibrated"),
            style="isotropic_fixn_calibrated",
            device="cpu",
            sample_cortex=True,
            ref_frame_side_length=8,
            kernel_backend="baseline",
            bias=True,
            transposed=True,
        )
        with torch.no_grad():
            scalar.weight.copy_(patch_bank.weight)
            scalar.bias.copy_(patch_bank.bias)
        torch.testing.assert_close(
            actual[index : index + 1],
            scalar(x[index : index + 1]),
            atol=2e-6,
            rtol=2e-5,
        )
    assert set(patch_bank.state_dict()) == {"weight", "bias"}
    assert (patch_bank.geometry.reverse_ptr.diff(dim=1) > 0).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_cuda_forward_and_gradients(
    patch_bank: OmniIsotropicPatchEmbedding, dtype: torch.dtype
) -> None:
    bank = deepcopy(patch_bank).cuda()
    levels = torch.tensor([6, 0, 4, 1, 5, 2, 3, 0], device="cuda")
    x = torch.randn(8, 3, 1024, device="cuda", dtype=dtype, requires_grad=True)
    oracle_x = x.detach().clone().requires_grad_(True)
    with torch.cuda.stream(torch.cuda.Stream()):
        # Wait explicitly for tensors initialized on the caller's stream.
        torch.cuda.current_stream().wait_stream(torch.cuda.default_stream())
        actual = bank.geometry(x, levels)
        expected = bank.geometry.reference(oracle_x, levels)
        grad = torch.randn_like(actual)
        actual.backward(grad)
        expected.backward(grad)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        actual, expected, atol=0 if dtype != torch.float32 else 1e-6, rtol=0
    )
    torch.testing.assert_close(
        x.grad,
        oracle_x.grad,
        atol=0.032 if dtype == torch.bfloat16 else 0.004,
        rtol=0.008,
    )
    # Exercise the shared linear projection's weight gradient under autocast.
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
        output = bank(x.float(), levels)
    output.float().square().mean().backward()
    assert torch.isfinite(bank.weight.grad).all()


@pytest.mark.parametrize(
    "config_name",
    [
        "dinov3_isotropic_multi_cmf",
        "dinov3_isotropic_multi_cmf_geodesic",
        "dinov3_isotropic_multi_cmf_geodesic_kernel24",
        "dinov3_isotropic_multi_cmf_geodesic_inset",
        "dinov3_isotropic_multi_cmf_geodesic_inset_kernel24",
        "dinov3_isotropic_multi_cmf_radial_log",
        "dinov3_isotropic_multi_cmf_manifold_3d",
    ],
)
@pytest.mark.parametrize("field_geometry", ["planar", "spherical"])
def test_image_to_logits_mixed_levels_and_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    config_name: str,
    field_geometry: str,
) -> None:
    native = DINOv3ViTModel(
        DINOv3ViTConfig(
            image_size=32,
            patch_size=8,
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=64,
            num_register_tokens=2,
        )
    )
    monkeypatch.setattr(
        dinov3,
        "load_dinov3",
        lambda path, device, pretrained: (deepcopy(native).to(device), None),
    )
    cfg, _, _ = load_config(
        config_name,
        False,
        Path(__file__).parents[1] / "config",
        device="cpu",
    )
    cfg.pretrained_model.lora.layers = [-1, 0]
    cfg.saccades.field_geometry = field_geometry
    if field_geometry == "spherical":
        cfg.saccades.fov = 120
        cfg.saccades.rescale_fov = 0
        with open_dict(cfg.saccades):
            cfg.saccades.camera_model = asdict(
                CameraModel("fisheye", (32, 32), (12, 12, 15.5, 15.5))
            )
    cfg.pretrained_model.patch_size = 8
    cfg.model.vit.patch_size = 8
    cfg.model.mlp = "16"
    cfg.saccades.resize_size = cfg.saccades.fixation_size = cfg.training.resolution = 32
    cfg.saccades.n_fixations = 2
    cfg.model.dropout = 0
    cfg.transforms.color_jitter = cfg.transforms.gray = cfg.transforms.flip = 0
    model = FoviNet(cfg, device="cpu").eval()
    images = torch.rand(7, 3, 32, 32)
    levels = torch.arange(7)
    fixes = [torch.full((7, 2), 0.5), torch.full((7, 2), 0.6)]
    with pytest.raises(ValueError, match="explicit cmf_indices"):
        model(images)
    actual, _, views = model(images, fixations=fixes, cmf_indices=levels)
    # Independent level calls catch both fixation ordering and RoPE broadcasting errors.
    for index in range(7):
        expected = model(
            images[index : index + 1],
            fixations=[f[index : index + 1] for f in fixes],
            cmf_indices=levels[index : index + 1],
        )[0]
        torch.testing.assert_close(
            actual[index : index + 1], expected, atol=1e-5, rtol=1e-4
        )
    torch.testing.assert_close(
        actual, model(views, fixated_inputs=True, cmf_indices=levels)[0]
    )
    restored = FoviNet(deepcopy(cfg), device="cpu").eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(
        actual, restored(images, fixations=fixes, cmf_indices=levels)[0], atol=0, rtol=0
    )
    model.train()
    with pytest.raises(ValueError, match="Already-fixated.*explicit cmf_indices"):
        model(views.detach(), fixated_inputs=True)
    loss = model(images, cmf_indices=levels)[0].square().mean()
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in gradients)
    assert any(g.abs().max() > 0 for g in gradients)
