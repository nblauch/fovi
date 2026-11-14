"""Regression coverage for omni model boundaries, SSL, and precision."""

from copy import deepcopy
from pathlib import Path

import pytest
import torch
from omegaconf import DictConfig
from transformers import DINOv3ViTConfig, DINOv3ViTModel

from fovi.models import dinov3
from fovi.models.fovinet import FoviNet
from fovi.models.knnvit import FoviDinoV3RoPE
from fovi.models.loading import load_config
from fovi.models.omni_isotropic import OmniIsotropicPatchEmbedding
from fovi.sensing import geodesic
from fovi.sensing.coords import SamplingCoords, SamplingCoordsBank


@pytest.fixture
def omni_config(monkeypatch: pytest.MonkeyPatch) -> DictConfig:
    native = DINOv3ViTModel(
        DINOv3ViTConfig(
            image_size=16,
            patch_size=4,
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=64,
            num_register_tokens=2,
            pos_embed_shift=None,
            pos_embed_jitter=None,
            pos_embed_rescale=None,
        )
    )
    monkeypatch.setattr(
        dinov3,
        "load_dinov3",
        lambda path, device, pretrained: (deepcopy(native).to(device), None),
    )
    cfg = load_config(
        "dinov3_isotropic_multi_cmf",
        False,
        Path(__file__).parents[1] / "config",
        device="cpu",
    )[0]
    cfg.pretrained_model.lora = None
    cfg.pretrained_model.freeze_backbone = False
    cfg.pretrained_model.patch_size = cfg.model.vit.patch_size = 4
    cfg.saccades.resize_size = cfg.saccades.fixation_size = cfg.training.resolution = 16
    cfg.saccades.cmf_a = [0.1, 1, 10]
    cfg.saccades.n_fixations = 2
    cfg.saccades.fixation_size_min_frac = cfg.saccades.fixation_size_max_frac = 1
    cfg.model.mlp = "16"
    cfg.model.dropout = 0
    cfg.transforms.color_jitter = cfg.transforms.gray = cfg.transforms.flip = 0
    return cfg


@pytest.mark.parametrize("output_mode", ["pooled", "patch_tokens"])
@pytest.mark.parametrize(
    "style", ["isotropic_fixn_calibrated", "warped_cartesian_as_grid"]
)
def test_ssl_forward_backward_and_fixation_order(
    omni_config: DictConfig, output_mode: str, style: str
) -> None:
    cfg = omni_config
    cfg.model.output_mode = output_mode
    cfg.saccades.mode = style
    if style == "warped_cartesian_as_grid":
        cfg.model.vit.position_coordinate_space = "cortical"
    cfg.training.loss = "simclr"
    cfg.saccades.ssl_policy = "multi_random"
    model = FoviNet(cfg, device="cpu").train()
    images = torch.rand(3, 3, 16, 16)
    levels = torch.tensor([2, 0, 1])
    f1 = torch.full((3, 2), 0.5)
    embeddings, _, views = model(images, f1=f1, cmf_indices=levels)
    assert embeddings.shape == (3, 2, 16)
    # The fixed first fixation has the same level as its image's second view.
    expected_view = model.retinal_transform(images, f1, cmf_indices=levels)
    torch.testing.assert_close(views[:3], expected_view)
    kwargs = (
        {"cmf_indices": levels.repeat(2)}
        if style == "isotropic_fixn_calibrated"
        else {}
    )
    features = model.network(views, apply_mlp=False, **kwargs)
    expected, _ = model.network.projector_ssl(features)
    torch.testing.assert_close(embeddings, expected.reshape(2, 3, 16).transpose(0, 1))
    # A loss that couples the two views must reach the backbone and SSL head.
    loss = (embeddings[:, 0] - embeddings[:, 1].detach()).square().mean()
    loss.backward()
    for module in (model.network.backbone, model.network.projector_ssl):
        grads = [
            p.grad
            for p in module.parameters()
            if p.requires_grad and p.grad is not None
        ]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert any(g.abs().max() > 0 for g in grads)


def test_unsupported_omni_architecture_fails_before_build(
    omni_config: DictConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fovi.models.fovinet import ARCHITECTURE_REGISTRY

    omni_config.model.arch = "fovi_alexnet"
    monkeypatch.setattr(
        ARCHITECTURE_REGISTRY,
        "get",
        lambda name: pytest.fail("Builder accessed before validation"),
    )
    with pytest.raises(ValueError, match="fovi_dinov3"):
        FoviNet(omni_config, device="cpu")


def test_ssl_training_step_accepts_batch_larger_than_two(
    omni_config: DictConfig,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("FOVI_SAVE_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("FOVI_DATASETS_DIR", str(tmp_path / "datasets"))
    from fovi.training.trainer import Trainer

    cfg = omni_config
    cfg.training.loss = "vicreg"
    cfg.saccades.ssl_policy = "multi_random"
    cfg.saccades.fixation_size_min_frac = cfg.saccades.fixation_size_max_frac = 0.5
    cfg.training.use_amp = False
    cfg.training.no_probes = True
    cfg.training.grad_accum_steps = 1
    cfg.logging.log_level = 0
    model = FoviNet(cfg, device="cpu")
    # Exercise the real training step without datasets, distributed setup or logging.
    trainer = Trainer.__new__(Trainer)
    trainer.cfg = cfg
    trainer.model = trainer.model_ = model
    trainer.probes = torch.nn.Linear(16, 2)
    trainer.optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    trainer.optimizer_probes = torch.optim.SGD(trainer.probes.parameters(), lr=0.01)
    trainer.train_loader = [
        (torch.rand(3, 3, 16, 16), torch.zeros(3, dtype=torch.long))
    ]
    trainer.lr_schedule = False
    trainer.supervised_loss = False
    trainer.do_network_training = True
    trainer.gpu = "cpu"
    trainer.amp_dtype = torch.bfloat16
    trainer.loss_name = "vicreg"
    trainer.ssl_loss = torch.nn.MSELoss()
    trainer.scaler = torch.amp.GradScaler("cpu", init_scale=128)
    before = [p.detach().clone() for p in model.network.projector_ssl.parameters()]
    trainer.train_loop(epoch=0, max_batches=1)
    after = list(model.network.projector_ssl.parameters())
    assert all(torch.isfinite(p).all() for p in after)
    assert any(not torch.equal(old, new) for old, new in zip(before, after))


def test_backbone_requires_explicit_cmf(omni_config: DictConfig) -> None:
    model = FoviNet(omni_config, device="cpu").eval()
    pixels = torch.rand(3, 3, 256)
    for kwargs in ({}, {"cmf_indices": None}):
        with pytest.raises(ValueError, match="explicit cmf_indices"):
            model.network.backbone(pixels, **kwargs)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("shared", [False, True])
def test_explicit_fixations_for_two_images(
    omni_config: DictConfig, device: str, shared: bool
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    model = FoviNet(omni_config, device=device).eval()
    images = torch.rand(2, 3, 16, 16, device=device)
    levels = torch.tensor([2, 0], device=device)
    fixation = torch.tensor([[0.4, 0.5], [0.7, 0.8]], device=device)
    supplied = fixation[0] if shared else fixation
    expected = fixation[:1].expand(2, 2) if shared else fixation
    result = model.sup_fixator(
        images, fixations=[supplied], n_fixations=1, cmf_indices=levels
    )
    torch.testing.assert_close(result["fixations"][:, 0], expected)
    expected_view = model.retinal_transform(images, expected, cmf_indices=levels)
    torch.testing.assert_close(result["x_fixs"][:, 0], expected_view)


@pytest.mark.parametrize("dimensions", [2, 3])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rope_dtype_conversion_preserves_frequency_and_angles(
    dimensions: int, dtype: torch.dtype
) -> None:
    rope = FoviDinoV3RoPE(100, 64, torch.randn(3, 16, dimensions), device="cpu")
    pixels, indices = torch.zeros(3, 3, 256), torch.arange(3)
    original = rope(pixels, indices)
    frequencies = rope.inv_freq.clone()
    rope.to(dtype=dtype)
    assert rope.inv_freq.dtype == torch.float32
    torch.testing.assert_close(rope.inv_freq, frequencies, atol=0, rtol=0)
    actual = rope(pixels, indices)
    for expected, observed in zip(original, actual):
        torch.testing.assert_close(observed, expected, atol=0, rtol=0)


@pytest.mark.parametrize(
    "style,fov_type,message",
    [
        ("warped_cartesian_as_grid", "invalid", "fov_type"),
        ("isotropic_fixn_calibrated", "square", "circular"),
    ],
)
def test_bank_rejects_invalid_field_combination(
    style: str, fov_type: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        SamplingCoordsBank(16, [1, 10], 8, style=style, fov_type=fov_type)


def test_calibrated_plotting_rejects_unsupported_layout() -> None:
    with pytest.raises(ValueError, match="isotropic_plotting_type"):
        SamplingCoords(
            16,
            1,
            8,
            style="isotropic_fixn_calibrated",
            isotropic_plotting_type="schwartz",
        )


def test_omni_geodesics_use_default_user_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("FOVI_GEODESIC_CACHE_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    kwargs = {
        "fov": 16,
        "cmf_a": [0.1, 1],
        "resolution": 16,
        "patch_size": 4,
        "reference_side": 4,
        "in_channels": 1,
        "embed_dim": 1,
        "sample_cortex": "geodesic",
        "device": "cpu",
    }
    first = OmniIsotropicPatchEmbedding(**kwargs)
    assert len(list((tmp_path / "fovi" / "geodesic").glob("*.npz"))) == 2
    monkeypatch.setattr(
        geodesic,
        "_solve_surface_geodesics",
        lambda *args: pytest.fail("Repeated geodesic solve"),
    )
    second = OmniIsotropicPatchEmbedding(**kwargs)
    for name, value in first.geometry.named_buffers():
        torch.testing.assert_close(
            dict(second.geometry.named_buffers())[name], value, atol=0, rtol=0
        )


def test_geodesic_cache_override_and_opt_out(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("FOVI_GEODESIC_CACHE_DIR", str(tmp_path))
    assert geodesic.geodesic_cache_directory() == tmp_path
    monkeypatch.setenv("FOVI_GEODESIC_CACHE_DIR", "")
    assert geodesic.geodesic_cache_directory() is None
