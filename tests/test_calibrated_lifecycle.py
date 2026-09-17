"""Precision, backend selection, and ownership of calibrated sampler state."""

import copy
import io
from dataclasses import replace

import pytest
import torch

from fovi.sensing.projection import CameraModel
from fovi.sensing.samplers import GridSampler


def make_sampler(device: str = "cpu", backend: str = "auto") -> GridSampler:
    return GridSampler(
        30,
        0.5,
        8,
        device=device,
        mode="bilinear",
        backend=backend,
        field_geometry="spherical",
        camera_model=CameraModel("fisheye", (80, 120), (75, 75, 59.5, 39.5)),
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float64])
def test_dtype_conversion_preserves_ray_values(dtype: torch.dtype) -> None:
    sampler = make_sampler()
    rays = sampler.canonical_directions.clone()
    pixels, valid = sampler.calibrated_pixels(torch.tensor([[0.4, 0.6]]))
    sampler.to(dtype=dtype)
    assert sampler.canonical_directions.dtype == torch.float32
    assert torch.equal(sampler.canonical_directions, rays)
    actual, actual_valid = sampler.calibrated_pixels(torch.tensor([[0.4, 0.6]]))
    assert torch.equal(actual, pixels)
    assert torch.equal(actual_valid, valid)


@pytest.mark.parametrize("selection", ["argument", "environment"])
def test_explicit_cuda_rejects_cpu_input(
    selection: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    sampler = make_sampler(backend="cuda" if selection == "argument" else "auto")
    if selection == "environment":
        monkeypatch.setenv("FOVI_GRID_SAMPLER_BACKEND", "cuda")
    with pytest.raises(RuntimeError, match="CUDA.*gradients"):
        sampler(torch.ones(1, 3, 80, 120), [0.5, 0.5])


def test_overscan_warns_and_nodes_can_reenter_image() -> None:
    camera = CameraModel("pinhole", (80, 120), (110, 110, 59.5, 39.5))
    with pytest.warns(UserWarning, match="central gaze.*zero"):
        sampler = GridSampler(
            90, 2, 8, device="cpu", field_geometry="spherical", camera_model=camera
        )
    _, central = sampler.calibrated_pixels(torch.tensor([[0.5, 0.5]]))
    _, shifted = sampler.calibrated_pixels(torch.tensor([[0.5, 0.85]]))
    assert (~central & shifted).any()
    samples = sampler(torch.ones(1, 1, 80, 120), [0.5, 0.5])
    assert torch.isfinite(samples).all()
    assert (samples[..., ~central[0]] == 0).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("warm", [False, True])
def test_compiled_sampler_deepcopy_and_pickle_own_their_state(warm: bool) -> None:
    sampler = make_sampler("cuda", "compiled")
    image = torch.arange(120, device="cuda", dtype=torch.float32)[
        None, None, None
    ].expand(1, 1, 80, 120)
    if warm:
        sampler(image, [0.5, 0.5])
    copied = copy.deepcopy(sampler)
    copied.camera_model = replace(copied.camera_model, intrinsics=(50, 50, 59.5, 39.5))
    expected = copied(image, [0.5, 0.6], direct=True)
    torch.testing.assert_close(copied(image, [0.5, 0.6]), expected)
    assert not torch.allclose(expected, sampler(image, [0.5, 0.6]))
    stream = io.BytesIO()
    torch.save(copied, stream)
    stream.seek(0)
    restored = torch.load(stream, weights_only=False)
    torch.testing.assert_close(restored(image, [0.5, 0.6]), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_selection_gradients_and_mixed_precision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sampler = make_sampler("cuda", "cuda").bfloat16()
    image = torch.ones(2, 3, 80, 120, device="cuda", dtype=torch.bfloat16)
    fix = torch.full((2, 2), 0.5, device="cuda")
    expected = sampler(image, fix, direct=True)
    torch.testing.assert_close(sampler(image, fix), expected)
    assert sampler.last_backend == "cuda_calibrated"
    fix.requires_grad_(True)
    with pytest.raises(RuntimeError, match="CUDA.*gradients"):
        sampler(image, fix)
    monkeypatch.setenv("FOVI_GRID_SAMPLER_BACKEND", "torch")
    sampler(image, fix).float().sum().backward()
    assert sampler.last_backend == "torch_calibrated"
    assert torch.isfinite(fix.grad).all()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Two GPUs required")
def test_compiled_deepcopy_moves_to_another_device() -> None:
    sampler = make_sampler("cuda:0", "compiled")
    original_rays = sampler.canonical_directions.clone()
    copied = copy.deepcopy(sampler).to(device="cuda:1", dtype=torch.float16)
    image = torch.ones(1, 1, 80, 120, device="cuda:1")
    expected = copied(image, [0.5, 0.5], direct=True)
    torch.testing.assert_close(copied(image, [0.5, 0.5]), expected)
    assert torch.equal(sampler.canonical_directions, original_rays)
