"""Mixed-level sampling compared with independently constructed scalar sensors."""

import pytest
import torch
from fovi.sensing.coords import SamplingCoordsBank, is_cmf_sequence
from fovi.sensing.projection import CameraModel
from fovi.sensing.retina import RetinalTransform
from fovi.sensing.samplers import GridSampler
from fovi.utils.fastaugs import transforms as fastT
from omegaconf import OmegaConf

LEVELS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]


@pytest.mark.parametrize("geometry", ["planar", "spherical"])
@pytest.mark.parametrize("fov_type", ["square", "circular"])
@pytest.mark.parametrize("mode", ["nearest", "bilinear"])
@pytest.mark.parametrize("backend", ["torch", "cuda"])
@pytest.mark.parametrize(
    "dtype", [torch.uint8, torch.float16, torch.float32, torch.float64]
)
def test_mixed_sampling(
    geometry: str, fov_type: str, mode: str, backend: str, dtype: torch.dtype
) -> None:
    if backend == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = "cuda" if backend == "cuda" else "cpu"
    kwargs = {
        "device": device,
        "backend": backend,
        "mode": mode,
        "style": "warped_cartesian_as_grid",
        "fov_type": fov_type,
        "field_geometry": geometry,
    }
    if geometry == "spherical":
        kwargs["camera_model"] = CameraModel("pinhole", (32, 40), (24, 24, 19.5, 15.5))
    bank = GridSampler(16.0, LEVELS, 8, **kwargs)
    indices = torch.tensor([6, 0, 3, 1, 5, 2, 4, 0], device=device)
    torch.manual_seed(42)
    images = torch.randint(0, 256, (len(indices), 3, 32, 40), device=device).to(dtype)
    fix = torch.rand(len(indices), 2, device=device)
    size = torch.full_like(fix, 32) if geometry == "planar" else None
    actual, pixels = bank(images, fix, size, cmf_indices=indices, return_coords=True)
    for image_index, level in enumerate(indices.cpu().tolist()):
        scalar = GridSampler(16.0, LEVELS[level], 8, **kwargs)
        region = slice(image_index, image_index + 1)
        # Match batch size so calibrated projection uses the same matmul arithmetic.
        expected, expected_pixels = scalar(images, fix, size, return_coords=True)
        torch.testing.assert_close(actual[region], expected[region], atol=0, rtol=0)
        torch.testing.assert_close(
            pixels[region], expected_pixels[region], atol=0, rtol=0
        )


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
def test_retina_fast_augmentation_and_color(dtype: torch.dtype) -> None:
    pre = fastT.Compose([fastT.NormalizeGPU([0.5] * 3, [0.25] * 3)])
    retina = RetinalTransform(
        8,
        start_res=32,
        fov=16,
        cmf_a=LEVELS,
        style="warped_cartesian_as_grid",
        device="cpu",
        sampler_backend="torch",
        pre_transforms=pre,
        sigma=0.5,
    )
    images = torch.randint(0, 256, (3, 3, 32, 32)).to(dtype)
    if dtype != torch.uint8:
        images /= 255
    indices = torch.tensor([0, 3, 6])
    fix = torch.tensor([[0.05, 0.95], [0.5, 0.5], [0.8, 0.2]])
    actual = retina(images, fix, cmf_indices=indices)
    retina.fast_pre_transforms = False
    expected = retina(images, fix, cmf_indices=indices)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for i, level in enumerate(indices.tolist()):
        scalar = RetinalTransform(
            8,
            start_res=32,
            fov=16,
            cmf_a=LEVELS[level],
            style="warped_cartesian_as_grid",
            device="cpu",
            sampler_backend="torch",
            pre_transforms=pre,
            sigma=0.5,
        )
        scalar.fast_pre_transforms = False
        torch.testing.assert_close(
            actual[i : i + 1], scalar(images[i : i + 1], fix[i : i + 1]), atol=0, rtol=0
        )


def test_bank_validation_and_state() -> None:
    assert is_cmf_sequence(OmegaConf.create(LEVELS))
    for invalid in ([], [0], [float("inf")], [1, 1], [True], ["auto"]):
        with pytest.raises(ValueError, match="cmf_a"):
            SamplingCoordsBank(16, invalid, 8)
    bank = SamplingCoordsBank(16, LEVELS, 8)
    assert not bank.state_dict()
    assert bank.for_level(3).cmf_a == 1
    sampler = GridSampler(16, LEVELS, 8, style="warped_cartesian_as_grid", device="cpu")
    image = torch.zeros(2, 3, 32, 32)
    with pytest.raises(ValueError, match="explicit cmf_indices"):
        sampler(image)
    with pytest.raises(RuntimeError, match="out of range"):
        sampler(image, cmf_indices=torch.tensor([-1, 0]))


def test_compiled_bank_on_cpu_matches_eager() -> None:
    kwargs = {
        "style": "warped_cartesian_as_grid",
        "device": "cpu",
        "field_geometry": "spherical",
        "camera_model": CameraModel("pinhole", (32, 40), (24, 24, 19.5, 15.5)),
        "mode": "bilinear",
    }
    compiled = GridSampler(16, LEVELS, 8, backend="compiled", **kwargs)
    eager = GridSampler(16, LEVELS, 8, backend="torch", **kwargs)
    indices = torch.tensor([6, 0, 3, 1])
    torch.manual_seed(0)
    images = torch.rand(4, 3, 32, 40)
    fixation = torch.rand(4, 2) * 0.4 + 0.3
    torch.testing.assert_close(
        compiled(images, fixation, cmf_indices=indices),
        eager(images, fixation, cmf_indices=indices),
    )
    with pytest.raises(RuntimeError, match="out of range"):
        compiled(images, fixation, cmf_indices=torch.tensor([0, 1, 2, 7]))


def test_mixed_sampler_gradients() -> None:
    indices = torch.tensor([0, 6])
    images = torch.rand(2, 3, 32, 32, requires_grad=True)
    fix = torch.tensor([[0.3, 0.4], [0.7, 0.6]], requires_grad=True)
    sampler = GridSampler(
        16,
        LEVELS,
        8,
        style="warped_cartesian_as_grid",
        device="cpu",
        backend="torch",
        mode="bilinear",
    )
    sampler(images, fix, 32, cmf_indices=indices).sum().backward()
    for i, level in enumerate(indices.tolist()):
        image = images[i : i + 1].detach().requires_grad_()
        fixation = fix[i : i + 1].detach().requires_grad_()
        scalar = GridSampler(
            16,
            LEVELS[level],
            8,
            style="warped_cartesian_as_grid",
            device="cpu",
            backend="torch",
            mode="bilinear",
        )
        scalar(image, fixation, 32).sum().backward()
        torch.testing.assert_close(images.grad[i : i + 1], image.grad)
        torch.testing.assert_close(fix.grad[i : i + 1], fixation.grad)


@pytest.mark.parametrize("geometry", ["planar", "spherical"])
def test_bank_dtype_move_preserves_geometry(geometry: str) -> None:
    kwargs = {"field_geometry": geometry}
    if geometry == "spherical":
        kwargs["camera_model"] = CameraModel("fisheye", (32, 32), (24, 24, 15.5, 15.5))
    sampler = GridSampler(
        16, LEVELS, 8, style="warped_cartesian_as_grid", device="cpu", **kwargs
    )
    original = sampler.sampling_grid.clone()
    sampler.to(dtype=torch.float16)
    torch.testing.assert_close(sampler.sampling_grid, original, atol=0, rtol=0)
    assert sampler.coords.cartesian.dtype == torch.float32
    assert sampler.canonical_directions.dtype == torch.float32
    assert not sampler.state_dict()


@pytest.mark.parametrize("backend", ["torch", "compiled", "cuda"])
def test_spherical_fisheye_square_shells_and_strided_selection(backend: str) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    kwargs = {
        "style": "warped_cartesian_as_grid",
        "device": "cuda",
        "field_geometry": "spherical",
        "fov_type": "square",
        "radius_norm": float("inf"),
        "camera_model": CameraModel("fisheye", (32, 32), (24, 24, 15.5, 15.5)),
        "backend": backend,
        "mode": "bilinear",
    }
    sampler = GridSampler(16, LEVELS, 8, **kwargs)
    indices = torch.tensor([0, 2, 6, 3], device="cuda")[::2]
    images = torch.rand(2, 3, 32, 32, device="cuda", dtype=torch.bfloat16)
    fixation = torch.tensor([[0.3, 0.7], [0.5, 0.5]], device="cuda")
    actual = sampler(images, fixation, cmf_indices=indices)
    for i, level in enumerate(indices.cpu().tolist()):
        scalar = GridSampler(16, LEVELS[level], 8, **kwargs)
        expected = scalar(images, fixation)
        torch.testing.assert_close(actual[i], expected[i], atol=0, rtol=0)
