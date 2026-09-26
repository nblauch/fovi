"""Calibrated native sampling agrees with the independent Torch geometry path."""

import math

import pytest
import torch

from fovi.sensing.projection import CameraModel
from fovi.sensing.samplers import GridSampler

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

CAMERAS = [
    ("pinhole", ()),
    ("pinhole", (-0.08, 0.01, 0.001, -0.001, 0.0, 0.001, 0.0, 0.0)),
    ("fisheye", ()),
    ("fisheye", (0.02, -0.003, 0.0002, 0.0)),
]


@pytest.mark.parametrize("model,distortion", CAMERAS)
@pytest.mark.parametrize("mode", ["nearest", "bilinear"])
@pytest.mark.parametrize("convention", ["camera_xyz", "pan_tilt"])
@pytest.mark.parametrize(
    "dtype", [torch.uint8, torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_native_matches_eager(
    model: str,
    distortion: tuple[float, ...],
    mode: str,
    convention: str,
    dtype: torch.dtype,
) -> None:
    camera = CameraModel(
        model,
        (96, 128),
        (70, 71, 63.5, 47.5),
        distortion,
        image_circle=(63.5, 47.5, 60),
    )
    sampler = GridSampler(
        70,
        2,
        12,
        device="cuda",
        mode=mode,
        field_geometry="spherical",
        camera_model=camera,
        gaze_convention=convention,
    )
    generator = torch.Generator(device="cuda").manual_seed(321)
    image = torch.randint(
        0, 256, (3, 3, 96, 256), device="cuda", generator=generator, dtype=torch.uint8
    )[..., ::2]
    if dtype != torch.uint8:
        image = image.to(dtype) / 256
    fixation = torch.tensor(
        [[0.5013, 0.5017], [0.3013, 0.7217], [0.1013, 0.9117]], device="cuda"
    )
    expected, expected_grid = sampler(image, fixation, direct=True, return_coords=True)
    actual, actual_grid = sampler(image, fixation, return_coords=True)
    assert sampler.last_backend == "cuda_calibrated"
    assert actual.dtype == expected.dtype
    if mode == "nearest":
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    else:
        tolerances = {
            torch.uint8: 0.02,
            torch.float16: 0.001,
            torch.bfloat16: 0.008,
            torch.float32: 5e-5,
            torch.float64: 1e-10,
        }
        torch.testing.assert_close(actual, expected, rtol=0, atol=tolerances[dtype])
    torch.testing.assert_close(actual_grid, expected_grid, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("model,distortion", CAMERAS)
@pytest.mark.parametrize("channels", [1, 4])
def test_rotation_override_and_graph_replay(
    model: str, distortion: tuple[float, ...], channels: int
) -> None:
    camera = CameraModel(model, (96, 128), (70, 70, 63.5, 47.5), distortion)
    sampler = GridSampler(
        60,
        2,
        12,
        device="cuda",
        mode="bilinear",
        field_geometry="spherical",
        camera_model=camera,
    )
    image = torch.ones(2, channels, 96, 128, device="cuda")
    rotation = torch.eye(3, device="cuda").expand(2, 3, 3).clone().transpose(1, 2)
    torch.testing.assert_close(
        sampler(image, rotation=rotation),
        sampler(image, rotation=rotation, direct=True),
    )
    # Explicit rotations can point out of the calibrated domain.
    rotation.copy_(torch.diag(torch.tensor([-1.0, 1.0, -1.0], device="cuda")))
    assert torch.count_nonzero(sampler(image, rotation=rotation)) == 0
    fixation = torch.full((2, 2), 0.5, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            sampler(image, fixation)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = sampler(image, fixation)
    image.mul_(2)
    fixation.copy_(torch.tensor([[0.3, 0.8], [0.9, 0.1]], device="cuda"))
    graph.replay()
    torch.testing.assert_close(
        captured, sampler(image, fixation, direct=True), atol=2e-5, rtol=1e-5
    )


def test_auto_preserves_image_and_fixation_gradients() -> None:
    camera = CameraModel("pinhole", (32, 40), (30, 30, 19.5, 15.5))
    sampler = GridSampler(
        20,
        1,
        4,
        device="cuda",
        mode="bilinear",
        field_geometry="spherical",
        camera_model=camera,
    )
    image = (
        torch.arange(40, device="cuda", dtype=torch.float32)[None, None, None]
        .expand(1, 1, 32, 40)
        .clone()
        .requires_grad_()
    )
    fixation = torch.tensor([[0.45, 0.55]], device="cuda", requires_grad=True)
    expected = sampler(image, fixation, direct=True)
    expected_grad = torch.autograd.grad(expected.sum(), (image, fixation))
    actual = sampler(image, fixation)
    actual_grad = torch.autograd.grad(actual.sum(), (image, fixation))
    assert sampler.last_backend == "compiled_calibrated"
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    for result, reference in zip(actual_grad, expected_grad):
        torch.testing.assert_close(result, reference, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("model,distortion", CAMERAS)
@pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
def test_large_image_peripheral_gaze_precision(
    model: str, distortion: tuple[float, ...], dtype: torch.dtype
) -> None:
    camera = CameraModel(model, (480, 640), (300, 300, 319.5, 239.5), distortion)
    sampler = GridSampler(
        60,
        1.875,
        40,
        device="cuda",
        mode="bilinear",
        field_geometry="spherical",
        camera_model=camera,
    )
    image = torch.randint(
        0,
        256,
        (1, 3, 480, 640),
        device="cuda",
        dtype=torch.uint8,
        generator=torch.Generator(device="cuda").manual_seed(2026),
    )
    if dtype == torch.float32:
        image = image.float() / 256
    fixation = torch.tensor([[0.3013, 0.8017]], device="cuda")
    actual, grid = sampler(image, fixation, return_coords=True)
    expected, expected_grid = sampler(image, fixation, direct=True, return_coords=True)
    # Fused dot products and Torch's single-image matmul round differently.
    # Bound that error in source pixels as well as interpolated intensities.
    # align_corners=False maps a normalized span of two to the full raster.
    pixel_scale = grid.new_tensor((image.shape[-1] / 2, image.shape[-2] / 2))
    coordinate_atol = 2e-4
    torch.testing.assert_close(
        (grid - expected_grid) * pixel_scale,
        torch.zeros_like(grid),
        rtol=0,
        atol=coordinate_atol,
    )
    # Unit-range pixels (including zero padding) bound each bilinear partial
    # derivative by one. Add both axis errors plus interpolation rounding.
    intensity_atol = 2 * coordinate_atol + 8 * torch.finfo(torch.float32).eps
    torch.testing.assert_close(
        actual,
        expected,
        rtol=0 if dtype == torch.float32 else 1e-4,
        atol=intensity_atol if dtype == torch.float32 else 0.02,
    )


@pytest.mark.parametrize("model,distortion", CAMERAS)
@pytest.mark.parametrize("mode", ["nearest", "bilinear"])
def test_half_pixel_ties_and_invalid_fixations(
    model: str, distortion: tuple[float, ...], mode: str
) -> None:
    camera = CameraModel(model, (96, 128), (70, 70, 63.5, 47.5), distortion)
    sampler = GridSampler(
        60,
        2,
        12,
        device="cuda",
        mode=mode,
        field_geometry="spherical",
        camera_model=camera,
    )
    image = (
        torch.arange(96 * 128, device="cuda", dtype=torch.float32)
        .reshape(1, 1, 96, 128)
        .expand(4, -1, -1, -1)
    )
    fixation = torch.tensor(
        [[0.5, 0.5], [float("nan"), 0.5], [0.5, float("inf")], [-0.5, 1.5]],
        device="cuda",
    )
    actual = sampler(image, fixation)
    expected = sampler(image, fixation, direct=True)
    torch.testing.assert_close(
        actual, expected, rtol=0, atol=0.002 if mode == "bilinear" else 0
    )
    assert torch.isfinite(actual).all()
    assert torch.count_nonzero(actual[1:]) == 0
    # The optical axis lands exactly between four pixel centers.
    expected_center = (
        image[0, 0, 48, 64] if mode == "nearest" else image[0, 0, 47:49, 63:65].mean()
    )
    torch.testing.assert_close(actual[0, 0, 0], expected_center, rtol=0, atol=0)


@pytest.mark.parametrize("model,distortion", CAMERAS)
def test_angular_calibration_limit(model: str, distortion: tuple[float, ...]) -> None:
    camera = CameraModel(
        model, (96, 128), (70, 70, 63.5, 47.5), distortion, max_angle_deg=20
    )
    sampler = GridSampler(
        60, 2, 12, device="cuda", field_geometry="spherical", camera_model=camera
    )
    image = torch.ones(1, 1, 96, 128, device="cuda")
    expected = sampler.canonical_directions[:, 2] >= math.cos(math.radians(20))
    assert expected.any() and not expected.all()
    torch.testing.assert_close(
        sampler(image, (0.5, 0.5))[0, 0], expected.float(), rtol=0, atol=0
    )


def test_native_fisheye_samples_fixation_behind_optical_hemisphere() -> None:
    camera = CameraModel(
        "fisheye", (240, 320), (110, 110, 159.5, 119.5), max_angle_deg=110
    )
    sampler = GridSampler(
        60, 2, 12, device="cuda", field_geometry="spherical", camera_model=camera
    )
    image = torch.ones((1, 1, 240, 320), device="cuda")
    fixation = torch.tensor([[0.05, 0.05]], device="cuda")
    actual = sampler(image, fixation)
    assert sampler.last_backend == "cuda_calibrated"
    expected = sampler(image, fixation, direct=True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-5)
    assert actual[0, 0, 0] == 1
