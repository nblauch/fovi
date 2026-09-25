"""Calibrated image sampling against independent camera and image oracles."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from fovi.sensing.projection import CameraModel, angular_directions, gaze_rotation
from fovi.sensing.samplers import GridSampler


@pytest.mark.parametrize(
    "model,distortion",
    [
        ("pinhole", (-0.08, 0.02, 0.001, -0.002, 0.003)),
        ("fisheye", (0.02, -0.003, 0.001, 0.0001)),
    ],
)
def test_projection_matches_opencv_and_roundtrips(
    model: str, distortion: tuple[float, ...]
) -> None:
    cv2 = pytest.importorskip("cv2")
    camera = CameraModel(model, (480, 640), (300, 290, 317, 237), distortion)
    xyz = np.array([[0, 0, 1], [0.4, -0.3, 1], [-0.8, 0.5, 1]], dtype=np.float64)
    k = np.array([[300, 0, 317], [0, 290, 237], [0, 0, 1]], dtype=np.float64)
    project = cv2.projectPoints if model == "pinhole" else cv2.fisheye.projectPoints
    expected, _ = project(
        xyz.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), k, np.array(distortion)
    )
    pixels, valid = camera.project(torch.tensor(xyz))
    np.testing.assert_allclose(pixels.numpy(), expected[:, 0], atol=1e-8)
    rays, inverse_valid = camera.unproject(pixels)
    torch.testing.assert_close(
        rays,
        torch.nn.functional.normalize(torch.tensor(xyz), dim=-1),
        atol=1e-8,
        rtol=1e-8,
    )
    assert (valid & inverse_valid).all()


@pytest.mark.parametrize("distortion", [(), (0.01, -0.001, 0.0002, 0.0)])
def test_fisheye_projects_full_frame_beyond_front_hemisphere(
    distortion: tuple[float, ...],
) -> None:
    camera = CameraModel(
        "fisheye",
        (240, 320),
        (110, 110, 159.5, 119.5),
        distortion,
        max_angle_deg=110,
    )
    corners = torch.tensor(
        [[0, 0], [319, 0], [0, 239], [319, 239]], dtype=torch.float64
    )
    rays, inverse_valid = camera.unproject(corners)
    assert inverse_valid.all()
    assert (rays[:, 2] < 0).all()
    recovered, project_valid = camera.project(rays)
    assert project_valid.all()
    torch.testing.assert_close(recovered, corners, atol=1e-8, rtol=0)
    _, limited_valid = replace(camera, max_angle_deg=90).project(rays)
    assert not limited_valid.any()


def test_fisheye_angular_domain_keeps_pinhole_limit() -> None:
    with pytest.raises(ValueError, match="max_angle_deg"):
        CameraModel("pinhole", (240, 320), (110, 110, 159.5, 119.5), max_angle_deg=110)
    with pytest.raises(ValueError, match="max_angle_deg"):
        CameraModel("fisheye", (240, 320), (110, 110, 159.5, 119.5), max_angle_deg=180)


def test_fisheye_sampler_retains_rearward_fixation() -> None:
    camera = CameraModel(
        "fisheye", (240, 320), (110, 110, 159.5, 119.5), max_angle_deg=110
    )
    sampler = GridSampler(
        60, 2, 12, device="cpu", field_geometry="spherical", camera_model=camera
    )
    image = torch.ones((1, 1, 240, 320))
    fixation = torch.tensor([[0.05, 0.05]])
    gaze, valid = camera.unproject(torch.tensor([[15.5, 5.5]]))
    assert valid.all() and (gaze[:, 2] < 0).all()
    assert sampler(image, fixation)[0, 0, 0] == 1


def test_window_uses_selected_angular_axis() -> None:
    fx = 640 / (2 * math.tan(math.radians(30)))
    fy = 480 / (2 * math.tan(math.radians(20)))
    camera = CameraModel("pinhole", (480, 640), (fx, fy, 319.5, 239.5))
    assert camera.field_of_view("long") == pytest.approx(60)
    assert camera.field_of_view("short") == pytest.approx(40)
    assert camera.field_of_view("long", 0.5) == pytest.approx(
        math.degrees(2 * math.atan(math.tan(math.radians(30)) * 0.5))
    )


@pytest.mark.parametrize("convention", ["camera_xyz", "pan_tilt"])
def test_gaze_rotation_preserves_angles(convention: str) -> None:
    target = torch.nn.functional.normalize(torch.tensor([[0.5, -0.3, 1.0]]), dim=-1)
    rotation = gaze_rotation(target, convention)
    torch.testing.assert_close(rotation[..., 2], target)
    rays = angular_directions(torch.tensor([[0.0, 0.0], [1.0, 0.0]]), 120)
    rotated = rays @ rotation[0].T
    assert float(rotated[0] @ rotated[1]) == pytest.approx(0.5, abs=1e-6)


@pytest.mark.parametrize("convention", ["camera_xyz", "pan_tilt"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("autocast", [False, True])
def test_gaze_rotation_retains_float32_precision(
    convention: str,
    device: str,
    autocast: bool,
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    # Near-axis commands still need the small second-order diagonal terms.
    x = torch.linspace(-0.02, 0.02, 48)
    targets = torch.stack((x, x.flip(0) * 0.7, torch.ones_like(x)), -1)
    targets = torch.cat((targets, torch.tensor([[0.5, -0.3, 1.0], [-0.8, 0.4, 0.2]])))
    expected = gaze_rotation(targets.double(), convention)
    targets = targets.to(device)
    previous_precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("high")
        with torch.autocast(device, dtype=torch.bfloat16, enabled=autocast):
            actual = gaze_rotation(targets, convention)
    finally:
        torch.set_float32_matmul_precision(previous_precision)
    assert actual.dtype == torch.float32
    actual = actual.double().cpu()
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-7)
    torch.testing.assert_close(
        actual.transpose(-1, -2) @ actual,
        torch.eye(3, dtype=torch.float64).expand_as(actual),
        atol=3e-7,
        rtol=3e-7,
    )
    torch.testing.assert_close(
        actual[..., 2],
        torch.nn.functional.normalize(targets.double().cpu(), dim=-1),
        atol=2e-7,
        rtol=2e-7,
    )


def test_off_axis_sampling_uses_rotated_rays() -> None:
    camera = CameraModel("pinhole", (80, 120), (60, 60, 59.5, 39.5))
    sampler = GridSampler(
        30,
        0.5,
        8,
        device="cpu",
        mode="bilinear",
        field_geometry="spherical",
        camera_model=camera,
    )
    image = torch.arange(120, dtype=torch.float32)[None, None, None].expand(
        1, 1, 80, 120
    )
    fixation = torch.tensor([[0.5, 0.7]])
    values = sampler(image, fixation)
    center = math.atan((0.7 * 120 - 0.5 - 59.5) / 60)
    # The center node follows the fixation, while other rays rotate with it.
    assert values[0, 0, 0] == pytest.approx(83.5, abs=1e-4)
    rays = sampler.canonical_directions
    x = math.cos(center) * rays[:, 0] + math.sin(center) * rays[:, 2]
    z = -math.sin(center) * rays[:, 0] + math.cos(center) * rays[:, 2]
    expected = 60 * x / z + 59.5
    torch.testing.assert_close(values[0, 0], expected, atol=2e-4, rtol=1e-5)


def test_outside_calibration_produces_finite_zero_samples() -> None:
    camera = CameraModel(
        "pinhole", (80, 120), (60, 60, 59.5, 39.5), (-0.08, 0.02, 0, 0)
    )
    sampler = GridSampler(
        120,
        2,
        8,
        device="cpu",
        mode="bilinear",
        field_geometry="spherical",
        camera_model=camera,
    )
    image = torch.ones(1, 1, 80, 120)
    rotation = torch.diag(torch.tensor([-1.0, 1.0, -1.0]))[None]
    actual = sampler(image, rotation=rotation)
    assert torch.isfinite(actual).all()
    assert torch.count_nonzero(actual) == 0


def test_resized_calibration_preserves_normalized_pixel_and_fov() -> None:
    camera = CameraModel(
        "fisheye", (480, 640), (300, 300, 317, 237), (0.02, -0.003, 0.001, 0.0001)
    )
    resized = camera.resized((240, 320))
    directions = torch.tensor([[0.4, -0.3, 1.0]])
    pixels, _ = camera.project(directions)
    smaller, _ = resized.project(directions)
    torch.testing.assert_close(smaller + 0.5, (pixels + 0.5) / 2)
    assert camera.field_of_view("long", 0.75) == pytest.approx(
        resized.field_of_view("long", 0.75)
    )


@pytest.mark.parametrize("model", ["pinhole", "fisheye"])
def test_optical_axis_has_correct_projection_and_inverse_gradients(model: str) -> None:
    camera = CameraModel(model, (80, 120), (75, 79, 59.5, 39.5))
    rays = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64, requires_grad=True)
    pixels = torch.tensor([[59.5, 39.5]], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x: camera.project(x)[0], rays)
    assert torch.autograd.gradcheck(lambda x: camera.unproject(x)[0], pixels)


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_calibrated_bilinear_preserves_floating_dtype(dtype: torch.dtype) -> None:
    camera = CameraModel("pinhole", (80, 120), (60, 60, 59.5, 39.5))
    sampler = GridSampler(
        30,
        0.5,
        8,
        device="cpu",
        mode="bilinear",
        field_geometry="spherical",
        camera_model=camera,
    )
    image = torch.full((1, 3, 80, 120), 0.375, dtype=dtype)
    actual = sampler(image, torch.tensor([[0.5, 0.6]]))
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, torch.full_like(actual, 0.375))
    with pytest.raises(ValueError, match="fixation_size"):
        sampler(image, torch.tensor([[0.5, 0.6]]), torch.tensor([[40, 40]]))
