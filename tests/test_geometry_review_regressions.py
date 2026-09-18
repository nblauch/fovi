"""Configuration boundaries must preserve the selected geometry and calibration."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from fovi.arch.knn import get_in_out_coords
from fovi.models.architectures import rescale_fov
from fovi.sensing.coords import SamplingCoords
from fovi.sensing.projection import CameraModel
from fovi.sensing.retina import RetinalTransform
from fovi.sensing.samplers import GridSampler


def test_yaml_camera_compiles_with_fullgraph() -> None:
    reference = CameraModel(
        "fisheye",
        (80, 120),
        (75, 75, 59.5, 39.5),
        (0.01, 0.001, 0.0, 0.0),
        (59.5, 39.5, 65.0),
    )
    camera = CameraModel.from_config(OmegaConf.create(asdict(reference)))
    rays = torch.tensor([[0.1, 0.2, 1.0], [0.0, 0.0, 1.0]])
    actual = torch.compile(camera.project, backend="eager", fullgraph=True)(rays)
    expected = reference.project(rays)
    for value, target in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, target)
    assert hash(camera) == hash(reference)


def test_yaml_sampler_fullgraph_preserves_fixation_gradients() -> None:
    camera = CameraModel("pinhole", (40, 60), (40, 40, 29.5, 19.5))
    sampler = GridSampler(
        20,
        0.5,
        6,
        device="cpu",
        mode="bilinear",
        field_geometry="spherical",
        camera_model=OmegaConf.create(asdict(camera)),
    )
    image = torch.arange(60, dtype=torch.float32)[None, None, None].expand(1, 1, 40, 60)
    fixation = torch.tensor([[0.4, 0.6]], requires_grad=True)
    compiled = torch.compile(
        sampler.calibrated_forward, backend="eager", fullgraph=True
    )
    actual, pixels = compiled(image, fixation, None)
    expected, expected_pixels = sampler.calibrated_forward(image, fixation, None)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(pixels, expected_pixels)
    gradient = torch.autograd.grad(actual.sum(), fixation, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected.sum(), fixation)[0]
    torch.testing.assert_close(gradient, expected_gradient)
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


def test_direct_config_builder_requires_resolved_geometry() -> None:
    cfg = OmegaConf.create({"saccades": {}})
    with pytest.raises(AttributeError, match="field_geometry"):
        rescale_fov(cfg)


@pytest.mark.parametrize("geometry", ["legacy", "spherical"])
def test_supplied_coordinates_must_match_geometry(geometry: str) -> None:
    coords = SamplingCoords(16, 0.5, 8, device="cpu", field_geometry=geometry)
    with pytest.raises(ValueError, match="field_geometry"):
        get_in_out_coords(
            8,
            16,
            0.5,
            2,
            in_coords=coords,
            device="cpu",
            auto_match_cart_resources=False,
            field_geometry="planar",
        )
    actual, output, _ = get_in_out_coords(
        8,
        16,
        0.5,
        2,
        in_coords=coords,
        device="cpu",
        auto_match_cart_resources=False,
        field_geometry=geometry,
    )
    assert actual is coords
    assert output.field_geometry == geometry


def test_planar_calibrated_pixels_reports_geometry_precondition() -> None:
    sampler = GridSampler(16, 0.5, 8, device="cpu")
    with pytest.raises(ValueError, match="requires spherical field_geometry"):
        sampler.calibrated_pixels(torch.tensor([[0.5, 0.5]]))


@pytest.mark.parametrize("geometry", ["planar", "legacy"])
@pytest.mark.parametrize("pooling", ["pooling", "gaussian_pooling"])
def test_retinal_pooling_accepts_model_camera_defaults(
    geometry: str, pooling: str
) -> None:
    kwargs = {"gauss_sigma": 1.0} if pooling == "gaussian_pooling" else {}
    transform = RetinalTransform(
        8,
        start_res=32,
        fixation_size=32,
        device="cpu",
        auto_match_cart_resources=False,
        sampler=pooling,
        sampler_backend="torch",
        field_geometry=geometry,
        camera_model=None,
        gaze_convention="camera_xyz",
        **kwargs,
    )
    image = torch.ones(1, 3, 32, 32)
    actual = transform(image, torch.tensor([[0.5, 0.5]]))
    assert actual.shape == (1, 3, len(transform.sampler.coords))
    assert torch.isfinite(actual).all()
    assert actual.max() > 0
    assert transform.sampler.coords.field_geometry == geometry


def test_mode_change_rejects_integer_bilinear_output() -> None:
    sampler = GridSampler(16, 0.5, 8, device="cpu", output_dtype=torch.uint8)
    with pytest.raises(
        ValueError, match="bilinear sampling requires a floating output dtype"
    ):
        sampler.mode = "bilinear"
    assert sampler.mode == "nearest"


@pytest.mark.parametrize(
    "field,value",
    [
        ("intrinsics", ("75", 75, 59.5, 39.5)),
        ("distortion", (b"0", 0, 0, 0)),
        ("image_circle", (59.5, 39.5, "65")),
        ("image_size", ("80", 120)),
        ("max_angle_deg", "80"),
    ],
)
def test_camera_rejects_string_calibration(
    field: str, value: str | tuple[str | bytes | float, ...]
) -> None:
    config = asdict(CameraModel("pinhole", (80, 120), (75, 75, 59.5, 39.5)))
    config[field] = value
    with pytest.raises(TypeError, match=field):
        CameraModel.from_config(config)


def test_camera_accepts_numpy_and_tensor_real_scalars() -> None:
    camera = CameraModel(
        "pinhole",
        (np.int64(80), torch.tensor(120)),
        (np.float32(75), torch.tensor(75.0), 59.5, 39.5),
        max_angle_deg=torch.tensor(80.0),
    )
    assert camera == CameraModel(
        "pinhole", (80, 120), (75, 75, 59.5, 39.5), max_angle_deg=80
    )
