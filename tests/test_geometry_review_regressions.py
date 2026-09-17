"""Configuration boundaries must preserve the selected geometry and calibration."""

from dataclasses import asdict

import pytest
import torch
from omegaconf import OmegaConf

from fovi.arch.knn import get_in_out_coords
from fovi.models.architectures import rescale_fov
from fovi.sensing.coords import SamplingCoords
from fovi.sensing.projection import CameraModel
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
