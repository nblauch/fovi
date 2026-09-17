"""Calibrated crop extent and central-gaze source-pixel density."""

import math
from dataclasses import asdict

import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from fovi.models import FoviNet
from fovi.models.architectures import rescale_fov
from fovi.sensing.calibration import calibrated_cmf_a
from fovi.sensing.coords import SamplingCoords, find_desired_res, isotropic_foveal_ring
from fovi.sensing.projection import CameraModel
from fovi.sensing.retina import RetinalTransform


def configuration(
    camera: CameraModel, side: str = "long", fraction: float = 1.0
) -> DictConfig:
    extent = max(camera.image_size) if side == "long" else min(camera.image_size)
    return OmegaConf.create(
        {
            "saccades": {
                "field_geometry": "spherical",
                "mode": "isotropic",
                "fov_type": "circular",
                "camera_model": asdict(camera),
                "fov_reference_side": side,
                "fov": 16,
                "rescale_fov": 1,
                "fixation_size": extent,
                "fixation_size_min_frac": fraction**2,
                "fixation_size_max_frac": fraction**2,
                "resize_size": 16,
                "auto_match_cart_resources": 0,
                "cmf_a": 0.5,
            }
        }
    )


@pytest.mark.parametrize("size", [(480, 640), (640, 480)])
@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("fraction", [1.0, 0.5])
def test_crop_size_resolves_angular_extent(
    size: tuple[int, int], side: str, fraction: float
) -> None:
    h, w = size
    camera = CameraModel(
        "pinhole",
        size,
        (
            w / (2 * math.tan(math.radians(30))),
            h / (2 * math.tan(math.radians(20))),
            (w - 1) / 2,
            (h - 1) / 2,
        ),
    )
    cfg = rescale_fov(configuration(camera, side, fraction))
    horizontal = (w >= h) if side == "long" else (w < h)
    half_angle = math.radians(30 if horizontal else 20)
    assert cfg.saccades.fov == pytest.approx(
        math.degrees(2 * math.atan(fraction * math.tan(half_angle)))
    )
    assert cfg.saccades.cmf_a == 0.5
    assert rescale_fov(cfg).saccades.fov == cfg.saccades.fov


@pytest.mark.parametrize(
    "model,distortion",
    [
        ("pinhole", ()),
        ("fisheye", ()),
        ("pinhole", (-0.08, 0.01, 0.001, -0.001)),
        ("fisheye", (0.02, -0.003, 0.0002, 0.0)),
    ],
)
@pytest.mark.parametrize("match_resources", [0, 1])
def test_auto_cmf_matches_projected_central_spacing(
    model: str, distortion: tuple[float, ...], match_resources: int
) -> None:
    camera = CameraModel(model, (80, 120), (75, 79, 57.5, 41.5), distortion)
    cfg = configuration(camera, fraction=0.5)
    cfg.saccades.cmf_a = "auto"
    cfg.saccades.auto_match_cart_resources = match_resources
    cfg = rescale_fov(cfg)
    assert isinstance(cfg.saccades.cmf_a, float) and cfg.saccades.cmf_a > 0
    retina = RetinalTransform(
        16,
        fov=cfg.saccades.fov,
        cmf_a=cfg.saccades.cmf_a,
        device="cpu",
        auto_match_cart_resources=match_resources,
        field_geometry="spherical",
        camera_model=asdict(camera),
    )
    assert retina.camera_model is retina.sampler.camera_model
    pixels, valid = retina.sampler.calibrated_pixels(torch.tensor([[0.5, 0.5]]))
    radii = retina.sampler.coords.polar[:, 0]
    first_ring = torch.isclose(radii, torch.unique(radii).sort().values[1])
    spacing = (pixels[0, first_ring] - pixels[0, 0]).norm(dim=-1).min()
    assert valid[0, first_ring].all()
    assert float(spacing) == pytest.approx(1.0, abs=0.05)
    a = retina.sampler.cmf_a
    retina(torch.ones(3, 3, 80, 120), torch.tensor([[0.4, 0.6]]).expand(3, -1))
    assert retina.sampler.cmf_a == a


def test_spherical_identity_transform_fails_before_architecture() -> None:
    cfg = OmegaConf.create({"saccades": {"field_geometry": "spherical", "mode": None}})
    with pytest.raises(ValueError, match="requires a retinal transform"):
        FoviNet(cfg, device="cpu")


def test_padding_error_reports_geometry_and_padding_extent() -> None:
    with pytest.raises(ValueError, match="FoV.*cmf_a.*padding.*radius"):
        SamplingCoords(180, 0.759, 30, field_geometry="spherical")


def test_explicit_spherical_extent_is_preserved() -> None:
    cfg = configuration(CameraModel("pinhole", (80, 120), (75, 75, 59.5, 39.5)))
    cfg.saccades.rescale_fov = 0
    cfg.saccades.fov = 110
    assert rescale_fov(cfg).saccades.fov == 110


@pytest.mark.parametrize("side", ["long", "short"])
def test_rectangular_pixel_crop_selects_reference_axis(side: str) -> None:
    camera = CameraModel("fisheye", (80, 120), (75, 75, 59.5, 39.5))
    cfg = configuration(camera, side)
    cfg.saccades.fixation_size = [40, 90]
    expected = math.degrees((90 if side == "long" else 40) / 75)
    assert rescale_fov(cfg).saccades.fov == pytest.approx(expected)


def test_variable_spherical_crop_requires_explicit_fixed_extent() -> None:
    cfg = configuration(CameraModel("pinhole", (80, 120), (75, 75, 59.5, 39.5)))
    cfg.saccades.fixation_size_min_frac = 0.25
    with pytest.raises(ValueError, match="fixed positive crop"):
        rescale_fov(cfg)


def test_unattainable_auto_density_fails_at_configuration() -> None:
    cfg = configuration(
        CameraModel("pinhole", (80, 120), (75, 75, 59.5, 39.5)), fraction=0.01
    )
    cfg.saccades.cmf_a = "auto"
    with pytest.raises(ValueError, match="could not attain one-pixel"):
        rescale_fov(cfg)


@pytest.mark.parametrize("side,dimension", [("long", "width"), ("short", "height")])
def test_crop_larger_than_camera_names_the_crop(side: str, dimension: str) -> None:
    cfg = configuration(CameraModel("pinhole", (80, 120), (75, 75, 59.5, 39.5)), side)
    cfg.saccades.fixation_size = 160
    with pytest.raises(ValueError, match=f"fixation_size.*{dimension}"):
        rescale_fov(cfg)


@pytest.mark.parametrize("resolution", [8, 16, 32])
@pytest.mark.parametrize("shape", ["circular", "square"])
def test_auto_density_finds_a_known_feasible_resource_matched_grid(
    resolution: int, shape: str
) -> None:
    fov = 60
    known_a = fov * math.exp(-2.5)

    def first_angle(a: float) -> float:
        rings, _ = find_desired_res(
            fov,
            a,
            resolution**2,
            style="isotropic",
            bounds=(1, 1000),
            force_less_than=True,
            quiet=True,
            fov_type=shape,
            field_geometry="spherical",
        )
        ring = isotropic_foveal_ring(fov, a, rings, shape, "spherical")
        return float(ring.norm(dim=-1).min()) * math.radians(fov / 2)

    # Choose a camera for which this discrete grid has exactly one-pixel spacing.
    focal = 1 / first_angle(known_a)
    camera = CameraModel("fisheye", (480, 640), (focal, focal, 319.5, 239.5))
    fitted = calibrated_cmf_a(
        camera,
        fov,
        resolution,
        auto_match_cart_resources=True,
        style="isotropic",
        fov_type=shape,
        gaze_convention="camera_xyz",
    )
    assert focal * first_angle(fitted) == pytest.approx(1.0, abs=0.05)


def test_auto_density_skips_single_ring_candidates() -> None:
    fitted = calibrated_cmf_a(
        CameraModel("fisheye", (80, 120), (75, 75, 59.5, 39.5)),
        30,
        2,
        auto_match_cart_resources=True,
        style="isotropic",
        fov_type="circular",
        gaze_convention="camera_xyz",
    )
    rings, count = find_desired_res(
        30,
        fitted,
        4,
        style="isotropic",
        force_less_than=True,
        quiet=True,
        field_geometry="spherical",
    )
    assert rings >= 2 and count <= 4
    radius = (
        isotropic_foveal_ring(30, fitted, rings, field_geometry="spherical")
        .norm(dim=-1)
        .min()
    )
    assert 75 * float(radius) * math.radians(15) == pytest.approx(1.0, abs=0.05)
