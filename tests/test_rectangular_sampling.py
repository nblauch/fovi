"""Rectangular retinal footprints and grids."""

import math

import pytest
import torch

from fovi.sensing.calibration import calibrated_cmf_a
from fovi.sensing.coords import SamplingCoords, get_isotropic_sampling_coords
from fovi.sensing.projection import CameraModel
from fovi.sensing.retina import RetinalTransform
from fovi.sensing.samplers import GridSampler


def test_two_axis_fov_derives_native_grid_from_pixel_budget() -> None:
    coords = SamplingCoords(
        (96.697, 155.184),
        96.697 * 0.036021,
        128,
        style="warped_cartesian_as_grid",
        fov_type="square",
        field_geometry="spherical",
    )
    assert coords.grid_shape == (119, 138)
    assert len(coords) == 119 * 138
    native_x, native_y = coords.native_half_extents
    assert native_x / native_y == pytest.approx(1.16569, rel=1e-4)
    edge = torch.tensor([[native_x, 0.0], [0.0, native_y]])
    torch.testing.assert_close(
        coords.native_to_visual(edge),
        torch.tensor([[155.184 / 96.697, 0.0], [0.0, 1.0]]),
        rtol=1e-5,
        atol=1e-5,
    )


def test_square_two_axis_fov_preserves_square_native_grid() -> None:
    for radius_norm in (2.0, math.inf):
        coords = SamplingCoords(
            (96.697, 96.697),
            3.5,
            128,
            style="warped_cartesian_as_grid",
            fov_type="square",
            field_geometry="spherical",
            radius_norm=radius_norm,
        )
        assert coords.grid_shape == (128, 128)
        assert coords.native_half_extents == pytest.approx((1.0, 1.0))
        native = torch.tensor([[0.5, 0.0], [0.0, 0.5]])
        visual = coords.native_to_visual(native)
        torch.testing.assert_close(visual[0, 0], visual[1, 1])


def test_tuple_resolution_is_rejected() -> None:
    with pytest.raises(ValueError, match="Resolution"):
        SamplingCoords((96.697, 155.184), 3.5, (100, 160), style="warped_cartesian")


def test_asymmetric_wang_field_is_rejected() -> None:
    with pytest.raises(ValueError, match="Asymmetric FoV"):
        SamplingCoords(
            (96.697, 155.184), 3.5, 128, style="warped_cartesian", fov_type="wang"
        )


@pytest.mark.parametrize(
    "style",
    [
        "isotropic",
        "isotropic_fixn",
        "logpolar",
        "logpolar_as_grid",
        "warped_cartesian",
        "warped_cartesian_as_grid",
        "uniform",
        "uniform_as_grid",
    ],
)
@pytest.mark.parametrize("fov_type", ["circular", "square"])
def test_rectangular_footprint_covers_all_sampling_styles(
    style: str, fov_type: str
) -> None:
    coords = SamplingCoords(
        (96.7, 154.72),
        3.5,
        13,
        style=style,
        fov_type=fov_type,
        field_geometry="spherical",
    )
    if style.startswith("uniform"):
        assert coords.grid_shape == (10, 16)
    elif style.startswith("warped_cartesian"):
        assert coords.grid_shape == (12, 14)
    else:
        assert coords.grid_shape == (13, 13)
    if style != "isotropic":
        assert len(coords) == math.prod(coords.grid_shape)
    if style.endswith("_as_grid"):
        assert coords.as_grid(coords.cartesian, sample_dim=0).shape == (
            *coords.grid_shape,
            2,
        )
    active = coords.cartesian[coords.valid_mask]
    if fov_type == "circular":
        assert torch.all(
            torch.linalg.vector_norm(active / active.new_tensor((1.6, 1.0)), dim=1)
            <= 1 + 1e-5
        )
    else:
        assert active[:, 0].abs().max() <= 1.6 + 1e-5
        assert active[:, 1].abs().max() <= 1 + 1e-5


@pytest.mark.parametrize("fov_type", ["circular", "square"])
@pytest.mark.parametrize("radius_norm", [2.0, math.inf])
def test_rectangular_native_warp_roundtrip_and_grid_orientation(
    fov_type: str, radius_norm: float
) -> None:
    coords = SamplingCoords(
        (96.7, 154.72),
        3.5,
        13,
        style="warped_cartesian_as_grid",
        fov_type=fov_type,
        field_geometry="spherical",
        radius_norm=radius_norm,
    )
    native = torch.tensor([[0.0, 0.0], [0.7, -0.4], [1.4, 0.8]])
    torch.testing.assert_close(
        coords.visual_to_native(coords.native_to_visual(native)),
        native,
        rtol=1e-5,
        atol=1e-5,
    )
    grid = coords.as_grid(coords.plotting, sample_dim=0)
    assert grid[0, 0, 0] < grid[0, -1, 0]
    assert grid[0, 0, 1] > grid[-1, 0, 1]


@pytest.mark.parametrize("fov_type", ["circular", "square"])
def test_rectangular_l2_warp_preserves_scalar_radial_cmf(fov_type: str) -> None:
    rectangular = SamplingCoords(
        (96.7, 154.72),
        3.5,
        13,
        style="warped_cartesian_as_grid",
        fov_type=fov_type,
        field_geometry="spherical",
        radius_norm=2.0,
    )
    scalar = SamplingCoords(
        96.7,
        3.5,
        10,
        style="warped_cartesian_as_grid",
        fov_type=fov_type,
        field_geometry="spherical",
        radius_norm=2.0,
    )
    native = torch.tensor([[0.5, 0.0], [0.0, 0.5], [0.3, 0.4]])
    torch.testing.assert_close(
        rectangular.native_to_visual(native), scalar.native_to_visual(native)
    )
    visual = torch.tensor([[0.5, 0.0], [0.0, 0.5], [0.3, 0.4]])
    torch.testing.assert_close(
        rectangular.visual_to_native(visual), scalar.visual_to_native(visual)
    )
    native_x, native_y = rectangular.native_half_extents
    field_edges = torch.tensor([[native_x, 0.0], [0.0, native_y]])
    torch.testing.assert_close(
        rectangular.native_to_visual(field_edges),
        torch.tensor([[1.6, 0.0], [0.0, 1.0]]),
    )


def test_rectangular_isotropic_keeps_physical_angles() -> None:
    coords, _, _ = get_isotropic_sampling_coords(
        (96.7, 154.72), 3.5, 13, fov_type="square", field_geometry="spherical"
    )
    assert torch.any(coords[:, 0].abs() > 1)
    assert torch.any(torch.isclose(coords[:, 1].abs(), torch.tensor(1.0)))
    assert torch.any(
        torch.all(torch.isclose(coords.abs(), coords.new_tensor((1.6, 1.0))), dim=1)
    )


def test_planar_crop_applies_rectangular_aspect_once() -> None:
    sampler = GridSampler(
        (16, 25.6),
        0.5,
        13,
        device="cpu",
        style="warped_cartesian_as_grid",
        fov_type="square",
        backend="torch",
    )
    grid = sampler.sampling_grid[0, 0]
    active_grid = grid[sampler.valid_mask]
    assert active_grid[:, 0].abs().max() <= 1
    assert active_grid[:, 1].abs().max() <= 1
    image = torch.rand(1, 3, 80, 128)
    output = sampler(image, fix_loc=(0.5, 0.5), fixation_size=(80, 128))
    assert output.shape[-1] == len(sampler.coords)


@pytest.mark.parametrize(
    "style", ["uniform_as_grid", "logpolar_as_grid", "warped_cartesian_as_grid"]
)
def test_rectangular_retinal_transform_returns_requested_image(style: str) -> None:
    retina = RetinalTransform(
        13,
        fov=(16, 25.6),
        cmf_a=0.5,
        style=style,
        device="cpu",
        fixation_size=(80, 128),
        sampler_backend="torch",
    )
    image = torch.rand(1, 3, 80, 128)
    result = retina(image, torch.tensor([[0.5, 0.5]]))
    assert result.shape == (1, 3, *retina.sampler.coords.grid_shape)


def test_spherical_rectangular_logpolar_samples_calibrated_image() -> None:
    camera = CameraModel("fisheye", (120, 192), (70, 70, 95.5, 59.5), max_angle_deg=90)
    retina = RetinalTransform(
        13,
        fov=(96.7, 154.72),
        cmf_a=3.5,
        style="logpolar_as_grid",
        field_geometry="spherical",
        camera_model=camera,
        device="cpu",
        sampler_backend="torch",
    )
    image = torch.rand(1, 3, 120, 192)
    result = retina(image, torch.tensor([[0.5, 0.5]]))
    assert result.shape == (1, 3, 13, 13)
    assert torch.isfinite(result).all()


def test_rectangular_pooling_preserves_public_grid_shape() -> None:
    retina = RetinalTransform(
        13,
        fov=(16, 25.6),
        cmf_a=0.5,
        style="warped_cartesian_as_grid",
        sampler="pooling",
        device="cpu",
        fixation_size=(80, 128),
    )
    result = retina(torch.rand(1, 3, 80, 128), torch.tensor([[0.5, 0.5]]))
    assert result.shape == (1, 3, *retina.sampler.coords.grid_shape)


def test_rectangular_isotropic_auto_calibration_uses_first_ring() -> None:
    camera = CameraModel("fisheye", (120, 192), (70, 70, 95.5, 59.5), max_angle_deg=90)
    cmf_a = calibrated_cmf_a(
        camera,
        30,
        8,
        auto_match_cart_resources=True,
        style="isotropic",
        fov_type="circular",
        gaze_convention="camera_xyz",
    )
    assert cmf_a > 0
