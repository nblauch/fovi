"""Isotropic plotting layouts must not change the sampled manifold."""

import math

import pytest
import torch

from fovi.models.omni_isotropic import patch_positions
from fovi.sensing.coords import SamplingCoords, SamplingCoordsBank, get_sampling_coords


@pytest.mark.parametrize("field_geometry", ["planar", "spherical"])
@pytest.mark.parametrize("extent", [0.75, 1.0])
@pytest.mark.parametrize("layout", ["v1like", "schwartz", "warp"])
def test_calibrated_layout_preserves_geometry(
    field_geometry: str, extent: float, layout: str
) -> None:
    baseline = SamplingCoords(
        16,
        1,
        8,
        style="isotropic_fixn_calibrated",
        max_val=extent,
        field_geometry=field_geometry,
    )
    coords = baseline.clone(isotropic_plotting_type=layout)
    for attribute in (
        "cartesian",
        "polar",
        "cortical",
        "cartesian_pad_coords",
        "cortical_pad_coords",
    ):
        torch.testing.assert_close(
            getattr(coords, attribute), getattr(baseline, attribute), atol=0, rtol=0
        )
    for space in ("cartesian", "cortical_radial_log", "cortical_manifold_3d"):
        torch.testing.assert_close(
            patch_positions(coords, space),
            patch_positions(baseline, space),
            atol=0,
            rtol=0,
        )
    direct = get_sampling_coords(
        16,
        1,
        8,
        style="isotropic_fixn_calibrated",
        max_val=extent,
        field_geometry=field_geometry,
        isotropic_plotting_type=layout,
    )
    torch.testing.assert_close(direct[2], coords.plotting, atol=0, rtol=0)
    assert torch.isfinite(coords.plotting).all()
    visual = coords.cartesian * 8
    if layout == "warp":
        expected_radius = torch.log1p(visual.norm(dim=-1)) / math.log1p(8)
        torch.testing.assert_close(coords.plotting.norm(dim=-1), expected_radius)
    else:
        expected_angle = torch.atan2(visual[:, 1], visual[:, 0].abs() + 1)
        if layout == "v1like":
            expected_angle = -expected_angle
        torch.testing.assert_close(coords.plotting[:, 1], expected_angle)
        left = coords.cartesian[:, 0] < 0
        if layout == "v1like":
            assert (coords.plotting[left, 0] > 0).all()
            assert (coords.plotting[~left, 0] < 0).all()
        else:
            assert (coords.plotting[left, 0] < 0).all()
            assert (coords.plotting[~left, 0] > 0).all()


@pytest.mark.parametrize("layout", ["v1like", "schwartz", "warp"])
def test_isotropic_bank_preserves_requested_layout(layout: str) -> None:
    bank = SamplingCoordsBank(
        16,
        [0.1, 1],
        8,
        style="isotropic_fixn_calibrated",
        isotropic_plotting_type=layout,
    )
    for index, cmf_a in enumerate(bank.cmf_a):
        scalar = SamplingCoords(
            16,
            cmf_a,
            8,
            style="isotropic_fixn_calibrated",
            isotropic_plotting_type=layout,
        )
        torch.testing.assert_close(
            bank.for_level(index).plotting, scalar.plotting, atol=0, rtol=0
        )


@pytest.mark.parametrize("layout", ["v1like", "schwartz", "warp", "unused"])
def test_isotropic_plotting_option_does_not_affect_warped_grid(layout: str) -> None:
    baseline = SamplingCoords(16, 1, 8, style="warped_cartesian_as_grid")
    actual = baseline.clone(isotropic_plotting_type=layout)
    for attribute in ("cartesian", "polar", "plotting", "cortical"):
        torch.testing.assert_close(
            getattr(actual, attribute), getattr(baseline, attribute), atol=0, rtol=0
        )
