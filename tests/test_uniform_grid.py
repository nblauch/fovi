"""Uniform grids preserve their footprint and analytic renderer mapping."""

import pytest
import torch

from fovi.sensing.coords import SamplingCoords


@pytest.mark.parametrize("style", ["uniform", "uniform_as_grid"])
@pytest.mark.parametrize("max_val", [0.5, 1.0])
def test_uniform_circular_mask_keeps_grid_and_masks_corners(
    style: str, max_val: float
) -> None:
    coords = SamplingCoords(
        120.0, 4.0, 16, style=style, max_val=max_val, fov_type="circular"
    )
    expected = coords.cartesian.norm(dim=-1) <= max_val

    assert len(coords) == 16**2
    assert expected.any() and (~expected).any()
    torch.testing.assert_close(coords.valid_mask, expected)
    assert coords.fov_padding_coords.shape == (0, 2)


def test_uniform_square_mask_keeps_full_image() -> None:
    coords = SamplingCoords(120.0, 4.0, 16, style="uniform_as_grid", fov_type="square")

    assert coords.valid_mask.all()


@pytest.mark.parametrize("style", ["uniform", "uniform_as_grid"])
@pytest.mark.parametrize("field", ["planar", "spherical"])
def test_uniform_native_mapping_is_identity_with_padding(
    style: str, field: str
) -> None:
    coords = SamplingCoords(120.0, 4.0, 16, style=style, field_geometry=field)
    points = torch.tensor([[[0.0, 0.0], [-0.25, 0.75]], [[1.5, -2.0], [0.9, 0.9]]])

    torch.testing.assert_close(coords.native_to_visual(points), points, atol=0, rtol=0)
    torch.testing.assert_close(coords.visual_to_native(points), points, atol=0, rtol=0)


def test_uniform_native_mapping_rejects_invalid_coordinate_axis() -> None:
    coords = SamplingCoords(120.0, 4.0, 16, style="uniform_as_grid")
    with pytest.raises(ValueError, match="two-coordinate axis"):
        coords.native_to_visual(torch.zeros(4, 3))
    with pytest.raises(ValueError, match="two-coordinate axis"):
        coords.visual_to_native(torch.zeros(4, 3))
