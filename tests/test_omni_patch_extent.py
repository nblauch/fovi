"""Fixed-count patch centers must honor the established boundary inset."""

import pytest
import torch

from fovi.arch.knn import get_in_out_coords
from fovi.models.omni_isotropic import OmniIsotropicPatchEmbedding
from fovi.sensing.coords import SamplingCoords


@pytest.mark.parametrize("a", [0.001, 0.01, 0.1, 1, 10, 100, 1000])
def test_auto_patch_extent_matches_standard_placement(a: float) -> None:
    inputs = SamplingCoords(16, a, 32, style="isotropic_fixn_calibrated")
    provisional = SamplingCoords(16, a, 8, style="isotropic_fixn_calibrated")
    outer_rings = provisional.polar[:, 0].unique()[-2:]
    expected_extent = outer_rings.mean()
    bank = OmniIsotropicPatchEmbedding(
        fov=16,
        cmf_a=[a],
        resolution=32,
        patch_size=4,
        reference_side=4,
        in_channels=1,
        embed_dim=1,
        sample_cortex=True,
        device="cpu",
        patch_max_coord_val="auto",
    )
    _, standard, _ = get_in_out_coords(
        32,
        16,
        a,
        4,
        style=inputs.style,
        in_coords=inputs,
        in_cart_res=32,
        device="cpu",
        max_out_coord_val="auto",
    )
    assert bank.num_patches == 64
    assert bank.geometry.samples == 1024
    torch.testing.assert_close(
        bank.positions[0], standard.cartesian_rowcol, atol=0, rtol=0
    )
    torch.testing.assert_close(bank.positions.norm(dim=-1).max(), expected_extent)
    assert bank.positions.norm(dim=-1).max() < 1
    assert (torch.diff(bank.geometry.reverse_ptr) > 0).all(), (
        "Every real sample must remain covered"
    )


def test_legacy_patch_extent_remains_at_perimeter() -> None:
    bank = OmniIsotropicPatchEmbedding(
        fov=16,
        cmf_a=[1],
        resolution=16,
        patch_size=4,
        reference_side=4,
        in_channels=1,
        embed_dim=1,
        sample_cortex=True,
        device="cpu",
    )
    torch.testing.assert_close(bank.positions.norm(dim=-1).max(), torch.tensor(1.0))


def test_inset_plotting_coordinates_use_actual_visual_radius() -> None:
    coords = SamplingCoords(16, 1, 8, style="isotropic_fixn_calibrated", max_val=0.75)
    expected_radius = torch.log1p(coords.polar[:, 0] * 8)
    torch.testing.assert_close(coords.plotting.norm(dim=-1), expected_radius)
