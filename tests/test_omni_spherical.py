"""Spherical omni geometry uses angular circumference and one consistent metric."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from fovi.models.omni_isotropic import OmniIsotropicPatchEmbedding, patch_positions
from fovi.sensing.coords import SamplingCoords, SamplingCoordsBank
from fovi.sensing.geodesic import patch_geodesic_distances
from fovi.sensing.isotropic_rings import calibrated_isotropic_rings
from fovi.sensing.manifold import CorticalSensorManifold


@pytest.mark.parametrize("resolution", [8, 128])
def test_spherical_bank_retains_exact_budget_and_finite_geometry(
    resolution: int,
) -> None:
    bank = SamplingCoordsBank(
        120,
        [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        resolution,
        style="isotropic_fixn_calibrated",
        field_geometry="spherical",
    )
    assert bank.cartesian.shape == (7, resolution**2, 2)
    assert bank.valid_mask.all()
    for level in range(7):
        coords = bank.for_level(level)
        assert torch.unique(coords.cartesian, dim=0).shape[0] == resolution**2
        torch.testing.assert_close(coords.polar[:, 0].max(), torch.tensor(1.0))
        assert torch.isfinite(coords.cortical).all()
        assert torch.isfinite(coords.cortical_pad_coords).all()


def test_spherical_calibration_accounts_for_ring_circumference() -> None:
    a, radius, count = 1000.0, 80.0, 256
    planar_r, planar_n = calibrated_isotropic_rings(a, count, radius)
    spherical_r, spherical_n = calibrated_isotropic_rings(
        a, count, radius, field_geometry="spherical"
    )

    def spherical_spacing_error(radii: np.ndarray, counts: np.ndarray) -> float:
        radial_step = math.log1p(radius / a) / (len(radii) - 1)
        circumference = 360 * np.sin(np.deg2rad(radii[1:])) / (a + radii[1:])
        return float(
            np.average(
                np.log(circumference / counts[1:] / radial_step) ** 2,
                weights=counts[1:],
            )
        )

    assert spherical_spacing_error(spherical_r, spherical_n) < spherical_spacing_error(
        planar_r, planar_n
    )
    assert spherical_n.sum() == count
    assert spherical_r[-1] == radius
    # In a sufficiently narrow field, the angular metric converges to planar.
    small_r, small_n = calibrated_isotropic_rings(
        0.001, count, 0.00008, field_geometry="spherical"
    )
    np.testing.assert_allclose(small_r, planar_r / 1e6, atol=1e-18)
    np.testing.assert_array_equal(small_n, planar_n)


@pytest.mark.parametrize(
    "space", ["cartesian", "cortical_radial_log", "cortical_manifold_3d"]
)
def test_spherical_positions_and_geodesics_use_the_same_surface(space: str) -> None:
    coords = SamplingCoords(
        120, 10, 8, style="isotropic_fixn_calibrated", field_geometry="spherical"
    )
    patches = SamplingCoords(
        120, 10, 4, style="isotropic_fixn_calibrated", field_geometry="spherical"
    )
    positions = patch_positions(patches, space)
    assert torch.isfinite(positions).all()
    if space == "cortical_manifold_3d":
        expected = torch.stack(
            (-patches.cortical[:, 1], patches.cortical[:, 0], patches.cortical[:, 2]),
            dim=-1,
        )
        expected /= 10 * math.log1p(60 / 10)
        torch.testing.assert_close(positions, expected, atol=1e-6, rtol=1e-5)
        manifold = CorticalSensorManifold(10, 120, field_geometry="spherical")
        np.testing.assert_allclose(
            patches.cortical[:, :2].norm(dim=-1).numpy(),
            manifold.rho_3d(patches.polar[:, 0].numpy() * 60),
            atol=2e-6,
        )
    distances = patch_geodesic_distances(coords, patches)
    chord = torch.cdist(
        torch.cat((coords.cortical, coords.cortical_pad_coords)), patches.cortical
    )
    assert (distances + 1e-4 >= chord).all()
    assert (distances - chord).max() > 0.01
    planar = SamplingCoords(120, 10, 4, style="isotropic_fixn_calibrated")
    with pytest.raises(ValueError, match="geometry"):
        patch_geodesic_distances(coords, planar)


@pytest.mark.parametrize("metric", [True, "geodesic"])
def test_spherical_projection_covers_samples_and_backpropagates(
    metric: bool | str,
) -> None:
    bank = OmniIsotropicPatchEmbedding(
        fov=120,
        cmf_a=[0.01, 10, 1000],
        resolution=16,
        patch_size=4,
        reference_side=4,
        in_channels=3,
        embed_dim=8,
        sample_cortex=metric,
        field_geometry="spherical",
        patch_max_coord_val="auto",
        device="cpu",
    )
    assert (bank.geometry.reverse_ptr.diff(dim=1) > 0).all()
    pixels = torch.randn(3, 3, 256, requires_grad=True)
    output = bank(pixels, torch.arange(3))
    assert output.shape == (3, 8, 16)
    output.square().sum().backward()
    assert torch.isfinite(pixels.grad).all()
    assert torch.isfinite(bank.weight.grad).all()


def test_spherical_calibration_rejects_unsupported_diameter() -> None:
    with pytest.raises(ValueError, match="180"):
        SamplingCoords(
            181, 10, 8, style="isotropic_fixn_calibrated", field_geometry="spherical"
        )
