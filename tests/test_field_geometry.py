"""Visual-field metric regressions, independent of rendering backends."""

from pathlib import Path

import numpy as np
import pytest
import torch
from fovi.sensing.coords import SamplingCoords, _compute_isotropic_r_and_num_theta
from fovi.sensing.manifold import CorticalSensorManifold
from omegaconf import OmegaConf


def test_new_training_config_declares_planar_geometry() -> None:
    cfg = OmegaConf.load(Path(__file__).resolve().parents[1] / "config/default.yaml")
    assert cfg.saccades.field_geometry == "planar"


def test_legacy_preserves_historical_manifold_and_planar_sampling() -> None:
    from scipy.integrate import cumulative_trapezoid
    from scipy.interpolate import interp1d

    a, fov, k = 0.5, 16.0, 10.0
    mesh = np.arange(0, 2 * fov, 0.0001)
    m = k / (a + mesh)
    derivative = -k * (a + mesh) ** -2 * np.sin(np.deg2rad(mesh)) * (
        180 / np.pi
    ) + m * np.cos(np.deg2rad(mesh))
    historical_z = interp1d(
        mesh, cumulative_trapezoid((m**2 - derivative**2) ** 0.5, x=mesh, initial=0)
    )
    radius = np.array([0.0, 0.00015, 0.2, 1.23456, 7.8, 12.1])
    legacy = CorticalSensorManifold(a, fov, k=k, field_geometry="legacy")
    np.testing.assert_array_equal(legacy.z_3d(radius), historical_z(radius))
    np.testing.assert_array_equal(
        legacy.rho_3d(radius),
        k / (a + radius) * np.sin(np.deg2rad(radius)) * (180 / np.pi),
    )
    planar = SamplingCoords(fov, a, 16, device="cpu")
    historical = SamplingCoords(fov, a, 16, device="cpu", field_geometry="legacy")
    torch.testing.assert_close(planar.cartesian, historical.cartesian, rtol=0, atol=0)
    assert not torch.equal(planar.cortical, historical.cortical)
    assert historical.clone().field_geometry == "legacy"
    assert historical.get_strided_coords(2)[0].field_geometry == "legacy"
    assert planar.field_geometry == "planar"


def test_planar_manifold_is_scale_invariant() -> None:
    small = CorticalSensorManifold(0.01, 1, field_geometry="planar")
    large = CorticalSensorManifold(1.5, 150, field_geometry="planar")
    r = np.array([0.05, 0.2, 0.5])
    np.testing.assert_allclose(small.rho_3d(r), large.rho_3d(r * 150))
    np.testing.assert_allclose(small.z_3d(r), large.z_3d(r * 150), atol=1e-3)


def test_spherical_ring_spacing_matches_manifold() -> None:
    radii, counts = _compute_isotropic_r_and_num_theta(
        151.8, 0.759, 250, field_geometry="spherical"
    )
    model = CorticalSensorManifold(0.759, 151.8, field_geometry="spherical")
    r = radii.numpy() * 75.9
    radial = model.m(r[1:-1]) * (r[2:] - r[:-2]) / 2
    tangential = 2 * np.pi * model.rho_3d(r[1:-1]) / counts.numpy()[1:-1]
    np.testing.assert_allclose(tangential[-30:] / radial[-30:], 1, atol=0.025)


def test_wide_spherical_mesh_is_finite_and_rejects_invalid_padding() -> None:
    with np.errstate(invalid="raise"):
        model = CorticalSensorManifold(0.759, 151.8, field_geometry="spherical")
        assert np.isfinite(model.z_3d(np.array([0, 75.9, 100]))).all()
    with pytest.raises(ValueError, match="domain|radius|extent"):
        model.z_3d(150)


def test_geometry_propagates_through_clone_and_stride() -> None:
    coords = SamplingCoords(90, 0.9, 30, field_geometry="spherical")
    assert coords.clone().field_geometry == "spherical"
    stride, _, _ = coords.get_strided_coords(2)
    assert stride.field_geometry == "spherical"
    assert torch.isfinite(stride.cortical).all()


def test_spherical_absolute_fov_changes_ring_counts() -> None:
    _, narrow = _compute_isotropic_r_and_num_theta(
        1, 0.01, 80, field_geometry="spherical"
    )
    _, wide = _compute_isotropic_r_and_num_theta(
        150, 1.5, 80, field_geometry="spherical"
    )
    assert wide[-1] < narrow[-1] * 0.8


def test_spherical_padding_margin_starts_at_retinal_rim() -> None:
    fov = np.rad2deg(1920 * 0.00345 / 2.5)
    coords = SamplingCoords(fov, 0.1 * fov, 5, device="cpu", field_geometry="spherical")
    radii = torch.unique(coords.polar[:, 0]).sort().values
    step = radii[-1] - radii[-2]
    # Only the first outer ring fits within the half-radius padding margin.
    expected_radius = radii[-1] + step
    torch.testing.assert_close(
        coords.cartesian_pad_coords.norm(dim=1),
        expected_radius.expand(len(coords.cartesian_pad_coords)),
    )
    assert coords.cartesian.norm(dim=1).max().item() == pytest.approx(1.0)
    assert torch.isfinite(coords.cortical_pad_coords).all()


def test_coarse_spherical_padding_keeps_one_complete_neighbor_ring() -> None:
    coords = SamplingCoords(150, 15, 3, device="cpu", field_geometry="spherical")
    radii = torch.unique(coords.polar[:, 0]).sort().values
    step = radii[-1] - radii[-2]
    assert step > 0.5
    torch.testing.assert_close(
        coords.cartesian_pad_coords.norm(dim=1),
        (radii[-1] + step).expand(len(coords.cartesian_pad_coords)),
    )
