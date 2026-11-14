"""Geometry and attention invariants for cortical RoPE and surface neighborhoods."""

import math
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from fovi.models.knnvit import FoviDinoV3RoPE
from fovi.models.loading import load_config
from fovi.models.omni_isotropic import OmniIsotropicPatchEmbedding, patch_positions
from fovi.sensing import geodesic as surface_geodesic
from fovi.sensing.coords import SamplingCoords
from fovi.sensing.geodesic import patch_geodesic_distances


@pytest.mark.parametrize("a", [0.001, 0.01, 0.1, 1, 10, 100, 1000])
@pytest.mark.parametrize(
    "space", ["cartesian", "cortical_radial_log", "cortical_manifold_3d"]
)
def test_position_geometry_and_scale_invariance(a: float, space: str) -> None:
    patches = SamplingCoords(16, a, 8, style="isotropic_fixn_calibrated")
    scaled = SamplingCoords(160, a * 10, 8, style="isotropic_fixn_calibrated")
    positions = patch_positions(patches, space)
    assert positions.shape == (64, 3 if space.endswith("3d") else 2)
    assert torch.isfinite(positions).all()
    torch.testing.assert_close(
        positions[0], torch.zeros_like(positions[0]), atol=1e-7, rtol=0
    )
    torch.testing.assert_close(
        positions, patch_positions(scaled, space), atol=1e-6, rtol=1e-5
    )
    radius = patches.cartesian.norm(dim=-1)
    if space == "cortical_radial_log":
        expected = torch.log1p(radius * 8 / a) / math.log1p(8 / a)
        torch.testing.assert_close(positions.norm(dim=-1), expected)
        torch.testing.assert_close(positions.norm(dim=-1).max(), torch.tensor(1.0))
    if space == "cortical_manifold_3d":
        # The original manifold uses k=10. One common scale must preserve every chord.
        expected = torch.cdist(patches.cortical, patches.cortical) / (
            10 * math.log1p(8 / a)
        )
        actual = torch.cdist(
            positions, positions, compute_mode="donot_use_mm_for_euclid_dist"
        )
        expected.fill_diagonal_(0)
        torch.testing.assert_close(actual, expected, atol=2e-4, rtol=1e-3)


def test_surface_geodesic_uses_actual_patch_centers_on_flat_mesh() -> None:
    inputs = SamplingCoords(16, 1, 16, style="isotropic_fixn_calibrated")
    patches = SamplingCoords(16, 1, 4, style="isotropic_fixn_calibrated")
    # An exactly flat mesh has an independent analytic distance oracle.
    inputs.cortical = torch.nn.functional.pad(inputs.cartesian.double() * 3, (0, 1))
    inputs.cortical_pad_coords = torch.nn.functional.pad(
        inputs.cartesian_pad_coords.double() * 3, (0, 1)
    )
    patches.cortical = torch.nn.functional.pad(patches.cartesian.double() * 3, (0, 1))
    actual = patch_geodesic_distances(inputs, patches)
    expected = torch.cdist(
        torch.cat((inputs.cortical, inputs.cortical_pad_coords)), patches.cortical
    ).float()
    assert (expected.min(dim=0).values > 0.001).any(), (
        "Need off-sample sources to catch source snapping"
    )
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)


def test_surface_geodesic_differs_from_chord_distance() -> None:
    inputs = SamplingCoords(16, 0.01, 16, style="isotropic_fixn_calibrated")
    patches = SamplingCoords(16, 0.01, 4, style="isotropic_fixn_calibrated")
    actual = patch_geodesic_distances(inputs, patches)
    expected_lower_bound = torch.cdist(
        torch.cat((inputs.cortical, inputs.cortical_pad_coords)),
        patches.cortical,
        compute_mode="donot_use_mm_for_euclid_dist",
    )
    assert (actual + 1e-4 >= expected_lower_bound).all()
    assert (actual - expected_lower_bound).max() > 0.1


def test_surface_cache_reuses_exact_geometry_and_rejects_corruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = SamplingCoords(16, 1, 8, style="isotropic_fixn_calibrated")
    patches = SamplingCoords(16, 1, 4, style="isotropic_fixn_calibrated")
    expected = patch_geodesic_distances(inputs, patches, tmp_path)

    def unexpected_solve(*args: np.ndarray) -> np.ndarray:
        raise RuntimeError("Cache miss")

    monkeypatch.setattr(surface_geodesic, "_solve_surface_geodesics", unexpected_solve)
    torch.testing.assert_close(
        patch_geodesic_distances(inputs, patches, tmp_path), expected, atol=0, rtol=0
    )
    changed = SamplingCoords(16, 1, 5, style="isotropic_fixn_calibrated")
    with pytest.raises(RuntimeError, match="Cache miss"):
        patch_geodesic_distances(inputs, changed, tmp_path)
    cache_file = next(tmp_path.glob("*.npz"))
    with np.load(cache_file, allow_pickle=False) as cached:
        values = dict(cached)
    values["distances"][0, 0] += 1
    np.savez_compressed(cache_file, **values)
    with pytest.raises(ValueError, match="Invalid geodesic cache"):
        patch_geodesic_distances(inputs, patches, tmp_path)


@pytest.mark.parametrize("dimensions", [2, 3])
def test_rotary_width_relative_positions_and_level_selection(dimensions: int) -> None:
    coords = torch.randn(3, 16, dimensions)
    rope = FoviDinoV3RoPE(100, 64, coords, device="cpu")
    indices = torch.tensor([2, 0, 2, 1])
    pixels = torch.zeros(4, 3, 16)
    cos, sin = rope(pixels, indices)
    assert cos.shape == sin.shape == (4, 1, 16, 64)
    torch.testing.assert_close(cos[0], cos[2], atol=0, rtol=0)
    torch.testing.assert_close(cos.square() + sin.square(), torch.ones_like(cos))
    q, k = torch.randn_like(cos), torch.randn_like(cos)

    def rotate(x: torch.Tensor, c: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        return x * c + torch.cat((-x[..., half:], x[..., :half]), dim=-1) * s

    attention = rotate(q, cos, sin) @ rotate(k, cos, sin).transpose(-1, -2)
    shifted = FoviDinoV3RoPE(100, 64, coords + 0.25, device="cpu")
    shifted_cos, shifted_sin = shifted(pixels, indices)
    shifted_attention = rotate(q, shifted_cos, shifted_sin) @ rotate(
        k, shifted_cos, shifted_sin
    ).transpose(-1, -2)
    torch.testing.assert_close(attention, shifted_attention, atol=3e-5, rtol=1e-4)
    if dimensions == 2:
        frequencies = 1 / 100 ** torch.arange(0, 1, 4 / 64)
        angles = (2 * math.pi * coords[indices, ..., None] * frequencies).flatten(-2)
        angles = torch.cat((angles, angles), -1).unsqueeze(1)
        torch.testing.assert_close(cos, angles.cos(), atol=0, rtol=0)
        torch.testing.assert_close(sin, angles.sin(), atol=0, rtol=0)
    else:
        assert torch.bincount(rope.coordinate_axes).tolist() == [11, 11, 10]
        for axis in range(3):
            altered = coords.clone()
            altered[..., axis] += 0.1
            changed = FoviDinoV3RoPE(100, 64, altered, device="cpu")(pixels, indices)[0]
            assert not torch.equal(changed, cos)


def test_rope_variants_share_geodesic_projection_and_initial_weights() -> None:
    banks = []
    for space in ("cartesian", "cortical_radial_log", "cortical_manifold_3d"):
        torch.manual_seed(1)
        banks.append(
            OmniIsotropicPatchEmbedding(
                fov=16,
                cmf_a=(0.001, 1, 1000),
                resolution=16,
                patch_size=4,
                reference_side=4,
                in_channels=3,
                embed_dim=16,
                sample_cortex="geodesic",
                position_coordinate_space=space,
                device="cpu",
            )
        )
    x, levels = torch.randn(3, 3, 256), torch.arange(3)
    baseline = banks[0](x, levels)
    for bank in banks[1:]:
        torch.testing.assert_close(bank(x, levels), baseline, atol=0, rtol=0)
        assert (bank.geometry.reverse_ptr.diff(dim=1) > 0).all()
        for name, value in banks[0].geometry.named_buffers():
            torch.testing.assert_close(
                dict(bank.geometry.named_buffers())[name], value, atol=0, rtol=0
            )


@pytest.mark.parametrize(
    "config_name,space",
    [
        ("dinov3_isotropic_multi_cmf_geodesic", "cartesian"),
        ("dinov3_isotropic_multi_cmf_radial_log", "cortical_radial_log"),
        ("dinov3_isotropic_multi_cmf_manifold_3d", "cortical_manifold_3d"),
    ],
)
def test_geodesic_recipes_only_change_metric_and_positions(
    config_name: str, space: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("FOVI_DATASETS_DIR", str(tmp_path / "datasets"))
    directory = Path(__file__).parents[1] / "config"
    baseline = load_config(
        "dinov3_isotropic_multi_cmf", False, directory, device="cpu"
    )[0]
    candidate = load_config(config_name, False, directory, device="cpu")[0]
    assert candidate.model.vit.position_coordinate_space == space
    assert candidate.saccades.sample_cortex == "geodesic"
    candidate.model.vit.position_coordinate_space = (
        baseline.model.vit.position_coordinate_space
    )
    candidate.saccades.sample_cortex = baseline.saccades.sample_cortex
    assert OmegaConf.to_container(candidate, resolve=True) == OmegaConf.to_container(
        baseline, resolve=True
    )
