"""Square shells fill a native image and share analytic rendering geometry."""

from __future__ import annotations

import math

import pytest
import torch

from fovi.arch.knn import KNNGetterLayer
from fovi.sensing.coords import SamplingCoords, get_sampling_coords, num_sampling_coords
from fovi.sensing.retina import RetinalTransform


def make_coords(
    res: int = 16, max_val: float = 1.0, field: str = "planar"
) -> SamplingCoords:
    return SamplingCoords(
        16.0,
        0.5,
        res,
        style="square_foveated_as_grid",
        fov_type="square",
        max_val=max_val,
        field_geometry=field,
    )


@pytest.mark.parametrize("res", [1, 2, 15, 16])
@pytest.mark.parametrize("max_val", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("field", ["planar", "legacy", "spherical"])
def test_shells_and_padding(res: int, max_val: float, field: str) -> None:
    coords = make_coords(res, max_val, field)
    assert len(coords) == res * res
    assert coords.valid_mask.all()
    assert coords.cortical.shape == (res * res, 2)
    assert (coords.cartesian.abs() < max_val).all()
    torch.testing.assert_close(coords.polar[:, 0], coords.cartesian.norm(dim=-1))
    assert (
        num_sampling_coords(16, 0.5, res, style=coords.style, fov_type="square")
        == res * res
    )
    shells = coords.cortical.abs().amax(-1)
    for shell in shells.unique():
        radii = coords.cartesian[shells == shell].abs().amax(-1)
        torch.testing.assert_close(radii, radii[0].expand_as(radii))
    torch.testing.assert_close(
        coords.native_to_visual(-coords.cortical), -coords.cartesian
    )
    torch.testing.assert_close(
        coords.native_to_visual(coords.cortical.flip(-1)), coords.cartesian.flip(-1)
    )
    torch.testing.assert_close(
        coords.visual_to_native(coords.cartesian), coords.cortical
    )
    if res > 1:
        padding = coords.cartesian_pad_coords
        assert (padding.abs().amax(-1) > max_val).all()
        torch.testing.assert_close(
            coords.visual_to_native(padding), coords.cortical_pad_coords
        )


@pytest.mark.parametrize("max_val", [0.5, 1.0, 2.0])
def test_boundary_direction_and_growth(max_val: float) -> None:
    coords = make_coords(max_val=max_val)
    edge = torch.stack(
        (torch.full((21,), max_val), torch.linspace(-max_val, max_val, 21)), -1
    )
    torch.testing.assert_close(coords.native_to_visual(edge), edge)
    torch.testing.assert_close(coords.visual_to_native(edge), edge)
    native = torch.stack((torch.linspace(0, max_val, 21), torch.zeros(21)), -1)
    visual = coords.native_to_visual(native)
    spacing = visual[:, 0].diff()
    assert (spacing[1:] > spacing[:-1]).all()
    p = torch.tensor([[0.8, 0.3], [-0.2, 0.6]]) * max_val
    mapped = coords.native_to_visual(p)
    torch.testing.assert_close(mapped[:, 0] * p[:, 1], mapped[:, 1] * p[:, 0])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_roundtrip_and_derivatives(dtype: torch.dtype, device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    coords = make_coords()
    points = torch.tensor(
        [[0, 0], [0.2, 0.7], [-1.5, 0.3]],
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    visual = coords.native_to_visual(points)
    restored = coords.visual_to_native(visual)
    assert visual.dtype == dtype and visual.device == points.device
    torch.testing.assert_close(restored, points)
    gradient = torch.autograd.grad(restored.sum(), points)[0]
    torch.testing.assert_close(gradient, torch.ones_like(points))
    if dtype == torch.float64:
        assert torch.autograd.gradcheck(coords.native_to_visual, (points,))
        assert torch.autograd.gradcheck(coords.visual_to_native, (points,))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fov": 0},
        {"cmf_a": -1},
        {"cmf_a": float("nan")},
        {"max_val": float("inf")},
        {"max_val": 0},
        {"res": 0},
        {"res": 1.5},
        {"res": True},
        {"fov_type": "circular"},
        {"fov_type": "wang"},
        {"field_geometry": "spherical", "fov": 180, "max_val": 2},
    ],
)
def test_invalid_geometry_rejected_before_sampling(
    kwargs: dict[str, float | int | str],
) -> None:
    args = {
        "fov": 16,
        "cmf_a": 0.5,
        "res": 1,
        "style": "square_foveated",
        "fov_type": "square",
    }
    args.update(kwargs)
    with pytest.raises(ValueError):
        SamplingCoords(**args)
    with pytest.raises(ValueError):
        get_sampling_coords(**args)


def test_vector_grid_orientation_and_image_gradients() -> None:
    kwargs = {
        "resolution": 16,
        "start_res": 32,
        "fixation_size": 32,
        "fov": 16,
        "cmf_a": 0.5,
        "fov_type": "square",
        "sampler": "grid_bilinear",
        "device": "cpu",
        "auto_match_cart_resources": False,
        "sampler_backend": "torch",
    }
    flat = RetinalTransform(style="square_foveated", **kwargs).eval()
    grid = RetinalTransform(style="square_foveated_as_grid", **kwargs).eval()
    image = torch.rand(2, 3, 32, 32, requires_grad=True)
    fixations = torch.tensor([[0.5, 0.5], [0.4, 0.7]])
    expected = flat(image, fixations).reshape(2, 3, 16, 16).transpose(-1, -2).flip(-2)
    actual = grid(image, fixations)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    weights = torch.randn_like(actual)
    a = torch.autograd.grad((actual * weights).sum(), image, retain_graph=True)[0]
    b = torch.autograd.grad((expected * weights).sum(), image)[0]
    torch.testing.assert_close(a, b)
    xy = grid.sampler.coords.as_grid(grid.sampler.coords.cartesian, sample_dim=0)
    assert (xy[:, 1:, 0] > xy[:, :-1, 0]).all()
    assert (xy[1:, :, 1] < xy[:-1, :, 1]).all()


def test_native_receptive_field_includes_corner_padding() -> None:
    coords = make_coords()
    layer = KNNGetterLayer(25, coords, coords, device="cpu", sample_cortex=True)
    corner_neighbors = layer.knn_indices[:, 0]
    assert (corner_neighbors >= len(coords)).any()
    combined = torch.cat((coords.cortical, coords.cortical_pad_coords))
    distances = (combined[corner_neighbors] - coords.cortical[0]).norm(dim=-1)
    assert distances.max() <= math.sqrt(8) * 2 / coords.resolution + 1e-6
