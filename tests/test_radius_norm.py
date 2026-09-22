"""The radius norm selects circular or square shells in the Cartesian warp."""

from __future__ import annotations

import math

import pytest
import torch

from fovi.arch.knn import KNNGetterLayer
from fovi.sensing.coords import SamplingCoords, get_sampling_coords, num_sampling_coords
from fovi.sensing.retina import RetinalTransform

NORMS = [2.0, math.inf]


def make_coords(
    radius_norm: float = math.inf,
    res: int = 16,
    max_val: float = 1.0,
    field: str = "planar",
    fov_type: str = "square",
    style: str = "warped_cartesian_as_grid",
) -> SamplingCoords:
    return SamplingCoords(
        16.0,
        0.5,
        res,
        style=style,
        fov_type=fov_type,
        max_val=max_val,
        field_geometry=field,
        radius_norm=radius_norm,
    )


def shell_radius(points: torch.Tensor, radius_norm: float) -> torch.Tensor:
    """Measure radius in the norm whose level sets are the sensor's shells."""
    if radius_norm == math.inf:
        return points.abs().amax(-1)
    return points.norm(dim=-1)


@pytest.mark.parametrize("radius_norm", NORMS)
@pytest.mark.parametrize("res", [1, 2, 15, 16])
@pytest.mark.parametrize("max_val", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("field", ["planar", "legacy", "spherical"])
def test_shells_and_padding(
    radius_norm: float, res: int, max_val: float, field: str
) -> None:
    if field == "spherical" and radius_norm == 2.0 and max_val > 1.0:
        # The Euclidean corner overshoots past the sphere at this extent.
        pytest.skip("Euclidean shells exceed 180 degrees beyond the nominal FoV")
    coords = make_coords(radius_norm, res, max_val, field)
    assert len(coords) == res * res
    assert coords.cortical.shape == (res * res, 2)
    assert (
        num_sampling_coords(
            16, 0.5, res, style=coords.style, fov_type="square",
            radius_norm=radius_norm)
        == res * res
    )
    torch.testing.assert_close(coords.polar[:, 0], coords.cartesian.norm(dim=-1))

    # Points sharing a native shell share one visual shell radius.
    shells = shell_radius(coords.cortical, radius_norm)
    for shell in shells.unique():
        radii = shell_radius(coords.cartesian[shells == shell], radius_norm)
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
        assert (shell_radius(coords.cortical_pad_coords, radius_norm) > max_val).all()
        torch.testing.assert_close(
            coords.visual_to_native(padding), coords.cortical_pad_coords
        )


@pytest.mark.parametrize("max_val", [1.0, 2.0])
@pytest.mark.parametrize("res", [8, 16])
def test_square_shells_fill_the_native_image(max_val: float, res: int) -> None:
    """The infinity norm is what buys full square coverage with no masking."""
    square = make_coords(math.inf, res, max_val)
    assert square.valid_mask.all()
    assert (square.cartesian.abs() < max_val).all()

    # The Euclidean warp pushes the native corners past the square FoV, so it
    # has to mask them; this is the difference the radius norm makes.
    euclidean = make_coords(2.0, res, max_val)
    assert not euclidean.valid_mask.all()


@pytest.mark.parametrize("res", [8, 16])
def test_norms_agree_on_the_axes(res: int) -> None:
    """Both norms are the same one-dimensional CMF along the four axes.

    This is the claim that makes the two a single sensor family rather than
    two sensors, so pin it: the norms may only diverge off-axis. It holds at
    ``max_val == 1``; past that the infinity norm folds ``max_val`` into its
    normalizer and rescales the CMF, while the Euclidean norm does not.
    """
    max_val = 1.0
    square = make_coords(math.inf, res, max_val)
    euclidean = make_coords(2.0, res, max_val)
    axis = torch.linspace(-max_val, max_val, 21)
    zero = torch.zeros_like(axis)
    for native in (torch.stack((axis, zero), -1), torch.stack((zero, axis), -1)):
        torch.testing.assert_close(
            square.native_to_visual(native), euclidean.native_to_visual(native)
        )
        torch.testing.assert_close(
            square.visual_to_native(native), euclidean.visual_to_native(native)
        )

    # Off-axis they must differ: the Euclidean corner overshoots the square one.
    corner = torch.tensor([[max_val, max_val]])
    assert (
        square.native_to_visual(corner).abs().max()
        < euclidean.native_to_visual(corner).abs().max()
    )


@pytest.mark.parametrize("max_val", [0.5, 1.0, 2.0])
def test_square_boundary_is_a_fixed_point(max_val: float) -> None:
    coords = make_coords(math.inf, max_val=max_val)
    edge = torch.stack(
        (torch.full((21,), max_val), torch.linspace(-max_val, max_val, 21)), -1
    )
    torch.testing.assert_close(coords.native_to_visual(edge), edge)
    torch.testing.assert_close(coords.visual_to_native(edge), edge)


@pytest.mark.parametrize("radius_norm", NORMS)
@pytest.mark.parametrize("max_val", [0.5, 1.0, 2.0])
def test_growth_is_monotonic_and_radial(radius_norm: float, max_val: float) -> None:
    coords = make_coords(radius_norm, max_val=max_val)
    native = torch.stack((torch.linspace(0, max_val, 21), torch.zeros(21)), -1)
    spacing = coords.native_to_visual(native)[:, 0].diff()
    assert (spacing[1:] > spacing[:-1]).all()
    # The map is purely radial, so it never rotates a point off its own ray.
    p = torch.tensor([[0.8, 0.3], [-0.2, 0.6]]) * max_val
    mapped = coords.native_to_visual(p)
    torch.testing.assert_close(mapped[:, 0] * p[:, 1], mapped[:, 1] * p[:, 0])


@pytest.mark.parametrize("radius_norm", NORMS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_roundtrip_and_derivatives(
    radius_norm: float, dtype: torch.dtype, device: str
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    coords = make_coords(radius_norm)
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


@pytest.mark.parametrize("res", [8, 15, 16])
def test_square_diagonal_seam_still_roundtrips(res: int) -> None:
    """``amax`` is one-sided where ``|x| == |y|``, and the lattice lands there.

    Only the derivative is discontinuous on the diagonal; the forward/inverse
    pair must stay exact, so a future attempt to smooth the seam cannot
    silently break the fixed point.
    """
    coords = make_coords(math.inf, res=res)
    on_diagonal = coords.cortical[
        coords.cortical[:, 0].abs() == coords.cortical[:, 1].abs()
    ]
    assert len(on_diagonal) > 0, "diagonal points should be present at every res"
    visual = coords.native_to_visual(on_diagonal)
    torch.testing.assert_close(coords.visual_to_native(visual), on_diagonal)
    # The diagonal is the locus where both coordinates set the shell radius.
    torch.testing.assert_close(visual[:, 0].abs(), visual[:, 1].abs())


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
        # 'wang' normalizes the Euclidean radius, so it has no infinity-norm form.
        {"fov_type": "wang"},
        {"radius_norm": 1.0},
        {"radius_norm": "inf"},
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
        "style": "warped_cartesian",
        "fov_type": "square",
        "radius_norm": math.inf,
    }
    args.update(kwargs)
    with pytest.raises(ValueError):
        SamplingCoords(**args)
    with pytest.raises(ValueError):
        get_sampling_coords(**args)


@pytest.mark.parametrize("style", ["isotropic", "logpolar", "uniform"])
def test_radius_norm_rejected_on_styles_that_ignore_it(style: str) -> None:
    """Only the Cartesian warp measures a native radius; fail loudly elsewhere."""
    with pytest.raises(ValueError, match="only applies to"):
        SamplingCoords(16.0, 0.5, 8, style=style, radius_norm=math.inf)
    with pytest.raises(ValueError, match="only applies to"):
        get_sampling_coords(16.0, 0.5, 8, style=style, radius_norm=math.inf)
    # The default norm stays accepted everywhere.
    assert SamplingCoords(16.0, 0.5, 8, style=style, radius_norm=2.0) is not None


def test_square_shells_accept_a_circular_fov() -> None:
    """The warp norm and the mask norm are independent choices."""
    coords = make_coords(math.inf, fov_type="circular")
    assert not coords.valid_mask.all()
    assert coords.valid_mask.any()


def test_vector_grid_orientation_and_image_gradients() -> None:
    kwargs = {
        "resolution": 16,
        "start_res": 32,
        "fixation_size": 32,
        "fov": 16,
        "cmf_a": 0.5,
        "fov_type": "square",
        "radius_norm": math.inf,
        "sampler": "grid_bilinear",
        "device": "cpu",
        "auto_match_cart_resources": False,
        "sampler_backend": "torch",
    }
    flat = RetinalTransform(style="warped_cartesian", **kwargs).eval()
    grid = RetinalTransform(style="warped_cartesian_as_grid", **kwargs).eval()
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
    coords = make_coords(math.inf)
    layer = KNNGetterLayer(25, coords, coords, device="cpu", sample_cortex=True)
    corner_neighbors = layer.knn_indices[:, 0]
    assert (corner_neighbors >= len(coords)).any()
    combined = torch.cat((coords.cortical, coords.cortical_pad_coords))
    distances = (combined[corner_neighbors] - coords.cortical[0]).norm(dim=-1)
    assert distances.max() <= math.sqrt(8) * 2 / coords.resolution + 1e-6
