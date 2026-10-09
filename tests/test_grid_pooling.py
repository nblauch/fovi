"""Mask-normalized 2D retinal pooling and automatic sampler selection."""

import pytest
import torch
from fovi.arch.knn import KNNPoolingLayer
from fovi.arch.pooling import GridPoolingLayer
from fovi.sensing.coords import SamplingCoords
from fovi.sensing.projection import CameraModel
from fovi.sensing.retina import RetinalTransform
from fovi.sensing.samplers import PoolingGridSampler


def grids(
    style: str = "warped_cartesian_as_grid", fov_type: str = "circular"
) -> tuple[SamplingCoords, SamplingCoords]:
    low = SamplingCoords(50, 5, 4, style=style, fov_type=fov_type)
    return low.clone(resolution=8), low


def canonical(grid: torch.Tensor) -> torch.Tensor:
    return grid.flip(-2).transpose(-2, -1).flatten(-2)


@pytest.mark.parametrize("style", ["warped_cartesian_as_grid", "uniform_as_grid"])
def test_null_kernel_is_exact_downscale_factor(style: str) -> None:
    high, low = grids(style)
    if style == "warped_cartesian_as_grid":
        high.valid_mask.fill_(True)
        low.valid_mask.fill_(True)
    pool = GridPoolingLayer(None, high, low)
    image = torch.arange(64, dtype=torch.float32).reshape(1, 1, 8, 8)
    assert pool.kernel_size == pool.stride == 2
    assert pool.k == 4
    actual = low.as_grid(pool(canonical(image)))
    torch.testing.assert_close(actual, torch.nn.functional.avg_pool2d(image, 2))


def test_explicit_kernel_uses_overlapping_windows_and_preserves_shape() -> None:
    high, low = grids("uniform_as_grid")
    pool = GridPoolingLayer(9, high, low)
    assert pool.kernel_size == 3
    image = torch.arange(64, dtype=torch.float32).reshape(1, 1, 8, 8)
    expected = torch.tensor(
        [
            [9, 11, 13, 14.5],
            [25, 27, 29, 30.5],
            [41, 43, 45, 46.5],
            [53, 55, 57, 58.5],
        ]
    )[None, None]
    torch.testing.assert_close(low.as_grid(pool(canonical(image))), expected)


def test_kernel_smaller_than_stride_selects_center_with_same_padding_convention() -> (
    None
):
    high, low = grids("uniform_as_grid")
    image = torch.arange(64, dtype=torch.float32).reshape(1, 1, 8, 8)
    pool = GridPoolingLayer(1, high, low)
    torch.testing.assert_close(
        low.as_grid(pool(canonical(image))), image[..., 1::2, 1::2]
    )


@pytest.mark.parametrize("mode", ["avg", "gaussian", "max"])
@pytest.mark.parametrize("fov_type", ["circular", "square"])
def test_masked_values_never_contribute_and_boundaries_are_normalized(
    mode: str, fov_type: str
) -> None:
    high, low = grids(fov_type=fov_type)
    pool = GridPoolingLayer(16, high, low, mode=mode, gauss_sigma=0.5)
    constants = torch.tensor([[-2.0, 4.0], [5.0, 10.0]])
    values = constants[..., None].expand(2, 2, len(high)).clone()
    values[..., ~high.valid_mask] = torch.nan
    result = pool(values)
    expected = constants[..., None] * low.valid_mask
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize("mode", ["avg", "gaussian", "max"])
def test_empty_neighborhoods_return_finite_zero(mode: str) -> None:
    high, low = grids()
    high.valid_mask.fill_(False)
    result = GridPoolingLayer(None, high, low, mode)(torch.ones(1, 2, len(high)))
    torch.testing.assert_close(result, torch.zeros_like(result))


def test_gaussian_weights_emphasize_the_center() -> None:
    high, low = grids("uniform_as_grid")
    image = torch.zeros(1, 1, 8, 8)
    image[..., 1, 1] = 1
    actual = low.as_grid(
        GridPoolingLayer(9, high, low, "gaussian", 0.5)(canonical(image))
    )
    # A full 3x3 window has center weight 1, four axial exp(-1), four corner exp(-2).
    expected = 1 / (
        1 + 4 * torch.exp(torch.tensor(-1.0)) + 4 * torch.exp(torch.tensor(-2.0))
    )
    torch.testing.assert_close(actual[0, 0, 0, 0], expected)
    assert actual[0, 0, 0, 0] > 1 / 9


@pytest.mark.parametrize("mode", ["avg", "gaussian"])
def test_masked_input_gradients_are_zero(mode: str) -> None:
    high, low = grids()
    values = torch.randn(1, 2, len(high), requires_grad=True)
    GridPoolingLayer(None, high, low, mode)(values).sum().backward()
    assert torch.count_nonzero(values.grad[..., ~high.valid_mask]) == 0
    assert torch.count_nonzero(values.grad[..., high.valid_mask]) > 0


@pytest.mark.parametrize(
    "style,expected",
    [
        ("uniform_as_grid", GridPoolingLayer),
        ("warped_cartesian_as_grid", GridPoolingLayer),
        ("logpolar_as_grid", GridPoolingLayer),
        ("warped_cartesian", KNNPoolingLayer),
        ("logpolar", KNNPoolingLayer),
        ("isotropic", KNNPoolingLayer),
    ],
)
def test_sensor_style_selects_pooler(style: str, expected: type) -> None:
    transform = RetinalTransform(
        4,
        fov=50,
        cmf_a=5,
        style=style,
        sampler="pooling",
        res_mult=2,
        auto_match_cart_resources=0,
        device="cpu",
    )
    assert isinstance(transform.sampler.pooler, expected)
    if expected is GridPoolingLayer:
        assert transform.sampler.k == 4


@pytest.mark.parametrize("high_res,cmf_a,error", [(9, 5, "integer"), (8, 6, "CMF")])
def test_grid_alignment_is_required(high_res: int, cmf_a: float, error: str) -> None:
    _, low = grids()
    high = low.clone(resolution=high_res, cmf_a=cmf_a)
    with pytest.raises(ValueError, match=error):
        GridPoolingLayer(None, high, low)


def test_sample_set_sensors_are_rejected() -> None:
    high = SamplingCoords(50, 5, 8, style="warped_cartesian")
    with pytest.raises(ValueError, match="_as_grid"):
        GridPoolingLayer(None, high, high.clone(resolution=4))


def polar_grids() -> tuple[SamplingCoords, SamplingCoords]:
    low = SamplingCoords(50, 5, 4, style="logpolar_as_grid")
    return low.clone(resolution=8), low


def test_polar_null_kernel_pools_angle_eccentricity_blocks() -> None:
    high, low = polar_grids()
    pool = GridPoolingLayer(None, high, low)
    image = torch.arange(64, dtype=torch.float32).reshape(1, 1, 8, 8)
    actual = low.as_grid(pool(high.as_grid(image.flatten(-2)).flatten(-2)))
    torch.testing.assert_close(actual, torch.nn.functional.avg_pool2d(image, 2))


@pytest.mark.parametrize("mode", ["avg", "max"])
def test_polar_kernel_wraps_angles_and_excludes_radial_padding(mode: str) -> None:
    high, low = polar_grids()
    pool = GridPoolingLayer(9, high, low, mode=mode)
    image = torch.randn(1, 1, 8, 8)
    # Output angle i sits on high-res angle 2i, so its 3x3 window spans angles
    # 2i-1..2i+1 (wrapping) and eccentricities 2j..2j+2 (clipped at the edge).
    expected = torch.empty(1, 1, 4, 4)
    reduce = torch.mean if mode == "avg" else torch.amax
    for i in range(4):
        rows = [(2 * i + offset) % 8 for offset in (-1, 0, 1)]
        for j in range(4):
            cols = [col for col in (2 * j, 2 * j + 1, 2 * j + 2) if col < 8]
            expected[..., i, j] = reduce(image[..., rows, :][..., cols])
    torch.testing.assert_close(low.as_grid(pool(image.flatten(-2))), expected)


@pytest.mark.parametrize("k", [0, -1, 1.5, True, 3, 8])
def test_invalid_kernel_is_rejected(k: float | bool) -> None:
    high, low = grids()
    with pytest.raises(ValueError, match="perfect square"):
        GridPoolingLayer(k, high, low)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("mode", ["avg", "gaussian"])
def test_calibrated_grid_pooling_is_cuda_graph_capturable(mode: str) -> None:
    kwargs = {
        "fov": 50,
        "cmf_a": 5,
        "resolution": 8,
        "res_mult": 2,
        "style": "warped_cartesian_as_grid",
        "field_geometry": "spherical",
        "camera_model": CameraModel("fisheye", (48, 64), (35, 35, 31.5, 23.5)),
        "pool_mode": mode,
        "gauss_sigma": 0.5,
    }
    reference = PoolingGridSampler(**kwargs, device="cpu", backend="torch")
    sampler = PoolingGridSampler(**kwargs, device="cuda", backend="cuda")
    image = (
        torch.arange(48 * 64)
        .remainder(251)
        .to(torch.uint8)
        .reshape(1, 1, 48, 64)
        .cuda()
    )
    fixation = torch.tensor([[0.4, 0.9]], device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            sampler(image, fixation)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = sampler(image, fixation)
    graph.replay()
    torch.testing.assert_close(actual.cpu(), reference(image.cpu(), fixation.cpu()))
    image.fill_(100)
    fixation.fill_(0.5)
    graph.replay()
    torch.testing.assert_close(actual.cpu(), reference(image.cpu(), fixation.cpu()))
