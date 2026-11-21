"""Native inference pooling agrees with the independent Torch image-window path."""

import pytest
import torch

from fovi.arch.pooling import GridPoolingLayer
from fovi.sensing.coords import SamplingCoords

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize(
    "style,mode,k,dtype,visible",
    [
        ("warped_cartesian_as_grid", "avg", None, torch.uint8, True),
        ("uniform_as_grid", "avg", 9, torch.float32, True),
        ("warped_cartesian_as_grid", "avg", 1, torch.float32, False),
        ("logpolar_as_grid", "max", 9, torch.float32, True),
        ("logpolar_as_grid", "avg", 9, torch.float32, False),
        ("warped_cartesian_as_grid", "max", None, torch.uint8, False),
    ],
)
def test_native_pooling_matches_torch_for_strided_storage(
    style, mode, k, dtype, visible
) -> None:
    low = SamplingCoords(50, 5, 8, style=style)
    high = low.clone(resolution=16)
    reference = GridPoolingLayer(k, high, low, mode=mode)
    native = GridPoolingLayer(k, high, low, mode=mode).cuda()
    storage = torch.randint(
        0, 256, (3, len(high), 4), generator=torch.Generator().manual_seed(42)
    ).to(dtype)
    if dtype == torch.float32:
        storage = storage / 128 - 1
    # Strided RGB views of RGBA storage must not require a full-resolution contiguous copy.
    values = storage[..., :3].transpose(1, 2)
    cuda_values = storage.cuda()[..., :3].transpose(1, 2)
    visibility = None
    if visible:
        visibility = torch.stack(
            (
                high.cartesian[:, 0] < 0.3,
                torch.ones(len(high), dtype=torch.bool),
                torch.zeros(len(high), dtype=torch.bool),
            )
        )
    expected = reference(values, visibility)
    actual = native(cuda_values, None if visibility is None else visibility.cuda())
    assert native.last_backend == "warp_grid2d"
    assert actual.dtype == dtype
    torch.testing.assert_close(actual.cpu(), expected)
    torch.testing.assert_close(cuda_values.cpu(), values)


def test_native_rounding_and_nan_max_match_torch() -> None:
    low = SamplingCoords(50, 5, 4, style="uniform_as_grid")
    high = low.clone(resolution=8)
    byte_pool = GridPoolingLayer(None, high, low).cuda()
    stripe = torch.arange(len(high)).remainder(2).to(torch.uint8)
    values = torch.stack((stripe, stripe + 1))[None]
    expected = GridPoolingLayer(None, high, low)(values)
    assert torch.equal(byte_pool(values.cuda()).cpu(), expected)
    max_pool = GridPoolingLayer(None, high, low, mode="max").cuda()
    values = torch.randn(1, 1, len(high))
    values[..., 0] = torch.nan
    values[..., 20] = torch.inf
    expected = GridPoolingLayer(None, high, low, mode="max")(values)
    torch.testing.assert_close(max_pool(values.cuda()).cpu(), expected, equal_nan=True)


def test_grid_pooling_preserves_torch_autograd_and_compile() -> None:
    low = SamplingCoords(50, 5, 4, style="warped_cartesian_as_grid")
    high = low.clone(resolution=8)
    reference = GridPoolingLayer(9, high, low)
    native = GridPoolingLayer(9, high, low).cuda()
    values = torch.randn(2, 3, len(high), requires_grad=True)
    cuda_values = values.detach().cuda().requires_grad_()
    visible = torch.rand(2, len(high)) > 0.3
    expected = reference(values, visible)
    actual = native(cuda_values, visible.cuda())
    assert native.last_backend == "torch_grid2d"
    torch.testing.assert_close(actual.cpu(), expected)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(cuda_values.grad.cpu(), values.grad)
    compiled = torch.compile(native, backend="eager", fullgraph=True)
    result = compiled(cuda_values.detach(), visible.cuda())
    torch.testing.assert_close(result.cpu(), expected)
    assert native.last_backend == "torch_grid2d"


def test_grid_pooling_stream_and_graph_replay_use_updated_storage() -> None:
    low = SamplingCoords(50, 5, 4, style="logpolar_as_grid")
    high = low.clone(resolution=8)
    reference = GridPoolingLayer(9, high, low)
    native = GridPoolingLayer(9, high, low).cuda()
    values = torch.randn(2, len(high), 4, device="cuda")[..., :3].transpose(1, 2)
    visible = torch.ones(2, len(high), dtype=torch.bool, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            native(values, visible)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = native(values, visible)
    for use_mask in (False, True):
        values.fill_(-2.0)
        visible[0].fill_(use_mask)
        visible[1].zero_()
        graph.replay()
        expected = reference(values.cpu(), visible.cpu())
        torch.testing.assert_close(actual.cpu(), expected)
