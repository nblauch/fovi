"""Native inference pooling agrees with the independent Torch image-window path."""

import pytest
import torch

from fovi.arch.pooling import GridPoolingLayer
from fovi.sensing.coords import SamplingCoords

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("mode", ["avg", "max"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("transform", ["forward_ad", "jvp", "vmap"])
def test_native_dispatch_preserves_torch_transforms(mode, dtype, transform) -> None:
    low = SamplingCoords(50, 5, 4, style="uniform_as_grid")
    high = low.clone(resolution=8)
    reference = GridPoolingLayer(9, high, low, mode=mode)
    pool = GridPoolingLayer(9, high, low, mode=mode).cuda()
    values = torch.randn(2, 3, len(high), dtype=dtype, device="cuda")
    visible = torch.ones(2, len(high), dtype=torch.bool, device="cuda")
    visible[0, ::2] = False
    visible[1] = False
    expected = reference(values.cpu(), visible.cpu()).cuda()
    if transform == "vmap":
        actual = torch.vmap(lambda x: pool(x, visible))(
            values.unsqueeze(0).expand(2, -1, -1, -1)
        )
        torch.testing.assert_close(actual, expected.unsqueeze(0).expand_as(actual))
    else:
        tangent = torch.ones_like(values)
        expected_tangent = reference(tangent.cpu(), visible.cpu()).cuda()
        if transform == "forward_ad":
            with torch.autograd.forward_ad.dual_level():
                dual = torch.autograd.forward_ad.make_dual(values, tangent)
                actual, actual_tangent = torch.autograd.forward_ad.unpack_dual(
                    pool(dual, visible)
                )
        else:
            actual, actual_tangent = torch.func.jvp(
                lambda x: pool(x, visible), (values,), (tangent,)
            )
        torch.testing.assert_close(actual, expected)
        assert actual_tangent is not None
        torch.testing.assert_close(actual_tangent, expected_tangent)
    assert pool.last_backend == "torch_grid2d"


def test_native_low_precision_average_does_not_overflow() -> None:
    dtype = torch.float16
    low = SamplingCoords(50, 5, 4, style="uniform_as_grid")
    high = low.clone(resolution=8)
    pool = GridPoolingLayer(None, high, low).cuda()
    # A valid dtype maximum and smallest subnormal exercise both ends of its range.
    tiny = torch.nextafter(
        torch.tensor(0.0, dtype=dtype), torch.tensor(1.0, dtype=dtype)
    ).item()
    values = torch.tensor([torch.finfo(dtype).max, tiny], dtype=dtype, device="cuda")[
        None, :, None
    ].expand(2, -1, len(high))
    visible = torch.ones(2, len(high), dtype=torch.bool, device="cuda")
    visible[0, ::2] = False
    result = pool(values, visible)
    assert pool.last_backend == "warp_grid2d"
    torch.testing.assert_close(result, values[:, :, : len(low)], rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["avg", "max"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("context", [torch.no_grad, torch.inference_mode])
@pytest.mark.parametrize("existing_grad", [False, True])
def test_inference_does_not_allocate_or_change_input_grad(
    mode, dtype, context, existing_grad
) -> None:
    low = SamplingCoords(50, 5, 4, style="uniform_as_grid")
    high = low.clone(resolution=8)
    pool = GridPoolingLayer(None, high, low, mode=mode).cuda()
    values = (
        torch.ones(2, len(high), 4, device="cuda", dtype=dtype)[..., :3]
        .transpose(1, 2)
        .requires_grad_()
    )
    original_grad = torch.full_like(values, 7.0) if existing_grad else None
    values.grad = original_grad
    with context():
        result = pool(values)
    assert pool.last_backend == "warp_grid2d"
    assert not result.requires_grad
    assert values.grad is original_grad
    if original_grad is not None:
        torch.testing.assert_close(original_grad, torch.full_like(original_grad, 7.0))
    torch.testing.assert_close(result, torch.ones_like(result))


@pytest.mark.parametrize(
    "style,mode,k,dtype,visible",
    [
        ("warped_cartesian_as_grid", "avg", None, torch.uint8, True),
        ("uniform_as_grid", "avg", 9, torch.float32, True),
        ("warped_cartesian_as_grid", "avg", 1, torch.float32, False),
        ("logpolar_as_grid", "max", 9, torch.float32, True),
        ("logpolar_as_grid", "avg", 9, torch.float32, False),
        ("warped_cartesian_as_grid", "max", None, torch.uint8, False),
        ("uniform_as_grid", "avg", 9, torch.float16, True),
        ("logpolar_as_grid", "max", 9, torch.float16, True),
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
    if dtype.is_floating_point:
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


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_grid_pooling_preserves_torch_autograd_and_compile(dtype) -> None:
    low = SamplingCoords(50, 5, 4, style="warped_cartesian_as_grid")
    high = low.clone(resolution=8)
    reference = GridPoolingLayer(9, high, low)
    native = GridPoolingLayer(9, high, low).cuda()
    values = torch.randn(2, 3, len(high), dtype=dtype, requires_grad=True)
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


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_grid_pooling_stream_and_graph_replay_use_updated_storage(dtype) -> None:
    low = SamplingCoords(50, 5, 4, style="logpolar_as_grid")
    high = low.clone(resolution=8)
    reference = GridPoolingLayer(9, high, low)
    native = GridPoolingLayer(9, high, low).cuda()
    values = torch.randn(2, len(high), 4, device="cuda", dtype=dtype)[
        ..., :3
    ].transpose(1, 2)
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
