"""Fused inference pooling on Torch's CUDA stream, without full-resolution temporaries."""

from __future__ import annotations

from functools import cache

import torch
import warp as wp


@cache
def _stream(device: torch.device, handle: int) -> wp.Stream:
    wp.init()
    stream = wp.get_stream(wp.device_from_torch(device))
    # Reuse Warp-owned streams instead of creating aliases that unregister them when destroyed.
    if stream.cuda_stream == handle:
        return stream
    return wp.stream_from_torch(torch.cuda.current_stream(device))


@cache
def _kernel(dtype: torch.dtype, mode: str):
    scalar = wp.dtype_from_torch(dtype)
    use_max = mode == "max"
    integer = dtype == torch.uint8

    @wp.kernel(module="unique")
    def pool(
        samples: wp.array3d(dtype=scalar),
        neighbors: wp.array2d(dtype=wp.int64),
        visible: wp.array2d(dtype=wp.bool),
        result: wp.array3d(dtype=scalar),
    ):
        env, channel, output = wp.tid()
        value = float(0.0)  # noqa: UP018 - Warp dynamic loop variable
        if wp.static(use_max):
            value = float(-wp.inf)
        count = int(0)  # noqa: UP018, RUF046 - Warp dynamic loop variable
        for slot in range(neighbors.shape[0]):
            source = neighbors[slot, output]
            if source < samples.shape[2]:
                seen = bool(True)  # noqa: UP018 - Warp dynamic loop variable
                if visible:
                    seen = visible[env, source]
                if seen:
                    item = float(samples[env, channel, source])
                    if wp.static(use_max):
                        if wp.isnan(item) or item >= value:
                            value = item
                    else:
                        value += item
                    count += 1
        if count == 0:
            value = 0.0
        elif wp.static(not use_max):
            value /= float(count)
        if wp.static(integer):
            value = wp.rint(wp.clamp(value, 0.0, 255.0))
        result[env, channel, output] = scalar(value)

    return pool


def pool_forward(
    values: torch.Tensor,
    neighbors: torch.Tensor,
    visible: torch.Tensor | None,
    mode: str,
) -> torch.Tensor:
    """Pool eligible ``(B, C, N)`` CUDA storage; the public layer handles dispatch and validation."""
    stream = _stream(
        values.device, torch.cuda.current_stream(values.device).cuda_stream
    )
    result = torch.empty(
        (*values.shape[:2], neighbors.shape[1]),
        dtype=values.dtype,
        device=values.device,
    )
    wp.launch(
        _kernel(values.dtype, mode),
        dim=tuple(result.shape),
        inputs=[
            wp.from_torch(values),
            wp.from_torch(neighbors),
            None if visible is None else wp.from_torch(visible),
        ],
        outputs=[wp.from_torch(result)],
        device=str(values.device),
        stream=stream,
    )
    return result
