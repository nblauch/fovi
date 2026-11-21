fovi.arch.pooling module
=======================

``GridPoolingLayer`` pools aligned retinal grids. CUDA float32 and uint8
average/max inference uses a fused Warp kernel. It accepts strided ``(B, C, N)``
views, including views of channel-last renderer storage, without creating
full-resolution float or contiguous copies. CPU, Gaussian pooling, other dtypes,
autograd, and ``torch.compile`` retain the Torch path.

An optional boolean ``visible`` mask of shape ``(B, N)`` excludes unseen samples
and normalizes over the remaining neighbors. Empty neighborhoods are zero;
retinal masks and Cartesian/polar padding still apply. Without ``visible``,
existing retinal pooling behavior is unchanged. Uint8 inputs return uint8 with
clamping and ties-to-even rounding. Callers needing fractional averages should
pass floating-point values.

For example, pool an RGB view of an RGBA buffer without a layout copy::

    # rgba: (batch, high_sample_count, 4), uint8 CUDA storage
    rgb = rgba[..., :3].transpose(1, 2)
    pooled = pooler(rgb, visible=visible)  # (batch, 3, low_sample_count)

``last_backend`` reports ``warp_grid2d``, ``torch_grid2d``, or ``modal``. Warm up before
CUDA graph capture. For average, max and modal pooling, ``neighbor_table``
exposes the cached canonical neighborhood table; the high-resolution sample
count marks excluded entries.

.. automodule:: fovi.arch.pooling
   :members:
   :undoc-members:
   :show-inheritance:
