"""Optional Warp kernels for compact KNN convolution inference and training.

Forward kernels plus adjoint kernels. Warp autodiff is
never used (``enable_backward=False`` everywhere); gradients are explicit adjoint kernels
registered with the ``fovi.arch.knn_autograd`` ops registry as ``warp_train``.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
import warp as wp


wp.config.quiet = True
wp.init()
wp.set_module_options({"fast_math": True, "enable_backward": False})


@wp.func
def _float_to_half(value: wp.float32):
    return wp.float16(value)


@wp.func
def _half_to_float(value: wp.float16):
    return wp.float32(value)


def _make_cached_kernel(tile_m: int, tile_n: int):
    tile_k = 64

    @wp.kernel(module="unique", enable_backward=False)
    def kernel(
        x: wp.array2d(dtype=wp.float16),
        effective_weight: wp.array3d(dtype=wp.float16),
        bias: wp.array(dtype=wp.float16),
        input_linear: wp.array2d(dtype=wp.int32),
        output_nbo: wp.array3d(dtype=wp.float16),
    ):
        n, tile_b, tile_o = wp.tid()
        accumulator = wp.tile_zeros(shape=(tile_m, tile_n), dtype=wp.float32)
        for p_tile in range(input_linear.shape[1] // tile_k):
            p_offset = p_tile * tile_k
            x_indices = wp.tile_load(
                input_linear[n], shape=tile_k, offset=p_offset, storage="shared"
            )
            x_values = wp.tile_load_indexed(
                x,
                indices=x_indices,
                shape=(tile_m, tile_k),
                offset=(tile_b * tile_m, 0),
                axis=1,
                storage="shared",
            )
            weight_values = wp.tile_load(
                effective_weight[n],
                shape=(tile_k, tile_n),
                offset=(p_offset, tile_o * tile_n),
                storage="shared",
            )
            wp.tile_matmul(x_values, weight_values, accumulator)
        bias_values = wp.tile_map(
            _half_to_float,
            wp.tile_load(bias, shape=tile_n, offset=tile_o * tile_n),
        )
        accumulator += wp.tile_broadcast(bias_values, shape=(tile_m, tile_n))
        wp.tile_store(
            output_nbo[n],
            wp.tile_map(_float_to_half, accumulator),
            offset=(tile_b * tile_m, tile_o * tile_n),
        )

    return kernel


def _make_uncached_kernel(tile_m: int, tile_n: int):
    tile_k = 64

    @wp.kernel(module="unique", enable_backward=False)
    def kernel(
        x: wp.array2d(dtype=wp.float16),
        weight_t: wp.array2d(dtype=wp.float16),
        bias: wp.array(dtype=wp.float16),
        input_linear: wp.array2d(dtype=wp.int32),
        weight_linear: wp.array2d(dtype=wp.int32),
        output_nbo: wp.array3d(dtype=wp.float16),
    ):
        n, tile_b, tile_o = wp.tid()
        accumulator = wp.tile_zeros(shape=(tile_m, tile_n), dtype=wp.float32)
        for p_tile in range(input_linear.shape[1] // tile_k):
            p_offset = p_tile * tile_k
            x_indices = wp.tile_load(
                input_linear[n], shape=tile_k, offset=p_offset, storage="shared"
            )
            w_indices = wp.tile_load(
                weight_linear[n], shape=tile_k, offset=p_offset, storage="shared"
            )
            x_values = wp.tile_load_indexed(
                x,
                indices=x_indices,
                shape=(tile_m, tile_k),
                offset=(tile_b * tile_m, 0),
                axis=1,
                storage="shared",
            )
            weight_values = wp.tile_load_indexed(
                weight_t,
                indices=w_indices,
                shape=(tile_k, tile_n),
                offset=(0, tile_o * tile_n),
                axis=0,
                storage="shared",
            )
            wp.tile_matmul(x_values, weight_values, accumulator)
        bias_values = wp.tile_map(
            _half_to_float,
            wp.tile_load(bias, shape=tile_n, offset=tile_o * tile_n),
        )
        accumulator += wp.tile_broadcast(bias_values, shape=(tile_m, tile_n))
        wp.tile_store(
            output_nbo[n],
            wp.tile_map(_float_to_half, accumulator),
            offset=(tile_b * tile_m, tile_o * tile_n),
        )

    return kernel


def _make_uncached_r2_kernel(tile_m: int, tile_n: int):
    """Uncached kernel with a 2-way batch-tile weight-reuse loop.

    Each CTA gathers every indexed weight tile once and applies it to two consecutive batch
    tiles, halving weight-stream traffic relative to one-accumulator kernels at equal tile_m.
    """
    tile_k = 64

    @wp.kernel(module="unique", enable_backward=False)
    def kernel(
        x: wp.array2d(dtype=wp.float16),
        weight_t: wp.array2d(dtype=wp.float16),
        bias: wp.array(dtype=wp.float16),
        input_linear: wp.array2d(dtype=wp.int32),
        weight_linear: wp.array2d(dtype=wp.int32),
        output_nbo: wp.array3d(dtype=wp.float16),
    ):
        n, tile_bg, tile_o = wp.tid()
        b0 = tile_bg * (2 * tile_m)
        b1 = b0 + tile_m
        acc0 = wp.tile_zeros(shape=(tile_m, tile_n), dtype=wp.float32)
        acc1 = wp.tile_zeros(shape=(tile_m, tile_n), dtype=wp.float32)
        for p_tile in range(input_linear.shape[1] // tile_k):
            p_offset = p_tile * tile_k
            x_indices = wp.tile_load(
                input_linear[n], shape=tile_k, offset=p_offset, storage="shared"
            )
            w_indices = wp.tile_load(
                weight_linear[n], shape=tile_k, offset=p_offset, storage="shared"
            )
            weight_values = wp.tile_load_indexed(
                weight_t,
                indices=w_indices,
                shape=(tile_k, tile_n),
                offset=(0, tile_o * tile_n),
                axis=0,
                storage="shared",
            )
            x0 = wp.tile_load_indexed(
                x,
                indices=x_indices,
                shape=(tile_m, tile_k),
                offset=(b0, 0),
                axis=1,
                storage="shared",
            )
            wp.tile_matmul(x0, weight_values, acc0)
            x1 = wp.tile_load_indexed(
                x,
                indices=x_indices,
                shape=(tile_m, tile_k),
                offset=(b1, 0),
                axis=1,
                storage="shared",
            )
            wp.tile_matmul(x1, weight_values, acc1)
        bias_values = wp.tile_map(
            _half_to_float,
            wp.tile_load(bias, shape=tile_n, offset=tile_o * tile_n),
        )
        bias_tile = wp.tile_broadcast(bias_values, shape=(tile_m, tile_n))
        acc0 += bias_tile
        acc1 += bias_tile
        wp.tile_store(
            output_nbo[n], wp.tile_map(_float_to_half, acc0), offset=(b0, tile_o * tile_n)
        )
        wp.tile_store(
            output_nbo[n], wp.tile_map(_float_to_half, acc1), offset=(b1, tile_o * tile_n)
        )

    return kernel


def _make_uncached_r4_kernel(tile_m: int, tile_n: int):
    """Uncached kernel with a 4-way batch-tile weight-reuse loop."""
    tile_k = 64

    @wp.kernel(module="unique", enable_backward=False)
    def kernel(
        x: wp.array2d(dtype=wp.float16),
        weight_t: wp.array2d(dtype=wp.float16),
        bias: wp.array(dtype=wp.float16),
        input_linear: wp.array2d(dtype=wp.int32),
        weight_linear: wp.array2d(dtype=wp.int32),
        output_nbo: wp.array3d(dtype=wp.float16),
    ):
        n, tile_bg, tile_o = wp.tid()
        b0 = tile_bg * (4 * tile_m)
        b1 = b0 + tile_m
        b2 = b0 + 2 * tile_m
        b3 = b0 + 3 * tile_m
        acc0 = wp.tile_zeros(shape=(tile_m, tile_n), dtype=wp.float32)
        acc1 = wp.tile_zeros(shape=(tile_m, tile_n), dtype=wp.float32)
        acc2 = wp.tile_zeros(shape=(tile_m, tile_n), dtype=wp.float32)
        acc3 = wp.tile_zeros(shape=(tile_m, tile_n), dtype=wp.float32)
        for p_tile in range(input_linear.shape[1] // tile_k):
            p_offset = p_tile * tile_k
            x_indices = wp.tile_load(
                input_linear[n], shape=tile_k, offset=p_offset, storage="shared"
            )
            w_indices = wp.tile_load(
                weight_linear[n], shape=tile_k, offset=p_offset, storage="shared"
            )
            weight_values = wp.tile_load_indexed(
                weight_t,
                indices=w_indices,
                shape=(tile_k, tile_n),
                offset=(0, tile_o * tile_n),
                axis=0,
                storage="shared",
            )
            x0 = wp.tile_load_indexed(
                x,
                indices=x_indices,
                shape=(tile_m, tile_k),
                offset=(b0, 0),
                axis=1,
                storage="shared",
            )
            wp.tile_matmul(x0, weight_values, acc0)
            x1 = wp.tile_load_indexed(
                x,
                indices=x_indices,
                shape=(tile_m, tile_k),
                offset=(b1, 0),
                axis=1,
                storage="shared",
            )
            wp.tile_matmul(x1, weight_values, acc1)
            x2 = wp.tile_load_indexed(
                x,
                indices=x_indices,
                shape=(tile_m, tile_k),
                offset=(b2, 0),
                axis=1,
                storage="shared",
            )
            wp.tile_matmul(x2, weight_values, acc2)
            x3 = wp.tile_load_indexed(
                x,
                indices=x_indices,
                shape=(tile_m, tile_k),
                offset=(b3, 0),
                axis=1,
                storage="shared",
            )
            wp.tile_matmul(x3, weight_values, acc3)
        bias_values = wp.tile_map(
            _half_to_float,
            wp.tile_load(bias, shape=tile_n, offset=tile_o * tile_n),
        )
        bias_tile = wp.tile_broadcast(bias_values, shape=(tile_m, tile_n))
        acc0 += bias_tile
        acc1 += bias_tile
        acc2 += bias_tile
        acc3 += bias_tile
        wp.tile_store(
            output_nbo[n], wp.tile_map(_float_to_half, acc0), offset=(b0, tile_o * tile_n)
        )
        wp.tile_store(
            output_nbo[n], wp.tile_map(_float_to_half, acc1), offset=(b1, tile_o * tile_n)
        )
        wp.tile_store(
            output_nbo[n], wp.tile_map(_float_to_half, acc2), offset=(b2, tile_o * tile_n)
        )
        wp.tile_store(
            output_nbo[n], wp.tile_map(_float_to_half, acc3), offset=(b3, tile_o * tile_n)
        )

    return kernel


_CACHED_SMALL = _make_cached_kernel(16, 32)
_CACHED_LARGE = _make_cached_kernel(32, 64)
_UNCACHED_SMALL = _make_uncached_kernel(16, 32)
_UNCACHED_LARGE = _make_uncached_kernel(32, 64)


# Large-batch uncached configs: name -> (maker, tile_m, tile_n, r, block_dim). Kernels are
# JIT-built lazily on first use so importing this module does not compile every variant.
_UNCACHED_BATCH_CONFIG_SPECS = {
    "m32n64r1": (_make_uncached_kernel, 32, 64, 1, 128),  # large kernel, for comparison
    "m64n64r1": (_make_uncached_kernel, 64, 64, 1, 128),
    "m64n64r1b256": (_make_uncached_kernel, 64, 64, 1, 256),
    "m32n64r2": (_make_uncached_r2_kernel, 32, 64, 2, 128),
    "m32n64r4": (_make_uncached_r4_kernel, 32, 64, 4, 128),
    "m64n64r2": (_make_uncached_r2_kernel, 64, 64, 2, 128),
    "m64n64r2b256": (_make_uncached_r2_kernel, 64, 64, 2, 256),
    # NOTE: m64n64r4 (4 fp32 64x64 accumulators) exceeds the sm_89 CTA resource budget; the
    # launch fails with CUDA "invalid argument" and Warp does not raise. Do not re-add without
    # the canary validation below confirming it launches.
    "m32n128r2": (_make_uncached_r2_kernel, 32, 128, 2, 128),
    "m32n128r4": (_make_uncached_r4_kernel, 32, 128, 4, 256),
    "m64n128r2": (_make_uncached_r2_kernel, 64, 128, 2, 256),
}
# Kernel objects are device-agnostic (Warp JIT-compiles per arch on first launch); canary
# validation is a per-device property, so the two caches are keyed separately.
_UNCACHED_BATCH_KERNELS = {}
_UNCACHED_BATCH_VALIDATED = {}


def _canary_launch_ok(kernel, tile_m, tile_n, r, block_dim, device):
    """Return True when the kernel actually launches on ``device``.

    Kernel launches that exceed CTA resource limits fail with a CUDA "invalid argument" that
    Warp reports on stderr without raising, leaving the (uninitialized) output untouched. A
    NaN-prefilled single-node canary makes that failure mode detectable.
    """
    rows = tile_m * r
    x = torch.zeros((rows, 2), dtype=torch.float16, device=device)
    weight_t = torch.zeros((1, tile_n), dtype=torch.float16, device=device)
    bias = torch.zeros(tile_n, dtype=torch.float16, device=device)
    # All input indices out of bounds -> zero fill; weight indices all row 0.
    input_linear = torch.full((1, 64), 2, dtype=torch.int32, device=device)
    weight_linear = torch.zeros((1, 64), dtype=torch.int32, device=device)
    output = torch.full((rows, tile_n, 1), float("nan"), dtype=torch.float16, device=device)
    _launch(
        kernel,
        tile_m,
        tile_n,
        x,
        bias,
        input_linear,
        output,
        [
            _descriptor(x),
            _descriptor(weight_t),
            _descriptor(bias),
            _descriptor(input_linear),
            _descriptor(weight_linear),
        ],
        r=r,
        block_dim=block_dim,
    )
    torch.cuda.synchronize(device)
    return bool(torch.isfinite(output).all().item())


def _get_uncached_batch_config(name, device=None):
    """Build (and canary-validate per device) a large-batch config; None when unlaunchable."""
    maker, tile_m, tile_n, r, block_dim = _UNCACHED_BATCH_CONFIG_SPECS[name]
    kernel = _UNCACHED_BATCH_KERNELS.get(name)
    if kernel is None:
        kernel = maker(tile_m, tile_n)
        _UNCACHED_BATCH_KERNELS[name] = kernel
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    key = (name, device.index)
    validated = _UNCACHED_BATCH_VALIDATED.get(key)
    if validated is None:
        validated = _canary_launch_ok(kernel, tile_m, tile_n, r, block_dim, device)
        _UNCACHED_BATCH_VALIDATED[key] = validated
    if not validated:
        return None
    return (kernel, tile_m, tile_n, r, block_dim)


def _select_uncached_batch_config(batch, out_channels, out_nodes, p_padded):
    """Heuristic for the large-batch (B >= 64) uncached kernel.

    Fit to the Ada fp16 sweeps. With the coalesced
    weight operand the wide m64n128r2 tile wins or ties every measured shape at
    B >= 128 whenever Cout does not waste most of the 128-wide tile; narrower output tiles
    only pay off for small Cout. Below 128 rows the 2-way reuse kernels waste half their
    compute on out-of-bounds subtiles, so a single 64-row accumulator is the safe choice.
    ``out_nodes`` / ``p_padded`` are kept for future tuning (they decided earlier fits).
    """
    del out_nodes, p_padded
    if batch >= 128:
        if out_channels >= 96:
            return "m64n128r2"
        return "m64n64r2b256"
    return "m64n64r1"


def _descriptor(tensor: torch.Tensor):
    return wp.from_torch(tensor, return_ctype=True)


def _launch(kernel, tile_m, tile_n, x, bias, input_linear, output, inputs, r=1, block_dim=128):
    stream = wp.stream_from_torch(torch.cuda.current_stream(x.device))
    rows_per_block = tile_m * r
    wp.launch_tiled(
        kernel,
        dim=(
            input_linear.shape[0],
            (x.shape[0] + rows_per_block - 1) // rows_per_block,
            (output.shape[1] + tile_n - 1) // tile_n,
        ),
        inputs=inputs,
        outputs=[_descriptor(output.permute(2, 0, 1))],
        block_dim=block_dim,
        stream=stream,
    )
    return output


def _pad_bias_to_tiles(bias, cout, tile_n):
    """Zero-pad bias so 1D tail tile loads never read past the buffer (tile-overread doctrine)."""
    needed = ((cout + tile_n - 1) // tile_n) * tile_n
    if bias.shape[0] < needed:
        return F.pad(bias, (0, needed - bias.shape[0]))
    return bias


def run_cached(x, effective_weight, bias, input_linear, cout=None):
    """Cached-effective-weight fp16 forward.

    ``cout`` is the true output-channel count; it defaults to ``effective_weight.shape[2]``
    for backward compatibility. Pass it explicitly when the effective-weight cache is padded
    to a tile multiple (recommended: pad to a multiple of 64) so the weight tile loads never
    read past the buffer; unpadded caches with non-multiple Cout are padded here per call,
    which costs a full cache copy — pad the cache instead.
    """
    if cout is None:
        cout = effective_weight.shape[2]
    output = torch.empty(
        (x.shape[0], cout, input_linear.shape[0]),
        device=x.device,
        dtype=torch.float16,
    )
    if x.shape[0] <= 16:
        kernel, tile_m, tile_n = _CACHED_SMALL, 16, 32
    else:
        kernel, tile_m, tile_n = _CACHED_LARGE, 32, 64
    needed = ((cout + tile_n - 1) // tile_n) * tile_n
    if effective_weight.shape[2] < needed:
        effective_weight = F.pad(effective_weight, (0, needed - effective_weight.shape[2]))
    return _launch(
        kernel,
        tile_m,
        tile_n,
        x,
        bias,
        input_linear,
        output,
        [
            _descriptor(x.reshape(x.shape[0], -1)),
            _descriptor(effective_weight),
            _descriptor(_pad_bias_to_tiles(bias, cout, tile_n)),
            _descriptor(input_linear),
        ],
    )


# --------------------------------------------------------------------------------------
# Adjoint kernels. Never Warp autodiff: explicit dA / dx / dWeff kernels.
#
# grad_input:  dA_n^T [P64, B] = W_n [P64, Cout] @ g_n^T [Cout, B]  (tiled, tensor cores)
#              -> staged as dA_stage [Nout, P64, Bc], batch innermost so the reverse-CSR
#              gather reads coalesce; then a scalar CSR kernel writes
#              dx[b, c, m] = sum_j dA_stage[n_j, c*K + k_j, b]  (fp32, deterministic).
#              Feeding g as [Nout, Cout, B] means no in-kernel transpose is needed anywhere.
# grad_weight: dWeff_n [P64, Cout] = A_n^T [P64, B] @ g_n [B, Cout] via tile_transpose of the
#              forward's indexed x tile; staged per Nout chunk, finished with a torch fp32
#              index_add_ over weight_linear (same semantics as CompactTorchOps.grad_weight).
# --------------------------------------------------------------------------------------


def _make_grad_input_stage_kernel(tile_p: int, tile_b: int, fp32_stage: bool):
    tile_k = 64

    if fp32_stage:

        @wp.kernel(module="unique", enable_backward=False)
        def kernel(
            g_nob: wp.array3d(dtype=wp.float16),
            weight_t: wp.array2d(dtype=wp.float16),
            weight_linear: wp.array2d(dtype=wp.int32),
            da_stage: wp.array3d(dtype=wp.float32),
        ):
            n, p_tile, b_tile = wp.tid()
            accumulator = wp.tile_zeros(shape=(tile_p, tile_b), dtype=wp.float32)
            w_indices = wp.tile_load(
                weight_linear[n], shape=tile_p, offset=p_tile * tile_p, storage="shared"
            )
            for c_tile in range((g_nob.shape[1] + tile_k - 1) // tile_k):
                c_offset = c_tile * tile_k
                weight_values = wp.tile_load_indexed(
                    weight_t,
                    indices=w_indices,
                    shape=(tile_p, tile_k),
                    offset=(0, c_offset),
                    axis=0,
                    storage="shared",
                )
                g_values = wp.tile_load(
                    g_nob[n],
                    shape=(tile_k, tile_b),
                    offset=(c_offset, b_tile * tile_b),
                    storage="shared",
                )
                wp.tile_matmul(weight_values, g_values, accumulator)
            wp.tile_store(
                da_stage[n], accumulator, offset=(p_tile * tile_p, b_tile * tile_b)
            )

    else:

        @wp.kernel(module="unique", enable_backward=False)
        def kernel(
            g_nob: wp.array3d(dtype=wp.float16),
            weight_t: wp.array2d(dtype=wp.float16),
            weight_linear: wp.array2d(dtype=wp.int32),
            da_stage: wp.array3d(dtype=wp.float16),
        ):
            n, p_tile, b_tile = wp.tid()
            accumulator = wp.tile_zeros(shape=(tile_p, tile_b), dtype=wp.float32)
            w_indices = wp.tile_load(
                weight_linear[n], shape=tile_p, offset=p_tile * tile_p, storage="shared"
            )
            for c_tile in range((g_nob.shape[1] + tile_k - 1) // tile_k):
                c_offset = c_tile * tile_k
                weight_values = wp.tile_load_indexed(
                    weight_t,
                    indices=w_indices,
                    shape=(tile_p, tile_k),
                    offset=(0, c_offset),
                    axis=0,
                    storage="shared",
                )
                g_values = wp.tile_load(
                    g_nob[n],
                    shape=(tile_k, tile_b),
                    offset=(c_offset, b_tile * tile_b),
                    storage="shared",
                )
                wp.tile_matmul(weight_values, g_values, accumulator)
            wp.tile_store(
                da_stage[n],
                wp.tile_map(_float_to_half, accumulator),
                offset=(p_tile * tile_p, b_tile * tile_b),
            )

    return kernel


def _make_grad_input_csr_kernel(fp32_stage: bool):
    if fp32_stage:

        @wp.kernel(module="unique", enable_backward=False)
        def kernel(
            da_stage: wp.array3d(dtype=wp.float32),
            rev_rowptr: wp.array(dtype=wp.int32),
            rev_col: wp.array(dtype=wp.int32),
            k: int,
            dx: wp.array3d(dtype=wp.float32),
        ):
            m, c, b = wp.tid()
            nout = da_stage.shape[0]
            accumulator = float(0.0)
            for index in range(int(rev_rowptr[m]), int(rev_rowptr[m + 1])):
                j = int(rev_col[index])
                kk = j // nout
                n = j - kk * nout
                accumulator += da_stage[n, c * k + kk, b]
            dx[b, c, m] = accumulator

    else:

        @wp.kernel(module="unique", enable_backward=False)
        def kernel(
            da_stage: wp.array3d(dtype=wp.float16),
            rev_rowptr: wp.array(dtype=wp.int32),
            rev_col: wp.array(dtype=wp.int32),
            k: int,
            dx: wp.array3d(dtype=wp.float32),
        ):
            m, c, b = wp.tid()
            nout = da_stage.shape[0]
            accumulator = float(0.0)
            for index in range(int(rev_rowptr[m]), int(rev_rowptr[m + 1])):
                j = int(rev_col[index])
                kk = j // nout
                n = j - kk * nout
                accumulator += float(da_stage[n, c * k + kk, b])
            dx[b, c, m] = accumulator

    return kernel


def _make_grad_weight_stage_kernel(tile_p: int, tile_n: int, fp32_stage: bool):
    tile_k = 64

    if fp32_stage:

        @wp.kernel(module="unique", enable_backward=False)
        def kernel(
            x_flat: wp.array2d(dtype=wp.float16),
            g_nbo: wp.array3d(dtype=wp.float16),
            input_linear: wp.array2d(dtype=wp.int32),
            dweff: wp.array3d(dtype=wp.float32),
        ):
            n, p_tile, o_tile = wp.tid()
            accumulator = wp.tile_zeros(shape=(tile_p, tile_n), dtype=wp.float32)
            x_indices = wp.tile_load(
                input_linear[n], shape=tile_p, offset=p_tile * tile_p, storage="shared"
            )
            for b_tile in range((x_flat.shape[0] + tile_k - 1) // tile_k):
                b_offset = b_tile * tile_k
                a_values = wp.tile_load_indexed(
                    x_flat,
                    indices=x_indices,
                    shape=(tile_k, tile_p),
                    offset=(b_offset, 0),
                    axis=1,
                    storage="shared",
                )
                g_values = wp.tile_load(
                    g_nbo[n],
                    shape=(tile_k, tile_n),
                    offset=(b_offset, o_tile * tile_n),
                    storage="shared",
                )
                wp.tile_matmul(wp.tile_transpose(a_values), g_values, accumulator)
            wp.tile_store(
                dweff[n], accumulator, offset=(p_tile * tile_p, o_tile * tile_n)
            )

    else:

        @wp.kernel(module="unique", enable_backward=False)
        def kernel(
            x_flat: wp.array2d(dtype=wp.float16),
            g_nbo: wp.array3d(dtype=wp.float16),
            input_linear: wp.array2d(dtype=wp.int32),
            dweff: wp.array3d(dtype=wp.float16),
        ):
            n, p_tile, o_tile = wp.tid()
            accumulator = wp.tile_zeros(shape=(tile_p, tile_n), dtype=wp.float32)
            x_indices = wp.tile_load(
                input_linear[n], shape=tile_p, offset=p_tile * tile_p, storage="shared"
            )
            for b_tile in range((x_flat.shape[0] + tile_k - 1) // tile_k):
                b_offset = b_tile * tile_k
                a_values = wp.tile_load_indexed(
                    x_flat,
                    indices=x_indices,
                    shape=(tile_k, tile_p),
                    offset=(b_offset, 0),
                    axis=1,
                    storage="shared",
                )
                g_values = wp.tile_load(
                    g_nbo[n],
                    shape=(tile_k, tile_n),
                    offset=(b_offset, o_tile * tile_n),
                    storage="shared",
                )
                wp.tile_matmul(wp.tile_transpose(a_values), g_values, accumulator)
            wp.tile_store(
                dweff[n],
                wp.tile_map(_float_to_half, accumulator),
                offset=(p_tile * tile_p, o_tile * tile_n),
            )

    return kernel


def _make_grad_input_stage2_kernel(tile_p: int, tile_b: int, fp32_stage: bool):
    """v2 gi staging: one CTA per (n, b_tile), the P64 loop lives inside the CTA.

    Amortizes per-CTA fixed costs over P64/tile_p accumulator rounds (the earlier
    v1 layout ran P64/64 x b_tiles CTAs of only Cout/64 matmul steps each).
    """
    tile_k = 64

    if fp32_stage:

        @wp.kernel(module="unique", enable_backward=False)
        def kernel(
            g_nob: wp.array3d(dtype=wp.float16),
            weight_t: wp.array2d(dtype=wp.float16),
            weight_linear: wp.array2d(dtype=wp.int32),
            da_stage: wp.array3d(dtype=wp.float32),
        ):
            n, b_tile = wp.tid()
            b_offset = b_tile * tile_b
            for p_tile in range(weight_linear.shape[1] // tile_p):
                p_offset = p_tile * tile_p
                accumulator = wp.tile_zeros(shape=(tile_p, tile_b), dtype=wp.float32)
                w_indices = wp.tile_load(
                    weight_linear[n], shape=tile_p, offset=p_offset, storage="shared"
                )
                for c_tile in range((g_nob.shape[1] + tile_k - 1) // tile_k):
                    c_offset = c_tile * tile_k
                    weight_values = wp.tile_load_indexed(
                        weight_t,
                        indices=w_indices,
                        shape=(tile_p, tile_k),
                        offset=(0, c_offset),
                        axis=0,
                        storage="shared",
                    )
                    g_values = wp.tile_load(
                        g_nob[n],
                        shape=(tile_k, tile_b),
                        offset=(c_offset, b_offset),
                        storage="shared",
                    )
                    wp.tile_matmul(weight_values, g_values, accumulator)
                wp.tile_store(da_stage[n], accumulator, offset=(p_offset, b_offset))

    else:

        @wp.kernel(module="unique", enable_backward=False)
        def kernel(
            g_nob: wp.array3d(dtype=wp.float16),
            weight_t: wp.array2d(dtype=wp.float16),
            weight_linear: wp.array2d(dtype=wp.int32),
            da_stage: wp.array3d(dtype=wp.float16),
        ):
            n, b_tile = wp.tid()
            b_offset = b_tile * tile_b
            for p_tile in range(weight_linear.shape[1] // tile_p):
                p_offset = p_tile * tile_p
                accumulator = wp.tile_zeros(shape=(tile_p, tile_b), dtype=wp.float32)
                w_indices = wp.tile_load(
                    weight_linear[n], shape=tile_p, offset=p_offset, storage="shared"
                )
                for c_tile in range((g_nob.shape[1] + tile_k - 1) // tile_k):
                    c_offset = c_tile * tile_k
                    weight_values = wp.tile_load_indexed(
                        weight_t,
                        indices=w_indices,
                        shape=(tile_p, tile_k),
                        offset=(0, c_offset),
                        axis=0,
                        storage="shared",
                    )
                    g_values = wp.tile_load(
                        g_nob[n],
                        shape=(tile_k, tile_b),
                        offset=(c_offset, b_offset),
                        storage="shared",
                    )
                    wp.tile_matmul(weight_values, g_values, accumulator)
                wp.tile_store(
                    da_stage[n],
                    wp.tile_map(_float_to_half, accumulator),
                    offset=(p_offset, b_offset),
                )

    return kernel


def _make_grad_weight_stage3_kernel(tile_p: int, tile_n: int, fp32_stage: bool):
    """v3 gw staging: v1's 3D (n, p_tile, o_tile) grid with the v2 transposed-x operand.

    For small-Nout shapes the v2 (n, o_tile) grid under-occupies the GPU; this keeps per-CTA
    work small but parallel while still using contiguous xt gathers and no tile_transpose.
    """
    tile_k = 64

    if fp32_stage:

        @wp.kernel(module="unique", enable_backward=False)
        def kernel(
            xt: wp.array2d(dtype=wp.float16),
            g_nbo: wp.array3d(dtype=wp.float16),
            input_linear: wp.array2d(dtype=wp.int32),
            dweff: wp.array3d(dtype=wp.float32),
        ):
            n, p_tile, o_tile = wp.tid()
            p_offset = p_tile * tile_p
            o_offset = o_tile * tile_n
            accumulator = wp.tile_zeros(shape=(tile_p, tile_n), dtype=wp.float32)
            x_indices = wp.tile_load(
                input_linear[n], shape=tile_p, offset=p_offset, storage="shared"
            )
            for b_tile in range(xt.shape[1] // tile_k):
                b_offset = b_tile * tile_k
                at_values = wp.tile_load_indexed(
                    xt,
                    indices=x_indices,
                    shape=(tile_p, tile_k),
                    offset=(0, b_offset),
                    axis=0,
                    storage="shared",
                )
                g_values = wp.tile_load(
                    g_nbo[n],
                    shape=(tile_k, tile_n),
                    offset=(b_offset, o_offset),
                    storage="shared",
                )
                wp.tile_matmul(at_values, g_values, accumulator)
            wp.tile_store(dweff[n], accumulator, offset=(p_offset, o_offset))

    else:

        @wp.kernel(module="unique", enable_backward=False)
        def kernel(
            xt: wp.array2d(dtype=wp.float16),
            g_nbo: wp.array3d(dtype=wp.float16),
            input_linear: wp.array2d(dtype=wp.int32),
            dweff: wp.array3d(dtype=wp.float16),
        ):
            n, p_tile, o_tile = wp.tid()
            p_offset = p_tile * tile_p
            o_offset = o_tile * tile_n
            accumulator = wp.tile_zeros(shape=(tile_p, tile_n), dtype=wp.float32)
            x_indices = wp.tile_load(
                input_linear[n], shape=tile_p, offset=p_offset, storage="shared"
            )
            for b_tile in range(xt.shape[1] // tile_k):
                b_offset = b_tile * tile_k
                at_values = wp.tile_load_indexed(
                    xt,
                    indices=x_indices,
                    shape=(tile_p, tile_k),
                    offset=(0, b_offset),
                    axis=0,
                    storage="shared",
                )
                g_values = wp.tile_load(
                    g_nbo[n],
                    shape=(tile_k, tile_n),
                    offset=(b_offset, o_offset),
                    storage="shared",
                )
                wp.tile_matmul(at_values, g_values, accumulator)
            wp.tile_store(
                dweff[n],
                wp.tile_map(_float_to_half, accumulator),
                offset=(p_offset, o_offset),
            )

    return kernel


def _make_grad_weight_stage4_kernel(tile_p: int, tile_n: int, fp32_stage: bool):
    """v4 gw staging: v3's 3D grid, but each CTA computes TWO adjacent o tiles.

    Each gathered xt tile is reused for both output-column tiles, halving the x-gather
    stream (the forward's reuse principle applied to the weight-gradient's batch contraction).
    Requires g_nbo padded to a multiple of 2*tile_n columns.
    """
    tile_k = 64

    if fp32_stage:

        @wp.kernel(module="unique", enable_backward=False)
        def kernel(
            xt: wp.array2d(dtype=wp.float16),
            g_nbo: wp.array3d(dtype=wp.float16),
            input_linear: wp.array2d(dtype=wp.int32),
            dweff: wp.array3d(dtype=wp.float32),
        ):
            n, p_tile, o_pair = wp.tid()
            p_offset = p_tile * tile_p
            o0 = o_pair * (2 * tile_n)
            o1 = o0 + tile_n
            acc0 = wp.tile_zeros(shape=(tile_p, tile_n), dtype=wp.float32)
            acc1 = wp.tile_zeros(shape=(tile_p, tile_n), dtype=wp.float32)
            x_indices = wp.tile_load(
                input_linear[n], shape=tile_p, offset=p_offset, storage="shared"
            )
            for b_tile in range(xt.shape[1] // tile_k):
                b_offset = b_tile * tile_k
                at_values = wp.tile_load_indexed(
                    xt,
                    indices=x_indices,
                    shape=(tile_p, tile_k),
                    offset=(0, b_offset),
                    axis=0,
                    storage="shared",
                )
                g0 = wp.tile_load(
                    g_nbo[n], shape=(tile_k, tile_n), offset=(b_offset, o0), storage="shared"
                )
                wp.tile_matmul(at_values, g0, acc0)
                g1 = wp.tile_load(
                    g_nbo[n], shape=(tile_k, tile_n), offset=(b_offset, o1), storage="shared"
                )
                wp.tile_matmul(at_values, g1, acc1)
            wp.tile_store(dweff[n], acc0, offset=(p_offset, o0))
            wp.tile_store(dweff[n], acc1, offset=(p_offset, o1))

    else:

        @wp.kernel(module="unique", enable_backward=False)
        def kernel(
            xt: wp.array2d(dtype=wp.float16),
            g_nbo: wp.array3d(dtype=wp.float16),
            input_linear: wp.array2d(dtype=wp.int32),
            dweff: wp.array3d(dtype=wp.float16),
        ):
            n, p_tile, o_pair = wp.tid()
            p_offset = p_tile * tile_p
            o0 = o_pair * (2 * tile_n)
            o1 = o0 + tile_n
            acc0 = wp.tile_zeros(shape=(tile_p, tile_n), dtype=wp.float32)
            acc1 = wp.tile_zeros(shape=(tile_p, tile_n), dtype=wp.float32)
            x_indices = wp.tile_load(
                input_linear[n], shape=tile_p, offset=p_offset, storage="shared"
            )
            for b_tile in range(xt.shape[1] // tile_k):
                b_offset = b_tile * tile_k
                at_values = wp.tile_load_indexed(
                    xt,
                    indices=x_indices,
                    shape=(tile_p, tile_k),
                    offset=(0, b_offset),
                    axis=0,
                    storage="shared",
                )
                g0 = wp.tile_load(
                    g_nbo[n], shape=(tile_k, tile_n), offset=(b_offset, o0), storage="shared"
                )
                wp.tile_matmul(at_values, g0, acc0)
                g1 = wp.tile_load(
                    g_nbo[n], shape=(tile_k, tile_n), offset=(b_offset, o1), storage="shared"
                )
                wp.tile_matmul(at_values, g1, acc1)
            wp.tile_store(dweff[n], wp.tile_map(_float_to_half, acc0), offset=(p_offset, o0))
            wp.tile_store(dweff[n], wp.tile_map(_float_to_half, acc1), offset=(p_offset, o1))

    return kernel


def _make_grad_weight_stage2_kernel(tile_p: int, tile_n: int, fp32_stage: bool):
    """v2 gw staging: one CTA per (n, o_tile), P64 loop in-CTA, transposed x operand.

    Takes ``xt [Cin*Nin+1, Bpad]`` (batch contiguous, cross-pollinated from the CUDA track):
    the indexed gather then reads contiguous 128-byte rows AND directly yields the A^T tile,
    eliminating wp.tile_transpose entirely.
    """
    tile_k = 64

    if fp32_stage:

        @wp.kernel(module="unique", enable_backward=False)
        def kernel(
            xt: wp.array2d(dtype=wp.float16),
            g_nbo: wp.array3d(dtype=wp.float16),
            input_linear: wp.array2d(dtype=wp.int32),
            dweff: wp.array3d(dtype=wp.float32),
        ):
            n, o_tile = wp.tid()
            o_offset = o_tile * tile_n
            for p_tile in range(input_linear.shape[1] // tile_p):
                p_offset = p_tile * tile_p
                accumulator = wp.tile_zeros(shape=(tile_p, tile_n), dtype=wp.float32)
                x_indices = wp.tile_load(
                    input_linear[n], shape=tile_p, offset=p_offset, storage="shared"
                )
                for b_tile in range(xt.shape[1] // tile_k):
                    b_offset = b_tile * tile_k
                    at_values = wp.tile_load_indexed(
                        xt,
                        indices=x_indices,
                        shape=(tile_p, tile_k),
                        offset=(0, b_offset),
                        axis=0,
                        storage="shared",
                    )
                    g_values = wp.tile_load(
                        g_nbo[n],
                        shape=(tile_k, tile_n),
                        offset=(b_offset, o_offset),
                        storage="shared",
                    )
                    wp.tile_matmul(at_values, g_values, accumulator)
                wp.tile_store(dweff[n], accumulator, offset=(p_offset, o_offset))

    else:

        @wp.kernel(module="unique", enable_backward=False)
        def kernel(
            xt: wp.array2d(dtype=wp.float16),
            g_nbo: wp.array3d(dtype=wp.float16),
            input_linear: wp.array2d(dtype=wp.int32),
            dweff: wp.array3d(dtype=wp.float16),
        ):
            n, o_tile = wp.tid()
            o_offset = o_tile * tile_n
            for p_tile in range(input_linear.shape[1] // tile_p):
                p_offset = p_tile * tile_p
                accumulator = wp.tile_zeros(shape=(tile_p, tile_n), dtype=wp.float32)
                x_indices = wp.tile_load(
                    input_linear[n], shape=tile_p, offset=p_offset, storage="shared"
                )
                for b_tile in range(xt.shape[1] // tile_k):
                    b_offset = b_tile * tile_k
                    at_values = wp.tile_load_indexed(
                        xt,
                        indices=x_indices,
                        shape=(tile_p, tile_k),
                        offset=(0, b_offset),
                        axis=0,
                        storage="shared",
                    )
                    g_values = wp.tile_load(
                        g_nbo[n],
                        shape=(tile_k, tile_n),
                        offset=(b_offset, o_offset),
                        storage="shared",
                    )
                    wp.tile_matmul(at_values, g_values, accumulator)
                wp.tile_store(
                    dweff[n],
                    wp.tile_map(_float_to_half, accumulator),
                    offset=(p_offset, o_offset),
                )

    return kernel


# name -> (maker, tile_p, tile_b_or_n, block_dim, version). Version 1 launches a 3D
# (n, p_tile, wide_tile) grid; version 2 launches (n, wide_tile) with the P loop in-CTA.
_GRAD_INPUT_STAGE_SPECS = {
    "gi_p64b64": (_make_grad_input_stage_kernel, 64, 64, 256, 1),
    "gi_p64b64bd128": (_make_grad_input_stage_kernel, 64, 64, 128, 1),
    "gi_p64b128": (_make_grad_input_stage_kernel, 64, 128, 256, 1),
    "gi2_p64b64": (_make_grad_input_stage2_kernel, 64, 64, 256, 2),
    "gi2_p64b64bd128": (_make_grad_input_stage2_kernel, 64, 64, 128, 2),
    "gi2_p64b128": (_make_grad_input_stage2_kernel, 64, 128, 256, 2),
}
_GRAD_WEIGHT_STAGE_SPECS = {
    "gw_p64n64": (_make_grad_weight_stage_kernel, 64, 64, 256, 1),
    "gw_p64n64bd128": (_make_grad_weight_stage_kernel, 64, 64, 128, 1),
    "gw_p64n128": (_make_grad_weight_stage_kernel, 64, 128, 256, 1),
    "gw2_p64n64": (_make_grad_weight_stage2_kernel, 64, 64, 256, 2),
    "gw2_p64n64bd128": (_make_grad_weight_stage2_kernel, 64, 64, 128, 2),
    "gw2_p64n128": (_make_grad_weight_stage2_kernel, 64, 128, 256, 2),
    "gw3_p64n64": (_make_grad_weight_stage3_kernel, 64, 64, 256, 3),
    "gw3_p64n128": (_make_grad_weight_stage3_kernel, 64, 128, 256, 3),
    # Negative result, kept for reproducibility: the two-o-accumulator reuse is
    # uniformly ~5-9% slower than gw3 (the xt layout already coalesced the gather, so extra
    # reuse only adds register pressure); gw4_p64n128 exceeds the sm_89 CTA budget entirely.
    "gw4_p64n64": (_make_grad_weight_stage4_kernel, 64, 64, 256, 4),
    "gw4_p64n128": (_make_grad_weight_stage4_kernel, 64, 128, 256, 4),
}
# Defaults fit on the Ada sweeps: the in-CTA P-loop gi2 kernel with 128-wide
# batch tiles, and the 3D-grid xt-operand gw3 kernel (best or tied on every measured shape).
_GRAD_INPUT_STAGE_DEFAULT = "gi2_p64b128"
_GRAD_WEIGHT_STAGE_DEFAULT = "gw3_p64n128"
_GRAD_KERNEL_OBJECTS = {}
_GRAD_KERNEL_VALIDATED = {}


def _grad_stage_dtype():
    return torch.float32 if os.environ.get("FOVI_KNN_GRAD_STAGE_FP32") == "1" else torch.float16


def _canary_grad_stage_ok(kernel, tile_p, tile_wide, block_dim, version, weight_operand, device):
    """NaN-canary a grad staging kernel (same silent-launch trap as the forward)."""
    stage_dtype = _grad_stage_dtype()
    g_canary = torch.zeros((1, 64, tile_wide), dtype=torch.float16, device=device)
    table = torch.zeros((1, tile_p), dtype=torch.int32, device=device)
    stage = torch.full((1, tile_p, tile_wide), float("nan"), dtype=stage_dtype, device=device)
    if weight_operand:
        # grad-input staging: (g_nob [1, 64, tile_b], weight_t [2, 2], weight_linear)
        inputs = [
            _descriptor(g_canary),
            _descriptor(torch.zeros((2, 2), dtype=torch.float16, device=device)),
            _descriptor(table),
        ]
    else:
        # grad-weight staging: v1 takes x_flat [B=64, 2]; v2/v3/v4 take xt [2, Bpad=64].
        # v4 reads two adjacent o tiles, so its g/stage canaries must be 2*tile_wide wide.
        if version == 4:
            g_canary = torch.zeros((1, 64, 2 * tile_wide), dtype=torch.float16, device=device)
            stage = torch.full(
                (1, tile_p, 2 * tile_wide), float("nan"), dtype=stage_dtype, device=device
            )
        x_canary_shape = (64, 2) if version == 1 else (2, 64)
        inputs = [
            _descriptor(torch.zeros(x_canary_shape, dtype=torch.float16, device=device)),
            _descriptor(g_canary),
            _descriptor(table),
        ]
    stream = wp.stream_from_torch(torch.cuda.current_stream(device))
    wp.launch_tiled(
        kernel,
        dim=(1, 1) if version == 2 else (1, 1, 1),
        inputs=inputs,
        outputs=[_descriptor(stage)],
        block_dim=block_dim,
        stream=stream,
    )
    torch.cuda.synchronize(device)
    return bool(torch.isfinite(stage).all().item())


def _get_grad_kernel(kind, name, device):
    """kind in {'gi', 'gw', 'csr'}; returns a per-device-validated kernel config or None."""
    fp32_stage = _grad_stage_dtype() is torch.float32
    object_key = (kind, name, fp32_stage)
    cached = _GRAD_KERNEL_OBJECTS.get(object_key)
    if cached is None:
        if kind == "csr":
            cached = (_make_grad_input_csr_kernel(fp32_stage),)
        else:
            specs = _GRAD_INPUT_STAGE_SPECS if kind == "gi" else _GRAD_WEIGHT_STAGE_SPECS
            maker, tile_p, tile_wide, block_dim, version = specs[name]
            cached = (maker(tile_p, tile_wide, fp32_stage), tile_p, tile_wide, block_dim, version)
        _GRAD_KERNEL_OBJECTS[object_key] = cached
    if kind == "csr":
        return cached
    validated_key = object_key + (device.index,)
    validated = _GRAD_KERNEL_VALIDATED.get(validated_key)
    if validated is None:
        kernel, tile_p, tile_wide, block_dim, version = cached
        validated = _canary_grad_stage_ok(
            kernel, tile_p, tile_wide, block_dim, version, kind == "gi", device
        )
        _GRAD_KERNEL_VALIDATED[validated_key] = validated
    return cached if validated else None


def _stage_budget_bytes():
    return int(os.environ.get("FOVI_KNN_STAGE_MIB", "512")) * 1024 * 1024


def _pad_flat_input_fp16(x, meta):
    """[B, Cin, Nin] fp16 -> [B, Cin*Nin + 1] with a trailing zero pad column."""
    x_flat = x.contiguous().reshape(x.shape[0], meta.cin * meta.nin)
    return F.pad(x_flat, (0, 1))


def grad_input(meta, grad_y, weight, config=None):
    """dx [B, Cin, Nin] fp32 for the compact operator; deterministic reverse-CSR reduction.

    ``grad_y`` [B, Cout, Nout] and ``weight`` [Cout, Q] must be fp16 (the autograd Function
    guarantees contiguity and compute dtype).
    """
    device = grad_y.device
    batch = grad_y.shape[0]
    explicit = config is not None
    config = config or _GRAD_INPUT_STAGE_DEFAULT
    stage_config = _get_grad_kernel("gi", config, device)
    if stage_config is None:
        if explicit:
            raise RuntimeError(f"Warp grad_input config {config!r} cannot launch on {device}")
        from .knn_autograd import CompactTorchOps

        return CompactTorchOps.grad_input(meta, grad_y, weight)
    kernel, tile_p, tile_b, block_dim, version = stage_config
    (csr_kernel,) = _get_grad_kernel("csr", "csr", device)
    stage_dtype = _grad_stage_dtype()

    # Plain wp.tile_load OVERREADS when a tile extends past the source extent: every
    # operand of the Cout contraction must be padded to full tiles so out-of-bounds memory
    # (which may hold NaNs) can never enter a product. Batch is padded to the tile width so
    # chunk windows are always tile-aligned; the CSR kernel only reads real batch columns.
    # NOTE: F.pad with all-zero padding returns the input VIEW unchanged, so contiguity must
    # be forced explicitly (a permuted view would silently feed strided, uncoalesced loads).
    cout_pad = (-meta.cout) % 64
    batch_pad = (-batch) % tile_b
    weight_t = F.pad(weight.reshape(meta.cout, meta.q).t(), (0, cout_pad)).contiguous()
    g_nob = (
        F.pad(grad_y.permute(2, 1, 0), (0, batch_pad, 0, cout_pad)).contiguous()
    )  # [Nout, Cout64, Bpad]
    batch_padded = batch + batch_pad
    dx = torch.empty(batch, meta.cin, meta.nin, device=device, dtype=torch.float32)
    stride_bytes = meta.nout * meta.p64 * stage_dtype.itemsize
    batch_chunk = max(tile_b, (_stage_budget_bytes() // max(stride_bytes, 1)) // tile_b * tile_b)
    stream = wp.stream_from_torch(torch.cuda.current_stream(device))
    for b0 in range(0, batch, batch_chunk):
        b_alloc = min(batch_chunk, batch_padded - b0)  # multiple of tile_b
        b1 = min(b0 + b_alloc, batch)
        da_stage = torch.empty(
            meta.nout, meta.p64, b_alloc, device=device, dtype=stage_dtype
        )
        if version == 1:
            stage_dim = (meta.nout, meta.p64 // tile_p, b_alloc // tile_b)
        else:
            stage_dim = (meta.nout, b_alloc // tile_b)
        wp.launch_tiled(
            kernel,
            dim=stage_dim,
            inputs=[
                _descriptor(g_nob[:, :, b0 : b0 + b_alloc]),
                _descriptor(weight_t),
                _descriptor(meta.weight_linear),
            ],
            outputs=[_descriptor(da_stage)],
            block_dim=block_dim,
            stream=stream,
        )
        wp.launch(
            csr_kernel,
            dim=(meta.nin, meta.cin, b1 - b0),
            inputs=[
                _descriptor(da_stage),
                _descriptor(meta.rev_rowptr),
                _descriptor(meta.rev_col),
                meta.k,
            ],
            outputs=[_descriptor(dx[b0:b1])],
            block_dim=256,
            stream=stream,
        )
    return dx


def grad_weight(meta, grad_y, x, config=None):
    """dW [Cout, Q] fp32; per-node dWeff staged on GPU, finished with fp32 index_add_.

    Mirrors ``CompactTorchOps.grad_weight`` semantics exactly: input pad entries gather the
    zero pad column (exact-zero contributions) and weight pad entries scatter zeros into
    row 0 of the [Q, Cout] accumulator.
    """
    device = grad_y.device
    batch = x.shape[0]
    explicit = config is not None
    config = config or _GRAD_WEIGHT_STAGE_DEFAULT
    stage_config = _get_grad_kernel("gw", config, device)
    if stage_config is None:
        if explicit:
            raise RuntimeError(f"Warp grad_weight config {config!r} cannot launch on {device}")
        from .knn_autograd import CompactTorchOps

        return CompactTorchOps.grad_weight(meta, grad_y, x)
    kernel, tile_p, tile_n, block_dim, version = stage_config
    stage_dtype = _grad_stage_dtype()

    # Pad both operands of the batch contraction to full tiles (see grad_input) and
    # force contiguity (zero-width F.pad returns views). v2+ kernels take the transposed
    # xt [Cin*Nin+1, Bpad] so the indexed gather reads contiguous rows without tile_transpose.
    # v4 processes o tiles in pairs, so g columns pad to 2*tile_n.
    batch_pad = (-batch) % 64
    o_group = tile_n * (2 if version == 4 else 1)
    cout_pad = (-meta.cout) % o_group
    x_flat = F.pad(_pad_flat_input_fp16(x, meta), (0, 0, 0, batch_pad))  # [Bpad, Cin*Nin+1]
    # v1 consumes x_flat directly; v2/v3 consume the transposed xt [Cin*Nin+1, Bpad].
    x_operand = x_flat.contiguous() if version == 1 else x_flat.t().contiguous()
    g_nbo = (
        F.pad(grad_y.permute(2, 0, 1), (0, cout_pad, 0, batch_pad)).contiguous()
    )  # [Nout, Bpad, Cout_n]
    dw_t = torch.zeros(meta.q, meta.cout, device=device, dtype=torch.float32)
    per_node_bytes = meta.p64 * meta.cout * stage_dtype.itemsize
    nout_chunk = max(1, min(meta.nout, _stage_budget_bytes() // max(per_node_bytes, 1)))
    stream = wp.stream_from_torch(torch.cuda.current_stream(device))
    for n0 in range(0, meta.nout, nout_chunk):
        n1 = min(n0 + nout_chunk, meta.nout)
        dweff = torch.empty(n1 - n0, meta.p64, meta.cout, device=device, dtype=stage_dtype)
        if version == 2:
            stage_dim = (n1 - n0, (meta.cout + tile_n - 1) // tile_n)
        else:
            stage_dim = (n1 - n0, meta.p64 // tile_p, (meta.cout + o_group - 1) // o_group)
        wp.launch_tiled(
            kernel,
            dim=stage_dim,
            inputs=[
                _descriptor(x_operand),
                _descriptor(g_nbo[n0:n1]),
                _descriptor(meta.input_linear[n0:n1]),
            ],
            outputs=[_descriptor(dweff)],
            block_dim=block_dim,
            stream=stream,
        )
        dw_t.index_add_(
            0,
            meta.weight_linear_flat[n0 * meta.p64 : n1 * meta.p64],
            dweff.reshape(-1, meta.cout).to(torch.float32),
        )
    return dw_t.t().contiguous().reshape(meta.cout, meta.q)


class WarpCompactOps:
    """Forward-only ops object matching the ``fovi.arch.knn_autograd`` registry interface."""

    name = "warp_compact"

    @staticmethod
    def forward(meta, x, weight, bias):
        if x.dtype != torch.float16 or weight.dtype != torch.float16:
            raise RuntimeError("warp_compact forward requires float16 compute dtype")
        if bias is None:
            bias = torch.zeros(meta.cout, device=x.device, dtype=torch.float16)
        return run_uncached(x, weight, bias, meta.input_linear, meta.weight_linear)

    @staticmethod
    def grad_input(meta, grad_y, weight):
        raise NotImplementedError("use warp_train for training; warp_compact is forward-only")

    @staticmethod
    def grad_weight(meta, grad_y, x):
        raise NotImplementedError("use warp_train for training; warp_compact is forward-only")


class WarpTrainOps:
    """Train-capable Warp ops: forward + adjoint kernels (fp16 compute, fp32 grads)."""

    name = "warp_train"

    forward = staticmethod(WarpCompactOps.forward)

    @staticmethod
    def grad_input(meta, grad_y, weight):
        if grad_y.dtype != torch.float16 or weight.dtype != torch.float16:
            raise RuntimeError("warp_train grad_input requires float16 compute dtype")
        return grad_input(meta, grad_y, weight)

    @staticmethod
    def grad_weight(meta, grad_y, x):
        if grad_y.dtype != torch.float16 or x.dtype != torch.float16:
            raise RuntimeError("warp_train grad_weight requires float16 compute dtype")
        return grad_weight(meta, grad_y, x)


try:
    from .knn_autograd import register_ops as _register_ops

    _register_ops(WarpTrainOps)
except ImportError:  # pragma: no cover - knn_autograd not present on older branches
    pass


def run_uncached(x, weight, bias, input_linear, weight_linear, config=None):
    """Uncached (per-call weight gather) fp16 forward.

    ``config`` optionally forces a large-batch kernel by name (see
    ``_UNCACHED_BATCH_CONFIG_SPECS``); by default a kernel is selected from the batch size.
    """
    output = torch.empty(
        (x.shape[0], weight.shape[0], input_linear.shape[0]),
        device=x.device,
        dtype=torch.float16,
    )
    batch = x.shape[0]
    explicit = config is not None
    if config is None and batch >= 64:
        config = _select_uncached_batch_config(
            batch, weight.shape[0], input_linear.shape[0], input_linear.shape[1]
        )
    resolved = _get_uncached_batch_config(config, device=x.device) if config is not None else None
    if resolved is None and explicit:
        raise RuntimeError(f"Warp KNN config {config!r} cannot launch on {x.device}")
    if resolved is not None:
        kernel, tile_m, tile_n, r, block_dim = resolved
    elif batch <= 16:
        kernel, tile_m, tile_n, r, block_dim = _UNCACHED_SMALL, 16, 32, 1, 128
    else:
        kernel, tile_m, tile_n, r, block_dim = _UNCACHED_LARGE, 32, 64, 1, 128
    # Materializing the [Q, Cout] transpose makes the per-node weight gather coalesced; the
    # copy costs microseconds and speeds the kernel up 1.3-2.0x at every batch size.
    # Columns are zero-padded to the o-tile grid so the indexed loads'
    # plain axis never reads past the buffer (tile-overread doctrine); the pad rides the same copy.
    cout = weight.shape[0]
    needed = ((cout + tile_n - 1) // tile_n) * tile_n
    weight_t = F.pad(weight.reshape(cout, -1).t(), (0, needed - cout))
    if not weight_t.is_contiguous():  # zero-width pad returns the strided view unchanged
        weight_t = weight_t.contiguous()
    return _launch(
        kernel,
        tile_m,
        tile_n,
        x,
        bias,
        input_linear,
        output,
        [
            _descriptor(x.reshape(x.shape[0], -1)),
            _descriptor(weight_t),
            _descriptor(_pad_bias_to_tiles(bias, cout, tile_n)),
            _descriptor(input_linear),
            _descriptor(weight_linear),
        ],
        r=r,
        block_dim=block_dim,
    )
