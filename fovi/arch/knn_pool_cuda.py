"""Native CUDA (CuPy NVRTC) kernels for :class:`KNNPoolingLayer`.

The baseline layer (``fovi/arch/knn.py``) materializes a ``[B, C, K, Nout]`` NaN-padded
gather (194 MiB at alexp0 B=512 fp16) and reduces it with NaN-aware torch ops; the max
path additionally performs a boolean-mask in-place fill whose ``nonzero`` triggers a host
sync every call, and its backward runs a max-scatter + index_put backward + fp16
``scatter_add_`` chain (~3x the forward). This module replaces both directions with one
kernel each:

- **Forward** (``knn_pool_forward``): fused gather + reduce. One CTA stages ``RS``
  consecutive ``(b, c)`` rows of ``x`` (rows are contiguous in global memory, so the
  staging copy is a single flat coalesced range) plus the persistent ``[K, Nout]``
  pad-token index table into shared memory, then reduces every ``(row, n)`` in registers.
  The ``B*C*K*Nout`` intermediate is never materialized and no NaN sentinel is needed
  (pad entries are detected by ``index == Nin``). ``max`` emits a ``uint8`` argmax slot
  per ``(b, c, n)`` for the backward, using the strict ``>`` / ascending-``k`` update
  whose tie rule (first occurrence of the maximum) matches ``torch.max`` on CUDA
  (verified empirically; the baseline backward routes ties identically). ``avg``
  accumulates valid neighbors in fp32 and divides by the precomputed valid count with an
  IEEE division, matching ``nanmean``'s sum-then-divide bit-for-bit (0/0 = NaN reproduces
  the empty-slice NaN for all-pad neighborhoods).
- **Backward** (``knn_pool_backward``): deterministic, atomic-free reverse-CSR gather.
  For each input node ``m`` the persistent CSR (built once per layer) lists the packed
  ``(k, n)`` pairs referencing it; every ``dx[b, c, m]`` is accumulated in fp32
  registers and written exactly once (so a NaN-prefill canary works for gradients too,
  unlike accumulate-into-zeros designs). ``max`` adds ``gy[b, c, n]`` where the stored
  argmax equals ``k``; ``avg`` adds ``gy[b, c, n] / count[n]``. This removes the
  baseline's fp16 scatter-add gradient noise entirely. Pooling is parameter-free and
  channel-wise, so there is no weight gradient and no ``Cin*K`` contraction anywhere.

Host-path doctrine from the conv kernels applies (sub-ms calls are host-bound): kernel
arguments are raw ``data_ptr()`` values, launches go through the conv module's cached
``ExternalStream`` path, and all derived index structures are built once per layer and
cached. Unlike the conv Function, the pooling backward needs neither ``x`` nor any
re-layout of ``grad_y`` — only the ``uint8`` argmax map is saved for training (max mode),
an activation-memory win over the baseline's saved gather intermediates.

NVRTC note: torch must be imported before CuPy (see :mod:`fovi.arch.knn_cuda`); importing
that module first guarantees the ordering and provides the shared launch machinery.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch  # noqa: F401  (must precede cupy; see NVRTC note in fovi.arch.knn_cuda)

from .knn_cuda import _include_options, _launch, _ptr  # imports torch, then cupy

import cupy as cp
import numpy as np


__all__ = [
    "PoolMeta",
    "pool_meta_from_indices",
    "ensure_pool_metadata",
    "pool_forward",
    "pool_backward",
    "KNNPoolFunction",
    "pool_function",
    "optimized_pool_forward",
    "clear_pool_cache",
]


_SOURCE = r"""
#include <cuda_fp16.h>
#include <cuda_bf16.h>

template <typename T> struct pool_traits;
template <> struct pool_traits<__half> {
    static __device__ __forceinline__ float to_float(__half v) { return __half2float(v); }
    static __device__ __forceinline__ __half from_float(float v) { return __float2half_rn(v); }
};
template <> struct pool_traits<__nv_bfloat16> {
    static __device__ __forceinline__ float to_float(__nv_bfloat16 v) {
        return __bfloat162float(v);
    }
    static __device__ __forceinline__ __nv_bfloat16 from_float(float v) {
        return __float2bfloat16_rn(v);
    }
};
template <> struct pool_traits<float> {
    static __device__ __forceinline__ float to_float(float v) { return v; }
    static __device__ __forceinline__ float from_float(float v) { return v; }
};

// Fused gather + reduce forward. One CTA stages RS consecutive (b*c) rows of x (a single
// contiguous global range) and the [K, Nout] pad-token index table in shared memory, then
// each thread reduces whole (row, n) neighborhoods in registers. Consecutive threads
// process consecutive n, so y (and aux) stores are coalesced.
//
// x:   [rows, nin]        rows = B*C, row-contiguous
// idx: [k, nout] int32    pad entries == nin
// count: [nout] f32       valid-neighbor count (MODE 1 only; 0 where all-pad)
// y:   [rows, nout]
// aux: [rows, nout] u8    argmax slot in [0, k) (MODE 0 only; may be null at inference)
//
// MODE 0 = max: strict > with ascending k picks the FIRST maximal neighbor, matching
// torch.max's CUDA tie rule so the backward routes ties exactly like the baseline.
// All-pad neighborhoods keep the -inf init (baseline parity). Input NaNs never win a
// strict > comparison, matching the baseline's NaN -> -inf masking for finite maxima.
// MODE 1 = avg: fp32 accumulate over valid neighbors, then IEEE division by the count
// (__fdiv_rn is immune to --use_fast_math), matching nanmean's fp32 sum-then-divide to
// the last bit and yielding 0/0 = NaN for all-pad neighborhoods like an empty nanmean.
template <typename T, int MODE, int RS, int THREADS>
__global__ void __launch_bounds__(THREADS) knn_pool_forward(
    const T* __restrict__ x,
    const int* __restrict__ idx,
    const float* __restrict__ count,
    T* __restrict__ y,
    unsigned char* __restrict__ aux,
    int rows, int nin, int nout, int k)
{
    extern __shared__ char smem_raw[];
    int* idx_s = reinterpret_cast<int*>(smem_raw);                   // [k * nout]
    float* cnt_s = reinterpret_cast<float*>(idx_s + k * nout);       // [nout] (MODE 1)
    T* xs = reinterpret_cast<T*>(cnt_s + (MODE == 1 ? nout : 0));    // [RS * nin]

    const int r0 = blockIdx.x * RS;
    const int rcount = min(RS, rows - r0);
    const int tid = threadIdx.x;

    for (int i = tid; i < k * nout; i += THREADS) idx_s[i] = idx[i];
    if (MODE == 1) {
        for (int i = tid; i < nout; i += THREADS) cnt_s[i] = count[i];
    }
    const T* src = x + (size_t)r0 * nin;
    for (int i = tid; i < rcount * nin; i += THREADS) xs[i] = src[i];
    __syncthreads();

    for (int e = tid; e < rcount * nout; e += THREADS) {
        const int n = e % nout;
        const int row = e / nout;
        const T* xr = xs + row * nin;
        const size_t off = ((size_t)r0 + row) * nout + n;
        if (MODE == 0) {
            float best = __int_as_float(0xff800000);  // -inf
            int barg = 0;
            for (int kk = 0; kk < k; ++kk) {
                const int m = idx_s[kk * nout + n];
                if (m < nin) {
                    const float v = pool_traits<T>::to_float(xr[m]);
                    if (v > best) { best = v; barg = kk; }
                }
            }
            y[off] = pool_traits<T>::from_float(best);
            if (aux) aux[off] = (unsigned char)barg;
        } else {
            float acc = 0.0f;
            for (int kk = 0; kk < k; ++kk) {
                const int m = idx_s[kk * nout + n];
                if (m < nin) acc += pool_traits<T>::to_float(xr[m]);
            }
            y[off] = pool_traits<T>::from_float(__fdiv_rn(acc, cnt_s[n]));
        }
    }
}

// Deterministic reverse-CSR backward: dx[row, m] is accumulated in an fp32 register from
// the CSR-listed (k, n) pairs referencing input node m and written exactly once — no
// atomics, no zero-initialized accumulator (a NaN-prefill canary is therefore valid for
// gradients here). grad_y (+ argmax for MODE 0) rows are staged like the forward; the CSR
// tables are broadcast reads shared by every CTA (L2-resident) with consecutive threads
// reading consecutive rowptr entries and overlapping rev_kn segments.
//
// gy:  [rows, nout]
// aux: [rows, nout] u8    (MODE 0)
// inv_count: [nout] f32   (MODE 1; all-pad NaN entries are never referenced by the CSR)
// rowptr: [nin + 1] int32; rev_kn: [nnz] int32 packed (k << 24) | n  (n < 2^24, k < 256)
// dx:  [rows, nin]
template <typename T, int MODE, int RS, int THREADS>
__global__ void __launch_bounds__(THREADS) knn_pool_backward(
    const T* __restrict__ gy,
    const unsigned char* __restrict__ aux,
    const float* __restrict__ inv_count,
    const int* __restrict__ rowptr,
    const int* __restrict__ rev_kn,
    T* __restrict__ dx,
    int rows, int nin, int nout)
{
    extern __shared__ char smem_raw[];
    float* invc_s = reinterpret_cast<float*>(smem_raw);              // [nout] (MODE 1)
    T* gys = reinterpret_cast<T*>(invc_s + (MODE == 1 ? nout : 0));  // [RS * nout]
    unsigned char* auxs = reinterpret_cast<unsigned char*>(gys + RS * nout);  // (MODE 0)

    const int r0 = blockIdx.x * RS;
    const int rcount = min(RS, rows - r0);
    const int tid = threadIdx.x;
    const int total = rcount * nout;

    if (MODE == 1) {
        for (int i = tid; i < nout; i += THREADS) invc_s[i] = inv_count[i];
    }
    const T* gsrc = gy + (size_t)r0 * nout;
    for (int i = tid; i < total; i += THREADS) gys[i] = gsrc[i];
    if (MODE == 0) {
        const unsigned char* asrc = aux + (size_t)r0 * nout;
        for (int i = tid; i < total; i += THREADS) auxs[i] = asrc[i];
    }
    __syncthreads();

    for (int e = tid; e < rcount * nin; e += THREADS) {
        const int m = e % nin;
        const int row = e / nin;
        const int j0 = __ldg(rowptr + m);
        const int j1 = __ldg(rowptr + m + 1);
        float acc = 0.0f;
        for (int j = j0; j < j1; ++j) {
            const unsigned int packed = (unsigned int)__ldg(rev_kn + j);
            const int n = (int)(packed & 0xFFFFFFu);
            const int kk = (int)(packed >> 24);
            if (MODE == 0) {
                if (auxs[row * nout + n] == kk)
                    acc += pool_traits<T>::to_float(gys[row * nout + n]);
            } else {
                acc += pool_traits<T>::to_float(gys[row * nout + n]) * invc_s[n];
            }
        }
        dx[((size_t)r0 + row) * nin + m] = pool_traits<T>::from_float(acc);
    }
}
"""


_CTYPE = {torch.float16: "__half", torch.bfloat16: "__nv_bfloat16", torch.float32: "float"}
_MODE_ID = {"max": 0, "avg": 1}
_THREADS = 256
_RS_CANDIDATES = (4, 2)

_KERNEL_CACHE = {}
_DEVICE_INFO = {}   # device index -> (sm count, opt-in smem limit)
_FWD_VALIDATED = set()
_BWD_VALIDATED = set()
_INDICES_META_CACHE = {}


def clear_pool_cache():
    """Drop compiled-module references, canary state, and the raw-indices meta cache
    (per-layer metadata lives on the layer; CuPy's on-disk compile cache is unaffected)."""
    _KERNEL_CACHE.clear()
    _FWD_VALIDATED.clear()
    _BWD_VALIDATED.clear()
    _INDICES_META_CACHE.clear()


# --------------------------------------------------------------------------------------
# Metadata
# --------------------------------------------------------------------------------------


@dataclass
class PoolMeta:
    """Derived, device-resident index tables for one pooling layer."""

    nin: int
    nout: int
    k: int
    indices: torch.Tensor      # [K, Nout] int32, pad entries == nin
    indices_i64: torch.Tensor  # [K, Nout] int64 twin (torch reference paths)
    count: torch.Tensor        # [Nout] float32 valid-neighbor count; 0 where all-pad
    inv_count: torch.Tensor    # [Nout] float32 1/valid-count; 0 where all-pad
    rev_rowptr: torch.Tensor   # [Nin + 1] int32 reverse-CSR row pointers
    rev_kn: torch.Tensor       # [nnz] int32 packed (k << 24) | n, deterministic order
    device: torch.device


def pool_meta_from_indices(indices, nin, device=None):
    """Build :class:`PoolMeta` from a ``[K, Nout]`` pad-token index table.

    ``indices`` follows the layer convention: entries ``>= nin`` (the layer uses exactly
    ``nin``) are padding. The reverse CSR drops padding entries, packs ``j = (k << 24) | n``
    and orders entries by (input node, then flattened ``k*Nout + n``) via a stable sort, so
    the backward's fp32 register accumulation is deterministic.
    """
    device = torch.device(device) if device is not None else indices.device
    idx64 = indices.detach().to(device=device, dtype=torch.int64).contiguous()
    if idx64.dim() != 2:
        raise ValueError(f"indices must be [K, Nout]; got shape {tuple(idx64.shape)}")
    k, nout = idx64.shape
    if not 1 <= k <= 255:
        raise ValueError(f"K={k} outside the uint8 argmax envelope [1, 255]")
    if nout >= 1 << 24:
        raise ValueError(f"Nout={nout} exceeds the 24-bit packed-CSR envelope")
    idx64 = torch.where(idx64 < nin, idx64, torch.full_like(idx64, nin))
    if bool((idx64 < 0).any()):
        raise ValueError("indices must be non-negative")

    valid = idx64 < nin
    counts = valid.sum(dim=0).to(torch.float32).contiguous()  # [Nout]; 0/0 -> NaN in-kernel
    inv_count = torch.where(
        counts > 0, 1.0 / counts.clamp(min=1.0), torch.zeros_like(counts)
    ).contiguous()  # all-pad neighborhoods have no CSR entries; 0 is never referenced

    flat = idx64.reshape(-1)
    valid_flat = flat < nin
    j = torch.nonzero(valid_flat, as_tuple=False).squeeze(1)
    m = flat[valid_flat]
    order = torch.argsort(m, stable=True)
    jj = j[order]
    kk = torch.div(jj, nout, rounding_mode="floor")
    nn = jj - kk * nout
    rev_kn = ((kk << 24) | nn).to(torch.int32).contiguous()
    node_counts = torch.bincount(m, minlength=nin)
    rowptr = torch.zeros(nin + 1, dtype=torch.int64, device=device)
    torch.cumsum(node_counts, dim=0, out=rowptr[1:])

    return PoolMeta(
        nin=nin,
        nout=nout,
        k=k,
        indices=idx64.to(torch.int32).contiguous(),
        indices_i64=idx64,
        count=counts,
        inv_count=inv_count,
        rev_rowptr=rowptr.to(torch.int32).contiguous(),
        rev_kn=rev_kn,
        device=device,
    )


def ensure_pool_metadata(layer, device):
    """Build (or fetch cached) :class:`PoolMeta` for a :class:`KNNPoolingLayer`.

    Cached on ``layer._knn_pool_cuda_meta`` (never enters ``state_dict``); the index table
    is fixed at layer construction, so the cache is keyed by device only.
    """
    device = torch.device(device)
    cached = getattr(layer, "_knn_pool_cuda_meta", None)
    if cached is not None and cached.device == device:
        return cached
    nin = int(getattr(layer, "knn_pad_token_val", layer.in_coords.shape[0]))
    meta = pool_meta_from_indices(layer.knn_indices_pad_token, nin, device)
    layer._knn_pool_cuda_meta = meta
    return meta


def _meta_for_indices(indices, nin, device):
    """Identity-keyed meta cache for raw index tables (benchmark-harness entry point)."""
    key = id(indices)
    hit = _INDICES_META_CACHE.get(key)
    if hit is not None and hit[0] is indices and hit[1] == nin and hit[2] == device:
        return hit[3]
    if len(_INDICES_META_CACHE) > 64:
        _INDICES_META_CACHE.clear()
    meta = pool_meta_from_indices(indices, nin, device)
    _INDICES_META_CACHE[key] = (indices, nin, device, meta)
    return meta


# --------------------------------------------------------------------------------------
# Kernel selection and launch
# --------------------------------------------------------------------------------------


def _device_info(device_index):
    info = _DEVICE_INFO.get(device_index)
    if info is None:
        sms = torch.cuda.get_device_properties(device_index).multi_processor_count
        limit = cp.cuda.runtime.deviceGetAttribute(
            cp.cuda.runtime.cudaDevAttrMaxSharedMemoryPerBlockOptin, device_index
        )
        info = (sms, limit)
        _DEVICE_INFO[device_index] = info
    return info


def _fwd_smem(meta, esize, rs, mode_id):
    table = meta.k * meta.nout * 4 + (meta.nout * 4 if mode_id == 1 else 0)
    return table + rs * meta.nin * esize


def _bwd_smem(meta, esize, rs, mode_id):
    per_row = meta.nout * esize + (meta.nout if mode_id == 0 else 0)
    return (meta.nout * 4 if mode_id == 1 else 0) + rs * per_row


def _pick_rs(rows, smem_of_rs, limit, device_index):
    """Largest RS in the swept set that fits shared memory while keeping >= 2 CTAs per SM.

    The RS sweep REFUTED large rows-per-CTA for pooling: with no
    weight stream to reuse, occupancy beats per-CTA staging amortization — rs=4 wins at
    training rows, rs=1-2 at small rows, and rs>=8 loses everywhere (the opposite of the
    conv kernels' rows/CTA=128 law).
    """
    sms, _ = _device_info(device_index)
    for rs in _RS_CANDIDATES:
        if smem_of_rs(rs) <= limit and rows >= rs * 2 * sms:
            return rs
    return 1


def _get_kernel(device_index, dtype, mode_id, rs, kind):
    key = (device_index, dtype, mode_id, rs, kind)
    cached = _KERNEL_CACHE.get(key)
    if cached is not None:
        return cached[0]
    name = f"knn_pool_{kind}<{_CTYPE[dtype]}, {mode_id}, {rs}, {_THREADS}>"
    options = ("--std=c++17", "--use_fast_math") + _include_options()
    with cp.cuda.Device(device_index):
        module = cp.RawModule(code=_SOURCE, options=options, name_expressions=(name,))
        kernel = module.get_function(name)
        # Shared-memory demand depends on the runtime shape, not just the template; opt the
        # kernel up to the device limit once so any fitting launch is legal.
        _, limit = _device_info(device_index)
        kernel.max_dynamic_shared_size_bytes = limit
    _KERNEL_CACHE[key] = (kernel, module)
    return kernel


def envelope_ok(meta, esize, device):
    """True when the shape fits the kernel design envelope at RS=1 on this device."""
    if not 1 <= meta.k <= 255 or meta.nout >= 1 << 24 or meta.nout < 1 or meta.nin < 1:
        return False
    _, limit = _device_info(device.index)
    for mode_id in (0, 1):
        if _fwd_smem(meta, esize, 1, mode_id) > limit:
            return False
        if _bwd_smem(meta, esize, 1, mode_id) > limit:
            return False
    return True


# --------------------------------------------------------------------------------------
# Forward / backward entry points
# --------------------------------------------------------------------------------------


def pool_forward(meta, x, mode, rs=None, need_aux=True):
    """Fused KNN pooling forward.

    Args:
        meta: :class:`PoolMeta` for the layer's index table.
        x: ``[B, C, Nin]`` CUDA tensor, float16 / bfloat16 / float32.
        mode: ``"max"`` or ``"avg"``.
        rs: optional rows-per-CTA override (default: heuristic).
        need_aux: emit the argmax map (max mode only; skip at inference to save traffic).

    Returns:
        ``(y, aux)``: ``y`` is ``[B, C, Nout]`` in ``x.dtype``; ``aux`` is the ``uint8``
        argmax map (max mode with ``need_aux``) or ``None``.
    """
    mode_id = _MODE_ID.get(mode)
    if mode_id is None:
        raise ValueError(f"unsupported pooling mode {mode!r} (kernel scope: max/avg)")
    if not x.is_cuda:
        raise ValueError("knn_pool_cuda.pool_forward requires CUDA tensors")
    if x.dtype not in _CTYPE:
        raise ValueError(f"unsupported dtype {x.dtype}")
    if x.dim() != 3 or x.shape[2] != meta.nin:
        raise ValueError(f"x must be [B, C, {meta.nin}]; got {tuple(x.shape)}")
    device = x.device
    x_c = x.contiguous()
    batch, channels, _ = x_c.shape
    rows = batch * channels
    esize = x_c.element_size()

    _, limit = _device_info(device.index)
    if rs is None:
        rs = _pick_rs(rows, lambda r: _fwd_smem(meta, esize, r, mode_id), limit, device.index)
    smem = _fwd_smem(meta, esize, rs, mode_id)
    if smem > limit:
        raise ValueError(
            f"forward smem {smem} B exceeds the device limit {limit} B "
            f"(nin={meta.nin}, nout={meta.nout}, k={meta.k}, rs={rs})"
        )

    # First-use NaN-prefill canary per (device, dtype, mode): catches silently skipped
    # launches recycling stale-but-plausible buffers (same doctrine as the conv kernels).
    canary_key = (device.index, x_c.dtype, mode_id)
    canary = canary_key not in _FWD_VALIDATED
    if canary:
        y = torch.full((batch, channels, meta.nout), float("nan"), device=device, dtype=x_c.dtype)
    else:
        y = torch.empty(batch, channels, meta.nout, device=device, dtype=x_c.dtype)
    aux = None
    aux_ptr = np.uint64(0)
    if mode_id == 0 and need_aux:
        aux = torch.empty(batch, channels, meta.nout, device=device, dtype=torch.uint8)
        aux_ptr = _ptr(aux)

    kernel = _get_kernel(device.index, x_c.dtype, mode_id, rs, "forward")
    grid = (-(-rows // rs),)
    _launch(
        kernel, grid, (_THREADS,),
        (
            _ptr(x_c), _ptr(meta.indices), _ptr(meta.count), _ptr(y), aux_ptr,
            np.int32(rows), np.int32(meta.nin), np.int32(meta.nout), np.int32(meta.k),
        ),
        smem, device,
    )
    if canary:
        bad = torch.isnan(y)
        if mode_id == 1:
            # avg legitimately emits NaN for all-pad neighborhoods (nanmean of an empty
            # slice); only NaNs in valid columns indicate a skipped launch.
            bad = bad & (meta.count > 0).reshape(1, 1, -1)
        if bool(bad.any()):
            raise RuntimeError(
                "knn_pool_cuda forward canary failed: NaNs remain in the output "
                f"(mode={mode}, rs={rs}); the kernel did not fully overwrite the buffer"
            )
        _FWD_VALIDATED.add(canary_key)
    return y, aux


def _backward_oracle(meta, grad_y, aux, mode):
    """fp32 torch reference for the backward (first-use canary): same routing semantics,
    scatter-add accumulation into a pad-spillway column."""
    batch, channels, nout = grad_y.shape
    gyf = grad_y.detach().reshape(-1, nout).to(torch.float32)
    rows = gyf.shape[0]
    dxf = torch.zeros(rows, meta.nin + 1, dtype=torch.float32, device=grad_y.device)
    if mode == "max":
        pos = torch.gather(meta.indices_i64, 0, aux.detach().reshape(-1, nout).to(torch.int64))
        dxf.scatter_add_(1, pos, gyf)
    else:
        contrib = gyf * meta.inv_count.reshape(1, -1)
        for kk in range(meta.k):
            pos = meta.indices_i64[kk].reshape(1, -1).expand(rows, -1)
            dxf.scatter_add_(1, pos, contrib)
    return dxf[:, : meta.nin].reshape(batch, channels, meta.nin)


def pool_backward(meta, grad_y, aux, mode, rs=None):
    """Deterministic reverse-CSR KNN pooling backward.

    Args:
        meta: :class:`PoolMeta` (must be the forward's meta).
        grad_y: ``[B, C, Nout]`` CUDA tensor, float16 / bfloat16 / float32.
        aux: the forward's ``uint8`` argmax map (required for ``max``; ignored for ``avg``).
        mode: ``"max"`` or ``"avg"``.
        rs: optional rows-per-CTA override.

    Returns:
        ``[B, C, Nin]`` gradient in ``grad_y.dtype`` (fp32 register accumulation, one
        final rounding — strictly less gradient noise than the baseline's fp16
        ``scatter_add_`` chain).
    """
    mode_id = _MODE_ID.get(mode)
    if mode_id is None:
        raise ValueError(f"unsupported pooling mode {mode!r} (kernel scope: max/avg)")
    if grad_y.dtype not in _CTYPE:
        raise ValueError(f"unsupported dtype {grad_y.dtype}")
    if grad_y.dim() != 3 or grad_y.shape[2] != meta.nout:
        raise ValueError(f"grad_y must be [B, C, {meta.nout}]; got {tuple(grad_y.shape)}")
    device = grad_y.device
    g_c = grad_y.contiguous()
    batch, channels, _ = g_c.shape
    rows = batch * channels
    esize = g_c.element_size()
    if mode_id == 0:
        if aux is None:
            raise ValueError("max-mode backward requires the forward's argmax map")
        if aux.shape != g_c.shape:
            raise ValueError("argmax map and grad_y shapes disagree")
        aux = aux.contiguous()
        aux_ptr = _ptr(aux)
    else:
        aux_ptr = np.uint64(0)

    _, limit = _device_info(device.index)
    if rs is None:
        rs = _pick_rs(rows, lambda r: _bwd_smem(meta, esize, r, mode_id), limit, device.index)
    smem = _bwd_smem(meta, esize, rs, mode_id)
    if smem > limit:
        raise ValueError(
            f"backward smem {smem} B exceeds the device limit {limit} B "
            f"(nout={meta.nout}, rs={rs})"
        )

    # dx is written exactly once per element (no accumulation), so the NaN-prefill canary
    # is valid for the gradient too; the oracle canary additionally pins the CSR semantics.
    canary_key = (device.index, g_c.dtype, mode_id)
    canary = canary_key not in _BWD_VALIDATED
    if canary:
        dx = torch.full((batch, channels, meta.nin), float("nan"), device=device, dtype=g_c.dtype)
    else:
        dx = torch.empty(batch, channels, meta.nin, device=device, dtype=g_c.dtype)

    kernel = _get_kernel(device.index, g_c.dtype, mode_id, rs, "backward")
    grid = (-(-rows // rs),)
    _launch(
        kernel, grid, (_THREADS,),
        (
            _ptr(g_c), aux_ptr, _ptr(meta.inv_count), _ptr(meta.rev_rowptr),
            _ptr(meta.rev_kn), _ptr(dx),
            np.int32(rows), np.int32(meta.nin), np.int32(meta.nout),
        ),
        smem, device,
    )
    if canary:
        if bool(torch.isnan(dx).any()):
            raise RuntimeError(
                "knn_pool_cuda backward canary failed: NaNs remain in dx "
                f"(mode={mode}, rs={rs}); the kernel did not fully overwrite the buffer"
            )
        oracle = _backward_oracle(meta, g_c, aux, mode)
        scale = max(oracle.abs().max().item(), 1.0)
        diff = (dx.float() - oracle).abs().max().item()
        if not diff <= 0.05 * scale:
            raise RuntimeError(
                f"knn_pool_cuda backward first-use canary failed: max_abs {diff:.3e} vs "
                f"oracle scale {scale:.3e} (mode={mode})"
            )
        _BWD_VALIDATED.add(canary_key)
    return dx


# --------------------------------------------------------------------------------------
# Autograd wrapper and layer-level dispatch
# --------------------------------------------------------------------------------------


class KNNPoolFunction(torch.autograd.Function):
    """Autograd wrapper: fused forward + deterministic rev-CSR backward.

    Pooling is parameter-free and not autocast-eligible (the baseline runs in the input
    dtype under AMP, consuming the preceding conv's half-precision output), so no dtype
    cast happens here: compute dtype == ``x.dtype``. Only the ``uint8`` argmax map is
    saved for max mode — neither ``x`` nor any gather intermediate is retained.
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, x, meta, mode, need_aux):
        x_c = x.contiguous()
        y, aux = pool_forward(meta, x_c, mode, need_aux=need_aux)
        ctx.meta = meta
        ctx.mode = mode
        ctx.compute_dtype = x_c.dtype
        if aux is not None:
            ctx.save_for_backward(aux)
        return y

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_y):
        saved = ctx.saved_tensors
        aux = saved[0] if saved else None
        g = grad_y.contiguous()
        if g.dtype != ctx.compute_dtype:
            g = g.to(ctx.compute_dtype)
        dx = pool_backward(ctx.meta, g, aux, ctx.mode)  # in compute (== x) dtype
        return dx, None, None, None


def _needs_aux(x, mode):
    return mode == "max" and torch.is_grad_enabled() and x.requires_grad


def pool_function(x, indices, mode):
    """Benchmark-harness-compatible entry point: ``(x, [K, Nout] indices, mode) -> y``
    with full autograd support (meta cached by index-table identity)."""
    meta = _meta_for_indices(indices, x.shape[2], x.device)
    return KNNPoolFunction.apply(x, meta, mode, _needs_aux(x, mode))


def optimized_pool_forward(layer, x):
    """Layer-level dispatch mirroring :func:`fovi.arch.knn_optimization.optimized_forward`.

    Returns the optimized result, or ``None`` when the caller should fall back to the
    baseline (unsupported mode/dtype/device, shape outside the shared-memory envelope).
    Reads only ``layer.mode``, ``layer.knn_indices_pad_token`` and
    ``layer.knn_pad_token_val`` (``in_coords.shape[0]`` fallback); caches metadata on
    ``layer._knn_pool_cuda_meta``.
    """
    mode = getattr(layer, "mode", None)
    if mode not in _MODE_ID:
        return None
    if not (torch.is_tensor(x) and x.is_cuda and x.dim() == 3 and x.dtype in _CTYPE):
        return None
    nin = int(getattr(layer, "knn_pad_token_val", layer.in_coords.shape[0]))
    if x.shape[2] != nin or x.shape[0] * x.shape[1] == 0:
        return None
    try:
        meta = ensure_pool_metadata(layer, x.device)
    except ValueError:  # index table outside the packing envelope (K > 255, huge Nout)
        return None
    if not envelope_ok(meta, x.element_size(), x.device):
        return None
    return KNNPoolFunction.apply(x, meta, mode, _needs_aux(x, mode))
