"""Native CUDA (CuPy NVRTC) kernels for compact KNN convolution.

This module implements the custom-CUDA forward path for :class:`KNNConvLayer`:

``y[b, o, n] = bias[o] + sum_{p} A_n[b, p] * W_n[p, o]``

where per output node ``n`` the operand tiles are gathered through two persistent int32 tables
(``input_linear``/``weight_linear``, shape ``[Nout, P64]`` with ``P64 = ceil(Cin*K/64)*64``; see
:mod:`fovi.arch.knn_optimization`). Out-of-range ``input_linear`` entries equal ``Cin*Nin`` and
must contribute zero.

Key design points versus an earlier WMMA prototype:

- Both gathered operands are re-laid-out host-side so the *contiguous* axis is the non-indexed
  one: ``x`` is transposed to ``xt[Cin*Nin + 1, Bpad]`` (last row zeros = padding row) and the
  weight to ``wt[Cin*V, Cpad]``. Each random index then selects a *row* whose copy is a clean
  run of 16-byte chunks, so staging is fully vectorized and coalesced.
- Shared-memory staging is double-buffered with ``cp.async`` (``__pipeline_memcpy_async``).
- One CTA processes ``RB`` consecutive batch sub-tiles against a single staged weight tile
  (multi-b-tile weight reuse; the killer was re-streaming ``W_n`` for every 16-row
  batch tile).
- Output is written to a ``[B, Nout, Cout]`` buffer (coalesced along ``Cout``) and returned as
  a ``[B, Cout, Nout]`` transposed view, matching the baseline layer's return convention.
- Everything is a C++ template over the scalar type (``__half``/``__nv_bfloat16``) and tile
  geometry, instantiated on demand through ``name_expressions``. Accumulation is always fp32.

Host-path note: kernel arguments are passed as raw ``data_ptr()`` values (``np.uint64`` packs
byte-identically to an ndarray pointer in CuPy's launch ABI), skipping per-call DLPack
export/stream negotiation — the small-shape training cells are host-bound.
This also sidesteps CuPy's lack of a bfloat16 dtype entirely.

NVRTC note: CuPy resolves ``libnvrtc.so.12`` from whatever is already loaded in the process.
``import torch`` (done below, before cupy) loads torch's bundled NVRTC (12.8 for this repo's
stack), which is required to emit native sm_120 (Blackwell) cubins. Importing cupy first in a
fresh process can silently bind an older NVRTC and fail with ``CUDA_ERROR_NO_BINARY_FOR_GPU``.
"""

from __future__ import annotations

import collections
import glob
import os
import sys

import torch  # noqa: F401  (must precede cupy import; see NVRTC note above)
import torch.nn.functional as F

try:
    import cupy as cp
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "fovi.arch.knn_cuda requires CuPy (pip install cupy-cuda12x). "
        "It is an optional native-CUDA backend for KNNConvLayer."
    ) from exc

import numpy as np


__all__ = [
    "forward",
    "grad_input",
    "grad_weight",
    "backward_combined",
    "CudaOps",
    "KernelConfig",
    "default_config",
    "default_grad_input_config",
    "default_grad_weight_config",
    "default_grad_weight_ksplit",
    "clear_kernel_cache",
]


_HEADER_ENV = "FOVI_CUDA_INCLUDE"
# Markers that must be visible through the include path for the kernel source to compile.
_REQUIRED_HEADERS = (
    "cuda_fp16.h",
    "cuda_bf16.h",
    "mma.h",
    "cuda_pipeline_primitives.h",
    os.path.join("crt", "host_defines.h"),
)


def _candidate_include_dirs():
    dirs = []
    override = os.environ.get(_HEADER_ENV)
    if override:
        dirs.extend(p for p in override.split(os.pathsep) if p)
    # Conda-packaged CUDA compiler headers (crt/ internals).
    conda_root = os.path.dirname(os.path.dirname(sys.prefix))
    for pattern in (
        os.path.join(conda_root, "pkgs", "cuda-crt-dev*", "targets", "x86_64-linux", "include"),
        os.path.join(sys.prefix, "targets", "x86_64-linux", "include"),
    ):
        dirs.extend(sorted(glob.glob(pattern), reverse=True))
    # Triton wheels bundle the full CUDA runtime header set (cuda_fp16/bf16, mma, pipeline).
    dirs.extend(
        sorted(
            glob.glob(
                os.path.join(
                    sys.prefix, "lib", "python*", "site-packages",
                    "triton", "backends", "nvidia", "include",
                )
            ),
            reverse=True,
        )
    )
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        dirs.append(os.path.join(cuda_home, "include"))
    if os.path.isdir("/usr/local/cuda/include"):
        dirs.append("/usr/local/cuda/include")
    # De-duplicate, preserving priority order.
    seen, unique = set(), []
    for d in dirs:
        if d not in seen and os.path.isdir(d):
            seen.add(d)
            unique.append(d)
    return unique


def _include_options():
    dirs = _candidate_include_dirs()
    missing = [
        h for h in _REQUIRED_HEADERS
        if not any(os.path.exists(os.path.join(d, h)) for d in dirs)
    ]
    if missing:
        raise ImportError(
            "fovi.arch.knn_cuda could not locate CUDA headers required for NVRTC "
            f"compilation (missing: {missing}; searched: {dirs or 'nothing'}). "
            "Remediation: `pip install triton` (bundles the CUDA runtime headers) and/or "
            "`conda install -c nvidia cuda-crt-dev`, or point the environment variable "
            f"{_HEADER_ENV} at one or more include directories (os.pathsep separated) "
            "containing cuda_fp16.h, cuda_bf16.h, mma.h, cuda_pipeline_primitives.h and crt/."
        )
    return tuple(f"-I{d}" for d in dirs)


_SOURCE = r"""
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cuda_pipeline_primitives.h>

using namespace nvcuda;

template <typename T> struct scalar_traits;
template <> struct scalar_traits<__half> {
    static __device__ __forceinline__ __half from_float(float v) { return __float2half_rn(v); }
};
template <> struct scalar_traits<__nv_bfloat16> {
    static __device__ __forceinline__ __nv_bfloat16 from_float(float v) {
        return __float2bfloat16_rn(v);
    }
};

// Copy VEC elements of T from global to shared. VEC*sizeof(T)==16 uses one 128-bit chunk and,
// when ASYNC, cp.async so the copy overlaps tensor-core work on the other buffer.
template <typename T, int ASYNC, int VEC>
__device__ __forceinline__ void copy_chunk(T* dst, const T* src) {
    if (VEC * sizeof(T) == 16) {
        if (ASYNC) {
            __pipeline_memcpy_async(dst, src, 16);
        } else {
            *reinterpret_cast<int4*>(dst) = *reinterpret_cast<const int4*>(src);
        }
    } else {
        #pragma unroll
        for (int i = 0; i < VEC; ++i) dst[i] = src[i];
    }
}

// Fused gather + GEMM forward for one output node per CTA z-slice:
//   y[b, n, o] = bias[o] + sum_p xt[input_linear[n, p]][b] * wt[weight_linear[n, p]][o]
//
// xt: [iw + 1, bpad]   input transposed; row iw is all zeros (padding row); bpad % BM == 0.
// wt: [q, cpad]        weight transposed; cpad % BN == 0, zero padded past cout.
// y:  [batch, nout, cout] (contiguous; the launcher returns the [B, Cout, Nout] transpose view).
//
// Template geometry: BM x BN output tile per batch sub-tile, BK contraction step, RB batch
// sub-tiles sharing each staged weight tile, WM x WN warps covering (BM, BN).
// KSTEPS > 0 bakes in the contraction trip count (requires p64 == KSTEPS * BK).
template <typename T, int BM, int BN, int BK, int RB, int WM, int WN,
          int KSTEPS, int ASYNC, int VEC>
__global__ void __launch_bounds__(WM * WN * 32) knn_conv_forward(
    const T* __restrict__ xt,
    const T* __restrict__ wt,
    const float* __restrict__ bias,
    const int* __restrict__ input_linear,
    const int* __restrict__ weight_linear,
    T* __restrict__ y,
    int batch, int bpad, int cout, int cpad, int nout, int p64)
{
    constexpr int SK = 8;             // skew (elements) against smem bank conflicts
    constexpr int LDA = BM + SK;      // a_smem leading dimension (col-major A tile)
    constexpr int LDW = BN + SK;      // w_smem leading dimension (row-major W tile)
    constexpr int A_TILE = BK * LDA;  // elements per (buffer, r) A tile
    constexpr int W_TILE = BK * LDW;  // elements per buffer W tile
    constexpr int THREADS = WM * WN * 32;
    constexpr int AM = BM / (WM * 16);  // 16x16 accumulators per warp along batch
    constexpr int AN = BN / (WN * 16);  // ... and along cout
    static_assert(BM % (WM * 16) == 0 && BN % (WN * 16) == 0, "warp tiling mismatch");
    static_assert(BM % VEC == 0 && BN % VEC == 0, "vector width must divide tile dims");

    extern __shared__ char smem_raw[];
    T* a_smem = reinterpret_cast<T*>(smem_raw);          // [2][RB][BK][LDA]
    T* w_smem = a_smem + 2 * RB * A_TILE;                // [2][BK][LDW]
    float* c_smem = reinterpret_cast<float*>(smem_raw);  // epilogue reuse: [BM][LDW]

    const int n = blockIdx.z;
    const int o_off = blockIdx.x * BN;
    const int bg_off = blockIdx.y * (RB * BM);
    const int r_count = min(RB, (bpad - bg_off) / BM);   // uniform across the CTA
    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int wm = warp / WN;
    const int wn = warp % WN;

    const int* il = input_linear + (size_t)n * p64;
    const int* wl = weight_linear + (size_t)n * p64;
    const int ksteps = (KSTEPS > 0) ? KSTEPS : (p64 / BK);

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[RB][AM][AN];
    #pragma unroll
    for (int r = 0; r < RB; ++r)
        #pragma unroll
        for (int i = 0; i < AM; ++i)
            #pragma unroll
            for (int j = 0; j < AN; ++j)
                wmma::fill_fragment(acc[r][i][j], 0.0f);

    // Stage the A tiles (r_count batch sub-tiles) and the W tile for contraction step kt into
    // shared-memory buffer `buf`. Consecutive threads copy consecutive chunks of one gathered
    // row, so global reads are coalesced along the contiguous (non-indexed) axis.
    auto stage = [&](int kt, int buf) {
        const int p0 = kt * BK;
        const int a_row_chunks = BM / VEC;
        const int a_chunks = r_count * BK * a_row_chunks;
        for (int c = tid; c < a_chunks; c += THREADS) {
            const int chunk = c % a_row_chunks;
            const int rp = c / a_row_chunks;
            const int p = rp % BK;
            const int r = rp / BK;
            const int idx = __ldg(il + p0 + p);
            const T* src = xt + (size_t)idx * bpad + bg_off + r * BM + chunk * VEC;
            T* dst = a_smem + ((size_t)buf * RB + r) * A_TILE + p * LDA + chunk * VEC;
            copy_chunk<T, ASYNC, VEC>(dst, src);
        }
        const int w_row_chunks = BN / VEC;
        const int w_chunks = BK * w_row_chunks;
        for (int c = tid; c < w_chunks; c += THREADS) {
            const int chunk = c % w_row_chunks;
            const int p = c / w_row_chunks;
            const int idx = __ldg(wl + p0 + p);
            const T* src = wt + (size_t)idx * cpad + o_off + chunk * VEC;
            T* dst = w_smem + (size_t)buf * W_TILE + p * LDW + chunk * VEC;
            copy_chunk<T, ASYNC, VEC>(dst, src);
        }
        if (ASYNC) __pipeline_commit();
    };

    stage(0, 0);
    for (int kt = 0; kt < ksteps; ++kt) {
        const int buf = kt & 1;
        if (kt + 1 < ksteps) stage(kt + 1, buf ^ 1);
        if (ASYNC) __pipeline_wait_prior((kt + 1 < ksteps) ? 1 : 0);
        __syncthreads();

        const T* a_base = a_smem + (size_t)buf * RB * A_TILE;
        const T* w_base = w_smem + (size_t)buf * W_TILE;
        #pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::row_major> b_frag[AN];
            #pragma unroll
            for (int j = 0; j < AN; ++j)
                wmma::load_matrix_sync(
                    b_frag[j], w_base + kk * LDW + wn * (AN * 16) + j * 16, LDW);
            #pragma unroll
            for (int r = 0; r < RB; ++r) {
                if (r >= r_count) break;
                #pragma unroll
                for (int i = 0; i < AM; ++i) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::col_major> a_frag;
                    wmma::load_matrix_sync(
                        a_frag,
                        a_base + r * A_TILE + kk * LDA + wm * (AM * 16) + i * 16,
                        LDA);
                    #pragma unroll
                    for (int j = 0; j < AN; ++j)
                        wmma::mma_sync(acc[r][i][j], a_frag, b_frag[j], acc[r][i][j]);
                }
            }
        }
        __syncthreads();  // buffer buf may be re-staged at kt + 2
    }

    // Epilogue: stage each batch sub-tile through fp32 smem, add bias, convert, and store
    // coalesced along cout into the [batch, nout, cout] buffer.
    constexpr int LDC = BN + SK;
    #pragma unroll
    for (int r = 0; r < RB; ++r) {
        if (r >= r_count) break;  // uniform: r_count is CTA-uniform
        #pragma unroll
        for (int i = 0; i < AM; ++i)
            #pragma unroll
            for (int j = 0; j < AN; ++j)
                wmma::store_matrix_sync(
                    c_smem + (wm * (AM * 16) + i * 16) * LDC + wn * (AN * 16) + j * 16,
                    acc[r][i][j], LDC, wmma::mem_row_major);
        __syncthreads();
        const int b0 = bg_off + r * BM;
        for (int e = tid; e < BM * BN; e += THREADS) {
            const int o = e % BN;
            const int m = e / BN;
            const int b = b0 + m;
            const int oo = o_off + o;
            if (b < batch && oo < cout) {
                const float v = c_smem[m * LDC + o] + bias[oo];
                y[((size_t)b * nout + n) * cout + oo] = scalar_traits<T>::from_float(v);
            }
        }
        __syncthreads();
    }
}

// Backward, input side. Per output node n and (batch, p) tile:
//     dA[b, p] = sum_o g_n[b, o] * W_n[p, o]
// with g_n read densely from gt[n] and W_n gathered by rows exactly like the forward; the
// result is scatter-accumulated with fp32 atomics into the TRANSPOSED gradient buffer
// dxt[input_linear[n, p], b] (layout inversion applied to the scatter itself: consecutive
// threads hit consecutive addresses along b, so the red.global traffic coalesces into full
// sectors instead of isolated 4-byte ops). The pad row iw is skipped; the host transposes
// dxt back to [B, Cin, Nin] once at the end. Gradients stay fp32 from the WMMA accumulators
// onward: no fp16/bf16 intermediates, so pre-scaled AMP gradients are strictly safer here
// than in the baseline autograd. No b guard is needed: pad batch rows of gt are zero, so
// their accumulated contributions are exactly 0.0 and land in dxt's pad columns.
//
// gt: [nout, bpad, cpad] (zero-padded rows/cols), wt: [q, cpad], dxt: [iw + 1, bpad] fp32
// zeroed. BM: batch tile (m), BN: p tile (must divide p64), BK: cout step (cpad % BK == 0).
template <typename T, int BM, int BN, int BK, int WM, int WN, int ASYNC>
__global__ void __launch_bounds__(WM * WN * 32) knn_conv_grad_input(
    const T* __restrict__ gt,
    const T* __restrict__ wt,
    const int* __restrict__ input_linear,
    const int* __restrict__ weight_linear,
    float* __restrict__ dxt,
    int bpad, int cpad, int nout, int p64, int iw)
{
    constexpr int SK = 8;
    constexpr int LDG = BK + SK;      // g_smem [BM][LDG] rows over b
    constexpr int LDW = BK + SK;      // w_smem [BN][LDW] rows over p
    constexpr int G_TILE = BM * LDG;
    constexpr int W_TILE = BN * LDW;
    constexpr int THREADS = WM * WN * 32;
    constexpr int AM = BM / (WM * 16);
    constexpr int AN = BN / (WN * 16);
    constexpr int VEC = 8;
    static_assert(BM % (WM * 16) == 0 && BN % (WN * 16) == 0, "warp tiling mismatch");

    extern __shared__ char smem_raw[];
    T* g_smem = reinterpret_cast<T*>(smem_raw);          // [2][BM][LDG]
    T* w_smem = g_smem + 2 * G_TILE;                     // [2][BN][LDW]
    float* c_smem = reinterpret_cast<float*>(smem_raw);  // epilogue reuse: [BM][BN + SK]

    const int n = blockIdx.z;
    const int p_off = blockIdx.x * BN;
    const int b_off = blockIdx.y * BM;
    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int wm = warp / WN;
    const int wn = warp % WN;

    const int* il = input_linear + (size_t)n * p64;
    const int* wl = weight_linear + (size_t)n * p64;
    const T* g_n = gt + (size_t)n * bpad * cpad;
    const int ksteps = cpad / BK;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[AM][AN];
    #pragma unroll
    for (int i = 0; i < AM; ++i)
        #pragma unroll
        for (int j = 0; j < AN; ++j)
            wmma::fill_fragment(acc[i][j], 0.0f);

    auto stage = [&](int kt, int buf) {
        const int o0 = kt * BK;
        const int row_chunks = BK / VEC;
        for (int c = tid; c < BM * row_chunks; c += THREADS) {
            const int chunk = c % row_chunks;
            const int b = c / row_chunks;
            const T* src = g_n + (size_t)(b_off + b) * cpad + o0 + chunk * VEC;
            T* dst = g_smem + (size_t)buf * G_TILE + b * LDG + chunk * VEC;
            copy_chunk<T, ASYNC, VEC>(dst, src);
        }
        for (int c = tid; c < BN * row_chunks; c += THREADS) {
            const int chunk = c % row_chunks;
            const int p = c / row_chunks;
            const int idx = __ldg(wl + p_off + p);
            const T* src = wt + (size_t)idx * cpad + o0 + chunk * VEC;
            T* dst = w_smem + (size_t)buf * W_TILE + p * LDW + chunk * VEC;
            copy_chunk<T, ASYNC, VEC>(dst, src);
        }
        if (ASYNC) __pipeline_commit();
    };

    stage(0, 0);
    for (int kt = 0; kt < ksteps; ++kt) {
        const int buf = kt & 1;
        if (kt + 1 < ksteps) stage(kt + 1, buf ^ 1);
        if (ASYNC) __pipeline_wait_prior((kt + 1 < ksteps) ? 1 : 0);
        __syncthreads();

        const T* g_base = g_smem + (size_t)buf * G_TILE;
        const T* w_base = w_smem + (size_t)buf * W_TILE;
        #pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            // b_frag holds W_n^T: element (k=o, n=p) lives at w_smem[p][o] -> col_major.
            wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::col_major> b_frag[AN];
            #pragma unroll
            for (int j = 0; j < AN; ++j)
                wmma::load_matrix_sync(
                    b_frag[j], w_base + (wn * (AN * 16) + j * 16) * LDW + kk, LDW);
            #pragma unroll
            for (int i = 0; i < AM; ++i) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major> a_frag;
                wmma::load_matrix_sync(
                    a_frag, g_base + (wm * (AM * 16) + i * 16) * LDG + kk, LDG);
                #pragma unroll
                for (int j = 0; j < AN; ++j)
                    wmma::mma_sync(acc[i][j], a_frag, b_frag[j], acc[i][j]);
            }
        }
        __syncthreads();
    }

    // Stage the accumulators COLUMN-major (c_smem[p][b], leading dim BM + SK) so the scatter
    // loop below reads linearly while consecutive threads write consecutive b addresses.
    constexpr int LDC = BM + SK;
    #pragma unroll
    for (int i = 0; i < AM; ++i)
        #pragma unroll
        for (int j = 0; j < AN; ++j)
            wmma::store_matrix_sync(
                c_smem + (wn * (AN * 16) + j * 16) * LDC + wm * (AM * 16) + i * 16,
                acc[i][j], LDC, wmma::mem_col_major);
    __syncthreads();
    for (int e = tid; e < BM * BN; e += THREADS) {
        const int b_l = e % BM;   // fastest: coalesced atomics along b
        const int p = e / BM;
        const int idx = __ldg(il + p_off + p);
        // Padding entries (idx == iw) contribute nothing; skipping them removes a
        // same-address atomic hotspot on pad-heavy shapes.
        if (idx < iw)
            atomicAdd(dxt + (size_t)idx * bpad + b_off + b_l, c_smem[p * LDC + b_l]);
    }
}

// Backward, weight side. Per output node n and (p, cout) tile:
//     dWeff[p, o] = sum_b A_n[b, p] * g_n[b, o]
// A_n is re-gathered through input_linear from the same transposed xt layout as the forward
// (the layout-inversion trick applies unchanged), g_n is dense, and the tile is
// scatter-accumulated with fp32 atomics into dwt[weight_linear[n, p], o]. Padding p entries
// gather the zero xt row, so they add exact zeros to dwt row 0 (matching the torch oracle).
//
// Split-K over the batch contraction: gridDim.y = (p64 / BM) * S encodes a split
// factor S; slice s of a p tile contracts batch steps [s*chunk, min((s+1)*chunk, ksteps)).
// Each slice atomically adds its fp32 partial into dwt — the existing scatter IS the
// second-stage reduction (fp32 atomics are traffic-bound, not serialization-bound, on this
// operator), so S=1 reproduces the unsplit kernel exactly and no extra buffer or
// reduction pass exists. Used when nout is too small for the (p, o, n) grid to fill the GPU.
//
// xt: [iw + 1, bpad], gt: [nout, bpad, cpad], dwt: [q, cout] fp32 zeroed.
// BM: p tile (must divide p64), BN: cout tile (cpad % BN == 0), BK: batch step (bpad % BK == 0).
template <typename T, int BM, int BN, int BK, int WM, int WN, int ASYNC>
__global__ void __launch_bounds__(WM * WN * 32) knn_conv_grad_weight(
    const T* __restrict__ xt,
    const T* __restrict__ gt,
    const int* __restrict__ input_linear,
    const int* __restrict__ weight_linear,
    float* __restrict__ dwt,
    int cout, int bpad, int cpad, int nout, int p64, int iw)
{
    constexpr int SK = 8;
    constexpr int LDA = BK + SK;      // a_smem [BM][LDA] rows over p, cols over b
    constexpr int LDG = BN + SK;      // g_smem [BK][LDG] rows over b, cols over o
    constexpr int A_TILE = BM * LDA;
    constexpr int G_TILE = BK * LDG;
    constexpr int THREADS = WM * WN * 32;
    constexpr int AM = BM / (WM * 16);
    constexpr int AN = BN / (WN * 16);
    constexpr int VEC = 8;
    static_assert(BM % (WM * 16) == 0 && BN % (WN * 16) == 0, "warp tiling mismatch");

    extern __shared__ char smem_raw[];
    T* a_smem = reinterpret_cast<T*>(smem_raw);          // [2][BM][LDA]
    T* g_smem = a_smem + 2 * A_TILE;                     // [2][BK][LDG]
    float* c_smem = reinterpret_cast<float*>(smem_raw);  // epilogue reuse: [BM][BN + SK]

    const int n = blockIdx.z;
    const int o_off = blockIdx.x * BN;
    const int p_tiles = p64 / BM;
    const int p_off = (blockIdx.y % p_tiles) * BM;
    const int split = blockIdx.y / p_tiles;      // split-K slice (gridDim.y = p_tiles * S)
    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int wm = warp / WN;
    const int wn = warp % WN;

    const int* il = input_linear + (size_t)n * p64;
    const int* wl = weight_linear + (size_t)n * p64;
    const T* g_n = gt + (size_t)n * bpad * cpad;
    const int total_ksteps = bpad / BK;
    const int nsplit = gridDim.y / p_tiles;
    const int chunk_steps = (total_ksteps + nsplit - 1) / nsplit;
    const int kt0 = split * chunk_steps;
    const int kt1 = min(kt0 + chunk_steps, total_ksteps);
    if (kt0 >= kt1) return;  // only when nsplit does not divide total_ksteps

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[AM][AN];
    #pragma unroll
    for (int i = 0; i < AM; ++i)
        #pragma unroll
        for (int j = 0; j < AN; ++j)
            wmma::fill_fragment(acc[i][j], 0.0f);

    auto stage = [&](int kt, int buf) {
        const int b0 = kt * BK;
        const int a_row_chunks = BK / VEC;
        for (int c = tid; c < BM * a_row_chunks; c += THREADS) {
            const int chunk = c % a_row_chunks;
            const int p = c / a_row_chunks;
            const int idx = __ldg(il + p_off + p);
            const T* src = xt + (size_t)idx * bpad + b0 + chunk * VEC;
            T* dst = a_smem + (size_t)buf * A_TILE + p * LDA + chunk * VEC;
            copy_chunk<T, ASYNC, VEC>(dst, src);
        }
        const int g_row_chunks = BN / VEC;
        for (int c = tid; c < BK * g_row_chunks; c += THREADS) {
            const int chunk = c % g_row_chunks;
            const int b = c / g_row_chunks;
            const T* src = g_n + (size_t)(b0 + b) * cpad + o_off + chunk * VEC;
            T* dst = g_smem + (size_t)buf * G_TILE + b * LDG + chunk * VEC;
            copy_chunk<T, ASYNC, VEC>(dst, src);
        }
        if (ASYNC) __pipeline_commit();
    };

    stage(kt0, 0);
    for (int kt = kt0; kt < kt1; ++kt) {
        const int buf = (kt - kt0) & 1;
        if (kt + 1 < kt1) stage(kt + 1, buf ^ 1);
        if (ASYNC) __pipeline_wait_prior((kt + 1 < kt1) ? 1 : 0);
        __syncthreads();

        const T* a_base = a_smem + (size_t)buf * A_TILE;
        const T* g_base = g_smem + (size_t)buf * G_TILE;
        #pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::row_major> b_frag[AN];
            #pragma unroll
            for (int j = 0; j < AN; ++j)
                wmma::load_matrix_sync(
                    b_frag[j], g_base + kk * LDG + wn * (AN * 16) + j * 16, LDG);
            #pragma unroll
            for (int i = 0; i < AM; ++i) {
                // a_frag holds A_n^T: element (m=p, k=b) lives at a_smem[p][b] -> row_major.
                wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major> a_frag;
                wmma::load_matrix_sync(
                    a_frag, a_base + (wm * (AM * 16) + i * 16) * LDA + kk, LDA);
                #pragma unroll
                for (int j = 0; j < AN; ++j)
                    wmma::mma_sync(acc[i][j], a_frag, b_frag[j], acc[i][j]);
            }
        }
        __syncthreads();
    }

    constexpr int LDC = BN + SK;
    #pragma unroll
    for (int i = 0; i < AM; ++i)
        #pragma unroll
        for (int j = 0; j < AN; ++j)
            wmma::store_matrix_sync(
                c_smem + (wm * (AM * 16) + i * 16) * LDC + wn * (AN * 16) + j * 16,
                acc[i][j], LDC, wmma::mem_row_major);
    __syncthreads();
    for (int e = tid; e < BM * BN; e += THREADS) {
        const int o = e % BN;
        const int pl = e / BN;
        // Padding p entries (input_linear == iw) gathered the zero xt row: their exact-zero
        // contributions to dwt row 0 can be skipped, avoiding a same-row atomic hotspot.
        if (o_off + o < cout && __ldg(il + p_off + pl) < iw) {
            const int row = __ldg(wl + p_off + pl);
            atomicAdd(dwt + (size_t)row * cout + o_off + o, c_smem[pl * LDC + o]);
        }
    }
}
"""


KernelConfig = collections.namedtuple(
    "KernelConfig",
    ["bm", "bn", "bk", "rb", "wm", "wn", "ksteps", "async_copy", "vec"],
)

_SKEW = 8
_CTYPE = {torch.float16: "__half", torch.bfloat16: "__nv_bfloat16"}
_KERNEL_CACHE = {}
_VALIDATED = set()


def clear_kernel_cache():
    """Drop compiled-module references and derived-tensor caches (CuPy's on-disk cache is
    unaffected)."""
    _KERNEL_CACHE.clear()
    _VALIDATED.clear()
    _GRAD_VALIDATED.clear()
    _GT_CACHE[0] = None
    _XT_CACHE[0] = None
    _ZERO_BIAS.clear()
    _STREAM_CACHE.clear()
    _WT_CACHE.clear()


def _smem_bytes(cfg, kind="forward"):
    epilogue = cfg.bm * (cfg.bn + _SKEW) * 4  # fp32 staging, reuses the same allocation
    if kind == "forward":
        a_tile = cfg.bk * (cfg.bm + _SKEW)
        w_tile = cfg.bk * (cfg.bn + _SKEW)
        main = 2 * (cfg.rb * a_tile + w_tile) * 2  # two buffers, 2-byte scalars
    elif kind == "grad_input":
        main = 2 * (cfg.bm * (cfg.bk + _SKEW) + cfg.bn * (cfg.bk + _SKEW)) * 2
        epilogue = cfg.bn * (cfg.bm + _SKEW) * 4  # column-major staging for the dxt scatter
    elif kind == "grad_weight":
        main = 2 * (cfg.bm * (cfg.bk + _SKEW) + cfg.bk * (cfg.bn + _SKEW)) * 2
    else:
        raise ValueError(f"unknown kernel kind {kind!r}")
    return max(main, epilogue)


def _instantiation(dtype, cfg, kind="forward"):
    if kind == "forward":
        return (
            f"knn_conv_forward<{_CTYPE[dtype]}, {cfg.bm}, {cfg.bn}, {cfg.bk}, {cfg.rb}, "
            f"{cfg.wm}, {cfg.wn}, {cfg.ksteps}, {int(cfg.async_copy)}, {cfg.vec}>"
        )
    return (
        f"knn_conv_{kind}<{_CTYPE[dtype]}, {cfg.bm}, {cfg.bn}, {cfg.bk}, "
        f"{cfg.wm}, {cfg.wn}, {int(cfg.async_copy)}>"
    )


def _get_kernel(device_index, dtype, cfg, kind="forward"):
    key = (device_index, dtype, cfg, kind)
    kernel = _KERNEL_CACHE.get(key)
    if kernel is not None:
        return kernel
    name = _instantiation(dtype, cfg, kind)
    options = ("--std=c++17", "--use_fast_math") + _include_options()
    with cp.cuda.Device(device_index):
        smem = _smem_bytes(cfg, kind)
        limit = cp.cuda.runtime.deviceGetAttribute(
            cp.cuda.runtime.cudaDevAttrMaxSharedMemoryPerBlockOptin, device_index
        )
        if smem > limit:
            raise ValueError(
                f"KernelConfig {cfg} needs {smem} B of shared memory per block but the device "
                f"limit is {limit} B; reduce bm/bn/rb"
            )
        module = cp.RawModule(code=_SOURCE, options=options, name_expressions=(name,))
        kernel = module.get_function(name)
        if smem > 48 * 1024:
            kernel.max_dynamic_shared_size_bytes = smem
    _KERNEL_CACHE[key] = (kernel, module)
    return _KERNEL_CACHE[key]


def default_config(batch, cout, p64, dtype=torch.float16, nout=None):
    """Heuristic geometry from the tuning sweep.

    Large batches want ~128 batch rows per CTA (weight-stream reuse saturates there) with a
    128-wide cout tile; small batches want the minimal 16-row tile with synchronous staging
    (cp.async overhead does not amortize with a single k-pipeline in flight per CTA).

    A small-Nout occupancy fallback (shrinking bm/bn to raise CTA count) was REFUTED by
    measurement: even at Nout=16, B=512 (128-CTA grid on 188
    SMs) the big tiles win — per-CTA staging efficiency beats residency. ``nout`` is accepted
    for signature stability but no longer alters the choice.
    """
    del nout
    if batch <= 16:
        cfg = KernelConfig(16, 128, 64, 1, 1, 8, 0, False, 8)
    elif batch <= 64:
        cfg = KernelConfig(64, 128, 64, 2, 2, 4, 0, True, 8)
    else:
        cfg = KernelConfig(128, 128, 64, 1, 4, 4, 0, True, 8)
    if cout <= 64:
        cfg = cfg._replace(bn=64, wn=4)
    return cfg


def _ptr(tensor):
    """Raw device pointer as a 64-bit kernel argument.

    CuPy's launch packs numpy scalars byte-wise, so a uint64 is ABI-identical to passing an
    ndarray's pointer — this skips per-call DLPack export/stream negotiation entirely (the
    losing cells are host-bound) and needs no bf16 dtype workaround. Lifetime
    is stream-ordered-safe: every argument is allocated by torch and consumed on the same
    stream. Kernels index raw dense row-major buffers, so contiguity is enforced (a
    non-contiguous view would be silently misread, e.g. as an untransposed weight).
    """
    if not tensor.is_contiguous():
        raise ValueError("knn_cuda kernel arguments must be contiguous")
    return np.uint64(tensor.data_ptr())


def _ceil_to(value, multiple):
    return -(-value // multiple) * multiple


def forward(x, weight, bias, input_linear, weight_linear, config=None):
    """Compute the compact KNN convolution forward pass with the native CUDA kernel.

    Args:
        x: ``[B, Cin, Nin]`` CUDA tensor, float16 or bfloat16 (fp32 inputs are cast).
        weight: ``[Cout, Cin * V]`` tensor on the same device (cast to ``x``'s compute dtype).
        bias: ``[Cout]`` tensor or ``None``; accumulated in fp32.
        input_linear: ``[Nout, P64]`` int32 table; entries ``== Cin * Nin`` select the zero row.
        weight_linear: ``[Nout, P64]`` int32 table into the transposed weight.
        config: optional :class:`KernelConfig` override.

    Returns:
        ``[B, Cout, Nout]`` tensor (transposed view of a contiguous ``[B, Nout, Cout]`` buffer,
        matching the baseline layer's return convention).
    """
    if not x.is_cuda:
        raise ValueError("knn_cuda.forward requires CUDA tensors")
    compute_dtype = x.dtype if x.dtype in _CTYPE else torch.float16
    device = x.device
    batch, cin, nin = x.shape
    cout = weight.shape[0]
    nout, p64 = input_linear.shape
    iw = cin * nin

    cfg = config or default_config(batch, cout, p64, compute_dtype, nout=nout)
    if p64 % cfg.bk != 0:
        raise ValueError(f"P64={p64} must be a multiple of BK={cfg.bk}")
    if cfg.ksteps and cfg.ksteps * cfg.bk != p64:
        raise ValueError(f"KSTEPS={cfg.ksteps} does not match P64={p64} / BK={cfg.bk}")
    if cfg.vec != 8 and cfg.async_copy:
        raise ValueError("cp.async staging requires 16-byte chunks (vec=8)")

    bpad = _ceil_to(max(batch, cfg.bm), cfg.bm)
    cpad = _ceil_to(cout, cfg.bn)

    # [iw + 1, bpad]: one zero pad row for out-of-range gathers, zero pad columns past batch.
    xt = _shared_xt(x, compute_dtype, bpad)
    wt = _shared_wt(weight, compute_dtype, cpad)
    if bias is None:
        bias_f = _zero_bias(device, cout)
    else:
        bias_f = bias.detach().to(device=device, dtype=torch.float32).contiguous()

    # First use of a (device, dtype, config) triple runs with a NaN-prefilled output as a canary
    # against silent launch failures recycling stale-but-plausible buffers.
    canary_key = (device.index, compute_dtype, cfg)
    canary = canary_key not in _VALIDATED
    if canary:
        y = torch.full((batch, nout, cout), float("nan"), device=device, dtype=compute_dtype)
    else:
        y = torch.empty(batch, nout, cout, device=device, dtype=compute_dtype)

    kernel, _module = _get_kernel(device.index, compute_dtype, cfg)
    grid = (cpad // cfg.bn, -(-bpad // (cfg.bm * cfg.rb)), nout)
    block = (cfg.wm * cfg.wn * 32,)
    _launch(
        kernel, grid, block,
        (
            _ptr(xt),
            _ptr(wt),
            _ptr(bias_f),
            _ptr(input_linear),
            _ptr(weight_linear),
            _ptr(y),
            np.int32(batch),
            np.int32(bpad),
            np.int32(cout),
            np.int32(cpad),
            np.int32(nout),
            np.int32(p64),
        ),
        _smem_bytes(cfg), device,
    )
    # CuPy raises on launch-API errors (bad config / excess smem); the first-use NaN canary
    # below additionally catches silently skipped launches recycling stale buffers.
    if canary:
        if bool(torch.isnan(y).any()):
            raise RuntimeError(
                f"knn_cuda forward canary failed: NaNs remain in the output (config={cfg}); "
                "the kernel did not fully overwrite the output buffer"
            )
        _VALIDATED.add(canary_key)
    return y.permute(0, 2, 1)


def default_grad_input_config(batch, cout):
    """dx kernel geometry: batch-tiled like the forward (rows/CTA law), p-tile fixed at 64."""
    if batch <= 16:
        return KernelConfig(16, 64, 64, 1, 1, 4, 0, False, 8)
    if batch <= 64:
        return KernelConfig(64, 64, 64, 1, 2, 4, 0, True, 8)
    return KernelConfig(128, 64, 64, 1, 4, 2, 0, True, 8)


def default_grad_weight_config(batch, cout, p64):
    """dW kernel geometry: p-tile 128 when P64 allows, cout-tile 128 for wide layers."""
    bm, wm = (128, 4) if p64 % 128 == 0 else (64, 2)
    bn, wn = (64, 4) if cout <= 64 else (128, 4)
    return KernelConfig(bm, bn, 64, 1, wm, wn, 0, batch > 64, 8)


_NUM_SMS = {}


def _num_sms(device_index):
    count = _NUM_SMS.get(device_index)
    if count is None:
        count = cp.cuda.runtime.deviceGetAttribute(
            cp.cuda.runtime.cudaDevAttrMultiProcessorCount, device_index
        )
        _NUM_SMS[device_index] = count
    return count


def default_grad_weight_ksplit(cfg, bpad, cpad, p64, nout, device_index):
    """Split-K-over-batch factor for the dW kernel.

    At small Nout the (o-tile, p-tile, node) grid cannot fill the GPU (the dW smem
    footprint caps residency at ~1 CTA/SM for the 128x128 geometry), so idle SMs are
    bought with extra contraction slices; each slice adds one [BM, BN] fp32 atomic pass,
    which is cheap for the high-Cin mid-tier shapes (tens of MiB) but NOT for wide-Q
    alexnet shapes — hence the grid-based gate rather than an unconditional split.
    Splitting below one BK batch step is impossible (ksteps caps the factor).
    """
    base_ctas = (cpad // cfg.bn) * (p64 // cfg.bm) * nout
    ksteps = bpad // cfg.bk
    if ksteps <= 1:
        return 1
    target = 2 * _num_sms(device_index)
    if base_ctas >= target:
        return 1
    return min(ksteps, -(-target // base_ctas))


_STREAM_CACHE = {}


def _current_stream(device):
    """Cached CuPy view of torch's current stream (host-path doctrine):
    raw stream pointer from the private fast path, ExternalStream cached per
    (device, raw stream)."""
    try:
        stream_ptr = torch._C._cuda_getCurrentRawStream(device.index)
    except AttributeError:  # pragma: no cover - older torch
        stream_ptr = torch.cuda.current_stream(device).cuda_stream
    key = (device.index, stream_ptr)
    stream = _STREAM_CACHE.get(key)
    if stream is None:
        stream = cp.cuda.ExternalStream(stream_ptr, device_id=device.index)
        _STREAM_CACHE[key] = stream
    return stream


def _launch(kernel, grid, block, args, smem, device):
    """Launch on torch's current stream with minimal per-call host overhead.

    The losing benchmark cells are host-bound: the CuPy device context
    push is skipped when the runtime device already matches (the common case under
    torch's device management).
    """
    stream = _current_stream(device)
    if cp.cuda.runtime.getDevice() == device.index:
        with stream:
            kernel(grid, block, args, shared_mem=smem)
    else:
        with cp.cuda.Device(device.index), stream:
            kernel(grid, block, args, shared_mem=smem)


_WT_CACHE = {}


def _shared_wt(weight, dtype, cpad):
    """Transposed, padded weight [Q, cpad] cached by (identity, version, dtype, cpad).

    One entry per parameter: valid across forward and grad_input within a step (the version
    counter only bumps at the optimizer update) and across all inference forwards until the
    weights change.
    """
    key = id(weight)
    cached = _WT_CACHE.get(key)
    version = weight._version
    if (
        cached is not None
        and cached[0] is weight
        and cached[1] == version
        and cached[2] == dtype
        and cached[3] == cpad
    ):
        return cached[4]
    if len(_WT_CACHE) > 256:  # bound growth from per-step AMP weight casts
        _WT_CACHE.clear()
    cout = weight.shape[0]
    w2 = weight.detach().reshape(cout, -1)
    if w2.dtype != dtype:
        w2 = w2.to(dtype)
    wt = F.pad(w2.transpose(0, 1), (0, cpad - cout)).contiguous()
    _WT_CACHE[key] = (weight, version, dtype, cpad, wt)
    return wt


def _grad_pads(batch, cout):
    """Shared (bpad, cpad) for both gradient kernels so one gt re-layout serves both.

    bpad: multiple of 64 (dW's batch contraction step) and of 128 above 64 (dx's batch tile).
    cpad: multiple of 64 (dx's cout contraction step) and of 128 above 64 (dW's cout tile).
    """
    bpad = _ceil_to(batch, 64) if batch <= 64 else _ceil_to(batch, 128)
    cpad = 64 if cout <= 64 else _ceil_to(cout, 128)
    return bpad, cpad


_GT_CACHE = [None]  # one slot: (grad_y ref, bpad, cpad, gt)
_XT_CACHE = [None]  # one slot: (x ref, dtype, bpad, xt)


def _shared_xt(x, dtype, bpad):
    """xt[iw + 1, bpad] re-layout of the flattened input, shared between the forward and
    grad_weight of one step (KNNConvFunction saves and passes the same x tensor).

    Identity-keyed one-slot cache like :func:`_shared_gt`.
    """
    cached = _XT_CACHE[0]
    if (
        cached is not None
        and cached[0] is x
        and cached[1] == dtype
        and cached[2] == bpad
    ):
        return cached[3]
    batch = x.shape[0]
    x2 = x.detach().reshape(batch, -1)
    if x2.dtype != dtype:
        x2 = x2.to(dtype)
    # .contiguous() is essential: with zero-width padding F.pad returns the transposed *view*,
    # which the raw kernel would misread as the untransposed buffer.
    xt = F.pad(x2.transpose(0, 1), (0, bpad - batch, 0, 1)).contiguous()
    _XT_CACHE[0] = (x, dtype, bpad, xt)
    return xt


def _shared_gt(grad_y, bpad, cpad):
    """gt[Nout, bpad, cpad] re-layout of grad_y, shared between grad_input and grad_weight.

    Identity-keyed one-slot cache: KNNConvFunction.backward calls grad_input then grad_weight
    with the same grad_y tensor, so the second call reuses the first call's re-layout. Keying
    on object identity (holding a reference) is essential — a data_ptr key could alias a
    recycled allocation from a later, different gradient tensor.
    """
    cached = _GT_CACHE[0]
    if cached is not None and cached[0] is grad_y and cached[1] == bpad and cached[2] == cpad:
        return cached[3]
    batch, cout, _nout = grad_y.shape
    gt = F.pad(grad_y.permute(2, 0, 1), (0, cpad - cout, 0, bpad - batch)).contiguous()
    _GT_CACHE[0] = (grad_y, bpad, cpad, gt)
    return gt


_ZERO_BIAS = {}


def _zero_bias(device, cout):
    key = (device.index, cout)
    bias = _ZERO_BIAS.get(key)
    if bias is None:
        bias = torch.zeros(cout, device=device, dtype=torch.float32)
        _ZERO_BIAS[key] = bias
    return bias


def grad_input(grad_y, weight, input_linear, weight_linear, cin, nin, config=None):
    """Input gradient: dx[b, c, m] = sum over (n, p) with input_linear[n, p] == c*Nin + m of
    (grad_y[:, :, n] @ W_n^T)[b, p], accumulated in fp32 via atomics.

    Args:
        grad_y: ``[B, Cout, Nout]`` CUDA tensor, float16 or bfloat16, contiguous.
        weight: ``[Cout, Cin * V]`` tensor (cast to ``grad_y``'s dtype for the gather).
        input_linear / weight_linear: ``[Nout, P64]`` int32 tables.
        cin, nin: input geometry (``iw = cin * nin`` addresses the pad-absorbing column).

    Returns:
        ``[B, Cin, Nin]`` float32 gradient.
    """
    if grad_y.dtype not in _CTYPE:
        raise ValueError("knn_cuda.grad_input requires float16 or bfloat16 grad_y")
    device = grad_y.device
    batch, cout, nout = grad_y.shape
    nout2, p64 = input_linear.shape
    if nout2 != nout:
        raise ValueError("grad_y and index tables disagree on Nout")
    iw = cin * nin

    cfg = config or default_grad_input_config(batch, cout)
    if p64 % cfg.bn != 0:
        raise ValueError(f"P64={p64} must be a multiple of the p tile BN={cfg.bn}")
    bpad, cpad = _grad_pads(batch, cout)
    if bpad % cfg.bm != 0 or cpad % cfg.bk != 0:
        raise ValueError(f"grad_input config {cfg} incompatible with bpad={bpad}, cpad={cpad}")

    gt = _shared_gt(grad_y, bpad, cpad)
    wt = _shared_wt(weight, grad_y.dtype, cpad)
    # Transposed accumulator: consecutive b along rows makes the atomic scatter coalesce.
    dxt = torch.zeros(iw + 1, bpad, device=device, dtype=torch.float32)

    kernel, _module = _get_kernel(device.index, grad_y.dtype, cfg, "grad_input")
    grid = (p64 // cfg.bn, bpad // cfg.bm, nout)
    block = (cfg.wm * cfg.wn * 32,)
    _launch(
        kernel, grid, block,
        (
            _ptr(gt), _ptr(wt), _ptr(input_linear), _ptr(weight_linear), _ptr(dxt),
            np.int32(bpad), np.int32(cpad), np.int32(nout), np.int32(p64), np.int32(iw),
        ),
        _smem_bytes(cfg, "grad_input"), device,
    )
    return dxt[:iw, :batch].t().reshape(batch, cin, nin)


def grad_weight(grad_y, x, input_linear, weight_linear, q, config=None, ksplit=None):
    """Weight gradient: dW[o, weight_linear[n, p]] += (A_n^T @ grad_y[:, :, n])[p, o],
    accumulated in fp32 via atomics into the transposed ``[Q, Cout]`` buffer.

    Padding p entries gather the zero xt row and therefore add exact zeros to row 0,
    matching ``CompactTorchOps.grad_weight`` semantics.

    Args:
        grad_y: ``[B, Cout, Nout]`` CUDA tensor, float16 or bfloat16, contiguous.
        x: ``[B, Cin, Nin]`` tensor (cast to ``grad_y``'s dtype for the gather).
        input_linear / weight_linear: ``[Nout, P64]`` int32 tables.
        q: ``Cin * V`` (the weight's flattened second dimension).
        ksplit: split-K-over-batch factor (``None`` = heuristic, see
            :func:`default_grad_weight_ksplit`; 1 = unsplit behavior). Encoded in
            ``gridDim.y = p_tiles * ksplit`` — no separate kernel instantiation.

    Returns:
        ``[Cout, Q]`` float32 gradient.
    """
    if grad_y.dtype not in _CTYPE:
        raise ValueError("knn_cuda.grad_weight requires float16 or bfloat16 grad_y")
    device = grad_y.device
    batch, cout, nout = grad_y.shape
    _, cin, nin = x.shape
    nout2, p64 = input_linear.shape
    if nout2 != nout:
        raise ValueError("grad_y and index tables disagree on Nout")
    iw = cin * nin

    cfg = config or default_grad_weight_config(batch, cout, p64)
    if p64 % cfg.bm != 0:
        raise ValueError(f"P64={p64} must be a multiple of the p tile BM={cfg.bm}")
    bpad, cpad = _grad_pads(batch, cout)
    if bpad % cfg.bk != 0 or cpad % cfg.bn != 0:
        raise ValueError(f"grad_weight config {cfg} incompatible with bpad={bpad}, cpad={cpad}")
    if ksplit is None:
        ksplit = default_grad_weight_ksplit(cfg, bpad, cpad, p64, nout, device.index)
    if not 1 <= ksplit <= bpad // cfg.bk:
        raise ValueError(f"ksplit={ksplit} outside [1, ksteps={bpad // cfg.bk}]")

    xt = _shared_xt(x, grad_y.dtype, bpad)
    gt = _shared_gt(grad_y, bpad, cpad)
    dwt = torch.zeros(q, cout, device=device, dtype=torch.float32)

    kernel, _module = _get_kernel(device.index, grad_y.dtype, cfg, "grad_weight")
    grid = (cpad // cfg.bn, (p64 // cfg.bm) * ksplit, nout)
    block = (cfg.wm * cfg.wn * 32,)
    _launch(
        kernel, grid, block,
        (
            _ptr(xt), _ptr(gt), _ptr(input_linear), _ptr(weight_linear), _ptr(dwt),
            np.int32(cout), np.int32(bpad), np.int32(cpad), np.int32(nout), np.int32(p64),
            np.int32(iw),
        ),
        _smem_bytes(cfg, "grad_weight"), device,
    )
    return dwt.t().contiguous()


def backward_combined(grad_y, x, weight, input_linear, weight_linear, cin, nin, q):
    """Both gradients from ONE host-side staging build.

    Semantically identical to ``grad_input(...)`` + ``grad_weight(...)`` — the same two
    kernels are launched with the same default configs — but the host path runs once:
    one pad computation, one ``gt`` re-layout (no cache round-trips), one ``xt``/``wt``
    fetch, one stream lookup shared by both launches. The separate entries stay for
    ``needs_input_grad`` edge cases (the autograd Function routes there when only one
    gradient is required).

    Returns:
        ``(dx [B, Cin, Nin] fp32, dW [Cout, Q] fp32)``.
    """
    if grad_y.dtype not in _CTYPE:
        raise ValueError("knn_cuda.backward_combined requires float16 or bfloat16 grad_y")
    device = grad_y.device
    batch, cout, nout = grad_y.shape
    nout2, p64 = input_linear.shape
    if nout2 != nout:
        raise ValueError("grad_y and index tables disagree on Nout")
    iw = cin * nin

    cfg_dx = default_grad_input_config(batch, cout)
    cfg_dw = default_grad_weight_config(batch, cout, p64)
    if p64 % cfg_dx.bn != 0 or p64 % cfg_dw.bm != 0:
        raise ValueError(f"P64={p64} incompatible with p tiles {cfg_dx.bn}/{cfg_dw.bm}")
    bpad, cpad = _grad_pads(batch, cout)
    ksplit = default_grad_weight_ksplit(cfg_dw, bpad, cpad, p64, nout, device.index)

    gt = _shared_gt(grad_y, bpad, cpad)
    wt = _shared_wt(weight, grad_y.dtype, cpad)
    xt = _shared_xt(x, grad_y.dtype, bpad)
    dxt = torch.zeros(iw + 1, bpad, device=device, dtype=torch.float32)
    dwt = torch.zeros(q, cout, device=device, dtype=torch.float32)

    kernel_dx, _m1 = _get_kernel(device.index, grad_y.dtype, cfg_dx, "grad_input")
    kernel_dw, _m2 = _get_kernel(device.index, grad_y.dtype, cfg_dw, "grad_weight")
    stream = _current_stream(device)
    args_dx = (
        _ptr(gt), _ptr(wt), _ptr(input_linear), _ptr(weight_linear), _ptr(dxt),
        np.int32(bpad), np.int32(cpad), np.int32(nout), np.int32(p64), np.int32(iw),
    )
    args_dw = (
        _ptr(xt), _ptr(gt), _ptr(input_linear), _ptr(weight_linear), _ptr(dwt),
        np.int32(cout), np.int32(bpad), np.int32(cpad), np.int32(nout), np.int32(p64),
        np.int32(iw),
    )
    if cp.cuda.runtime.getDevice() == device.index:
        with stream:
            kernel_dx(
                (p64 // cfg_dx.bn, bpad // cfg_dx.bm, nout),
                (cfg_dx.wm * cfg_dx.wn * 32,), args_dx,
                shared_mem=_smem_bytes(cfg_dx, "grad_input"),
            )
            kernel_dw(
                (cpad // cfg_dw.bn, (p64 // cfg_dw.bm) * ksplit, nout),
                (cfg_dw.wm * cfg_dw.wn * 32,), args_dw,
                shared_mem=_smem_bytes(cfg_dw, "grad_weight"),
            )
    else:
        with cp.cuda.Device(device.index), stream:
            kernel_dx(
                (p64 // cfg_dx.bn, bpad // cfg_dx.bm, nout),
                (cfg_dx.wm * cfg_dx.wn * 32,), args_dx,
                shared_mem=_smem_bytes(cfg_dx, "grad_input"),
            )
            kernel_dw(
                (cpad // cfg_dw.bn, (p64 // cfg_dw.bm) * ksplit, nout),
                (cfg_dw.wm * cfg_dw.wn * 32,), args_dw,
                shared_mem=_smem_bytes(cfg_dw, "grad_weight"),
            )
    dx = dxt[:iw, :batch].t().reshape(batch, cin, nin)
    return dx, dwt.t().contiguous()


# --------------------------------------------------------------------------------------
# Ops-registry integration (see fovi.arch.knn_autograd)
# --------------------------------------------------------------------------------------


_GRAD_VALIDATED = set()


def _grad_canary(kind, native, oracle_fn, device, dtype):
    """First-use-per-(kind, device, dtype) parity check against the torch compact oracle.

    Gradient kernels accumulate into zero-initialized buffers, so the forward's NaN-prefill
    canary cannot detect a silently skipped launch (all-zero gradients look plausible). One
    oracle comparison per process per configuration class closes that hole.
    """
    key = (kind, device, dtype)
    if key in _GRAD_VALIDATED:
        return
    oracle = oracle_fn()
    scale = max(oracle.abs().max().item(), 1.0)
    diff = (native - oracle).abs().max().item()
    if not diff <= 0.05 * scale:
        raise RuntimeError(
            f"knn_cuda {kind} first-use canary failed: max_abs {diff:.3e} vs oracle scale "
            f"{scale:.3e}; kernel output does not match the torch compact oracle"
        )
    _GRAD_VALIDATED.add(key)


class CudaOps:
    """Registry ops backed by the native CUDA kernels (forward + fused-atomic backward).

    fp32 compute (no AMP) delegates to the torch compact oracle: the WMMA kernels are
    fp16/bf16 only, and dense cuBLAS serves fp32 well. Gradients are
    computed fully fused (fp16/bf16 operand staging -> fp32 WMMA accumulators -> fp32 global
    atomics) with no reduced-precision intermediates, so pre-scaled AMP gradients are strictly
    more overflow-robust here than in the baseline autograd.
    """

    name = "cuda"

    @staticmethod
    def forward(meta, x, weight, bias):
        if x.dtype not in _CTYPE:
            from .knn_autograd import CompactTorchOps

            return CompactTorchOps.forward(meta, x, weight, bias)
        return forward(x, weight, bias, meta.input_linear, meta.weight_linear)

    @staticmethod
    def grad_input(meta, grad_y, weight):
        from .knn_autograd import CompactTorchOps

        if grad_y.dtype not in _CTYPE:
            return CompactTorchOps.grad_input(meta, grad_y, weight)
        dx = grad_input(
            grad_y, weight, meta.input_linear, meta.weight_linear, meta.cin, meta.nin
        )
        _grad_canary(
            "grad_input", dx, lambda: CompactTorchOps.grad_input(meta, grad_y, weight),
            grad_y.device.index, grad_y.dtype,
        )
        return dx

    @staticmethod
    def grad_weight(meta, grad_y, x):
        from .knn_autograd import CompactTorchOps

        if grad_y.dtype not in _CTYPE:
            return CompactTorchOps.grad_weight(meta, grad_y, x)
        dw = grad_weight(grad_y, x, meta.input_linear, meta.weight_linear, meta.q)
        _grad_canary(
            "grad_weight", dw, lambda: CompactTorchOps.grad_weight(meta, grad_y, x),
            grad_y.device.index, grad_y.dtype,
        )
        return dw

    @staticmethod
    def backward_combined(meta, grad_y, x, weight):
        """Both grads from one staging build.

        ``KNNConvFunction`` calls this automatically when BOTH input and weight grads are
        needed; the split ``grad_input``/``grad_weight`` entries above continue to serve
        the ``needs_input_grad`` edge cases. Same kernels, same configs, same math — the
        saving is host-path only (one prep instead of two op dispatches).
        """
        from .knn_autograd import CompactTorchOps

        if grad_y.dtype not in _CTYPE:
            return (
                CompactTorchOps.grad_input(meta, grad_y, weight),
                CompactTorchOps.grad_weight(meta, grad_y, x),
            )
        dx, dw = backward_combined(
            grad_y, x, weight, meta.input_linear, meta.weight_linear,
            meta.cin, meta.nin, meta.q,
        )
        _grad_canary(
            "grad_input", dx, lambda: CompactTorchOps.grad_input(meta, grad_y, weight),
            grad_y.device.index, grad_y.dtype,
        )
        _grad_canary(
            "grad_weight", dw, lambda: CompactTorchOps.grad_weight(meta, grad_y, x),
            grad_y.device.index, grad_y.dtype,
        )
        return dx, dw


def _register():
    try:
        from .knn_autograd import register_ops
    except ImportError:  # pragma: no cover - registry not present on older branches
        return
    register_ops(CudaOps)


_register()
