"""CuPy/NVRTC kernels for sampling CUDA images at moving foveated grids.

The public sampler owns coordinate preparation and backend selection.  This module is a
small optional implementation detail: it accepts raw Torch storage and launches on Torch's
current CUDA stream without constructing a CuPy array or a full-resolution float tensor.
The floating nearest specializations are private and exist to benchmark the fused kernel;
public floating sampling remains on ``torch.grid_sample``.
"""

from __future__ import annotations

import torch  # must be imported before CuPy so its bundled NVRTC is selected

try:
    import cupy as cp
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "The native uint8 grid sampler requires CuPy (pip install cupy-cuda12x)."
    ) from exc

import numpy as np


__all__ = ["sample_uint8", "clear_kernel_cache"]


_SOURCE = r"""
extern "C" __global__ void fovi_uint8_nearest(
    const unsigned char* __restrict__ image,
    const float* __restrict__ base_grid,
    const float* __restrict__ fix_loc,
    const float* __restrict__ fix_size,
    unsigned char* __restrict__ output,
    long long stride_b, long long stride_c,
    long long stride_h, long long stride_w,
    int batch, int channels, int height, int width, int points)
{
    const long long total = (long long)batch * channels * points;
    for (long long linear = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         linear < total; linear += (long long)blockDim.x * gridDim.x) {
        const int n = linear % points;
        const int c = (linear / points) % channels;
        const int b = linear / ((long long)points * channels);

        const float pixel_x = base_grid[2 * n] * (0.5f * fix_size[2 * b + 1])
                            + fix_loc[2 * b + 1] * width;
        const float pixel_y = base_grid[2 * n + 1] * (0.5f * fix_size[2 * b])
                            + fix_loc[2 * b] * height;
        const int x = __float2int_rn(pixel_x - 0.5f);
        const int y = __float2int_rn(pixel_y - 0.5f);
        unsigned char value = 0;
        if ((unsigned int)x < (unsigned int)width &&
            (unsigned int)y < (unsigned int)height) {
            const long long offset = (long long)b * stride_b + (long long)c * stride_c
                                   + (long long)y * stride_h + (long long)x * stride_w;
            value = image[offset];
        }
        output[linear] = value;
    }
}

extern "C" __global__ void fovi_float32_nearest(
    const float* __restrict__ image,
    const float* __restrict__ base_grid,
    const float* __restrict__ fix_loc,
    const float* __restrict__ fix_size,
    float* __restrict__ output,
    long long stride_b, long long stride_c,
    long long stride_h, long long stride_w,
    int batch, int channels, int height, int width, int points)
{
    const long long total = (long long)batch * channels * points;
    for (long long linear = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         linear < total; linear += (long long)blockDim.x * gridDim.x) {
        const int n = linear % points;
        const int c = (linear / points) % channels;
        const int b = linear / ((long long)points * channels);

        const float scale_x = __fmul_rn(fix_size[2 * b + 1], 0.5f);
        const float scale_y = __fmul_rn(fix_size[2 * b], 0.5f);
        const float center_x = __fmul_rn(fix_loc[2 * b + 1], (float)width);
        const float center_y = __fmul_rn(fix_loc[2 * b], (float)height);
        const float pixel_x = __fadd_rn(
            __fmul_rn(base_grid[2 * n], scale_x), center_x);
        const float pixel_y = __fadd_rn(
            __fmul_rn(base_grid[2 * n + 1], scale_y), center_y);
        const int x = __float2int_rn(pixel_x - 0.5f);
        const int y = __float2int_rn(pixel_y - 0.5f);
        float value = 0.0f;
        if ((unsigned int)x < (unsigned int)width &&
            (unsigned int)y < (unsigned int)height) {
            const long long offset = (long long)b * stride_b + (long long)c * stride_c
                                   + (long long)y * stride_h + (long long)x * stride_w;
            value = image[offset];
        }
        output[linear] = value;
    }
}

extern "C" __global__ void fovi_float16_nearest(
    const unsigned short* __restrict__ image,
    const float* __restrict__ base_grid,
    const float* __restrict__ fix_loc,
    const float* __restrict__ fix_size,
    unsigned short* __restrict__ output,
    long long stride_b, long long stride_c,
    long long stride_h, long long stride_w,
    int batch, int channels, int height, int width, int points)
{
    const long long total = (long long)batch * channels * points;
    for (long long linear = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         linear < total; linear += (long long)blockDim.x * gridDim.x) {
        const int n = linear % points;
        const int c = (linear / points) % channels;
        const int b = linear / ((long long)points * channels);

        const float scale_x = __fmul_rn(fix_size[2 * b + 1], 0.5f);
        const float scale_y = __fmul_rn(fix_size[2 * b], 0.5f);
        const float center_x = __fmul_rn(fix_loc[2 * b + 1], (float)width);
        const float center_y = __fmul_rn(fix_loc[2 * b], (float)height);
        const float pixel_x = __fadd_rn(
            __fmul_rn(base_grid[2 * n], scale_x), center_x);
        const float pixel_y = __fadd_rn(
            __fmul_rn(base_grid[2 * n + 1], scale_y), center_y);
        const int x = __float2int_rn(pixel_x - 0.5f);
        const int y = __float2int_rn(pixel_y - 0.5f);
        unsigned short value = 0;
        if ((unsigned int)x < (unsigned int)width &&
            (unsigned int)y < (unsigned int)height) {
            const long long offset = (long long)b * stride_b + (long long)c * stride_c
                                   + (long long)y * stride_h + (long long)x * stride_w;
            value = image[offset];
        }
        output[linear] = value;
    }
}

extern "C" __global__ void fovi_float64_nearest(
    const double* __restrict__ image,
    const double* __restrict__ base_grid,
    const double* __restrict__ fix_loc,
    const double* __restrict__ fix_size,
    double* __restrict__ output,
    long long stride_b, long long stride_c,
    long long stride_h, long long stride_w,
    int batch, int channels, int height, int width, int points)
{
    const long long total = (long long)batch * channels * points;
    for (long long linear = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         linear < total; linear += (long long)blockDim.x * gridDim.x) {
        const int n = linear % points;
        const int c = (linear / points) % channels;
        const int b = linear / ((long long)points * channels);

        const double scale_x = __dmul_rn(fix_size[2 * b + 1], 0.5);
        const double scale_y = __dmul_rn(fix_size[2 * b], 0.5);
        const double center_x = __dmul_rn(fix_loc[2 * b + 1], (double)width);
        const double center_y = __dmul_rn(fix_loc[2 * b], (double)height);
        const double pixel_x = __dadd_rn(
            __dmul_rn(base_grid[2 * n], scale_x), center_x);
        const double pixel_y = __dadd_rn(
            __dmul_rn(base_grid[2 * n + 1], scale_y), center_y);
        const int x = __double2int_rn(pixel_x - 0.5);
        const int y = __double2int_rn(pixel_y - 0.5);
        double value = 0.0;
        if ((unsigned int)x < (unsigned int)width &&
            (unsigned int)y < (unsigned int)height) {
            const long long offset = (long long)b * stride_b + (long long)c * stride_c
                                   + (long long)y * stride_h + (long long)x * stride_w;
            value = image[offset];
        }
        output[linear] = value;
    }
}

__device__ __forceinline__ float load_uint8_or_zero(
    const unsigned char* image, int b, int c, int y, int x,
    long long stride_b, long long stride_c, long long stride_h, long long stride_w,
    int height, int width)
{
    if ((unsigned int)x >= (unsigned int)width ||
        (unsigned int)y >= (unsigned int)height) return 0.0f;
    const long long offset = (long long)b * stride_b + (long long)c * stride_c
                           + (long long)y * stride_h + (long long)x * stride_w;
    return (float)image[offset];
}

extern "C" __global__ void fovi_uint8_bilinear(
    const unsigned char* __restrict__ image,
    const float* __restrict__ base_grid,
    const float* __restrict__ fix_loc,
    const float* __restrict__ fix_size,
    float* __restrict__ output,
    long long stride_b, long long stride_c,
    long long stride_h, long long stride_w,
    int batch, int channels, int height, int width, int points)
{
    const long long total = (long long)batch * channels * points;
    for (long long linear = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         linear < total; linear += (long long)blockDim.x * gridDim.x) {
        const int n = linear % points;
        const int c = (linear / points) % channels;
        const int b = linear / ((long long)points * channels);

        const float pixel_x = base_grid[2 * n] * (0.5f * fix_size[2 * b + 1])
                            + fix_loc[2 * b + 1] * width;
        const float pixel_y = base_grid[2 * n + 1] * (0.5f * fix_size[2 * b])
                            + fix_loc[2 * b] * height;
        const float source_x = pixel_x - 0.5f;
        const float source_y = pixel_y - 0.5f;
        const int x0 = __float2int_rd(source_x);
        const int y0 = __float2int_rd(source_y);
        const float wx = source_x - x0;
        const float wy = source_y - y0;

        const float v00 = load_uint8_or_zero(
            image, b, c, y0, x0, stride_b, stride_c, stride_h, stride_w, height, width);
        const float v01 = load_uint8_or_zero(
            image, b, c, y0, x0 + 1, stride_b, stride_c, stride_h, stride_w, height, width);
        const float v10 = load_uint8_or_zero(
            image, b, c, y0 + 1, x0, stride_b, stride_c, stride_h, stride_w, height, width);
        const float v11 = load_uint8_or_zero(
            image, b, c, y0 + 1, x0 + 1, stride_b, stride_c, stride_h, stride_w, height, width);
        output[linear] =
            v00 * ((1.0f - wy) * (1.0f - wx)) +
            v01 * ((1.0f - wy) * wx) +
            v10 * (wy * (1.0f - wx)) +
            v11 * (wy * wx);
    }
}
"""


_MODULES = {}
_KERNELS = {}
_STREAM_CACHE = {}


def _kernel(name, device):
    key = (device.index, name)
    kernel = _KERNELS.get(key)
    if kernel is None:
        with cp.cuda.Device(device.index):
            module = _MODULES.get(device.index)
            if module is None:
                module = cp.RawModule(
                    code=_SOURCE,
                    options=("--std=c++14",),
                    name_expressions=(
                        "fovi_uint8_nearest", "fovi_uint8_bilinear",
                        "fovi_float16_nearest", "fovi_float32_nearest",
                        "fovi_float64_nearest"),
                )
                _MODULES[device.index] = module
            kernel = module.get_function(name)
        _KERNELS[key] = kernel
    return kernel


def _current_stream(device):
    try:
        stream_ptr = torch._C._cuda_getCurrentRawStream(device.index)
    except AttributeError:  # pragma: no cover - older Torch
        stream_ptr = torch.cuda.current_stream(device).cuda_stream
    key = (device.index, stream_ptr)
    stream = _STREAM_CACHE.get(key)
    if stream is None:
        if hasattr(cp.cuda.Stream, "from_external"):
            stream = cp.cuda.Stream.from_external(torch.cuda.current_stream(device))
        else:  # pragma: no cover - CuPy < 14
            stream = cp.cuda.ExternalStream(stream_ptr, device_id=device.index)
        _STREAM_CACHE[key] = stream
    return stream


def _ptr(tensor):
    return np.uint64(tensor.data_ptr())


def sample_uint8(image, base_grid, fix_loc, fix_size, mode="nearest"):
    """Sample ``image`` and return contiguous ``[B, C, N]`` native-scale output."""
    if image.dtype != torch.uint8 or not image.is_cuda:
        raise RuntimeError("native uint8 sampling requires a CUDA torch.uint8 tensor")
    if image.ndim != 4:
        raise ValueError(f"expected NCHW image, got {tuple(image.shape)}")
    if mode not in ("nearest", "bilinear"):
        raise ValueError(f"unsupported mode {mode!r}")

    device = image.device
    base_grid = base_grid[0, 0]
    if (base_grid.device != device or base_grid.dtype != torch.float32
            or not base_grid.is_contiguous()):
        base_grid = base_grid.to(device=device, dtype=torch.float32).contiguous()
    if (fix_loc.device != device or fix_loc.dtype != torch.float32
            or not fix_loc.is_contiguous()):
        fix_loc = fix_loc.to(device=device, dtype=torch.float32).contiguous()
    if (fix_size.device != device or fix_size.dtype != torch.float32
            or not fix_size.is_contiguous()):
        fix_size = fix_size.to(device=device, dtype=torch.float32).contiguous()
    batch, channels, height, width = image.shape
    points = base_grid.shape[0]
    dtype = torch.uint8 if mode == "nearest" else torch.float32
    output = torch.empty((batch, channels, points), device=device, dtype=dtype)

    total = batch * channels * points
    threads = 256
    blocks = min((total + threads - 1) // threads, 4096)
    args = (
        _ptr(image), _ptr(base_grid), _ptr(fix_loc), _ptr(fix_size), _ptr(output),
        np.int64(image.stride(0)), np.int64(image.stride(1)),
        np.int64(image.stride(2)), np.int64(image.stride(3)),
        np.int32(batch), np.int32(channels), np.int32(height), np.int32(width),
        np.int32(points),
    )
    stream = _current_stream(device)
    kernel = _kernel(f"fovi_uint8_{mode}", device)
    if cp.cuda.runtime.getDevice() == device.index:
        with stream:
            kernel((blocks,), (threads,), args)
    else:  # pragma: no cover - multi-GPU context mismatch
        with cp.cuda.Device(device.index), stream:
            kernel((blocks,), (threads,), args)
    return output


def _sample_float_nearest(image, base_grid, fix_loc, fix_size):
    """Benchmark the fused native nearest kernel with a CUDA floating image."""
    dtype_config = {
        torch.float16: ("fovi_float16_nearest", torch.float32),
        torch.float32: ("fovi_float32_nearest", torch.float32),
        torch.float64: ("fovi_float64_nearest", torch.float64),
    }
    if image.dtype not in dtype_config or not image.is_cuda:
        raise RuntimeError(
            "native floating sampling requires a CUDA float16/float32/float64 tensor")
    if image.ndim != 4:
        raise ValueError(f"expected NCHW image, got {tuple(image.shape)}")

    device = image.device
    kernel_name, coordinate_dtype = dtype_config[image.dtype]
    base_grid = base_grid[0, 0]
    if (base_grid.device != device or base_grid.dtype != coordinate_dtype
            or not base_grid.is_contiguous()):
        base_grid = base_grid.to(
            device=device, dtype=coordinate_dtype).contiguous()
    if (fix_loc.device != device or fix_loc.dtype != coordinate_dtype
            or not fix_loc.is_contiguous()):
        fix_loc = fix_loc.to(
            device=device, dtype=coordinate_dtype).contiguous()
    if (fix_size.device != device or fix_size.dtype != coordinate_dtype
            or not fix_size.is_contiguous()):
        fix_size = fix_size.to(
            device=device, dtype=coordinate_dtype).contiguous()
    batch, channels, height, width = image.shape
    points = base_grid.shape[0]
    output = torch.empty(
        (batch, channels, points), device=device, dtype=image.dtype)

    total = batch * channels * points
    threads = 256
    blocks = min((total + threads - 1) // threads, 4096)
    args = (
        _ptr(image), _ptr(base_grid), _ptr(fix_loc), _ptr(fix_size), _ptr(output),
        np.int64(image.stride(0)), np.int64(image.stride(1)),
        np.int64(image.stride(2)), np.int64(image.stride(3)),
        np.int32(batch), np.int32(channels), np.int32(height), np.int32(width),
        np.int32(points),
    )
    stream = _current_stream(device)
    kernel = _kernel(kernel_name, device)
    if cp.cuda.runtime.getDevice() == device.index:
        with stream:
            kernel((blocks,), (threads,), args)
    else:  # pragma: no cover - multi-GPU context mismatch
        with cp.cuda.Device(device.index), stream:
            kernel((blocks,), (threads,), args)
    return output


def clear_kernel_cache():
    """Clear Python-side kernel and stream handles (primarily for tests)."""
    _MODULES.clear()
    _KERNELS.clear()
    _STREAM_CACHE.clear()
