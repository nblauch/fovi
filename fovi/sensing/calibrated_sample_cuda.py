"""Fused calibrated ray projection and image gathering on Torch's CUDA stream."""

from __future__ import annotations

import math
from functools import cache
from pathlib import Path

import torch  # Import first so CuPy finds Torch's bundled NVRTC.

# isort: split
import cupy as cp
import numpy as np

from .projection import CameraModel

_SOURCE = Path(__file__).with_name("calibrated_sample.cu").read_text()
_IMAGE_TYPES = {
    torch.uint8: 0,
    torch.float16: 1,
    torch.bfloat16: 2,
    torch.float32: 3,
    torch.float64: 4,
}


@cache
def _kernels(
    device: int, options: tuple[str, ...]
) -> tuple[cp.RawKernel, cp.RawKernel]:
    with cp.cuda.Device(device):
        module = cp.RawModule(
            code=_SOURCE, options=("--std=c++17", "--fmad=false", *options)
        )
        return module.get_function("prepare_gaze"), module.get_function(
            "calibrated_sample"
        )


@cache
def _stream(device: int, pointer: int) -> cp.cuda.ExternalStream:
    return cp.cuda.ExternalStream(pointer, device_id=device)


class CalibratedCudaSampler:
    """Specialize a fixed lens and interpolation mode; retain dynamic image/gaze storage."""

    def __init__(self, camera: CameraModel, mode: str, convention: str) -> None:
        if mode not in ("nearest", "bilinear"):
            raise ValueError(f"Unknown sampling mode {mode!r}")
        if convention not in ("camera_xyz", "pan_tilt"):
            raise ValueError(f"Unknown gaze convention {convention!r}")
        self.camera = camera
        self.mode = mode
        self.options = (
            f"-DFISHEYE={int(camera.model == 'fisheye')}",
            f"-DDISTORTED={int(any(camera.distortion))}",
            f"-DPAN_TILT={int(convention == 'pan_tilt')}",
            f"-DBILINEAR={int(mode == 'bilinear')}",
        )
        coefficients = (*camera.distortion, *((0.0,) * (8 - len(camera.distortion))))
        circle = camera.image_circle or (0.0, 0.0, -1.0)
        values = (
            *camera.intrinsics,
            *coefficients,
            math.radians(camera.max_angle_deg),
            *circle,
        )
        dimensions = tuple(np.int32(v) for v in camera.image_size)
        self.parameters = {
            torch.float32: (*map(np.float32, values), *dimensions),
            torch.float64: (*map(np.float64, values), *dimensions),
        }

    def __call__(
        self,
        image: torch.Tensor,
        rays: torch.Tensor,
        fixation: torch.Tensor,
        rotation: torch.Tensor | None,
        return_pixels: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Sample NCHW images into (B, C, N); optionally return (B, N, 2) source pixels.

        Inputs must be on the same CUDA device. Rays are contiguous float32 (N, 3).
        Fixations and rotations use float64 for double images, float32 otherwise.
        This inference kernel has no backward; the public sampler retains autograd.
        """
        coordinate_dtype = (
            torch.float64 if image.dtype == torch.float64 else torch.float32
        )
        if not image.is_cuda or image.dtype not in _IMAGE_TYPES:
            raise ValueError(
                "Calibrated CUDA sampling requires a supported CUDA image dtype"
            )
        if (
            rays.device != image.device
            or rays.dtype != torch.float32
            or not rays.is_contiguous()
        ):
            raise ValueError(
                "Calibrated CUDA rays must be contiguous float32 on the image device"
            )
        if fixation.device != image.device or fixation.dtype != coordinate_dtype:
            raise ValueError(
                "Fixation device/dtype must match image coordinate arithmetic"
            )
        if rotation is not None and (
            rotation.device != image.device or rotation.dtype != coordinate_dtype
        ):
            raise ValueError(
                "Rotation device/dtype must match image coordinate arithmetic"
            )
        batch, channels, height, width = image.shape
        if (height, width) != tuple(self.camera.image_size):
            raise ValueError("Image dimensions do not match calibration")
        points = rays.shape[0]
        output_dtype = (
            torch.float32
            if image.dtype == torch.uint8 and self.mode == "bilinear"
            else image.dtype
        )
        output = torch.empty(
            (batch, channels, points), dtype=output_dtype, device=image.device
        )
        pixels = (
            torch.empty((batch, points, 2), dtype=coordinate_dtype, device=image.device)
            if return_pixels
            else None
        )
        device = image.device.index
        options = (
            *self.options,
            f"-DIMAGE_TYPE={_IMAGE_TYPES[image.dtype]}",
            f"-DEXPLICIT_ROTATION={int(rotation is not None)}",
        )
        gaze_kernel, sample_kernel = _kernels(device, options)
        parameters = self.parameters[coordinate_dtype]
        stream = _stream(device, torch.cuda.current_stream(image.device).cuda_stream)
        with cp.cuda.Device(device), stream:
            if rotation is None:
                rotation = torch.empty(
                    (batch, 10), dtype=coordinate_dtype, device=image.device
                )
                gaze_kernel(
                    ((batch + 127) // 128,),
                    (128,),
                    (
                        np.uint64(fixation.data_ptr()),
                        np.uint64(rotation.data_ptr()),
                        np.int64(fixation.stride(0)),
                        np.int64(fixation.stride(1)),
                        np.int32(batch),
                        *parameters,
                    ),
                )
                rotation_strides = (10, 3, 1)
            else:
                rotation_strides = rotation.stride()
            sample_kernel(
                (min((batch * points + 255) // 256, 4096),),
                (256,),
                (
                    np.uint64(image.data_ptr()),
                    np.uint64(rays.data_ptr()),
                    np.uint64(rotation.data_ptr()),
                    np.uint64(output.data_ptr()),
                    np.uint64(0 if pixels is None else pixels.data_ptr()),
                    *(np.int64(v) for v in image.stride()),
                    *(np.int64(v) for v in rotation_strides),
                    np.int32(batch),
                    np.int32(channels),
                    np.int32(points),
                    *parameters,
                ),
            )
        return output, pixels
