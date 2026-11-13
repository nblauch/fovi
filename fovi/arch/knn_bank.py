"""Per-image sparse geometry selection followed by a shared patch projection."""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn
from torch.autograd.function import FunctionCtx

if TYPE_CHECKING:
    from cupy import RawKernel


def incidence_tables(
    indices: np.ndarray, bins: np.ndarray, samples: int, reference_bins: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build forward and transpose CSR tables, excluding the zero-padding token."""
    patches = indices.shape[0]
    rows = np.arange(patches)[:, None] * reference_bins + bins
    valid = indices < samples
    src, dst = indices[valid], rows[valid]
    order = np.argsort(dst, kind="stable")
    ptr = np.r_[0, np.cumsum(np.bincount(dst, minlength=patches * reference_bins))]
    reverse_order = np.argsort(src, kind="stable")
    reverse_ptr = np.r_[0, np.cumsum(np.bincount(src, minlength=samples))]
    return tuple(
        array.astype(np.int32)
        for array in (ptr, src[order], reverse_ptr, dst[reverse_order])
    )


@cache
def cuda_kernels(device_index: int) -> dict[tuple[str, torch.dtype], RawKernel]:
    """Compile native CUDA reductions once per device; PyTorch owns the stream."""
    import cupy as cp

    from .knn_cuda import cuda_include_options

    dtypes = (
        (torch.float32, "float"),
        (torch.float16, "__half"),
        (torch.bfloat16, "__nv_bfloat16"),
    )
    names = [
        f"{fn}<{ctype}>"
        for fn in ("project", "project_backward")
        for _, ctype in dtypes
    ]
    with cp.cuda.Device(device_index):
        module = cp.RawModule(
            code=Path(__file__).with_suffix(".cu").read_text(),
            options=("--std=c++17",) + cuda_include_options(),
            name_expressions=names,
        )
        return {
            (fn, dtype): module.get_function(f"{fn}<{ctype}>")
            for fn in ("project", "project_backward")
            for dtype, ctype in dtypes
        }


class IncidenceBank(nn.Module):
    """Geometry-only tables; reconstructed from config, never checkpoint weights."""

    def __init__(
        self,
        tables: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
        samples: int,
        patches: int,
        reference_bins: int,
    ) -> None:
        super().__init__()
        self.samples, self.patches, self.reference_bins = (
            samples,
            patches,
            reference_bins,
        )
        self.levels = len(tables)
        self.width = max(len(level[1]) for level in tables)
        for column, name in enumerate(
            ("ptr", "indices", "reverse_ptr", "reverse_indices")
        ):
            arrays = [level[column] for level in tables]
            if column in (1, 3):
                arrays = [
                    np.pad(array, (0, self.width - len(array))) for array in arrays
                ]
            self.register_buffer(
                name, torch.from_numpy(np.stack(arrays)), persistent=False
            )

    def validate(self, x: torch.Tensor, levels: torch.Tensor) -> None:
        """Check dimensions and level range before passing raw pointers to CUDA."""
        if x.ndim != 3 or x.shape[-1] != self.samples or x.device != self.ptr.device:
            raise ValueError(
                "Expected [batch, channels, samples] on the geometry device"
            )
        if x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError("Projection supports float32, float16, or bfloat16")
        if (
            levels.dtype != torch.int64
            or levels.shape != (x.shape[0],)
            or levels.device != x.device
        ):
            raise ValueError("cmf_indices must be int64 [batch] on the input device")
        torch._assert_async(
            ((levels >= 0) & (levels < self.levels)).all(), "CMF index out of range"
        )

    def launch(
        self,
        x: torch.Tensor,
        levels: torch.Tensor,
        out: torch.Tensor,
        backward: bool = False,
    ) -> None:
        """Execute deterministic bin reductions without sorting or splitting the batch."""
        import cupy as cp

        ptr, indices = (
            (self.reverse_ptr, self.reverse_indices)
            if backward
            else (self.ptr, self.indices)
        )
        channels = out.shape[1] if backward else x.shape[1]
        name = "project_backward" if backward else "project"
        kernel = cuda_kernels(x.device.index)[name, x.dtype]
        args = tuple(np.uint64(t.data_ptr()) for t in (x, levels, ptr, indices, out))
        args += tuple(
            np.int32(n)
            for n in (
                levels.shape[0],
                channels,
                self.samples,
                self.patches,
                self.reference_bins,
                self.width,
            )
        )
        with (
            cp.cuda.Device(x.device.index),
            cp.cuda.ExternalStream(torch.cuda.current_stream(x.device).cuda_stream),
        ):
            kernel(((out.numel() + 255) // 256,), (256,), args)

    def reference(self, x: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
        """Slow differentiable CPU oracle for inspection and correctness tests."""
        outputs = []
        for image, level in zip(x, levels.tolist()):
            ptr = self.ptr[level].long()
            count = int(ptr[-1])
            rows = torch.repeat_interleave(
                torch.arange(self.patches * self.reference_bins, device=x.device),
                ptr.diff(),
            )
            values = image.float()[:, self.indices[level, :count].long()]
            bins = image.new_zeros(
                (image.shape[0], self.patches * self.reference_bins),
                dtype=torch.float32,
            )
            bins = bins.index_add(1, rows, values).to(x.dtype)
            outputs.append(
                bins.reshape(image.shape[0], self.patches, self.reference_bins)
                .transpose(0, 1)
                .flatten(1)
            )
        return torch.stack(outputs)

    def forward(self, x: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
        self.validate(x, levels)
        if x.is_cuda:
            return Project.apply(x.contiguous(), levels.contiguous(), self)
        return self.reference(x, levels)


class Project(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx, x: torch.Tensor, levels: torch.Tensor, bank: IncidenceBank
    ) -> torch.Tensor:
        ctx.bank, ctx.channels = bank, x.shape[1]
        ctx.save_for_backward(levels)
        out = x.new_empty(x.shape[0], bank.patches, x.shape[1] * bank.reference_bins)
        bank.launch(x, levels, out)
        return out

    @staticmethod
    def backward(
        ctx: FunctionCtx, grad: torch.Tensor
    ) -> tuple[torch.Tensor, None, None]:
        (levels,) = ctx.saved_tensors
        dx = grad.new_empty(levels.shape[0], ctx.channels, ctx.bank.samples)
        ctx.bank.launch(grad.contiguous(), levels, dx, backward=True)
        return dx, None, None
