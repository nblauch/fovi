"""Retinal pooling on sample sets and aligned Cartesian image grids."""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..sensing.coords import SamplingCoords
from .knn import KNNPoolingLayer


def resolve_pool_method(style: str, method: str) -> str:
    """Resolve automatic pooling without changing explicitly selected KNN behavior."""
    if method == "auto":
        return "grid2d" if style.endswith("_as_grid") else "knn"
    if method in ("grid2d", "knn"):
        return method
    raise ValueError(f"pool_method must be 'auto', 'grid2d', or 'knn', got {method!r}")


class GridPoolingLayer(nn.Module):
    """Mask-normalized pooling between aligned Cartesian grids.

    Inputs and outputs use canonical ``(B, C, N)`` sample ordering. Pooling
    operates on upright image grids with stride equal to the integer resolution
    ratio. ``k`` is the kernel side; None uses that ratio. Symmetric windows are
    used when kernel and stride have matching parity; otherwise SAME-style
    asymmetric padding selects the window half a sample toward larger image
    indices. Kernels smaller than the stride crop the corresponding margins.
    Gaussian sigma uses distances normalized to [0, 1], as in KNN pooling.
    """

    def __init__(
        self,
        k: int | None,
        in_coords: SamplingCoords,
        out_coords: SamplingCoords,
        mode: str = "avg",
        gauss_sigma: float = 10.0,
    ) -> None:
        super().__init__()
        styles = ("uniform_as_grid", "warped_cartesian_as_grid")
        if in_coords.style not in styles or out_coords.style != in_coords.style:
            raise ValueError(
                "grid2d pooling requires matching Cartesian _as_grid sensors; "
                "polar sensors require wrap-around handling and are unsupported"
            )
        geometry = ("fov", "cmf_a", "fov_type", "field_geometry", "radius_norm")
        if any(
            getattr(in_coords, name) != getattr(out_coords, name) for name in geometry
        ):
            raise ValueError(
                "grid2d pooling requires identical input/output geometry and CMF (cmf_a_mult=1)"
            )
        high, low = in_coords.resolution, out_coords.resolution
        if high <= low or high % low:
            raise ValueError(
                "grid2d pooling requires an integer high/low resolution ratio > 1"
            )
        self.stride = high // low
        if k is None:
            k = self.stride
        if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
            raise ValueError(
                "grid2d pooling k must be a positive integer kernel side or None"
            )
        if mode not in ("avg", "gaussian", "max"):
            raise ValueError(f"Unsupported grid2d pooling mode: {mode!r}")
        if mode == "gaussian" and (not math.isfinite(gauss_sigma) or gauss_sigma <= 0):
            raise ValueError("grid2d gauss_sigma must be finite and positive")
        self.k = k
        self.mode = mode
        self.in_coords = in_coords
        self.out_coords = out_coords
        before = (k - self.stride) // 2
        after = k - self.stride - before
        self.padding = (before, after, before, after)
        self.register_buffer(
            "input_mask",
            in_coords.as_grid(in_coords.valid_mask)[None, None],
            persistent=False,
        )
        output_mask = out_coords.as_grid(out_coords.valid_mask)[None, None]

        axis = (
            torch.arange(k, device=in_coords.cartesian.device, dtype=torch.float32)
            - (k - 1) / 2
        )
        radius = torch.sqrt(axis[:, None].square() + axis[None, :].square())
        span = radius.max() - radius.min()
        radius = (radius - radius.min()) / torch.where(span > 0, span, 1)
        if mode == "gaussian":
            weights = torch.exp(-radius.square() / (2 * gauss_sigma**2))
            weights = weights / weights.sum()
        else:
            weights = torch.full_like(radius, 1 / k**2)
        self.register_buffer("weights", weights[None, None], persistent=False)
        coverage = self._weighted_pool(self.input_mask.float())
        self.register_buffer(
            "output_mask", output_mask & (coverage > 0), persistent=False
        )
        normalizer = torch.where(coverage > 0, coverage, 1).reciprocal()
        self.register_buffer("normalizer", normalizer, persistent=False)

    @property
    def last_backend(self) -> str:
        """The pooling backend; image sampling has its own backend selection."""
        return "torch_grid2d"

    def _weighted_pool(self, grid: torch.Tensor) -> torch.Tensor:
        padded = F.pad(grid, self.padding)
        if self.mode in ("avg", "max"):
            return F.avg_pool2d(padded, self.k, self.stride)
        batch, channels = grid.shape[:2]
        pooled = F.conv2d(
            padded.reshape(batch * channels, 1, *padded.shape[-2:]),
            self.weights.to(dtype=grid.dtype),
            stride=self.stride,
        )
        return pooled.reshape(batch, channels, *pooled.shape[-2:])

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Pool ``(B, C, N_high)`` values to ``(B, C, N_low)``."""
        grid = self.in_coords.as_grid(values)
        if self.mode == "max":
            grid = torch.where(self.input_mask, grid, -torch.inf)
            result = F.max_pool2d(
                F.pad(grid, self.padding, value=-torch.inf), self.k, self.stride
            )
        else:
            grid = torch.where(self.input_mask, grid, 0)
            result = self._weighted_pool(grid) * self.normalizer.to(dtype=grid.dtype)
        result = torch.where(self.output_mask, result, 0)
        return result.flip(-2).transpose(-2, -1).flatten(-2)


def build_retinal_pooler(
    in_coords: SamplingCoords,
    out_coords: SamplingCoords,
    *,
    k: int | None,
    mode: str,
    method: str,
    sample_cortex: bool | str,
    gauss_sigma: float,
    device: str | torch.device,
) -> tuple[KNNPoolingLayer | GridPoolingLayer, int]:
    """Build a retinal pooler and return its resolved neighbor count/kernel side."""
    method = resolve_pool_method(out_coords.style, method)
    if method == "grid2d":
        pooler = GridPoolingLayer(k, in_coords, out_coords, mode, gauss_sigma)
        return pooler, pooler.k
    if k is None:
        k = int(np.round(len(in_coords) / len(out_coords)))
    return KNNPoolingLayer(
        k,
        in_coords,
        out_coords,
        mode=mode,
        device=device,
        sample_cortex=sample_cortex,
        gauss_sigma=gauss_sigma,
    ), k
