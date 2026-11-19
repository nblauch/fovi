"""Retinal pooling: 2D windows for ``_as_grid`` sensors, KNN for sample sets."""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..sensing.coords import SamplingCoords
from .knn import KNNPoolingLayer
from .modal import modal_pool

POLAR_GRID_STYLES = ("logpolar_as_grid",)


class GridPoolingLayer(nn.Module):
    """Mask-normalized pooling between aligned ``_as_grid`` sensor images.

    Inputs and outputs use canonical ``(B, C, N)`` sample ordering. Pooling
    operates on the sensor image (upright for Cartesian grids, angle x
    eccentricity for log-polar grids) with stride equal to the integer
    resolution ratio. ``k`` is the neighborhood size (squared kernel side),
    matching KNN pooling's neighbor count; None uses the squared ratio.

    Spatial axes use SAME-style padding: symmetric windows when kernel and
    stride have matching parity, otherwise the window sits half a sample toward
    larger indices; kernels smaller than the stride crop the margins. Padding
    is excluded from the normalization like masked retinal units. Polar grids
    pad the angular axis circularly (as ``PolarPadder``), centering each window
    on its output angle, which coincides with high-res angle ``i * stride``.
    Their eccentricity axis keeps the spatial convention; because log-polar
    radii include both endpoints at every resolution, windows there are
    index-aligned rather than centered on output radii.
    Gaussian sigma uses distances normalized to [0, 1], as in KNN pooling.

    ``mode="modal"`` gives each output the most common value in its window, for
    label maps (see :func:`~fovi.arch.modal.modal_pool`). Ties go to the sample
    nearest the window's center, then the first in row-major window order;
    padding and masked samples do not vote.
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
        if not in_coords.style.endswith("_as_grid") or out_coords.style != in_coords.style:
            raise ValueError("grid2d pooling requires matching _as_grid sensors")
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
            k = self.stride**2
        if (
            isinstance(k, bool)
            or not isinstance(k, int)
            or k <= 0
            or math.isqrt(k) ** 2 != k
        ):
            raise ValueError(
                "grid2d pooling k must be a positive perfect square "
                "(squared kernel side) or None"
            )
        if mode not in ("avg", "gaussian", "max", "modal"):
            raise ValueError(f"Unsupported grid2d pooling mode: {mode!r}")
        if mode == "gaussian" and (not math.isfinite(gauss_sigma) or gauss_sigma <= 0):
            raise ValueError("grid2d gauss_sigma must be finite and positive")
        self.k = k
        self.kernel_size = side = math.isqrt(k)
        self.mode = mode
        self.in_coords = in_coords
        self.out_coords = out_coords
        self.polar = in_coords.style in POLAR_GRID_STYLES
        before = (side - self.stride) // 2
        after = side - self.stride - before
        if self.polar:
            # Rows are angles: wrap around, centered on the output angle.
            # Trailing samples beyond the last window are ignored by pooling.
            angle_before = (side - 1) // 2
            self.angular_padding = (angle_before, max(side - self.stride - angle_before, 0))
            self.padding = (before, after, 0, 0)
        else:
            self.angular_padding = None
            self.padding = (before, after, before, after)
        self.register_buffer(
            "input_mask",
            in_coords.as_grid(in_coords.valid_mask)[None, None],
            persistent=False,
        )
        output_mask = out_coords.as_grid(out_coords.valid_mask)[None, None]

        axis = (
            torch.arange(side, device=in_coords.cartesian.device, dtype=torch.float32)
            - (side - 1) / 2
        )
        radius = torch.sqrt(axis[:, None].square() + axis[None, :].square())
        span = radius.max() - radius.min()
        radius = (radius - radius.min()) / torch.where(span > 0, span, 1)
        if mode == "gaussian":
            weights = torch.exp(-radius.square() / (2 * gauss_sigma**2))
            weights = weights / weights.sum()
        else:
            weights = torch.full_like(radius, 1 / k)
        self.register_buffer("weights", weights[None, None], persistent=False)
        coverage = self._weighted_pool(self.input_mask.float())
        self.register_buffer(
            "output_mask", output_mask & (coverage > 0), persistent=False
        )
        normalizer = torch.where(coverage > 0, coverage, 1).reciprocal()
        self.register_buffer("normalizer", normalizer, persistent=False)
        if mode == "modal":
            self.register_buffer("neighbor_table", self._neighbor_table(), persistent=False)

    def _neighbor_table(self) -> torch.Tensor:
        """Each output's window as a ``(k, N_low)`` table of input indices in tie-break order.

        Windows use the pooler's padding, stride and masks. Within a window,
        samples are ordered by distance from its center, then row-major;
        ``N_high`` marks padding, masked inputs and every slot of a masked output.
        """
        num_inputs = len(self.in_coords)
        device = self.input_mask.device
        # Pool sample indices (float64 is exact for any realistic count); -1 marks padding and masked samples.
        index = torch.arange(num_inputs, device=device, dtype=torch.float64)
        grid = self.in_coords.as_grid(index[None, None])
        grid = self._pad(torch.where(self.input_mask, grid, -1.0), -1.0)
        side, stride = self.kernel_size, self.stride
        windows = grid[0, 0].unfold(0, side, stride).unfold(1, side, stride)  # (H_lo, W_lo, side, side)
        windows = windows.flatten(-2).permute(2, 0, 1)  # (k, H_lo, W_lo)
        axis = torch.arange(side, device=device, dtype=torch.float64) - (side - 1) / 2
        radius = (axis[:, None].square() + axis[None, :].square()).flatten()
        windows = windows[torch.sort(radius, stable=True).indices]
        windows = torch.where(self.output_mask[0], windows, -1.0)
        if not self.polar:
            # Invert the upright Cartesian layout produced by ``as_grid``, as forward does.
            windows = windows.flip(-2).transpose(-2, -1)
        indices = windows.flatten(-2).long()
        return torch.where(indices < 0, num_inputs, indices)

    @property
    def last_backend(self) -> str:
        """The pooling backend; image sampling has its own backend selection."""
        return "torch_grid2d"

    def _pad(self, grid: torch.Tensor, value: float = 0.0) -> torch.Tensor:
        if self.angular_padding is not None:
            grid = F.pad(grid, (0, 0, *self.angular_padding), mode="circular")
        return F.pad(grid, self.padding, value=value)

    def _weighted_pool(self, grid: torch.Tensor) -> torch.Tensor:
        padded = self._pad(grid)
        if self.mode in ("avg", "max", "modal"):
            return F.avg_pool2d(padded, self.kernel_size, self.stride)
        batch, channels = grid.shape[:2]
        pooled = F.conv2d(
            padded.reshape(batch * channels, 1, *padded.shape[-2:]),
            self.weights.to(dtype=grid.dtype),
            stride=self.stride,
        )
        return pooled.reshape(batch, channels, *pooled.shape[-2:])

    def forward(self, values: torch.Tensor, visible: torch.Tensor | None = None) -> torch.Tensor:
        """Pool ``(B, C, N_high)`` values to ``(B, C, N_low)``.

        Args:
            values: Values to pool. Modal pooling takes any dtype and keeps it.
            visible: ``mode="modal"`` only: optional ``(B, N_high)`` mask of the
                inputs each image sees; the others do not vote.
        """
        if self.mode == "modal":
            return modal_pool(values, self.neighbor_table, visible)
        if visible is not None:
            raise ValueError("visible is supported only by modal pooling")
        grid = self.in_coords.as_grid(values)
        if self.mode == "max":
            grid = torch.where(self.input_mask, grid, -torch.inf)
            result = F.max_pool2d(
                self._pad(grid, -torch.inf), self.kernel_size, self.stride
            )
        else:
            grid = torch.where(self.input_mask, grid, 0)
            result = self._weighted_pool(grid) * self.normalizer.to(dtype=grid.dtype)
        result = torch.where(self.output_mask, result, 0)
        if not self.polar:
            # Invert the upright Cartesian layout produced by ``as_grid``.
            result = result.flip(-2).transpose(-2, -1)
        return result.flatten(-2)


def build_retinal_pooler(
    in_coords: SamplingCoords,
    out_coords: SamplingCoords,
    *,
    k: int | None,
    mode: str,
    sample_cortex: bool | str,
    gauss_sigma: float,
    device: str | torch.device,
) -> tuple[KNNPoolingLayer | GridPoolingLayer, int]:
    """Build a retinal pooler and return its resolved neighbor count.

    ``_as_grid`` sensors always use 2D grid pooling; sample sets use KNN.
    """
    if out_coords.style.endswith("_as_grid"):
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
