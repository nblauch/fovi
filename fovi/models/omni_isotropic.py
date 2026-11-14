"""Fixed-count isotropic patch embeddings with per-image CMF selection."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from omegaconf import DictConfig
from torch import nn
from torch.nn import functional as F

from ..arch.knn import get_in_out_coords
from ..arch.knn_bank import IncidenceBank, incidence_tables
from ..sensing.coords import SamplingCoords, is_cmf_sequence, validate_cmf_values
from ..sensing.geodesic import geodesic_cache_directory, patch_geodesic_distances
from ..sensing.manifold import vis_cartesian_to_cortical_cartesian_coords
from .knnvit import KNNPartitioningPatchEmbedding

POSITION_SPACES = ("cartesian", "cortical_radial_log", "cortical_manifold_3d")


def is_omni_isotropic(cfg: DictConfig) -> bool:
    """Whether the configured backbone needs a per-image isotropic geometry index."""
    return (
        is_cmf_sequence(cfg.saccades.cmf_a)
        and cfg.saccades.mode == "isotropic_fixn_calibrated"
    )


def patch_positions(patches: SamplingCoords, space: str) -> torch.Tensor:
    """Build fixed patch positions in row/column order, plus height for 3D.

    Both cortical representations share unit cortical radial length at the
    field boundary. The 3D surface uses a single scalar for every axis.
    """
    if space == "cartesian":
        return patches.cartesian_rowcol
    radius = patches.fov / 2
    radial_extent = math.log1p(radius / patches.cmf_a)
    if space == "cortical_radial_log":
        xy = patches.cartesian.double()
        norm = xy.norm(dim=-1, keepdim=True)
        log_radius = torch.log1p(norm * radius / patches.cmf_a) / radial_extent
        direction = xy / torch.where(norm > 0, norm, torch.ones_like(norm))
        warped = direction * log_radius
        return torch.stack((-warped[:, 1], warped[:, 0]), dim=-1).float()
    if space == "cortical_manifold_3d":
        surface = (
            vis_cartesian_to_cortical_cartesian_coords(
                patches.cartesian.double().cpu().numpy(),
                patches.cmf_a,
                patches.fov,
                k=1,
                field_geometry=patches.field_geometry,
                as_tensor=True,
            )
            / radial_extent
        )
        return torch.stack(
            (-surface[:, 1], surface[:, 0], surface[:, 2]), dim=-1
        ).float()
    raise ValueError(f"Unknown isotropic position space: {space!r}")


class OmniIsotropicPatchEmbedding(nn.Module):
    """Share one learned kernel across independently calibrated geometry levels."""

    def __init__(
        self,
        *,
        fov: float,
        cmf_a: Sequence[float],
        resolution: int,
        patch_size: int,
        reference_side: int,
        in_channels: int,
        embed_dim: int,
        sample_cortex: bool | str,
        device: str | torch.device,
        position_coordinate_space: str = "cartesian",
        patch_max_coord_val: float | str = 1.0,
        field_geometry: str = "planar",
    ) -> None:
        super().__init__()
        if field_geometry not in ("planar", "spherical"):
            raise ValueError("Omni isotropic requires planar or spherical geometry")
        if not isinstance(reference_side, int) or reference_side < 2:
            raise ValueError("reference_side must be an integer of at least two")
        if position_coordinate_space not in POSITION_SPACES:
            raise ValueError(f"Expected isotropic position space in {POSITION_SPACES}")
        if patch_max_coord_val != "auto" and (
            not isinstance(patch_max_coord_val, (float, int))
            or not 0 < patch_max_coord_val <= 1
        ):
            raise ValueError("patch_max_coord_val must be 'auto' or a radius in (0, 1]")
        self.patch_max_coord_val = patch_max_coord_val
        self.cmf_a = validate_cmf_values(cmf_a)
        if resolution % patch_size or resolution // patch_size < 2:
            raise ValueError(
                "Resolution must be divisible by patch_size with at least four patches"
            )
        self.in_channels, self.out_channels = in_channels, embed_dim
        self.reference_side = reference_side
        self.num_patches = (resolution // patch_size) ** 2
        self.weight = nn.Parameter(
            torch.empty(embed_dim, in_channels * reference_side**2, device=device)
        )
        self.bias = nn.Parameter(torch.zeros(embed_dim, device=device))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        tables, positions = [], []
        for value in self.cmf_a:
            coords = SamplingCoords(
                fov,
                value,
                resolution,
                style="isotropic_fixn_calibrated",
                field_geometry=field_geometry,
            )
            _, patches, _ = get_in_out_coords(
                resolution,
                fov,
                value,
                patch_size,
                style="isotropic_fixn_calibrated",
                in_coords=coords,
                in_cart_res=resolution,
                device="cpu",
                max_out_coord_val=patch_max_coord_val,
                field_geometry=field_geometry,
            )
            distances = None
            if sample_cortex == "geodesic":
                distances = patch_geodesic_distances(
                    coords, patches, geodesic_cache_directory()
                )
            # Reuse the scalar layer's covering-kNN and reference-bin definitions.
            layer = KNNPartitioningPatchEmbedding(
                in_channels=1,
                embed_dim=1,
                in_res=resolution,
                fov=fov,
                cmf_a=value,
                in_coords=coords,
                out_coords=patches,
                cart_patch_size=patch_size,
                style="isotropic_fixn_calibrated",
                device="cpu",
                sample_cortex=sample_cortex,
                ref_frame_side_length=reference_side,
                kernel_backend="baseline",
                bias=True,
                precomputed_distances=distances,
                field_geometry=field_geometry,
            )
            tables.append(
                incidence_tables(
                    layer.knn_indices_pad_token.T.cpu().numpy(),
                    layer.local_rf.argmax(-1).cpu().numpy(),
                    len(coords),
                    reference_side**2,
                )
            )
            positions.append(patch_positions(patches, position_coordinate_space))
        self.geometry = IncidenceBank(
            tables, resolution**2, self.num_patches, reference_side**2
        ).to(device)
        self.register_buffer(
            "positions", torch.stack(positions).to(device), persistent=False
        )

    @torch.no_grad()
    def load_conv2d_weights(self, conv: nn.Conv2d) -> None:
        """Orient and bilinearly resample the pretrained reference weight field.

        Matches the scalar KNN loader. Preserve pointwise amplitude: each real
        sample still contributes once per neighborhood, regardless of bin count.
        Area or L2 normalization would artificially attenuate the larger kernel.
        """
        expected = (self.out_channels, self.in_channels)
        if tuple(conv.weight.shape[:2]) != expected or conv.bias is None:
            raise ValueError(f"Expected biased Conv2d with channel shape {expected}")
        weight = conv.weight.transpose(-1, -2).flip(-1)
        target = (self.reference_side, self.reference_side)
        if weight.shape[-2:] != target:
            weight = F.interpolate(
                weight, size=target, mode="bilinear", align_corners=True
            )
        self.weight.copy_(weight.reshape_as(self.weight))
        self.bias.copy_(conv.bias)

    def forward(self, x: torch.Tensor, cmf_indices: torch.Tensor) -> torch.Tensor:
        """Return [batch, channels, patches], retaining the native DINO contract."""
        projected = self.geometry(x, cmf_indices)
        return F.linear(projected, self.weight, self.bias).transpose(1, 2)
