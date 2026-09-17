"""Construction-time foveal density fitting against a calibrated source camera."""

from __future__ import annotations

import math

import torch
from scipy.optimize import minimize_scalar

from .coords import find_desired_res, isotropic_foveal_ring
from .projection import CameraModel, angular_directions, gaze_rotation


def calibrated_cmf_a(
    camera: CameraModel,
    fov: float,
    resolution: int,
    *,
    auto_match_cart_resources: bool,
    style: str,
    fov_type: str,
    gaze_convention: str,
) -> float:
    """Fit minimum center-to-first-ring spacing to one source pixel at central gaze.

    The minimum is over the actual first-ring directions, so anisotropic focal
    lengths and lens distortion participate in the criterion. Resource matching
    selects the effective ring count for each candidate. Integer ring counts can
    make the optimum discontinuous; require spacing within 5% of one pixel.
    The returned degree-valued CMF parameter stays fixed across all later gazes.

    Args:
        camera: Calibration of the images that will be sampled.
        fov: Full angular diameter in degrees.
        resolution: Ring count, or Cartesian side length when resource matching.
        auto_match_cart_resources: Match the squared resolution as a node budget.
        style: Sampling layout; automatic calibration supports ``isotropic``.
        fov_type: ``circular`` or ``square`` retinal boundary.
        gaze_convention: Rotation convention at nominal central gaze.

    Returns:
        Positive CMF parameter in degrees.

    Raises:
        ValueError: The layout or calibration cannot meet the density criterion.
    """
    if style != "isotropic":
        raise ValueError(
            "Calibrated cmf_a='auto' requires isotropic sampling; provide an explicit cmf_a for other layouts"
        )
    if not math.isfinite(fov) or not 0 < fov <= 180 or resolution < 2:
        raise ValueError(
            "Calibrated cmf_a='auto' requires FoV in (0, 180] and resolution >= 2"
        )
    h, w = camera.image_size
    center = torch.tensor([(w - 1) / 2, (h - 1) / 2], dtype=torch.float64)
    target, valid = camera.unproject(center[None])
    if not bool(valid.all()):
        raise ValueError(
            "Calibrated cmf_a='auto' requires valid camera coverage at central gaze"
        )
    rotation = gaze_rotation(target, gaze_convention)[0]

    def spacing(log_fraction: float) -> float:
        a = fov * math.exp(log_fraction)
        rings = resolution
        if auto_match_cart_resources:
            rings, _ = find_desired_res(
                fov,
                a,
                resolution**2,
                style="isotropic",
                bounds=(2, 1000),
                force_less_than=True,
                quiet=True,
                fov_type=fov_type,
                field_geometry="spherical",
            )
        ring = isotropic_foveal_ring(fov, a, rings, fov_type, "spherical").double()
        rays = angular_directions(ring, fov) @ rotation.T
        pixels, ring_valid = camera.project(rays)
        if not bool(ring_valid.all()):
            return math.inf
        return float((pixels - center).norm(dim=-1).min())

    result = minimize_scalar(
        lambda log_a: abs(spacing(log_a) - 1),
        bounds=(-8, 4),
        method="bounded",
        options={"xatol": 1e-6, "maxiter": 80},
    )
    distance = spacing(float(result.x))
    if not result.success or not math.isfinite(distance) or abs(distance - 1) > 0.05:
        raise ValueError(
            "Calibrated cmf_a='auto' could not attain one-pixel central spacing "
            f"with this FoV/resolution (best spacing {distance:g} pixels). "
            "Choose an explicit cmf_a or adjust the sampling resolution."
        )
    return fov * math.exp(float(result.x))
