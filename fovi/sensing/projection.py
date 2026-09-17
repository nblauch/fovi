"""Calibrated central cameras and spherical gaze, without renderer dependencies.

Pixels use OpenCV's integer pixel-center convention. Directions use camera
coordinates X right, Y down, Z forward. External adapters convert frames once.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TypedDict

import torch
from torch import Tensor


class CameraCalibration(TypedDict, total=False):
    """Serializable CameraModel arguments; model, image_size, and intrinsics are required."""

    model: str
    image_size: tuple[int, int]
    intrinsics: tuple[float, float, float, float]
    distortion: tuple[float, ...]
    image_circle: tuple[float, float, float] | None
    max_angle_deg: float


@dataclass(frozen=True)
class CameraModel:
    """A calibrated pinhole or equidistant-polynomial fisheye camera.

    Args:
        model: ``pinhole`` or ``fisheye`` (OpenCV fisheye convention).
        image_size: Source image (height, width).
        intrinsics: (fx, fy, cx, cy), in integer-center pixel coordinates.
        distortion: Pinhole (k1, k2, p1, p2[, k3[, k4, k5, k6]]) or
            fisheye (k1, k2, k3, k4). Empty means the ideal model.
        image_circle: Optional usable disc (cx, cy, radius), in source pixels.
        max_angle_deg: Calibrated angular domain about the optical axis.
    """

    model: str
    image_size: tuple[int, int]
    intrinsics: tuple[float, float, float, float]
    distortion: tuple[float, ...] = ()
    image_circle: tuple[float, float, float] | None = None
    max_angle_deg: float = 90.0

    @classmethod
    def from_config(cls, camera: CameraModel | CameraCalibration) -> CameraModel:
        """Normalize serialized calibration once at an API boundary."""
        if isinstance(camera, cls):
            return camera
        if not isinstance(camera, Mapping):
            raise TypeError("camera_model must be a CameraModel or calibration mapping")
        return cls(**camera)

    def __post_init__(self) -> None:
        # Frozen calibration must also contain immutable, compiler-friendly values.
        # YAML/OmegaConf and direct dataclass construction share this boundary.
        for name in ("image_size", "intrinsics", "distortion", "image_circle"):
            values = getattr(self, name)
            if values is not None:
                convert = int if name == "image_size" else float
                if name == "image_size" and any(int(v) != v for v in values):
                    raise ValueError("image_size must contain integer height and width")
                object.__setattr__(self, name, tuple(convert(v) for v in values))
        object.__setattr__(self, "max_angle_deg", float(self.max_angle_deg))
        if self.model not in ("pinhole", "fisheye"):
            raise ValueError(f"Unsupported camera model {self.model!r}")
        if len(self.image_size) != 2 or any(v <= 0 for v in self.image_size):
            raise ValueError("image_size must contain positive height and width")
        if len(self.intrinsics) != 4 or not all(
            math.isfinite(v) for v in self.intrinsics
        ):
            raise ValueError("intrinsics must contain finite fx, fy, cx, cy")
        if min(self.intrinsics[:2]) <= 0:
            raise ValueError("Focal lengths must be positive")
        lengths = (0, 4, 5, 8) if self.model == "pinhole" else (0, 4)
        if len(self.distortion) not in lengths or not all(
            math.isfinite(v) for v in self.distortion
        ):
            raise ValueError(f"Invalid {self.model} distortion coefficients")
        if not 0 < self.max_angle_deg <= 90:
            raise ValueError("max_angle_deg must be in (0, 90]")
        if self.image_circle is not None and (
            len(self.image_circle) != 3
            or not all(math.isfinite(v) for v in self.image_circle)
            or self.image_circle[2] <= 0
        ):
            raise ValueError(
                "image_circle must contain finite center and positive radius"
            )

    def _pinhole_distort(self, xy: Tensor) -> Tensor:
        coefficients = (*self.distortion, *((0.0,) * (8 - len(self.distortion))))
        k1, k2, p1, p2, k3, k4, k5, k6 = coefficients
        x, y = xy.unbind(-1)
        r2 = x * x + y * y
        radial = (1 + r2 * (k1 + r2 * (k2 + r2 * k3))) / (
            1 + r2 * (k4 + r2 * (k5 + r2 * k6))
        )
        return torch.stack(
            (
                x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x),
                y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y,
            ),
            -1,
        )

    def resized(self, image_size: tuple[int, int]) -> CameraModel:
        """Calibrate a full-frame resize; cropping and image rotation need new intrinsics."""
        sy, sx = image_size[0] / self.image_size[0], image_size[1] / self.image_size[1]
        fx, fy, cx, cy = self.intrinsics
        circle = self.image_circle
        if circle is not None:
            if not math.isclose(sx, sy):
                raise ValueError("An image circle requires an isotropic resize")
            circle = (
                (circle[0] + 0.5) * sx - 0.5,
                (circle[1] + 0.5) * sy - 0.5,
                circle[2] * sx,
            )
        return replace(
            self,
            image_size=image_size,
            intrinsics=(fx * sx, fy * sy, (cx + 0.5) * sx - 0.5, (cy + 0.5) * sy - 0.5),
            image_circle=circle,
        )

    def _pinhole_undistort(self, target: Tensor) -> Tensor:
        """Invert radial/tangential distortion with a batched analytic Newton step."""
        if not self.distortion:
            return target
        k1, k2, p1, p2, k3, k4, k5, k6 = (
            *self.distortion,
            *((0.0,) * (8 - len(self.distortion))),
        )
        xy = target.clone()
        eps = torch.finfo(target.dtype).eps
        for _ in range(12):
            x, y = xy.unbind(-1)
            r2 = x * x + y * y
            numerator = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
            denominator = 1 + r2 * (k4 + r2 * (k5 + r2 * k6))
            radial = numerator / denominator
            derivative = (
                (k1 + r2 * (2 * k2 + r2 * 3 * k3)) * denominator
                - numerator * (k4 + r2 * (2 * k5 + r2 * 3 * k6))
            ) / denominator.square()
            jxx = radial + 2 * x * x * derivative + 2 * p1 * y + 6 * p2 * x
            jyy = radial + 2 * y * y * derivative + 6 * p1 * y + 2 * p2 * x
            jxy = 2 * x * y * derivative + 2 * p1 * x + 2 * p2 * y
            determinant = jxx * jyy - jxy.square()
            determinant = torch.where(
                determinant.abs() > eps, determinant, torch.full_like(determinant, eps)
            )
            residual = self._pinhole_distort(xy) - target
            dx, dy = residual.unbind(-1)
            step = (
                torch.stack((jyy * dx - jxy * dy, jxx * dy - jxy * dx), -1)
                / determinant[..., None]
            )
            xy = xy - step
        return xy

    def _fisheye_radius(self, theta: Tensor) -> tuple[Tensor, Tensor]:
        k1, k2, k3, k4 = self.distortion or (0.0, 0.0, 0.0, 0.0)
        t2 = theta * theta
        radius = theta * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))
        derivative = 1 + t2 * (3 * k1 + t2 * (5 * k2 + t2 * (7 * k3 + t2 * 9 * k4)))
        return radius, derivative

    def pixel_validity(self, pixels: Tensor) -> Tensor:
        """Return (...,) validity for (..., 2) source pixels."""
        h, w = self.image_size
        x, y = pixels.unbind(-1)
        valid = (
            torch.isfinite(pixels).all(-1)
            & (x >= -0.5)
            & (x < w - 0.5)
            & (y >= -0.5)
            & (y < h - 0.5)
        )
        if self.image_circle is not None:
            cx, cy, radius = self.image_circle
            valid = valid & ((x - cx) ** 2 + (y - cy) ** 2 <= radius * radius)
        return valid

    def project(self, directions: Tensor) -> tuple[Tensor, Tensor]:
        """Project (..., 3) directions to (..., 2) pixels and (...,) validity."""
        z = directions[..., 2]
        lateral = torch.linalg.vector_norm(directions[..., :2], dim=-1)
        theta = torch.atan2(lateral, z)
        eps = torch.finfo(directions.dtype).eps
        if self.model == "pinhole":
            xy = self._pinhole_distort(
                directions[..., :2] / z.clamp_min(eps)[..., None]
            )
        else:
            radius, _ = self._fisheye_radius(theta)
            scale = torch.where(
                lateral > eps, radius / lateral.clamp_min(eps), 1 / z.clamp_min(eps)
            )
            xy = directions[..., :2] * scale[..., None]
        fx, fy, cx, cy = self.intrinsics
        pixels = torch.stack((xy[..., 0] * fx + cx, xy[..., 1] * fy + cy), -1)
        valid = (
            self.pixel_validity(pixels)
            & (theta <= math.radians(self.max_angle_deg))
            & (z > 0)
        )
        return pixels, valid

    def unproject(self, pixels: Tensor) -> tuple[Tensor, Tensor]:
        """Invert calibrated pixels into unit directions; flag failed inversions."""
        fx, fy, cx, cy = self.intrinsics
        target = torch.stack(
            ((pixels[..., 0] - cx) / fx, (pixels[..., 1] - cy) / fy), -1
        )
        eps = torch.finfo(pixels.dtype).eps
        if self.model == "pinhole":
            xy = self._pinhole_undistort(target)
            directions = torch.nn.functional.normalize(
                torch.cat((xy, torch.ones_like(xy[..., :1])), -1), dim=-1
            )
        else:
            radius = torch.linalg.vector_norm(target, dim=-1)
            theta = radius.clone()
            for _ in range(12):
                mapped, derivative = self._fisheye_radius(theta)
                theta = theta - (mapped - radius) / derivative.clamp_min(eps)
            scale = torch.where(
                radius > eps,
                torch.sin(theta) / radius.clamp_min(eps),
                torch.ones_like(radius),
            )
            directions = torch.cat(
                (target * scale[..., None], torch.cos(theta)[..., None]), -1
            )
        reconstructed, angular_valid = self.project(directions)
        valid = (
            self.pixel_validity(pixels)
            & angular_valid
            & ((reconstructed - pixels).abs().amax(-1) < 1e-3)
        )
        return directions, valid

    def field_of_view(self, reference_side: str, fraction: float = 1.0) -> float:
        """Return the angular span of a centered short/long-side retinal window.

        Window endpoints are image boundaries in integer-center coordinates.
        The selected axis follows pixel dimensions, not an assumed aspect ratio.
        """
        if reference_side not in ("short", "long") or not 0 < fraction <= 1:
            raise ValueError(
                "Expected reference_side short/long and fraction in (0, 1]"
            )
        h, w = self.image_size
        horizontal = (w >= h) if reference_side == "long" else (w < h)
        center = torch.tensor([(w - 1) / 2, (h - 1) / 2], dtype=torch.float64)
        extent = (w if horizontal else h) * fraction / 2
        offset = torch.tensor(
            [extent if horizontal else 0, 0 if horizontal else extent],
            dtype=torch.float64,
        )
        rays, _ = self.unproject(torch.stack((center - offset, center + offset)))
        # Boundary endpoints are intentionally allowed; validate inversion separately.
        recovered, _ = self.project(rays)
        angular_valid = rays[:, 2] >= math.cos(math.radians(self.max_angle_deg)) - 1e-12
        if (
            not angular_valid.all()
            or not torch.isfinite(rays).all()
            or not torch.allclose(
                recovered,
                torch.stack((center - offset, center + offset)),
                atol=1e-3,
                rtol=0,
            )
        ):
            raise ValueError(
                "Retinal window extends outside the invertible calibration domain"
            )
        return math.degrees(float(torch.acos((rays[0] * rays[1]).sum().clamp(-1, 1))))


def angular_directions(cartesian: Tensor, fov_deg: float) -> Tensor:
    """Map normalized retinal (N, 2), X right/Y up, to camera unit rays."""
    xy = torch.stack((cartesian[..., 0], -cartesian[..., 1]), -1)
    radius = torch.linalg.vector_norm(xy, dim=-1)
    half = math.radians(fov_deg / 2)
    scale = half * torch.sinc(radius * half / math.pi)
    return torch.cat((xy * scale[..., None], torch.cos(radius * half)[..., None]), -1)


def gaze_rotation(directions: Tensor, convention: str = "camera_xyz") -> Tensor:
    """Aim +Z at (B, 3) directions using the declared zero-torsion convention.

    ``camera_xyz`` matches Rx(roll) Ry(pitch) in a Y-down/Z-forward camera.
    ``pan_tilt`` matches pan-then-tilt, Ry(pan) Rx(tilt).
    """
    x, y, z = directions.unbind(-1)
    if convention == "camera_xyz":
        pitch = torch.atan2(x, torch.sqrt(y * y + z * z))
        roll = torch.atan2(-y, z)
    elif convention == "pan_tilt":
        pitch = torch.atan2(x, z)
        roll = torch.atan2(-y, torch.sqrt(x * x + z * z))
    else:
        raise ValueError(f"Unknown gaze convention {convention!r}")
    cp, sp, cr, sr = (
        torch.cos(pitch),
        torch.sin(pitch),
        torch.cos(roll),
        torch.sin(roll),
    )
    zero, one = torch.zeros_like(cp), torch.ones_like(cp)
    rx = torch.stack((one, zero, zero, zero, cr, -sr, zero, sr, cr), -1).reshape(
        *x.shape, 3, 3
    )
    ry = torch.stack((cp, zero, sp, zero, one, zero, -sp, zero, cp), -1).reshape(
        *x.shape, 3, 3
    )
    return rx @ ry if convention == "camera_xyz" else ry @ rx
