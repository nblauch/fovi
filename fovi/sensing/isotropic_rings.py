"""Deterministic, exact-count rings for inverse-linear magnification."""

import heapq
import math

import numpy as np


def allocate_ring_counts(target: np.ndarray, count: int) -> np.ndarray:
    """Minimize a convex spacing surrogate with at least three points per ring."""
    lo, hi = 0.0, count / target.min()
    for _ in range(48):
        scale = (lo + hi) / 2
        if np.maximum(3, target * scale).sum() > count:
            hi = scale
        else:
            lo = scale
    populations = np.floor(np.maximum(3, target * lo)).astype(np.int64)
    heap = [
        ((2 * (populations[i] - target[i]) + 1) / target[i], i)
        for i in range(len(target))
    ]
    heapq.heapify(heap)
    for _ in range(count - int(populations.sum())):
        _, i = heapq.heappop(heap)
        populations[i] += 1
        heapq.heappush(heap, ((2 * (populations[i] - target[i]) + 1) / target[i], i))
    return populations


def calibrated_isotropic_rings(
    cmf_a: float, count: int, radius: float, field_geometry: str = "planar"
) -> tuple[np.ndarray, np.ndarray]:
    """Choose log-spaced radii and integer populations with exactly count samples.

    The center contributes one point. Nearby ring counts are compared by the
    population-weighted RMS log ratio of tangential to radial intrinsic spacing.
    Calibration changes neither the CMF nor the outer visual radius.
    """
    if not all(math.isfinite(v) and v > 0 for v in (cmf_a, radius)):
        raise ValueError("cmf_a and radius must be finite and positive")
    if isinstance(count, bool) or not isinstance(count, int) or count < 4:
        raise ValueError("Ring calibration requires an integer count of at least four")
    if field_geometry not in ("planar", "spherical"):
        raise ValueError("Ring calibration requires planar or spherical geometry")
    if field_geometry == "spherical" and radius > 90:
        raise ValueError("Spherical field diameter must not exceed 180 degrees")
    span = np.log1p(radius / cmf_a)

    def target_counts(s: np.ndarray) -> np.ndarray:
        target = 2 * np.pi * (-np.expm1(-s[1:])) / (span / (len(s) - 1))
        if field_geometry == "spherical":
            # Angular rings have circumference 2*pi*sin(r), with r in radians.
            # The radial cortical arc remains log1p(r/a); only the circumference changes.
            radii = cmf_a * np.expm1(s[1:])
            target *= np.sinc(radii / 180)
        return target

    def ideal(rings: int) -> float:
        return float(1 + np.sum(target_counts(np.linspace(0, span, rings))))

    lo, hi = 2, (count - 1) // 3 + 1
    while lo < hi:
        mid = (lo + hi) // 2
        if ideal(mid) < count:
            lo = mid + 1
        else:
            hi = mid
    best_score = math.inf
    for rings in range(max(2, lo - 6), min((count - 1) // 3 + 1, lo + 6) + 1):
        s = np.linspace(0, span, rings)
        target = target_counts(s)
        populations = np.r_[1, allocate_ring_counts(target, count - 1)]
        score = np.average(
            np.log(target / populations[1:]) ** 2, weights=populations[1:]
        )
        if score < best_score:
            best_score = score
            best_radii = cmf_a * np.expm1(s)
            best_populations = populations
    best_radii[-1] = radius
    return best_radii, best_populations


def calibrated_isotropic_coordinates(
    fov: float,
    cmf_a: float,
    resolution: int,
    max_val: float,
    field_geometry: str = "planar",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return normalized xy, polar, and unnormalized radial-log coordinates.

    The third return retains the low-level radial representation; SamplingCoords
    selects named visualization layouts from xy and polar coordinates instead.
    The dimensionless max_val scales the actual visual eccentricity, including
    the radius used in the radial-log representation.
    """
    max_val = float(max_val)
    count = resolution**2
    radii, populations = calibrated_isotropic_rings(
        cmf_a, count, fov / 2, field_geometry
    )
    ring = np.repeat(np.arange(len(radii)), populations)
    starts = np.repeat(np.cumsum(populations) - populations, populations)
    angles = (np.arange(count) - starts) * (2 * np.pi / populations[ring])
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    normalized_radius = radii[ring] / (fov / 2) * max_val
    return (
        normalized_radius[:, None] * directions,
        np.column_stack((normalized_radius, angles)),
        np.log1p(radii[ring] * max_val / cmf_a)[:, None] * directions,
    )
