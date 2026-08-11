#!/usr/bin/env python3
"""Compare local sampling anisotropy across sensor and FoV variants.

This extends the analysis used for Supplementary Figure S4: for every valid
sample, find its nearest positions in the sensor's native/cortical coordinate
system, compute their covariance in visual Cartesian space, and report the
ratio of the long to short principal-axis standard deviations. Spatial padding
and masked grid positions participate in matching so FoV boundaries do not
warp the measured neighborhood.
"""

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from scipy.spatial import cKDTree
import torch

from fovi.sensing.coords import (
    _inverse_warped_cartesian,
    _warped_cartesian_radius_normalizer,
)
from fovi.sensing.retina import RetinalTransform


SENSORS = (
    ("warped_cartesian", "Warped Cartesian",
     ("circular", "square", "wang")),
    ("logpolar", "Log-polar", ("circular", "square")),
    ("isotropic", "Fovi isotropic", ("circular", "square")),
)
FOV_TYPES = ("circular", "square", "wang")
FOV_TITLES = {
    "circular": "Circular FoV",
    "square": "Square FoV",
    "wang": "Wang FoV",
}
SENSOR_COLORS = {
    "Warped Cartesian": "#d55e00",
    "Log-polar": "#0072b2",
    "Fovi isotropic": "#009e73",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=Path(
            "docs/assets/sensor_fov_examples/local_isotropy.png"))
    parser.add_argument(
        "--eccentricity-output", type=Path,
        default=Path(
            "docs/assets/sensor_fov_examples/"
            "local_isotropy_by_eccentricity.png"))
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--fov", type=float, default=16.0)
    parser.add_argument("--cmf-a", type=float, default=0.5)
    parser.add_argument("--k-neighbors", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--clip-percentile", type=float, default=99.0)
    parser.add_argument(
        "--eccentricity-bin-width", type=float, default=0.1)
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def compute_local_anisotropy(
        query_native_coords, candidate_visual_coords, candidate_native_coords,
        k_neighbors=1000, batch_size=1024):
    """Return the S4 principal-axis ratio at every sampling location."""
    query_native_coords = np.asarray(
        query_native_coords, dtype=np.float64)
    candidate_visual_coords = np.asarray(
        candidate_visual_coords, dtype=np.float64)
    candidate_native_coords = np.asarray(
        candidate_native_coords, dtype=np.float64)
    if candidate_visual_coords.shape[0] != candidate_native_coords.shape[0]:
        raise ValueError("candidate coordinate arrays must have equal length")
    if query_native_coords.shape[1] != candidate_native_coords.shape[1]:
        raise ValueError("query and candidate native dimensions must match")
    if not 2 <= k_neighbors <= candidate_visual_coords.shape[0]:
        raise ValueError(
            "k_neighbors must be between 2 and the number of candidates")

    tree = cKDTree(candidate_native_coords)
    anisotropy = np.empty(query_native_coords.shape[0], dtype=np.float64)
    for start in range(0, query_native_coords.shape[0], batch_size):
        stop = min(start + batch_size, query_native_coords.shape[0])
        _, neighbor_indices = tree.query(
            query_native_coords[start:stop], k=k_neighbors, workers=-1)
        local_points = candidate_visual_coords[neighbor_indices]
        centered = local_points - local_points.mean(axis=1, keepdims=True)
        covariance = np.einsum(
            "bki,bkj->bij", centered, centered) / (k_neighbors - 1)
        eigenvalues = np.linalg.eigvalsh(covariance)
        eigenvalues = np.maximum(eigenvalues, np.finfo(np.float64).eps)
        anisotropy[start:stop] = np.sqrt(
            eigenvalues[:, 1] / eigenvalues[:, 0])
    return anisotropy


def extend_regular_axis(axis, padding_width):
    """Extend a uniformly spaced native-grid axis in both directions."""
    axis = np.asarray(axis, dtype=np.float64)
    if axis.size < 2:
        raise ValueError("regular grid axes must have at least two locations")
    deltas = np.diff(axis)
    step = np.median(deltas)
    np.testing.assert_allclose(deltas, step, rtol=2e-5, atol=2e-7)
    offsets = np.arange(-padding_width, axis.size + padding_width)
    return axis[0] + step * offsets


def get_padded_grid_coords(style, fov_type, coords, args):
    """Build the virtual zero-padding lattice implicit in a padded Conv2d."""
    # A disk containing k lattice sites has radius approximately sqrt(k / pi).
    # sqrt(k) layers is a conservative bound at sides and corners.
    padding_width = int(np.ceil(np.sqrt(args.k_neighbors)))
    real_visual = coords.cartesian.detach().cpu().numpy()

    if style == "warped_cartesian":
        real_native = coords.plotting.detach().cpu().numpy()
        native_grid = real_native.reshape(
            args.resolution, args.resolution, 2)
        first_axis = extend_regular_axis(
            native_grid[:, 0, 0], padding_width)
        second_axis = extend_regular_axis(
            native_grid[0, :, 1], padding_width)
        first, second = np.meshgrid(
            first_axis, second_axis, indexing="ij")
        candidate_native = np.stack(
            (first, second), axis=-1).reshape(-1, 2)
        plotting = torch.as_tensor(
            candidate_native, dtype=coords.cartesian.dtype)
        candidate_visual, _ = _inverse_warped_cartesian(
            plotting, args.fov, args.cmf_a,
            radius_normalizer=_warped_cartesian_radius_normalizer(
                fov_type, args.fov, args.cmf_a))
        candidate_visual = candidate_visual.detach().cpu().numpy()
        candidate_shape = (len(first_axis), len(second_axis), 2)
        inner_visual = candidate_visual.reshape(candidate_shape)[
            padding_width:-padding_width,
            padding_width:-padding_width].reshape(-1, 2)
    elif style == "logpolar":
        # Native log-polar pixels form a regular radius-angle image. The model's
        # PolarPadder wraps the angular axis and zero-pads the radial axis.
        real_native = coords.cortical.detach().cpu().numpy()
        native_grid = real_native.reshape(
            args.resolution, args.resolution, 2)
        radial_axis = extend_regular_axis(
            native_grid[:, 0, 0], padding_width)
        angular_axis = native_grid[0, :, 1]
        radial, angular = np.meshgrid(
            radial_axis, angular_axis, indexing="ij")
        base_native = np.stack(
            (radial, angular), axis=-1).reshape(-1, 2)

        max_visual_radius = coords.polar[:, 0].max().item()
        rho_max = np.log(
            (max_visual_radius * (args.fov / 2.0) + args.cmf_a)
            / args.cmf_a)
        visual_radius = args.cmf_a * np.expm1(
            base_native[:, 0] * rho_max) / (args.fov / 2.0)
        theta = base_native[:, 1] * (2.0 * np.pi)
        base_visual = np.stack((
            visual_radius * np.cos(theta),
            visual_radius * np.sin(theta)), axis=1)
        angular_step = np.median(np.diff(angular_axis))
        angular_period = angular_step * args.resolution
        candidate_native = np.concatenate((
            base_native - np.array((0.0, angular_period)),
            base_native,
            base_native + np.array((0.0, angular_period))), axis=0)
        candidate_visual = np.tile(base_visual, (3, 1))
        candidate_shape = (len(radial_axis), len(angular_axis), 2)
        inner_visual = base_visual.reshape(candidate_shape)[
            padding_width:-padding_width].reshape(-1, 2)
    else:
        raise ValueError(f"unsupported regular-grid style: {style}")

    np.testing.assert_allclose(
        inner_visual, real_visual, rtol=2e-5, atol=2e-6)
    return real_native, candidate_visual, candidate_native


def get_sensor_coords(style, fov_type, args):
    retinal_transform = RetinalTransform(
        resolution=args.resolution,
        start_res=args.resolution,
        fov=args.fov,
        cmf_a=args.cmf_a,
        style=style,
        sampler="grid_bilinear",
        device="cpu",
        auto_match_cart_resources=True,
        isotropic_plotting_type="schwartz",
        fov_type=fov_type,
    ).eval()
    coords = retinal_transform.sampler.coords
    valid = coords.valid_mask.detach().cpu().numpy()
    real_visual = coords.cartesian.detach().cpu().numpy()

    if style in ("warped_cartesian", "logpolar"):
        real_native, candidate_visual, candidate_native = (
            get_padded_grid_coords(style, fov_type, coords, args))
        return (
            real_visual[valid], real_native[valid],
            candidate_visual, candidate_native)

    real_native = coords.cortical.detach().cpu().numpy()
    pad_visual = coords.cartesian_pad_coords.detach().cpu().numpy()
    pad_native = coords.cortical_pad_coords.detach().cpu().numpy()
    candidate_visual = np.concatenate((real_visual, pad_visual), axis=0)
    candidate_native = np.concatenate((real_native, pad_native), axis=0)
    return (
        real_visual[valid], real_native[valid],
        candidate_visual, candidate_native)


def summarize_by_eccentricity(
        visual, anisotropy, fov, bin_width_degrees):
    """Aggregate anisotropy over angle within eccentricity bins."""
    eccentricity = np.linalg.norm(visual, axis=1) * (fov / 2.0)
    max_eccentricity = (
        np.ceil(eccentricity.max() / bin_width_degrees)
        * bin_width_degrees)
    edges = np.arange(
        0.0, max_eccentricity + bin_width_degrees,
        bin_width_degrees)
    if edges[-1] < max_eccentricity:
        edges = np.append(edges, max_eccentricity)
    centers = (edges[:-1] + edges[1:]) / 2.0
    bin_indices = np.clip(
        np.digitize(eccentricity, edges) - 1, 0, len(centers) - 1)
    summary = np.full((3, len(centers)), np.nan, dtype=np.float64)
    for bin_index in range(len(centers)):
        selected = anisotropy[bin_indices == bin_index]
        if selected.size:
            summary[:, bin_index] = np.percentile(
                selected, (25, 50, 75))
    return centers, summary


def render_eccentricity_summary(results, args):
    """Plot angle-aggregated anisotropy over eccentricity."""
    fig, axes = plt.subplots(
        1, 3, figsize=(17, 4.8), sharex=True, sharey=True,
        facecolor="white")
    max_eccentricity = max(
        np.linalg.norm(visual, axis=1).max() * (args.fov / 2.0)
        for _, _, visual, _ in results)
    for ax, fov_type in zip(axes, FOV_TYPES):
        for title, result_fov_type, visual, anisotropy in results:
            if result_fov_type != fov_type:
                continue
            centers, summary = summarize_by_eccentricity(
                visual, anisotropy, args.fov,
                args.eccentricity_bin_width)
            lower, median, upper = summary
            color = SENSOR_COLORS[title]
            ax.fill_between(
                centers, lower, upper, color=color, alpha=0.16,
                linewidth=0)
            ax.plot(
                centers, median, color=color, marker="o", markersize=2.8,
                linewidth=1.4, label=title)
        ax.set_title(FOV_TITLES[fov_type])
        ax.set_xlim(0, max_eccentricity)
        ax.set_xlabel("Eccentricity (degrees)")
        ax.grid(alpha=0.22, linewidth=0.7)
    axes[0].set_ylabel(
        "Local anisotropy √(λₘₐₓ/λₘᵢₙ)")
    axes[1].legend(frameon=False, loc="upper right")
    fig.suptitle(
        "Local sampling anisotropy over eccentricity\n"
        "polar angle aggregated; median and interquartile range within "
        f"{args.eccentricity_bin_width:g}° bins",
        fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    args.eccentricity_output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        args.eccentricity_output, dpi=args.dpi,
        facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    args = parse_args()
    if not 0 < args.clip_percentile <= 100:
        raise ValueError("clip_percentile must be in (0, 100]")
    if args.eccentricity_bin_width <= 0:
        raise ValueError("eccentricity_bin_width must be positive")

    results = []
    for style, title, fov_types in SENSORS:
        for fov_type in fov_types:
            (visual, query_native, candidate_visual,
             candidate_native) = get_sensor_coords(style, fov_type, args)
            anisotropy = compute_local_anisotropy(
                query_native, candidate_visual, candidate_native,
                k_neighbors=args.k_neighbors, batch_size=args.batch_size)
            results.append((title, fov_type, visual, anisotropy))
            print(
                f"{style}/{fov_type}: queries={len(visual)} "
                f"candidates={len(candidate_visual)} "
                f"median={np.median(anisotropy):.4f} "
                f"p95={np.percentile(anisotropy, 95):.4f} "
                f"max={anisotropy.max():.4f}")

    all_anisotropy = np.concatenate([result[3] for result in results])
    vmax = np.percentile(all_anisotropy, args.clip_percentile)
    norm = Normalize(vmin=1.0, vmax=vmax, clip=True)

    fig = plt.figure(figsize=(14, 14), facecolor="white")
    grid = fig.add_gridspec(
        4, 3, height_ratios=(1, 1, 1, 0.045),
        left=0.025, right=0.975, bottom=0.065, top=0.92,
        hspace=0.16, wspace=0.08)
    axes = np.array([
        [fig.add_subplot(grid[row, col]) for col in range(3)]
        for row in range(3)
    ])
    scatter = None
    results_by_key = {
        (title, fov_type): (visual, anisotropy)
        for title, fov_type, visual, anisotropy in results
    }
    coord_limit = 1.02 * max(
        1.0, max(np.abs(visual).max() for visual, _ in results_by_key.values()))
    for row, (_, title, _) in enumerate(SENSORS):
        for col, fov_type in enumerate(FOV_TYPES):
            ax = axes[row, col]
            result = results_by_key.get((title, fov_type))
            if result is None:
                ax.text(
                    0.5, 0.5, "Unsupported", ha="center", va="center",
                    color="0.45", transform=ax.transAxes)
            else:
                visual, anisotropy = result
                scatter = ax.scatter(
                    visual[:, 0], visual[:, 1], c=anisotropy,
                    s=0.65, cmap="viridis", norm=norm, linewidths=0,
                    rasterized=True)
            ax.set_title(
                f"{title} — {FOV_TITLES[fov_type]}", fontsize=11)
            ax.set_xlim(-coord_limit, coord_limit)
            ax.set_ylim(-coord_limit, coord_limit)
            ax.set_aspect("equal", adjustable="box")
            ax.axis("off")

    colorbar = fig.colorbar(
        scatter, cax=fig.add_subplot(grid[3, :]),
        orientation="horizontal", extend="max")
    colorbar.set_label(
        "Local anisotropy, principal-axis ratio √(λₘₐₓ/λₘᵢₙ)")
    fig.suptitle(
        f"Local sampling anisotropy ({args.k_neighbors} native-space neighbors)",
        fontsize=15, y=0.975)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(
        f"color_scale=[1, {vmax:.4f}] "
        f"({args.clip_percentile:g}th percentile; higher values clipped)")
    print(args.output)
    render_eccentricity_summary(results, args)
    print(args.eccentricity_output)


if __name__ == "__main__":
    main()
