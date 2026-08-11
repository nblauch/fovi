#!/usr/bin/env python3
"""Render reproducible FoV examples for each supported sensor topology."""

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Circle, Rectangle
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

from fovi.sensing.coords import get_warped_cartesian_sampling_coords
from fovi.sensing.retina import RetinalTransform


SENSORS = (
    ("warped_cartesian_as_grid", "warped_cartesian",
     ("circular", "square", "wang")),
    ("logpolar_as_grid", "logpolar", ("circular", "square")),
    ("isotropic", "fovi_isotropic_schwartz", ("circular", "square")),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image", type=Path,
        default=Path("web/foveated-player/runs/seoul/input.png"))
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("docs/assets/sensor_fov_examples"))
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--fov", type=float, default=16.0)
    parser.add_argument("--cmf-a", type=float, default=0.5)
    # Fixation 4 from web/foveated-player/runs/seoul/manifest.json.
    parser.add_argument(
        "--fixation-row", type=float, default=0.4753135183173694)
    parser.add_argument(
        "--fixation-col", type=float, default=0.3847964188143906)
    parser.add_argument("--fixation-size", type=int, default=512)
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def load_image(path):
    image = Image.open(path).convert("RGB")
    batch = pil_to_tensor(image).float().div_(255).unsqueeze(0)
    return image, batch


def save_source_reference(image, args):
    width, height = image.size
    x = args.fixation_col * width
    y = args.fixation_row * height
    radius = args.fixation_size / 2
    square_half_extent = radius

    # Keep the source image at its native 3:2 aspect ratio and reserve a
    # separate panel for the legend so it does not obscure the FoV outlines.
    fig = plt.figure(figsize=(10, 5), facecolor="black")
    ax = fig.add_axes((0.0, 0.0, 0.75, 1.0))
    legend_ax = fig.add_axes((0.75, 0.0, 0.25, 1.0))
    ax.set_facecolor("black")
    legend_ax.set_facecolor("black")
    ax.imshow(image)
    ax.add_patch(Rectangle(
        (x - square_half_extent, y - square_half_extent),
        2 * square_half_extent, 2 * square_half_extent,
        fill=False, edgecolor="#00e5ff", linewidth=2.0,
        label="outer square FoV"))
    ax.add_patch(Circle(
        (x, y), radius, fill=False, edgecolor="#ffd54f", linewidth=2.0,
        linestyle="--", label="circular FoV"))
    warped, _, _ = get_warped_cartesian_sampling_coords(
        args.fov, args.cmf_a, args.resolution,
        fov_type="wang")
    warped = warped.reshape(args.resolution, args.resolution, 2).numpy()
    for edge_index, edge in enumerate((
            warped[0], warped[-1], warped[:, 0], warped[:, -1])):
        ax.plot(
            x + edge[:, 0] * radius,
            y - edge[:, 1] * radius,
            color="#ff4fd8", linewidth=2.0, linestyle=":",
            label="Wang FoV" if edge_index == 0 else None)
    cross = max(width, height) * 0.018
    ax.plot([x - cross, x + cross], [y, y], color="#76ff03", linewidth=2.2)
    ax.plot([x, x], [y - cross, y + cross], color="#76ff03", linewidth=2.2)
    handles, labels = ax.get_legend_handles_labels()
    legend = legend_ax.legend(
        handles, labels, loc="center", frameon=False,
        fontsize=13, handlelength=2.5)
    for text in legend.get_texts():
        text.set_color("white")
    legend_ax.axis("off")
    # Do not let an out-of-image FoV outline change the reference canvas size.
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")
    path = args.output_dir / "source_fixation.png"
    fig.savefig(
        path, dpi=args.dpi, facecolor=fig.get_facecolor(), pad_inches=0)
    plt.close(fig)
    return path


def render_grid(samples, style, path, dpi):
    rgb = samples[0].detach().cpu()
    if style == "warped_cartesian_as_grid":
        # The native tensor axes are x then y. Orient x horizontally and +y up
        # for display without changing the stored downstream representation.
        rgb = rgb.permute(2, 1, 0).flip(0)
    else:
        # Native log-polar axes are cortical radius then angle. Transpose them
        # for display so eccentricity runs horizontally.
        rgb = rgb.permute(2, 1, 0)

    fig, ax = plt.subplots(figsize=(6, 4), facecolor="black")
    ax.set_facecolor("black")
    ax.imshow(rgb.clamp(0, 1).numpy(), interpolation="nearest", aspect="equal")
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor(),
                bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def render_isotropic(samples, retinal_transform, path, dpi):
    colors = samples[0].T.detach().cpu().clamp(0, 1).numpy()
    coords = retinal_transform.sampler.coords.plotting.detach().cpu().numpy()

    # Use an exact 1600x800 canvas at the selected DPI.
    fig, ax = plt.subplots(
        figsize=(1600 / dpi, 800 / dpi), facecolor="white")
    ax.set_facecolor("white")
    ax.scatter(
        coords[:, 0], coords[:, 1], c=colors, s=16,
        marker="o", linewidths=0)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor(), pad_inches=0)
    plt.close(fig)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image, batch = load_image(args.image)
    _, _, height, width = batch.shape
    fixation = torch.tensor(
        [[args.fixation_row, args.fixation_col]], dtype=batch.dtype)

    generated = [save_source_reference(image, args)]
    metadata = []
    for style, filename_stem, fov_types in SENSORS:
        for fov_type in fov_types:
            retinal_transform = RetinalTransform(
                resolution=args.resolution,
                start_res=max(height, width),
                fov=args.fov,
                cmf_a=args.cmf_a,
                style=style,
                sampler="grid_bilinear",
                fixation_size=args.fixation_size,
                device="cpu",
                auto_match_cart_resources=True,
                isotropic_plotting_type="schwartz",
                fov_type=fov_type,
            ).eval()
            with torch.no_grad():
                samples = retinal_transform(
                    batch, fixation, args.fixation_size)

            path = args.output_dir / f"{filename_stem}_{fov_type}.png"
            if style == "isotropic":
                render_isotropic(samples, retinal_transform, path, args.dpi)
            else:
                render_grid(samples, style, path, args.dpi)
            generated.append(path)
            metadata.append((
                filename_stem, fov_type,
                len(retinal_transform.sampler.coords), tuple(samples.shape)))

    print(f"source={args.image}")
    print(
        f"fixation=({args.fixation_row:.2f}, {args.fixation_col:.2f}) "
        f"fixation_size={args.fixation_size}px fov={args.fov:g}deg "
        f"cmf_a={args.cmf_a:g} target_resolution={args.resolution}")
    for name, shape, count, tensor_shape in metadata:
        print(f"{name}/{shape}: samples={count}, output={tensor_shape}")
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
