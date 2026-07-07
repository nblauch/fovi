#!/usr/bin/env python3
"""Render a panning video of the native 3D FOVI sensor manifold."""

import argparse
import os
import sys

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _repo_root)
_demo_cache = os.path.join(_repo_root, '.fovi_demo_cache')
os.environ.setdefault('FOVI_SAVE_DIR', _demo_cache)
os.environ.setdefault('FOVI_DATASETS_DIR', _demo_cache)

import matplotlib
if 'matplotlib.pyplot' not in sys.modules:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image

from fovi.demo import load_image_for_sampling
from fovi.sensing.retina import RetinalTransform
from fovi.utils import normalize


def fig_to_frame(fig):
    fig.canvas.draw()
    frame = np.array(fig.canvas.buffer_rgba())
    return frame[..., :3]


def crop_vertical_padding(frame, crop_frac=0.12):
    """Trim uniform top/bottom breathing room from a rendered manifold frame."""
    if crop_frac <= 0:
        return frame
    crop_px = int(round(frame.shape[0] * crop_frac))
    if crop_px <= 0 or 2 * crop_px >= frame.shape[0]:
        return frame
    return frame[crop_px:-crop_px]


def save_frames_as_video(frames, output_path, fps=15):
    frames = [
        np.asarray(frame.convert('RGB') if isinstance(frame, Image.Image) else frame).astype(np.uint8)
        for frame in frames
    ]
    imageio.mimsave(
        output_path,
        frames,
        fps=fps,
        codec='libx264',
        format='FFMPEG',
        macro_block_size=1,
    )


def _set_axes_equal_to_data(ax, xyz):
    """Use equal data ranges so the manifold shape is not distorted."""
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    centers = (mins + maxs) / 2
    radius = np.max(maxs - mins) / 2

    ax.set_xlim(centers[2] - radius, centers[2] + radius)
    ax.set_ylim(centers[0] - radius, centers[0] + radius)
    ax.set_zlim(centers[1] - radius, centers[1] + radius)
    ax.set_box_aspect((1, 1, 1))


def render_3d_manifold_video_from_colors(
    retinal_transform,
    colors,
    output_path,
    *,
    fps=15,
    duration=3.0,
    dpi=120,
    figsize=(4.8, 4.0),
    elev=15.0,
    azim_start=-55.0,
    azim_revolutions=1.0,
    point_size=2.0,
    vertical_crop_frac=0.12,
    background_color='#ffffff',
):
    """Render the cortical manifold as a short azimuth-orbit video.

    Args:
        retinal_transform: Built ``RetinalTransform`` whose sampler has
            ``coords.cortical``.
        colors: RGB array with shape ``(num_points, 3)`` in [0, 1].
        output_path: Destination video path.
    """
    cortical_xyz = retinal_transform.sampler.coords.cortical.detach().cpu().numpy()
    colors = np.clip(np.asarray(colors), 0, 1)
    if colors.shape[0] != cortical_xyz.shape[0]:
        raise ValueError(
            f'Expected {cortical_xyz.shape[0]} colors, got {colors.shape[0]}.',
        )

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor(background_color)
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_facecolor(background_color)

    # Notebook/movie convention: matplotlib axes are (cortical_z, cortical_x, cortical_y).
    ax.scatter(
        cortical_xyz[:, 2],
        cortical_xyz[:, 0],
        cortical_xyz[:, 1],
        s=point_size,
        c=colors,
        depthshade=False,
    )
    _set_axes_equal_to_data(ax, cortical_xyz)
    ax.set_axis_off()
    ax.set_position([0, 0, 1, 1])

    num_frames = max(1, int(round(fps * duration)))
    denom = max(num_frames - 1, 1)
    frames = []
    for index in range(num_frames):
        azim = azim_start - (index / denom) * 360.0 * azim_revolutions
        ax.view_init(elev=elev, azim=azim)
        frames.append(crop_vertical_padding(fig_to_frame(fig), vertical_crop_frac))

    plt.close(fig)
    save_frames_as_video(frames, output_path, fps=fps)
    return output_path


def render_3d_manifold_video(
    image_batch,
    fixation,
    retinal_transform,
    output_path,
    *,
    fixation_size,
    **render_kwargs,
):
    """Foveate one image/fixation and render its 3D manifold video."""
    with torch.no_grad():
        output = retinal_transform(
            image_batch,
            fix_loc=list(fixation),
            fixation_size=fixation_size,
        ).cpu()
    colors = normalize(output[0].T, dim=0).numpy()
    return render_3d_manifold_video_from_colors(
        retinal_transform,
        colors,
        output_path,
        **render_kwargs,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description='Render a panning 3D cortical manifold video for one fixation.',
    )
    parser.add_argument('image', help='Path to the source image')
    parser.add_argument('-o', '--output', default='manifold-3d.mp4')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--resolution', type=int, default=64)
    parser.add_argument('--start-res', type=int, default=None, dest='start_res')
    parser.add_argument('--fov', type=float, default=16.0)
    parser.add_argument('--cmf-a', type=float, default=0.5, dest='cmf_a')
    parser.add_argument('--style', default='isotropic')
    parser.add_argument('--sampler', default='grid_nn')
    parser.add_argument('--fixation', type=float, nargs=2, default=[0.5, 0.5],
                        metavar=('ROW', 'COL'))
    parser.add_argument('--fixation-size-frac', type=float, default=1.0,
                        dest='fixation_size_frac')
    parser.add_argument('--fps', type=int, default=15)
    parser.add_argument('--duration', type=float, default=3.0)
    parser.add_argument('--dpi', type=int, default=120)
    parser.add_argument('--elev', type=float, default=15.0)
    parser.add_argument('--azim-start', type=float, default=-55.0)
    parser.add_argument('--azim-revolutions', type=float, default=1.0)
    parser.add_argument('--point-size', type=float, default=2.0)
    parser.add_argument('--vertical-crop-frac', type=float, default=0.12,
                        help='Fraction to crop from top and bottom of rendered frames')
    parser.add_argument('--background-color', default='#ffffff')
    parser.add_argument('--center-crop', action='store_true')
    parser.add_argument('--no-normalize', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    batch, _display_rgb, height, width = load_image_for_sampling(
        args.image,
        device=args.device,
        normalize=not args.no_normalize,
        center_crop=args.center_crop,
    )
    start_res = args.start_res if args.start_res is not None else max(height, width)
    retinal_transform = RetinalTransform(
        resolution=args.resolution,
        start_res=start_res,
        fov=args.fov,
        cmf_a=args.cmf_a,
        style=args.style,
        sampler=args.sampler,
        device=args.device,
    ).eval()
    fix_side = np.sqrt(args.fixation_size_frac) * min(height, width)
    render_3d_manifold_video(
        batch,
        args.fixation,
        retinal_transform,
        args.output,
        fixation_size=fix_side,
        fps=args.fps,
        duration=args.duration,
        dpi=args.dpi,
        elev=args.elev,
        azim_start=args.azim_start,
        azim_revolutions=args.azim_revolutions,
        point_size=args.point_size,
        vertical_crop_frac=args.vertical_crop_frac,
        background_color=args.background_color,
    )
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()
