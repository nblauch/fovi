#!/usr/bin/env python3
"""Interactive demo: pick fixations on an image and run foveated sampling."""

import argparse
import json
import os
import sys

# Allow running without full FOVI environment configuration.
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _repo_root)
_demo_cache = os.path.join(_repo_root, '.fovi_demo_cache')
os.environ.setdefault('FOVI_SAVE_DIR', _demo_cache)
os.environ.setdefault('FOVI_DATASETS_DIR', _demo_cache)

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.widgets import Button
from PIL import Image

from fovi.demo import load_image_for_sampling
from fovi.sensing.coords import transform_sampling_grid
from fovi.sensing.retina import RetinalTransform
from fovi.utils import normalize
from scripts.render_3d_manifold_video import render_3d_manifold_video_from_colors

SAMPLER_CHOICES = ('grid_nn', 'grid_bilinear', 'pooling', 'gaussian_pooling')
PLOTTING_TYPE_CHOICES = ('v1like', 'schwartz', 'warp')
MANIFEST_VERSION = 1
DEFAULT_VIEW_ORDER = ('global_cartesian', 'manifold_3d', 'flat')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Pick fixations on an image and run foveated sampling.',
    )
    parser.add_argument('image', help='Path to a local image file')
    parser.add_argument('output_dir', help='Directory for output PNGs')

    parser.add_argument('--resolution', type=int, default=64,
                        help='RetinalTransform resolution (default: 64)')
    parser.add_argument('--start-res', type=int, default=None,
                        help='RetinalTransform start_res (default: image side length)')
    parser.add_argument('--fov', type=float, default=16.0,
                        help='Field of view in degrees (default: 16.0)')
    parser.add_argument('--cmf-a', type=float, default=0.5, dest='cmf_a',
                        help='Cortical magnification parameter (default: 0.5)')
    parser.add_argument('--style', default='isotropic',
                        help='Sampling style (default: isotropic)')
    parser.add_argument('--sampler', choices=SAMPLER_CHOICES, default='grid_nn',
                        help='Sampler type (default: grid_nn)')
    parser.add_argument('--fixation-size', type=int, default=None, dest='fixation_size',
                        help='Max fixation size in pixels (default: start_res)')
    parser.add_argument('--fixation-size-frac', type=float, default=1.0,
                        dest='fixation_size_frac',
                        help='Fraction of image used per fixation (default: 1.0)')
    parser.add_argument('--auto-match-cart-resources', action=argparse.BooleanOptionalAction,
                        default=True, dest='auto_match_cart_resources',
                        help='Auto-match cartesian resources (default: True)')
    parser.add_argument('--sigma', type=float, default=None,
                        help='Gaussian color decay sigma (default: None)')
    parser.add_argument('--no-color-val', action='store_true', dest='no_color_val',
                        help='Disable color in eval mode')
    parser.add_argument('--isotropic-plotting-type', choices=PLOTTING_TYPE_CHOICES,
                        default='v1like', dest='isotropic_plotting_type',
                        help='Plotting layout for isotropic coords (default: v1like)')

    parser.add_argument('--res-mult', type=int, default=None, dest='res_mult',
                        help='KNN sampler resolution multiplier')
    parser.add_argument('--cmf-a-mult', type=int, default=None, dest='cmf_a_mult',
                        help='KNN sampler cmf_a multiplier')
    parser.add_argument('--k', type=int, default=None,
                        help='KNN sampler number of neighbors')

    parser.add_argument('--device', default='cuda',
                        help='Device for sampling (default: cuda)')
    parser.add_argument('--size-mult', type=int, default=10,
                        help='Scatter size multiplier for cartesian panels (default: 10)')
    parser.add_argument('--no-normalize', action='store_true',
                        help='Skip ImageNet normalization on input tensor')
    parser.add_argument('--center-crop', action='store_true',
                        help='Center-crop to a square before fixation picking')
    parser.add_argument('--dpi', type=int, default=150,
                        help='DPI for saved PNGs (default: 150)')
    parser.add_argument('--manifold-video-fps', type=int, default=15,
                        help='FPS for 3D manifold videos (default: 15)')
    parser.add_argument('--manifold-video-duration', type=float, default=3.0,
                        help='Duration in seconds for 3D manifold videos (default: 3.0)')
    parser.add_argument('--regenerate', action='store_true',
                        help='Skip fixation picker; reload fixations from output_dir manifest.json')

    return parser.parse_args()


def resolve_device(requested):
    if requested == 'cuda' and not torch.cuda.is_available():
        print('CUDA unavailable; using CPU.', file=sys.stderr)
        return 'cpu'
    return requested


def save_input_image(display_rgb, output_path):
    """Save the source image without fixation markers."""
    Image.fromarray((np.clip(display_rgb, 0, 1) * 255).astype(np.uint8)).save(output_path)


def sampling_params_dict(args, height, width, start_res):
    """Collect RetinalTransform parameters for the manifest."""
    return {
        'resolution': args.resolution,
        'start_res': start_res,
        'fov': args.fov,
        'cmf_a': args.cmf_a,
        'style': args.style,
        'sampler': args.sampler,
        'fixation_size': args.fixation_size,
        'fixation_size_frac': args.fixation_size_frac,
        'auto_match_cart_resources': args.auto_match_cart_resources,
        'sigma': args.sigma,
        'no_color_val': args.no_color_val,
        'isotropic_plotting_type': args.isotropic_plotting_type,
        'res_mult': args.res_mult,
        'cmf_a_mult': args.cmf_a_mult,
        'k': args.k,
        'size_mult': args.size_mult,
        'center_crop': args.center_crop,
        'normalize': not args.no_normalize,
        'image_height': height,
        'image_width': width,
        'manifold_video_fps': getattr(args, 'manifold_video_fps', 15),
        'manifold_video_duration': getattr(args, 'manifold_video_duration', 3.0),
    }


def write_manifest(output_dir, fixations, args, height, width, start_res, title=None):
    """Write manifest.json for the static web player."""
    fixation_entries = []
    for idx, (row, col) in enumerate(fixations, start=1):
        fixation_entries.append({
            'index': idx,
            'row': row,
            'col': col,
            'views': {
                'flat': f'flat-{idx}.png',
                'flat_schwartz': f'flat-schwartz-{idx}.png',
                'manifold_3d': f'manifold-3d-{idx}.mp4',
                'manifold_3d_plotly': f'manifold-3d-{idx}.json',
                'local_cartesian': f'local-cartesian-{idx}.png',
                'global_cartesian': f'global-cartesian-{idx}.png',
            },
        })

    manifest = {
        'version': MANIFEST_VERSION,
        'title': title or os.path.basename(os.path.normpath(output_dir)),
        'source_image': 'input.png',
        'source_with_fixations': 'input-with-fixations.png',
        'image_size': [height, width],
        'fixations': fixation_entries,
        'params': sampling_params_dict(args, height, width, start_res),
        'timing': {
            'fixation_reveal_ms': 900,
            'view_hold_ms': 1400,
            'view_order': list(DEFAULT_VIEW_ORDER),
            'pause_between_fixations_ms': 400,
        },
    }
    manifest_path = os.path.join(output_dir, 'manifest.json')
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    return manifest_path


def load_fixations_from_output(output_dir):
    """Load fixation (row, col) pairs from a previous demo run.

    Returns:
        tuple: (fixations, manifest or None)
    """
    manifest_path = os.path.join(output_dir, 'manifest.json')
    if os.path.isfile(manifest_path):
        with open(manifest_path, encoding='utf-8') as f:
            manifest = json.load(f)
        fixations = []
        for entry in manifest.get('fixations', []):
            if 'row' in entry and 'col' in entry:
                fixations.append((float(entry['row']), float(entry['col'])))
        if fixations:
            return fixations, manifest
        raise SystemExit(
            f'{manifest_path} has no fixation coordinates; re-run without --regenerate '
            'or add fixations.json and backfill the manifest.',
        )

    fixations_path = os.path.join(output_dir, 'fixations.json')
    if os.path.isfile(fixations_path):
        with open(fixations_path, encoding='utf-8') as f:
            data = json.load(f)
        fixations = [(float(row), float(col)) for row, col in data]
        if fixations:
            return fixations, None
        raise SystemExit(f'{fixations_path} is empty.')

    raise SystemExit(
        f'Cannot --regenerate: no manifest.json or fixations.json in {output_dir}',
    )


def figure_size_for_image(height, width, max_size=8):
    """Figure size preserving native image aspect ratio."""
    if width >= height:
        return (max_size, max_size * height / width)
    return (max_size * width / height, max_size)


def pick_fixations(display_rgb):
    """Open an interactive window to collect fixation points.

    Returns:
        list[tuple[float, float]]: Normalized (row, col) fixations in [0, 1].
    """
    height, width = display_rgb.shape[:2]
    fixations = []
    cross_artists = []
    label_artists = []

    fig, ax = plt.subplots(figsize=figure_size_for_image(height, width))
    plt.subplots_adjust(bottom=0.12)
    ax.imshow(display_rgb, extent=(0, width, height, 0), aspect='equal')
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_title('Left-click to add fixations, right-click to undo, then Done')
    ax.set_xticks([])
    ax.set_yticks([])

    def redraw_crosses():
        for artist in cross_artists + label_artists:
            artist.remove()
        cross_artists.clear()
        label_artists.clear()
        for idx, (row, col) in enumerate(fixations, start=1):
            x = col * width
            y = row * height
            size = max(height, width) * 0.03
            h_line, = ax.plot([x - size, x + size], [y, y], color='lime', linewidth=2)
            v_line, = ax.plot([x, x], [y - size, y + size], color='lime', linewidth=2)
            cross_artists.extend([h_line, v_line])
            label = ax.text(x + size, y - size, str(idx), color='lime', fontsize=10,
                            fontweight='bold', va='bottom')
            label_artists.append(label)
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        if event.button == 1:
            row = np.clip(event.ydata / height, 0.0, 1.0)
            col = np.clip(event.xdata / width, 0.0, 1.0)
            fixations.append((float(row), float(col)))
            redraw_crosses()
        elif event.button == 3 and fixations:
            fixations.pop()
            redraw_crosses()

    done_state = {'finished': False}

    def on_done(_event):
        done_state['finished'] = True
        plt.close(fig)

    fig.canvas.mpl_connect('button_press_event', on_click)
    done_ax = fig.add_axes([0.4, 0.02, 0.2, 0.05])
    done_button = Button(done_ax, 'Done')
    done_button.on_clicked(on_done)

    plt.show()

    if not done_state['finished']:
        raise SystemExit('Fixation picker closed without pressing Done.')
    if not fixations:
        raise SystemExit('No fixations selected. Left-click at least one point, then Done.')

    return fixations


def _make_figure_transparent(fig, ax):
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)


def save_input_with_fixations(display_rgb, fixations, output_path, dpi):
    height, width = display_rgb.shape[:2]
    fig, ax = plt.subplots(figsize=figure_size_for_image(height, width, max_size=6))
    _make_figure_transparent(fig, ax)
    ax.imshow(display_rgb, extent=(0, width, height, 0), aspect='equal')
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis('off')
    for idx, (row, col) in enumerate(fixations, start=1):
        x = col * width
        y = row * height
        size = max(height, width) * 0.03
        ax.plot([x - size, x + size], [y, y], color='lime', linewidth=2)
        ax.plot([x, x], [y - size, y + size], color='lime', linewidth=2)
        ax.text(x + size, y - size, str(idx), color='lime', fontsize=10,
                fontweight='bold', va='bottom')
    fig.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=dpi, transparent=True)
    plt.close(fig)


def render_scatter(
    coords,
    color,
    sizes,
    *,
    equal_axes=True,
    fixed_square_limits=False,
    figsize=None,
    image_size=None,
):
    """Render a scatter plot of foveated samples.

    Local cartesian panels use fixed [-1, 1] square limits. Global cartesian
    panels use the same limits but aspect ratio matched to ``image_size``
    (height, width) so the plot is not squeezed. Manifold (flat) panels use
    data-driven limits with equal aspect.
    """
    if figsize is None:
        if image_size is not None:
            figsize = figure_size_for_image(image_size[0], image_size[1])
        else:
            figsize = (4, 4) if fixed_square_limits else (8, 4)

    fig, ax = plt.subplots(figsize=figsize)
    _make_figure_transparent(fig, ax)
    ax.scatter(coords[:, 0], coords[:, 1], c=color, s=sizes)
    if equal_axes:
        if fixed_square_limits or image_size is not None:
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
        if image_size is not None:
            height, width = image_size
            ax.set_aspect(height / width, adjustable='box')
        else:
            ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    fig.subplots_adjust(0, 0, 1, 1)
    return fig


def save_plotly_manifold_data(retinal_transform, color, output_path):
    """Write compact data for a browser-side Plotly 3D manifold scatter."""
    cortical_xyz = retinal_transform.sampler.coords.cortical.detach().cpu().numpy()
    plot_xyz = np.column_stack([
        cortical_xyz[:, 2],
        cortical_xyz[:, 0],
        cortical_xyz[:, 1],
    ])
    mins = plot_xyz.min(axis=0)
    maxs = plot_xyz.max(axis=0)
    center = (mins + maxs) / 2
    scale = np.max(maxs - mins)
    plot_xyz = (plot_xyz - center) / scale

    rgb = np.clip(np.asarray(color), 0, 1)
    rgb_u8 = np.rint(rgb * 255).astype(np.uint8)
    colors = [f'#{r:02x}{g:02x}{b:02x}' for r, g, b in rgb_u8]

    data = {
        'version': 1,
        # Match the matplotlib/movie convention: (cortical_z, cortical_x, cortical_y).
        'x': np.round(plot_xyz[:, 0], 5).tolist(),
        'y': np.round(plot_xyz[:, 1], 5).tolist(),
        'z': np.round(plot_xyz[:, 2], 5).tolist(),
        'range': [-0.51, 0.51],
        'color': colors,
        'marker_size': 2.4,
        'camera': {
            'eye': {'x': -1.45, 'y': -0.28, 'z': 0.36},
            'center': {'x': 0, 'y': 0, 'z': 0},
            'up': {'x': 0, 'y': 0, 'z': 1},
        },
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, separators=(',', ':'))
    return output_path


def build_retinal_transform(args, start_res, device):
    sampler_kwargs = {}
    if args.res_mult is not None:
        sampler_kwargs['res_mult'] = args.res_mult
    if args.cmf_a_mult is not None:
        sampler_kwargs['cmf_a_mult'] = args.cmf_a_mult
    if args.k is not None:
        sampler_kwargs['k'] = args.k

    return RetinalTransform(
        resolution=args.resolution,
        start_res=start_res,
        fov=args.fov,
        cmf_a=args.cmf_a,
        style=args.style,
        sampler=args.sampler,
        fixation_size=args.fixation_size,
        device=device,
        auto_match_cart_resources=args.auto_match_cart_resources,
        sigma=args.sigma,
        no_color_val=args.no_color_val,
        isotropic_plotting_type=args.isotropic_plotting_type,
        **sampler_kwargs,
    ).eval()


def run_sampling_and_save(batch, fixations, retinal_transform, args, height, width, output_dir):
    scale = np.sqrt(args.fixation_size_frac)
    fix_side = scale * min(height, width)
    rel_cart_coords = retinal_transform.sampler.coords.cartesian.cpu().numpy()
    flat_v1like_coords = retinal_transform.sampler.coords.clone(
        isotropic_plotting_type='v1like',
    ).plotting.cpu().numpy()
    flat_schwartz_coords = retinal_transform.sampler.coords.clone(
        isotropic_plotting_type='schwartz',
    ).plotting.cpu().numpy()
    cart_sizes = args.size_mult * retinal_transform.scatter_sizes

    saved_paths = []

    with torch.no_grad():
        for idx, (row, col) in enumerate(fixations, start=1):
            fix_loc = [row, col]
            output = retinal_transform(
                batch, fix_loc=fix_loc, fixation_size=fix_side,
            ).cpu()
            color = normalize(output[0].T, dim=0).numpy()

            fix_loc_tensor = torch.tensor([[row, col]], dtype=batch.dtype, device=batch.device)
            fix_size_tensor = torch.tensor([[fix_side, fix_side]], dtype=batch.dtype, device=batch.device)
            abs_cart_coords = transform_sampling_grid(
                retinal_transform.sampler.out_sampling_grid,
                fix_loc_tensor,
                fix_size_tensor,
                (height, width),
            ).squeeze().cpu().numpy()
            abs_cart_coords[:, 1] = -abs_cart_coords[:, 1]

            outputs = [
                ('flat', flat_v1like_coords, 4, False, False, None),
                ('flat-schwartz', flat_schwartz_coords, 4, False, False, None),
                ('local-cartesian', rel_cart_coords, cart_sizes, True, True, None),
                ('global-cartesian', abs_cart_coords, cart_sizes, True, False, (height, width)),
            ]

            for name, coords, sizes, equal_axes, fixed_square, img_size in outputs:
                path = os.path.join(output_dir, f'{name}-{idx}.png')
                fig = render_scatter(
                    coords, color, sizes,
                    equal_axes=equal_axes,
                    fixed_square_limits=fixed_square,
                    image_size=img_size,
                )
                fig.savefig(path, bbox_inches='tight', pad_inches=0, dpi=args.dpi, transparent=True)
                plt.close(fig)
                saved_paths.append(path)

            manifold_video_path = os.path.join(output_dir, f'manifold-3d-{idx}.mp4')
            render_3d_manifold_video_from_colors(
                retinal_transform,
                color,
                manifold_video_path,
                fps=getattr(args, 'manifold_video_fps', 15),
                duration=getattr(args, 'manifold_video_duration', 3.0),
            )
            saved_paths.append(manifold_video_path)

            manifold_plotly_path = os.path.join(output_dir, f'manifold-3d-{idx}.json')
            save_plotly_manifold_data(retinal_transform, color, manifold_plotly_path)
            saved_paths.append(manifold_plotly_path)

    return saved_paths


def main():
    args = parse_args()
    device = resolve_device(args.device)

    if not os.path.isfile(args.image):
        raise SystemExit(f'Image not found: {args.image}')

    if args.regenerate and not os.path.isdir(args.output_dir):
        raise SystemExit(f'Cannot --regenerate: output directory not found: {args.output_dir}')

    batch, display_rgb, height, width = load_image_for_sampling(
        args.image, device=device, normalize=not args.no_normalize,
        center_crop=args.center_crop,
    )

    prev_manifest = None
    if args.regenerate:
        fixations, prev_manifest = load_fixations_from_output(args.output_dir)
        if prev_manifest is not None:
            prev_h, prev_w = prev_manifest.get('image_size', [None, None])
            if prev_h is not None and (prev_h, prev_w) != (height, width):
                print(
                    f'Warning: image size ({height}x{width}) differs from previous run '
                    f'({prev_h}x{prev_w}); fixation positions are reused as normalized coords.',
                    file=sys.stderr,
                )
        print(f'Regenerating {len(fixations)} fixation(s) from {args.output_dir}')
    else:
        fixations = pick_fixations(display_rgb)

    os.makedirs(args.output_dir, exist_ok=True)
    start_res = args.start_res if args.start_res is not None else max(height, width)
    retinal_transform = build_retinal_transform(args, start_res, device)

    input_path = os.path.join(args.output_dir, 'input-with-fixations.png')
    save_input_with_fixations(display_rgb, fixations, input_path, args.dpi)
    save_input_image(display_rgb, os.path.join(args.output_dir, 'input.png'))

    saved_paths = run_sampling_and_save(
        batch, fixations, retinal_transform, args, height, width, args.output_dir,
    )

    manifest_title = prev_manifest.get('title') if prev_manifest else None
    manifest_path = write_manifest(
        args.output_dir, fixations, args, height, width, start_res, title=manifest_title,
    )

    print(f'{"Regenerated" if args.regenerate else "Selected"} {len(fixations)} fixation(s).')
    print(f'Output directory: {args.output_dir}')
    print(f'Wrote {manifest_path}')
    print(f'Wrote {input_path}')
    print('Static player: https://nblauch.github.io/fovi/ or web/foveated-player/index.html?run=<path-to-output-dir>')


if __name__ == '__main__':
    main()
