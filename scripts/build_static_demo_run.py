#!/usr/bin/env python3
"""Build a bundled example run for the static foveated-player demo."""

import argparse
import os
import sys

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _repo_root)
_demo_cache = os.path.join(_repo_root, '.fovi_demo_cache')
os.environ.setdefault('FOVI_SAVE_DIR', _demo_cache)
os.environ.setdefault('FOVI_DATASETS_DIR', _demo_cache)

import numpy as np
from PIL import Image

from scripts.interactive_sampling_demo import (
    build_retinal_transform,
    load_fixations_from_output,
    run_sampling_and_save,
    save_input_image,
    save_input_with_fixations,
    write_manifest,
)
from fovi.demo import load_image_for_sampling


def parse_args():
    parser = argparse.ArgumentParser(description='Build a static demo run for the web player.')
    parser.add_argument(
        '--output-dir',
        default=os.path.join(_repo_root, 'web', 'foveated-player', 'runs', 'example'),
        help='Directory for the example run (default: web/foveated-player/runs/example)',
    )
    parser.add_argument('--device', default='cpu')
    return parser.parse_args()


def make_synthetic_image(path, height=240, width=360):
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[:, : width // 2] = [210, 80, 60]
    arr[:, width // 2 :] = [40, 90, 180]
    y, x = np.ogrid[:height, :width]
    mask = (x - width * 0.35) ** 2 + (y - height * 0.45) ** 2 < (height * 0.12) ** 2
    arr[mask] = [240, 210, 80]
    Image.fromarray(arr).save(path)
    return path


def make_args_namespace():
    return argparse.Namespace(
        resolution=64,
        start_res=None,
        fov=16.0,
        cmf_a=0.5,
        style='isotropic',
        sampler='grid_nn',
        fixation_size=None,
        fixation_size_frac=1.0,
        auto_match_cart_resources=True,
        sigma=None,
        no_color_val=False,
        isotropic_plotting_type='v1like',
        res_mult=None,
        cmf_a_mult=None,
        k=None,
        size_mult=10,
        dpi=120,
        manifold_video_fps=15,
        manifold_video_duration=3.0,
        center_crop=False,
        no_normalize=False,
    )


def main():
    cli = parse_args()
    sampling_args = make_args_namespace()
    os.makedirs(cli.output_dir, exist_ok=True)

    img_path = os.path.join(cli.output_dir, '_source_synthetic.png')
    make_synthetic_image(img_path)

    batch, display_rgb, height, width = load_image_for_sampling(
        img_path, device=cli.device, center_crop=False,
    )
    try:
        fixations, _prev_manifest = load_fixations_from_output(cli.output_dir)
    except SystemExit:
        fixations = [(0.45, 0.35), (0.55, 0.72), (0.28, 0.55)]
    start_res = max(height, width)
    retinal_transform = build_retinal_transform(sampling_args, start_res, cli.device)

    save_input_image(display_rgb, os.path.join(cli.output_dir, 'input.png'))
    save_input_with_fixations(
        display_rgb, fixations,
        os.path.join(cli.output_dir, 'input-with-fixations.png'),
        sampling_args.dpi,
    )
    run_sampling_and_save(
        batch, fixations, retinal_transform, sampling_args, height, width, cli.output_dir,
    )
    manifest_path = write_manifest(
        cli.output_dir, fixations, sampling_args, height, width, start_res, title='Example',
    )

    os.remove(img_path)
    print(f'Built example run at {cli.output_dir}')
    print(f'Manifest: {manifest_path}')


if __name__ == '__main__':
    main()
