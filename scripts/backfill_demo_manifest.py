#!/usr/bin/env python3
"""Backfill manifest.json and input.png for demo runs created before manifest support."""

import argparse
import glob
import json
import os
import re
import sys

from PIL import Image

MANIFEST_VERSION = 1


def parse_args():
    parser = argparse.ArgumentParser(description='Backfill manifest.json for an existing demo output folder.')
    parser.add_argument('output_dir', help='Path to run folder (flat-*.png, global-cartesian-*.png, …)')
    parser.add_argument('--title', default=None)
    parser.add_argument('--fixations-file', default=None,
                        help='JSON file: [[row, col], …] normalized fixation coordinates')
    return parser.parse_args()


def load_fixations(path):
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    return [(float(r), float(c)) for r, c in data]


def count_fixations(output_dir):
    flats = glob.glob(os.path.join(output_dir, 'flat-*.png'))
    indices = []
    for path in flats:
        match = re.search(r'flat-(\d+)\.png$', os.path.basename(path))
        if match:
            indices.append(int(match.group(1)))
    return sorted(indices)


def main():
    args = parse_args()
    output_dir = os.path.abspath(args.output_dir)
    if not os.path.isdir(output_dir):
        raise SystemExit(f'Not a directory: {output_dir}')

    indices = count_fixations(output_dir)
    if not indices:
        raise SystemExit('No flat-*.png files found.')

    composite = os.path.join(output_dir, 'input-with-fixations.png')
    plain = os.path.join(output_dir, 'input.png')
    if os.path.isfile(composite) and not os.path.isfile(plain):
        Image.open(composite).convert('RGB').save(plain)

    source_image = 'input.png' if os.path.isfile(plain) else 'input-with-fixations.png'
    if not os.path.isfile(os.path.join(output_dir, source_image)):
        raise SystemExit('Need input.png or input-with-fixations.png')

    with Image.open(os.path.join(output_dir, source_image)) as img:
        width, height = img.size

    fixations_coords = None
    fixations_file = args.fixations_file or os.path.join(output_dir, 'fixations.json')
    if os.path.isfile(fixations_file):
        fixations_coords = load_fixations(fixations_file)

    fixation_entries = []
    for idx in indices:
        entry = {
            'index': idx,
            'views': {
                'flat': f'flat-{idx}.png',
                'global_cartesian': f'global-cartesian-{idx}.png',
            },
        }
        if fixations_coords and idx <= len(fixations_coords):
            row, col = fixations_coords[idx - 1]
            entry['row'] = row
            entry['col'] = col
        fixation_entries.append(entry)

    manifest = {
        'version': MANIFEST_VERSION,
        'title': args.title or os.path.basename(output_dir),
        'source_image': source_image,
        'source_with_fixations': 'input-with-fixations.png',
        'image_size': [height, width],
        'fixations': fixation_entries,
        'params': {},
        'timing': {
            'view_order': ['global_cartesian', 'flat'],
        },
        'display': {
            'use_composite_source': fixations_coords is None,
        },
    }

    manifest_path = os.path.join(output_dir, 'manifest.json')
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    print(f'Wrote {manifest_path} ({len(indices)} fixation(s))')
    if fixations_coords is None:
        print('No fixations.json — using composite source image for fixation beats.')


if __name__ == '__main__':
    main()
