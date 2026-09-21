# Versions and releases

See the [installation guide](read_me.rst) for installation options and
[package boundaries and migration](package_boundaries.md) for updated import paths.

## Checking your version

```bash
python -c 'import fovi; print(fovi.__version__)'
```

Source installs from `main` may include changes that are not in a published release.
For reproducible experiments and issue reports, also record the commit and any local
changes by running these commands from your fovi checkout:

```bash
git rev-parse HEAD
git status --short
```

## 2.2.0

Version 2.2.0 reorients grid-shaped sensor outputs, makes DINOv3 position
coordinates selectable, and adds checkpoint loading from W&B runs.

**Breaking: grid-shaped sensor outputs are now upright.** The `uniform_as_grid`
and `warped_cartesian_as_grid` styles return rows ordered top to bottom and
columns left to right. Previously they kept the sampler's
`meshgrid(indexing='ij')` order, which is a transpose and a vertical flip away
from the new layout. Nothing records which layout a checkpoint was trained with,
so a model trained on either style before 2.2.0 receives rotated and mirrored
input after upgrading, and its accuracy collapses without raising an error.
Retrain those models on 2.2.0, or pin 2.1.0 to keep using them.
Styles that do not end in `_as_grid` are unaffected.

DINOv3 position coordinates are selectable through
`model.vit.position_coordinate_space`. The defaults preserve earlier behavior:
grid sensors use `cortical` native coordinates and vector sensors use
`cartesian`.

Training checkpoints load directly from W&B with
`get_model_from_base_fn('wandb://entity/project/run')`, including while a run is
still training. A run publishes only its latest checkpoint, replacing the
previous upload. Training also writes `resolved_config.yaml` alongside its
checkpoints, recording the settings resolved while building the model;
checkpoint loading prefers it over the original Hydra launch config.

## 2.1.0

Version 2.1.0 corrects the planar field geometry and adds calibrated spherical
sampling, selected through the new `saccades.field_geometry` setting.

- `planar` uses the corrected unbounded planar field.
- `spherical` treats eccentricity as an angle on the sphere, bounded by the
  cortical-magnification limit. It requires a retinal transform, so
  `saccades.mode` cannot be null.
- `legacy` reproduces the pre-2.1.0 integration mesh and endpoints exactly.

A configuration without `saccades.field_geometry` resolves to `legacy` and warns,
so existing checkpoints keep the geometry they were trained with. Set the value
explicitly: choose `planar` or `spherical` for new training, and `legacy` for
weights trained before 2.1.0. Moving existing weights onto `planar` or
`spherical` changes KNN neighborhoods and can change predictions.

Foveal density can be fit at construction time against a calibrated source
camera model, with a CUDA sampling path for the calibrated grids.

## 2.0.1

Optional KNN backend discovery is cached outside forward passes.

## 2.0.0

Version 2.0.0 separates sensing, models, and training within a single `fovi`
distribution.

- Base `fovi` includes sensing, sampling grids, and KNN layers, with CuPy and Warp
  for optimized kernels.
- `fovi[models]` adds complete networks and pretrained checkpoint loading.
- `fovi[training]` adds model dependencies, training utilities, and research tools.
- `fovi[all]` selects the same dependencies as `fovi[models,training]`.

FFCV-SSL requires a separate manual installation for the built-in training and
validation loaders, including with `fovi[all]`. Pretrained inference does not require
FFCV, datasets, or research storage environment variables.

Complete networks and inference loaders now live in `fovi.models`; training code
lives in `fovi.training`. Previous model import paths have been removed, so existing
scripts and configuration targets must use the new paths. Training compatibility
imports remain available. See the [migration guide](package_boundaries.md) for
replacement imports and checkpoint compatibility.

## 1.0 — Source baseline

Version 1.0 identifies the source before the package reorganization, at commit
[`2916774`](https://github.com/nblauch/fovi/commit/2916774285399ea9d04026c65a828e9c1aef9923).
It contains the original combined sensing, model, training, and Hub-loading code.
This is a source baseline, not a release published on PyPI.
