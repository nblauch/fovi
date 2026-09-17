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

## 2.1.0 — Unreleased

Visual-field geometry is explicit: `field_geometry="planar"` (the default),
`"spherical"`, or `"legacy"` for compatibility with earlier checkpoints.
Planar and spherical sampling rings and cortical manifolds use consistent metrics.
The planar manifold corrects the previous small-angle approximation, so existing
checkpoint activations can change slightly even though weights are unchanged.
Pretrained validation compares full ImageNet validation accuracy on the same GPU
at each configured fixation count. Some changes exceed 0.1 percentage points;
these measurements explicitly use corrected planar geometry.
The [full validation report](geometry_validation.md) records every configured
fixation count for six published checkpoints. The largest measured top-1 decrease
is 0.738 percentage points for ResNet-18 at five fixations. Checkpoint weights are
unchanged; accuracy and activations are not guaranteed identical across this
geometry correction. [Activation diagnostics](geometry_compatibility.md) isolate
the changes to cortical processing, with identical sampled retinal inputs in
the tested cases. The explicit legacy option preserves historical geometry;
checkpoint configs select it through `saccades.field_geometry: legacy`. Existing
Hugging Face configs still need that metadata update as part of release rollout.

Spherical sampling uses an angular FoV and rotates retinal rays for saccades.
Calibrated pinhole and OpenCV-style fisheye projection support software fixation
away from the source camera's optical center. Window calibration uses the angular
span of the selected image side, including lens distortion. See
[calibrated spherical sampling](spherical_sampling.md) for configuration,
coordinate conventions, and scope.

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
