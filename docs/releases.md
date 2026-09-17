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
