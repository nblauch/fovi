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

## 2.7.0

Version 2.7.0 makes KNN neighbor tables independent of the device, adds modal
pooling for label maps, fuses grid pooling for CUDA inference, fixes float16
grid-pooling overflow, shrinks the base install, and speeds up automatic density
for square retinas.

### Device-independent KNN neighbors

Sampling coordinates differ between CPU and CUDA by float32 rounding, and
symmetric sensors such as isotropic rings place several candidates at equal
distances, so the neighbor chosen at a tied k-th slot used to depend on the
device. Every KNN search (visual, cortical, geodesic, and the native-grid tree)
now treats distances within `NEIGHBOR_TIE_TOLERANCE * (|query| + d_k)` as tied
(`fovi.arch.knn.NEIGHBOR_TIE_TOLERANCE`, a relative 5e-5), fills tied slots by
candidate index, and orders near-equal neighbors by index. CPU and CUDA now build
the same tables. Where candidates tie at the k-th distance, a table can differ
from the one 2.6.0 built on a given device.

### Modal pooling for label maps

`KNNPoolingLayer` and `GridPoolingLayer` accept `mode="modal"`: each output takes
the most common value among its pooling neighbors, compared bitwise and copied
in the input's dtype, so integer IDs of any width and colorized label vectors
come back exactly. Ties go to the earliest neighbor (nearest for KNN, nearest the
window center for grids). Padding, masked inputs and, optionally, inputs an image
does not see do not vote. With `pool_mode="modal"`, `PoolingGridSampler` keeps
the image's dtype, accepts an integer `output_dtype`, and leaves photoreceptors
outside the source image out of the vote. `GridSampler.sample_visibility`
reports which samples read the image at a fixation, and nearest sampling accepts
integer label maps of any dtype.

### Fused grid pooling for CUDA inference

`GridPoolingLayer` average and max pooling of CUDA float32 and uint8 inputs, at
inference, runs as one Warp kernel that reads strided `(B, C, N)` views directly
and writes only the pooled result, without full-resolution float, layout,
padding, or visibility intermediates. On 1,024 strided RGB views of 256x256
images pooled to 128x128, pooling takes 0.57 ms instead of 15 ms and 373 MiB of
PyTorch memory instead of 5.2 GiB. Arithmetic grid modes accept `visible=(B, N)`,
which excludes hidden samples and returns zero for empty neighborhoods; uint8
output rounds ties to even. Float16 inputs use the fused kernel too, accumulating
in float32 (0.1 ms instead of 1.3 ms for a batch of 128). Inference under
`no_grad()` or `inference_mode()` no longer allocates a gradient buffer for
inputs that require gradients. CPU, other dtypes and modes, autograd (including
forward-mode AD), `torch.func` transforms, and `torch.compile` keep the Torch
path. Reproduce the measurements with `benchmarks/benchmark_grid_pooling.py`.

Torch-path average pooling with a visibility mask now accumulates float16 inputs
in float32 until after normalization. Before, a window of large finite values
could overflow to infinity. This fallback uses more memory and time during
autograd.

### A smaller base install

The base no longer requires `scikit-image`, `fastcore`, `plum-dispatch`, or
`pygeodesic`, so applications that need only sensing, pooling, and KNN layers
pull in fewer packages:

- `scikit-image` moves to `fovi[training]`. Only `fovi.utils.analyze_rf` uses it,
  and imports it when called.
- `fastcore` and `plum-dispatch` move to `fovi[models]`, and so to
  `fovi[training]` and `fovi[all]`. Only the `fovi.utils.fastaugs` augmentation
  pipelines use them, which `FoviNet` and the trainer build.
  `RetinalTransform` imports `fastaugs` only when given `fastaugs`
  pre-transforms, so it constructs and samples with the base install.
- `pygeodesic` is replaced by fovi's own exact geodesic solver,
  `fovi.sensing.mesh_geodesic`, which uses only Warp and NumPy. It solves the
  same exact problem on the same meshes. Tests check it against `pygeodesic`
  (now a test-only dependency, in `requirements-test.txt`): distances agree to
  float64 rounding, and KNN pooling and patch-partition neighbor tables are
  identical. The solver runs many solves as one Warp launch, on CUDA for the
  many small local meshes of KNN layers, or concurrently on CPU cores for the
  shared mesh of omni-isotropic patch geodesics. Building geodesic KNN layers is
  several times faster, and filling the omni-isotropic patch geodesic cache at
  resolution 128 takes about 2 s instead of 84 s.

Geodesic neighborhoods are otherwise unchanged. Cached patch geodesics from 2.6.0
are recomputed once, since cache entries now record the solver.

### Faster automatic density for square retinas

`cmf_a='auto'` with a square isotropic retina is faster to construct. Resource
matching counts the samples of hundreds of candidate ring counts. Each count
tested every sample of every ring against the square, which dispatched
thousands of small torch operations. Rings within the square's half-width now
count whole, and only the corner rings are tested. Counts and calibrated
`cmf_a` values are unchanged. A square calibration takes 1.3–1.5 s instead of
2.3–2.7 s with one torch thread, and 5–13 s instead of 21–28 s with 24 threads
on a busy machine.

The test suite also runs faster, after trimming two test matrices whose cases
repeated the same coverage.

## 2.6.0

Version 2.6.0 adds fixed-count isotropic omni-foveation for DINOv3 with planar
and spherical geometry, configurable distance metrics and positional coordinates,
and a persistent geodesic cache. Isotropic visualization supports `v1like`,
`schwartz`, and `warp` layouts. It also fixes SSL view handling and fixation
normalization while preserving existing single-strength checkpoint geometry.
The supplied recipe uses geodesic neighborhoods, Cartesian RoPE, automatic
inward patch placement, and a 24-by-24 reference kernel.

See [isotropic omni-foveation](omni_isotropic.md) for configuration and usage.

## 2.4.0

Version 2.4.0 trains one encoder across several CMF levels. Setting
`saccades.cmf_a` to a list (for example
`[0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]`) builds a `SamplingCoordsBank`.
Training draws a level per image and reuses it across that image's fixations.
Validation assigns levels deterministically and reports per-level loss,
accuracy and image counts. List-valued `cmf_a` requires a
`warped_cartesian_as_grid` sensor with square or circular FoV and cortical ViT
positions. `configure_dinov3_positions` also rejects Cartesian positions for
checkpoints saved with a list of levels. Evaluating a multi-level model requires
explicit levels: pass `cmf_indices` to forward calls, or `cmf_level` to
`Trainer.compute_activations` and `knn_probe`.

Scalar `cmf_a` configs and checkpoints are unaffected, with one bug fix. The
fast pre-transform path, the training default, now applies the FoV mask, which
matches the reference and evaluation paths. Before this fix, circular-FoV
warped-Cartesian models were effectively trained with a square FoV and validated
with a circular one.

## 2.3.0

Version 2.3.0 adds square iso-eccentricity shells to the warped-Cartesian
sensor, through a new `radius_norm` argument.

`radius_norm` selects the norm that measures radius in the sensor's native
plane. The default `2.0` is the Euclidean radius and the existing behavior,
unchanged. `math.inf` uses the Chebyshev radius, so iso-eccentricity shells are
squares rather than circles and the native square maps exactly onto the visual
square, leaving every cell valid instead of masking the corners. Along the four
axes the two norms are the same map; they differ only off-axis.

The argument is accepted anywhere `fov_type` is, including `RetinalTransform`
and the `saccades` config block, where `.inf` is the YAML spelling. Existing
configs and checkpoints are unaffected: omitting `radius_norm` keeps the
Euclidean warp, which is byte-for-byte identical to 2.2.0. `fov_type='wang'`
normalizes the Euclidean radius at the native square's side centers, so it has
no infinity-norm counterpart and is rejected.

See [the radius norm](radius_norm.md) for the mapping, its behavior on the
diagonals, and related work.

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
