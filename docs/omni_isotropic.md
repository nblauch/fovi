# Fixed-count isotropic omni-foveation

`dinov3_isotropic_multi_cmf` adapts DINOv3-S+ with the standard ImageNet
multi-CMF recipe. `isotropic_fixn_calibrated` is a new, deterministic sampling
style; existing isotropic checkpoints retain their original geometry.

Each level has exactly `resolution**2` real retinal samples and
`(resolution // patch_size)**2` real patch centers. Integer ring populations
are calibrated while retaining logarithmic radial spacing, the specified CMF,
and the outer field radius. No intrinsic relaxation is applied. This reduces
radial/tangential spacing distortion; it does not guarantee perfect discrete
isotropy or eliminate the aligned angular seams between rings.

At construction, the existing covering-kNN patch builder supplies neighbors and
reference-bin assignments for every level. Forward and transposed incidence
tables are registered as nonpersistent buffers. Native CUDA reductions select
the correct table per image, then one shared linear projection applies the
learned patch weights. No per-level batch splitting, dense edge-feature tensor,
or expanded weight tensor is needed. The input-gradient kernel uses the reverse
incidence tables, with no atomic accumulation. A CPU reference implementation
supports inspection and tests.

The per-image `cmf_indices` also select the configured patch positions for RoPE.
All fixations of one image reuse its level; flattening fixations into the batch
repeats the indices in fixation-major order. Evaluation requires explicit level
indices, and existing per-CMF validation metrics are retained.

The initial configuration uses Euclidean distances on the embedded cortical
manifold (`sample_cortex: true`), matching the research prototype. This is a
chord-distance approximation, not the `geodesic` option used by historical
single-strength isotropic pretraining. The backbone uses visual Cartesian RoPE;
the warped omni baseline uses native grid RoPE. These differences must be
considered when interpreting comparisons.

## Geodesic and positional ablations

`dinov3_isotropic_multi_cmf_geodesic` retains Cartesian RoPE and changes
neighborhood membership to cortical surface geodesics. It triangulates the real
samples, spatial padding and actual patch centers together. PyGeodesic solves
distances on this piecewise-linear surface, approximating the smooth manifold.
This path does not snap patch origins to retinal samples or restrict candidate
neighbors. Historical single-level geodesic construction is unchanged. Reference
bin coordinates retain the existing chord-radius/visual-angle convention.

The optional `FOVI_GEODESIC_CACHE_DIR` stores checksummed distances keyed by exact
geometry, algorithm version and solver dependency versions. Corrupt entries fail
explicitly. This cache only avoids CPU setup work; the CUDA convolution is
unchanged, and its geometry remains fixed for the model lifetime.

`dinov3_isotropic_multi_cmf_radial_log` additionally uses the normalized radial
mapping `log1p(r/a) / log1p(R/a)` with the original polar angle, in row/column
order. This retains two-axis RoPE and its frequency allocation. Flattening
distorts circumferential distances; it changes only the positional signal, not
the isotropic sample locations or geodesic neighborhoods.

`dinov3_isotropic_multi_cmf_manifold_3d` instead uses the embedded cortical
row/column/height coordinates. A common scalar, the cortical radial arc length
to the field boundary, normalizes all axes together. The attention width remains
fixed: a 64-dimensional head has 32 rotary pairs, assigned 11/11/10 to the axes.
Each axis spans the same endpoint frequencies as the original two-axis RoPE.
This reallocates pretrained rotary channels and encodes extrinsic position
differences, not exact geodesic displacement or rotation-invariant attention.
All three ablations retain exactly the same sample and token counts.

## Geometry transfer

### Patch boundary placement

The original omni configs retain perimeter-centered outer patches for checkpoint
compatibility. The optional `model.vit.patch_max_coord_val` defaults to `1.0` when
absent from a saved config. `dinov3_isotropic_multi_cmf_geodesic_inset` explicitly
sets it to `auto`: the existing `get_in_out_coords` helper regenerates the patch
grid with maximum radius halfway between its provisional two outermost rings.
The corresponding `_inset_kernel24` config changes only `reference_side` to 24.
Both use Cartesian RoPE and geodesic neighborhoods.

Retinal coordinates and counts are unchanged, and every level still has 64
patches. Patch positions, neighborhoods, bin tables, and positional encodings
are regenerated consistently; the exact-geometry cache keys prevent reuse of
perimeter-centered distance tables. This is the standard boundary-placement
heuristic, not a solver that optimizes RF extent. Changing this option when
loading a checkpoint deliberately changes its geometry.

This implementation currently accepts circular **planar** geometry. Spherical
geometry fails explicitly; its ring calibration still needs implementation and
validation. A change in spherical field diameter or planar-to-spherical transfer
requires regenerating the sample/patch grids, neighbor and reference-bin tables,
and RoPE positions. Keep sample count, patch count, reference kernel dimensions,
and coordinate conventions fixed to reuse learned weights. In planar geometry,
scaling field diameter and unnormalized CMF `a` together preserves the normalized
layout; spherical curvature removes that general scale invariance.

The bank supports a declared finite list of strengths. An unseen strength
requires constructing another geometry, not interpolating integer neighbors.
Geometry metadata is rebuilt from saved configuration rather than loaded as
weights. Changes to the calibration algorithm must therefore preserve the
existing style's behavior or introduce a new versioned style.

## Validation

Run `pytest tests/test_omni_isotropic.py tests/test_omni_positions.py
tests/test_omni_patch_extent.py tests/test_omni_reference_kernel.py` for exact
counts, planar scale invariance, independent scalar-layer parity, CUDA
derivatives, mixed-level end-to-end routing, checkpoint reconstruction, patch
boundary placement, and reference-kernel resizing. Run
`python scripts/check_omni_isotropic.py --batch-size 128 --output /tmp/omni-smoke`
to qualify the actual pretrained model and optimizer on a GPU before training.
Pass `--config-name` to qualify one of the new geodesic/RoPE variants.
