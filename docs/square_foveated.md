# Square-foveated sensing

`square_foveated` places samples on concentric square shells, with progressively
larger spacing toward the periphery. `square_foveated_as_grid` returns the same
samples as an upright image. Both require `fov_type='square'`: every native cell
is valid, including the corners. Existing sampling styles are unchanged.

```python
from fovi.sensing.coords import SamplingCoords

coords = SamplingCoords(
    fov=16.0, cmf_a=0.5, res=64,
    style="square_foveated_as_grid", fov_type="square",
)
image_coordinates = coords.as_grid(coords.cartesian, sample_dim=0)
```

Use the same style and coverage arguments with `RetinalTransform`. Its grid
output has shape `(batch, channels, resolution, resolution)`; the vector style
has shape `(batch, channels, resolution * resolution)`.

## Mapping

Let p be a native coordinate, s = max(|pₓ|, |pᵧ|), R = fov/2,
a = cmf_a, and b = max_val. The visual square radius is

g(s) = (a/R) × expm1((s/b) × log1p(bR/a)).

The visual coordinate is p × g(s)/s, with its analytic limit at the origin.
The inverse square radius is b × log1p(Rt/a) / log1p(bR/a), where t is the
visual square radius. `native_to_visual` and `visual_to_native` expose these
maps for tensors shaped `(..., 2)`, including coordinates outside the footprint
used for padding. Pixel centers lie inside the boundary; the continuous square
boundary maps exactly to itself. `max_val` scales both chart boundaries.

The native manifold and receptive-field distance coordinates are two-dimensional.
Polar coordinates still contain ordinary Euclidean eccentricity and angle.
The map preserves direction, but is not locally isotropic and has derivative
seams along the diagonals. The CMF depends on square radius, so magnification
is not constant over circles of equal eccentricity.

`planar` and `legacy` use the existing planar visual-chart convention.
`spherical` interprets visual-chart Euclidean radius as angular eccentricity;
the footprint is square in that chart, and its entire boundary must remain
below the antipode. A square chart is not necessarily a square footprint in
another camera projection. The native sensor remains a two-dimensional lattice.

## Comparison images

Run `scripts/render_sensor_fov_examples.py` from the repository root. It produces
the new square image alongside existing sensor examples, plus
`square_shell_comparison.png` showing mapped shells and sample locations against
the Wang footprint. Use `--output-dir` to choose an artifact directory.
