# The radius norm: circular and square shells

The warped-Cartesian sensor lays a uniform lattice in a *native* plane and pushes
each sample outward along its own ray by a radial cortical magnification law.
`radius_norm` selects which norm measures radius in that native plane, and so
what shape the iso-eccentricity shells take:

| `radius_norm` | native radius | shells | pairs with |
| --- | --- | --- | --- |
| `2.0` (default) | `‖p‖₂` | concentric circles | `fov_type='circular'` |
| `math.inf` | `max(\|pₓ\|, \|pᵧ\|)` | concentric squares | `fov_type='square'` |

Both settings are the *same* sensor family running the same magnification law.
Along the four axes they are the same one-dimensional map; they can only differ
off-axis. What the infinity norm buys is full square coverage: the native square
maps exactly onto the visual square, so every cell is valid including the
corners, and nothing has to be masked away. Under the Euclidean norm the native
corners overshoot the square FoV and get masked.

```python
import math

from fovi.sensing.coords import SamplingCoords

coords = SamplingCoords(
    fov=16.0, cmf_a=0.5, res=64,
    style="warped_cartesian_as_grid", fov_type="square",
    radius_norm=math.inf,
)
image_coordinates = coords.as_grid(coords.cartesian, sample_dim=0)
```

`radius_norm` is accepted anywhere `fov_type` is, including `RetinalTransform`
and the `saccades` config block (`radius_norm: .inf` in YAML). The `_as_grid`
style returns the same samples as an upright image, shaped
`(batch, channels, resolution, resolution)`; the vector style is
`(batch, channels, resolution * resolution)`.

`fov_type='wang'` normalizes the *Euclidean* radius at the native square's side
centers, so it has no infinity-norm counterpart and is rejected with
`radius_norm=inf`.

## Mapping

Let p be a native coordinate, R = fov/2, a = cmf_a, b = max_val, and let s be
the native radius measured in `radius_norm`. The visual radius is

g(s) = (a/R) × expm1((s/b) × log1p(bR/a)),

and the visual coordinate is p × g(s)/s, with its analytic limit at the origin.
The inverse is b × log1p(Rt/a) / log1p(bR/a) for visual radius t.
`native_to_visual` and `visual_to_native` expose both directions for tensors
shaped `(..., 2)`, including coordinates outside the footprint used for padding.

Pixel centers lie inside the boundary. Under `radius_norm=inf` the continuous
square boundary maps exactly to itself.

`max_val` is the maximum radius in units of `fov/2`, measured in the norm that
`fov_type` selects, and it is also the half-extent of the native lattice — the
normalization that makes those two coincide. One asymmetry is worth knowing:
under `radius_norm=inf`, `max_val` enters the normalizer, so changing it
rescales the CMF; under `radius_norm=2.0` it does not. The two norms therefore
agree along the axes exactly when `max_val = 1`, which is the value used
throughout.

The native manifold and receptive-field distance coordinates are two-dimensional.
Polar coordinates always contain ordinary Euclidean eccentricity and angle,
whichever norm drives the warp.

### What the square shells cost

The map preserves direction but is not locally isotropic. Under
`radius_norm=inf` the CMF is exact only along the axes: differentiating along a
diagonal ray gives an effective foveal constant of √2·a rather than a, so the
fovea is about 1.41× coarser on the diagonals, converging to parity in the far
periphery. Magnification is therefore not constant over circles of equal
Euclidean eccentricity — it is constant over *squares*.

`max` is also not differentiable where `|pₓ| = |pᵧ|`, so the Jacobian has
seams along the diagonals, and lattice points land on those seams at every
resolution. The forward/inverse pair stays exact there; only the derivative is
one-sided.

## Field geometry

`planar` and `legacy` use the existing planar visual-chart convention.
`spherical` interprets visual-chart Euclidean radius as angular eccentricity.
Under `radius_norm=inf` the footprint is square in that chart and its entire
boundary must stay below the antipode — the corners reach √2 · max_val · fov/2,
which is what the geometry check enforces. A square chart is not necessarily a
square footprint in another camera projection. The native sensor remains a
two-dimensional lattice.

## Comparison images

Run `scripts/render_sensor_fov_examples.py` from the repository root. It renders
both radius norms alongside the other sensor examples, plus
`square_shell_comparison.png` showing mapped shells and sample locations against
the Wang footprint. Use `--output-dir` to choose an artifact directory.

## Related work

The two halves of this construction — a CMF-derived radial warp, and square
iso-eccentricity contours — each have precedent, but we did not find prior work
combining them, or treating the radius norm as a parameter.

**Square and max-norm foveation.** Martínez & Robles (2006), *A New Foveal
Cartesian Geometry Approach Used for Object Tracking* (SPPRA), is the closest
prior art: it samples concentric squares around the fovea so the result "fits
perfectly into a rectangular shape with no gaps," with distortion confined to
the diagonals. It is a discrete, piecewise-linear approximation to log-polar
rather than a continuous CMF-derived warp. Lukanov, König & Pipa (2021,
*Front. Comput. Neurosci.* 15:746204) build the deep-learning instantiation of
that geometry and report it outperforming log-polar foveation at matched
budgets. Shah & Raj (2023, *Training on Foveated Images Improves Robustness to
Adversarial Attacks*, NeurIPS; arXiv:2308.00854) use an explicit Chebyshev
eccentricity `max(|Δx|, |Δy|)/W` for its "un-rotated square level sets," though
theirs is a blur-and-desaturate foveation rather than a resampling warp. On the
graphics side, Li, Du, Babu, Brumar & Varshney (2021, *A Log-Rectilinear
Transformation for Foveated 360-degree Video Streaming*, IEEE TVCG 27(5)) replace
log-polar with a separable per-axis log warp, giving axis-aligned rectangular
iso-contours and removing log-polar's corner waste; NVIDIA's VRWorks Multi-Res
Shading and Lens Matched Shading are the shipped equivalents.

**Circle–square geometry.** Shirley & Chiu (1997), *A Low Distortion Map Between
Disk and Square* (JGT 2(3)), is the canonical concentric-square map, and is
itself an infinity-norm radial construction: it preserves the max-norm radius
and remaps angle. Fong's *Analytical Methods for Squaring the Disc*
(arXiv:1509.06344), *Squircular Calculations* (arXiv:1604.02174), and
*Elliptification of Rectangular Imagery* (arXiv:1709.07875) use the
Fernández-Guasti squircle, whose squareness parameter interpolates circle to
square continuously with a cheaper closed-form inverse than a Lamé curve —
relevant if `radius_norm` ever becomes continuous rather than a choice of two.
The Lamé superellipse is the textbook statement that the Lp unit ball is a
circle at p = 2 and a square as p → ∞.

**Other magnification laws.** Our inverse-linear CMF M(e) ∝ 1/(e + a) is what
connects this sensor to log-polar; it is not the only option. Meng, Du, Zwicker
& Varshney (2018), *Kernel Foveated Rendering* (I3D / PACM CGIT 1(1)), embed a
polynomial kernel x^α inside the log-polar map and sweep α, the clearest
existing precedent for treating the magnification law itself as a
hyperparameter. Zhang et al. (2024), *Retinotopic Foveated Rendering* (IEEE VR),
derive a radially *asymmetric* CMF from fMRI retinotopy — an orthogonal axis of
generalization to ours. Killick, Henderson, Siebert & Aragon-Camarasa (2023),
*Foveation in the Era of Deep Learning* (BMVC), use a piecewise square-root-then-
geometric radial law on a sunflower lattice. Deza & Konkle (2020), *Emergent
Properties of Foveated Perceptual Systems* (arXiv:2006.07991), treat the
receptive-field growth rate as the experimental variable.

Finally, note that FOVEA (Thavamani et al., ICCV 2021) and the saliency sampler
(Recasens et al., ECCV 2018) both implement their learned warps separably in x
and y, so their effective iso-magnification contours are already axis-aligned
rectangles. Neither remarks on it; the infinity-norm formulation is what makes
that geometry explicit.
