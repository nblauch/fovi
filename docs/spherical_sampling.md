# Calibrated spherical sampling

`field_geometry` selects the visual metric independently of `fov_type`, which
selects the field boundary. Both sampling density and the cortical manifold use
the chosen visual metric. The cortical magnification function remains
M(r) = k / (cmf_a + r).

With `planar`, jointly scaling `fov` and `cmf_a` leaves the normalized sampling
geometry unchanged. Only `cmf_a / fov` matters. With `spherical`, curvature also
depends on the actual angular `fov` in degrees; do not normalize it to one.
Convert a normalized parameter with `cmf_a = cmf_a_frac * fov` once, after
resolving the angular window. Isotropic angular sampling supports spherical
geometry; flattened log-polar and warped Cartesian grid layouts do not.

`field_geometry="legacy"` is an explicit compatibility option for checkpoints
trained with the earlier geometry. It preserves planar sampling rings together
with the historical manifold and its numerical integration. New configurations
default to `planar`; there is no automatic checkpoint-name or hash detection.
Legacy is not the calibrated spherical pipeline. Preserve the checkpoint's
original FoV and CMF settings when using it.

## Camera calibration and windows

`CameraModel` supports pinhole projection with OpenCV radial/tangential distortion
and equidistant-polynomial fisheye projection. Calibration describes the source
image after any image preprocessing. Pixels use integer centers, with the first
pixel centered at `(0, 0)`. Directions use X right, Y down, Z forward.
`image_size` is `(height, width)` and `intrinsics` is `(fx, fy, cx, cy)`.
An optional image circle and maximum angle restrict the usable source domain.
The supported angular domain is the front hemisphere.

```python
from fovi.sensing.projection import CameraModel
from fovi.sensing.retina import RetinalTransform

camera = CameraModel(
    model="fisheye",
    image_size=(480, 640),
    intrinsics=(300.0, 300.0, 319.5, 239.5),
    distortion=(0.02, -0.003, 0.0002, 0.0),
)
fov = camera.field_of_view("long", fraction=0.75)
retina = RetinalTransform(
    resolution=64,
    fov=fov,
    cmf_a=0.03 * fov,
    field_geometry="spherical",
    camera_model=camera,
    gaze_convention="pan_tilt",
    sampler="grid_nn",
    device="cuda",
)
# images: (B, C, H, W); fixation: (B, 2), normalized (row, column).
samples = retina(images, fixation)
```

The long or short side is selected using the source raster dimensions. A camera
with horizontal FoV 60° and vertical FoV 40° uses 60° with long-side calibration
when its raster is wider than it is tall. Partial windows unproject their actual
pixel endpoints; multiplying the full angular FoV by the pixel fraction is
generally incorrect. `CameraModel.resized()` handles full-frame image resizing
with the pixel-center convention preserved. Cropping and rotation require
appropriately transformed calibration supplied by the caller.

## Saccades and sampling

A fixation unprojects to a source-camera direction, defines a zero-torsion gaze
rotation, rotates the canonical retinal rays, and projects them through the
same source calibration. `pan_tilt` uses pan followed by tilt. `camera_xyz` uses
the camera XYZ rotation convention. `GridSampler` also accepts an explicit
`(B, 3, 3)` rotation mapping retinal-camera directions into source-camera space.
This supports calibrated off-center software fixation without changing the
physical camera pose or generating a dense rectified image.

Spherical retinal extent is fixed by `fov`. Per-call pixel `fixation_size`
overrides on `RetinalTransform` are rejected. The random saccade policy uses its
crop-size distribution to choose fixation locations, while keeping the angular
retinal extent fixed. Samples outside the calibrated image/domain are zero;
bilinear interpolation uses zero padding for missing neighboring pixels.

Sampling gathers only the compact output from uint8 images. The CUDA path fuses
rotation, calibrated projection, and gathering with `torch.compile`; first-use
compilation must be excluded from steady-state measurements. Use
`benchmarks/benchmark_calibrated_sampling.py` to compare calibrated dynamic gaze
against translation of a single canonical grid for the same sample count.
See [measured sampling performance](calibrated_sampling_performance.md) for
latency, first-call costs, and CUDA graph replay results.
Changing the gaze changes the source pixels; a translated central grid is an
approximation whose error grows with eccentricity and lens distortion.

Camera system identification, simulation rendering, optical blur, and camera
mount dynamics belong to external libraries. Matching ideal rays does not imply
pixel-level equivalence between direct rendering and sampling a rasterized,
filtered camera image.
