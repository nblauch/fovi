# Calibrated sampling performance

Measured on an NVIDIA RTX 6000 Ada with PyTorch 2.11.0+cu128. Inputs are uint8 RGB,
480×640, with 2,287 spherical retinal nodes spanning 60°. The baseline projects
the same nodes through the same lens once at central gaze, then translates that
canonical grid. Calibrated sampling unprojects each new fixation, rotates the
retinal rays, projects through the distorted camera model, and gathers pixels.
Both paths avoid expanding the source image to float.

The table shows warm bilinear sampling at normalized fixation (0.3013, 0.8017),
with times in milliseconds per batch. Graph columns measure CUDA graph replay
after capture. They are separate deployment options, not interchangeable timing
methods.

| Lens | Batch | Calibrated | Canonical | Calibrated graph | Canonical graph |
|---|---:|---:|---:|---:|---:|
| Pinhole + radial/tangential distortion | 1 | 0.2544 | 0.0199 | 0.0547 | 0.0037 |
| Pinhole + radial/tangential distortion | 32 | 0.2754 | 0.0198 | 0.0766 | 0.0069 |
| Pinhole + radial/tangential distortion | 256 | 0.2896 | 0.0368 | 0.1983 | 0.0357 |
| Fisheye + angular polynomial | 1 | 0.1237 | 0.0202 | 0.0221 | 0.0034 |
| Fisheye + angular polynomial | 32 | 0.1387 | 0.0198 | 0.0400 | 0.0070 |
| Fisheye + angular polynomial | 256 | 0.1551 | 0.0359 | 0.1386 | 0.0354 |

The calibrated path uses `torch.compile`; the canonical path uses the existing
native CUDA sampler. Calibrated output is checked against the eager calibrated
implementation before timing. The canonical approximation intentionally samples
different rays away from central gaze; this is a speed comparison, not a parity
claim. The pinhole inverse uses an iterative solve, which costs more than the
fisheye case in this configuration.

First calls in this run took 0.44–1.79 seconds, excluded from the table. These
include initialization and compilation/cache loading; a cold compiler cache can
take longer. CUDA graph replay requires fixed shapes and stable storage. These
measurements cover sampling only, excluding image acquisition, preprocessing,
model inference, and simulation rendering. Latency is sensitive to GPU load and
launch overhead.

The [raw results](../benchmarks/results/calibrated_sampling_ada.json) include
nearest and bilinear modes, central and peripheral gaze, calibration parameters,
all batch sizes, and backend labels. Reproduce with:

```bash
python benchmarks/benchmark_calibrated_sampling.py --device cuda:0 --output results.json
```
