# Geometry validation

Full ImageNet validation comparisons against the pre-correction source (1f5fd27).
Each pair uses the same GPU, checkpoint files, preprocessing, batch size, and
per-batch fixation RNG seed. Checkpoint hashes and label ordering are checked.
Inputs use centered square crops decoded to each model's configured resolution,
ImageNet normalization, the configured AMP mode, and the configured fixation policy.
These comparisons measure a geometry change; they are not a new training run.

The corrected planar geometry is used for every candidate below. Differences
above 0.1 percentage points are shown explicitly; checkpoint neighborhoods have
not been restored to their historical values. New models default to planar
geometry. The explicit legacy option is available for older checkpoints;
the measurements below use corrected planar geometry.

| Model | Fixations | Top-1 before | Top-1 corrected | Δ pp | Top-5 before | Top-5 corrected | Δ pp |
|---|---:|---:|---:|---:|---:|---:|---:|
| fovi-alexnet_a-0.5_res-64_rfmult-1_in1k | 20 | 45.404 | 45.328 | -0.076 | 69.908 | 69.840 | -0.068 |
| fovi-alexnet_a-0.5_res-64_rfmult-2_in1k | 20 | 48.006 | 47.762 | -0.244 | 72.022 | 71.854 | -0.168 |
| fovi-dinov3-splus_a-2.78_res-64_in1k | 1 | 69.058 | 68.846 | -0.212 | 89.178 | 89.006 | -0.172 |
| fovi-dinov3-splus_a-2.78_res-64_in1k | 2 | 71.654 | 71.512 | -0.142 | 90.270 | 90.148 | -0.122 |
| fovi-dinov3-splus_a-2.78_res-64_in1k | 3 | 72.958 | 72.742 | -0.216 | 91.000 | 90.926 | -0.074 |
| fovi-dinov3-splus_a-2.78_res-64_in1k | 5 | 74.246 | 74.068 | -0.178 | 91.724 | 91.724 | +0.000 |
| fovi-dinov3-splus_a-2.78_res-64_in1k | 10 | 75.274 | 75.158 | -0.116 | 92.572 | 92.502 | -0.070 |
| fovi-dinov3-splus_a-2.78_res-64_in1k | 20 | 75.848 | 75.744 | -0.104 | 93.094 | 93.032 | -0.062 |
| fovi-dinov3-splus_a-60.94_res-64_in1k | 1 | 65.076 | 64.984 | -0.092 | 86.372 | 86.354 | -0.018 |
| fovi-dinov3-splus_a-60.94_res-64_in1k | 2 | 67.466 | 67.590 | +0.124 | 87.760 | 87.636 | -0.124 |
| fovi-dinov3-splus_a-60.94_res-64_in1k | 3 | 68.872 | 68.792 | -0.080 | 88.616 | 88.532 | -0.084 |
| fovi-dinov3-splus_a-60.94_res-64_in1k | 5 | 70.066 | 70.044 | -0.022 | 89.294 | 89.294 | +0.000 |
| fovi-dinov3-splus_a-60.94_res-64_in1k | 10 | 71.174 | 71.178 | +0.004 | 90.212 | 90.130 | -0.082 |
| fovi-dinov3-splus_a-60.94_res-64_in1k | 20 | 71.786 | 71.782 | -0.004 | 90.706 | 90.698 | -0.008 |
| fovi-resnet18_a-0.5_res-64_rfmult-2_in1k | 1 | 33.098 | 32.610 | -0.488 | 55.732 | 54.928 | -0.804 |
| fovi-resnet18_a-0.5_res-64_rfmult-2_in1k | 2 | 39.186 | 38.740 | -0.446 | 61.974 | 61.278 | -0.696 |
| fovi-resnet18_a-0.5_res-64_rfmult-2_in1k | 3 | 42.236 | 41.694 | -0.542 | 64.822 | 64.286 | -0.536 |
| fovi-resnet18_a-0.5_res-64_rfmult-2_in1k | 5 | 45.498 | 44.760 | -0.738 | 67.674 | 67.222 | -0.452 |
| fovi-resnet18_a-0.5_res-64_rfmult-2_in1k | 10 | 48.632 | 48.018 | -0.614 | 70.810 | 70.246 | -0.564 |
| fovi-resnet18_a-0.5_res-64_rfmult-2_in1k | 20 | 50.364 | 49.950 | -0.414 | 73.064 | 72.548 | -0.516 |
| fovi-dinov3-hplus_a-2.78_res-64_in1k | 1 | 83.736 | 83.720 | -0.016 | 96.520 | 96.500 | -0.020 |
| fovi-dinov3-hplus_a-2.78_res-64_in1k | 2 | 84.540 | 84.558 | +0.018 | 96.960 | 96.956 | -0.004 |
| fovi-dinov3-hplus_a-2.78_res-64_in1k | 3 | 84.950 | 84.962 | +0.012 | 97.150 | 97.144 | -0.006 |
| fovi-dinov3-hplus_a-2.78_res-64_in1k | 5 | 85.534 | 85.452 | -0.082 | 97.386 | 97.378 | -0.008 |
| fovi-dinov3-hplus_a-2.78_res-64_in1k | 10 | 85.898 | 85.954 | +0.056 | 97.648 | 97.628 | -0.020 |
| fovi-dinov3-hplus_a-2.78_res-64_in1k | 20 | 85.994 | 85.992 | -0.002 | 97.764 | 97.760 | -0.004 |

## Execution

- fovi-alexnet_a-0.5_res-64_rfmult-1_in1k: NVIDIA RTX 6000 Ada Generation; PyTorch 2.11.0+cu128; 50000 images; batch 16; seed 2026.
- fovi-alexnet_a-0.5_res-64_rfmult-2_in1k: NVIDIA RTX 6000 Ada Generation; PyTorch 2.11.0+cu128; 50000 images; batch 16; seed 2026.
- fovi-dinov3-splus_a-2.78_res-64_in1k: NVIDIA RTX 6000 Ada Generation; PyTorch 2.11.0+cu128; 50000 images; batch 16; seed 2026.
- fovi-dinov3-splus_a-60.94_res-64_in1k: NVIDIA RTX 6000 Ada Generation; PyTorch 2.11.0+cu128; 50000 images; batch 16; seed 2026.
- fovi-resnet18_a-0.5_res-64_rfmult-2_in1k: NVIDIA RTX 6000 Ada Generation; PyTorch 2.11.0+cu128; 50000 images; batch 16; seed 2026.
- fovi-dinov3-hplus_a-2.78_res-64_in1k: NVIDIA RTX PRO 6000 Blackwell Workstation Edition; PyTorch 2.11.0+cu128; 50000 images; batch 16; seed 2026.

Use `scripts/validate_pretrained_geometry.py` in each source checkout, then
`scripts/summarize_geometry_validation.py` to reproduce this comparison.

The accompanying JSON contains checkpoint hashes and unrounded metrics.
