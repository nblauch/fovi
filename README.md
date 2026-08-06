# `fovi`

Welcome to the `fovi` codebase, a PyTorch library for implementing foveated vision. This library provides tools for foveated sampling and an interface to deep vision models, including CNNs and ViTs. 

We provide an interactive walkthrough of the methods and results at https://nblauch.github.io/fovi/

## 🛠️ Install

First, create a fresh conda environment:
```
conda create -n fovi python=3.9 # 3.9 is only necessary if using ffcv, see below
conda activate fovi
```

Clone the repo and enter it:
```
git clone https://github.com/nblauch/fovi.git
cd fovi
```

Now, for installing our package. The easiest installation is without `ffcv`, as `ffcv` rquires Python 3.9 and other harder dependencies. Installing without it will allow you to use everything in our code-base except the training functionality that leverages `ffcv`. If you want training functionality with `ffcv`, see below. You could also use your own training scripts with our models. 

For the easy install, with your new environment activated, just do:
```
# from within the fovi repo
pip install -e . # this will automatically install fovi/requirements.txt
```

To install with `ffcv` to allow fast training, we first follow the instructions to install `ffcv-ssl`, which has stricter requirements, and then install `fovi` and its requirements. With your `fovi` conda environment activated, do:
```
conda install pkg-config compilers libjpeg-turbo opencv pytorch torchvision torchaudio pytorch-cuda numba -c pytorch -c nvidia -c conda-forge
pip install git+https://github.com/facebookresearch/FFCV-SSL.git
# from within the fovi repo
pip install -e .
```

To use flash attention, install per the typical approach:
```
pip install packaging ninja
pip install flash-attn --no-build-isolation
```

## 🤗 Pretrained Models

Pretrained models are hosted on [HuggingFace Hub](https://huggingface.co/fovi-pytorch) and are automatically downloaded on first use:

| Model | Size | Description |
|-------|------|-------------|
| [`fovi-dinov3-hplus_a-2.78_res-64_in1k`](https://huggingface.co/fovi-pytorch/fovi-dinov3-hplus_a-2.78_res-64_in1k) | ~3.4 GB | ViT-H/16+ backbone, high foveation (a=2.78) |
| [`fovi-dinov3-splus_a-2.78_res-64_in1k`](https://huggingface.co/fovi-pytorch/fovi-dinov3-splus_a-2.78_res-64_in1k) | ~131 MB | ViT-S/16+ backbone, high foveation (a=2.78) |
| [`fovi-dinov3-splus_a-60.94_res-64_in1k`](https://huggingface.co/fovi-pytorch/fovi-dinov3-splus_a-60.94_res-64_in1k) | ~131 MB | ViT-S/16+ backbone, low foveation (a=60.94) |
| [`fovi-alexnet_a-0.5_res-64_rfmult-1_in1k`](https://huggingface.co/fovi-pytorch/fovi-alexnet_a-0.5_res-64_rfmult-1_in1k) | ~24 MB | AlexNet, high foveation (a=0.5), rfmult=1 (matched resolution kernel reference frame) |
| [`fovi-alexnet_a-0.5_res-64_rfmult-2_in1k`](https://huggingface.co/fovi-pytorch/fovi-alexnet_a-0.5_res-64_rfmult-2_in1k) | ~69 MB | AlexNet, high foveation (a=0.5), rfmult=2 (default higher-resolution kernel reference frame) |
| [`fovi-resnet18_a-0.5_res-64_rfmult-2_in1k`](https://huggingface.co/fovi-pytorch/fovi-resnet18_a-0.5_res-64_rfmult-2_in1k) | ~179 MB | ResNet18, high foveation (a=0.5), rfmult=2 |

```python
from fovi import get_model_from_base_fn

# Models are automatically downloaded from HuggingFace Hub on first use
model = get_model_from_base_fn('fovi-dinov3-splus_a-2.78_res-64_in1k')
```


## 📝 Example notebooks

`notebooks/step0_sensor_manifold` : explore the basic concepts involved in our foveated sensor

`notebooks/step1_sampling.ipynb` : learn how to do foveated sampling from images

`notebooks/step2_knnconv.ipynb` : learn how to build kNN-convolutional neural networks to process foveated sensor outputs

`notebooks/step3_dinov3.ipynb` : work with a state-of-the-art foveated vision system based on the DINOv3 ViT model, adapted to handle foveated inputs. 

`notebooks/step4_get_activations.ipynb`: use hooks to extract intermediate activations from a model, and explore the Trainer class

## 📚 Documentation

The docs are hosted at: https://nblauch.github.io/fovi/docs/

You can also build locally. Docs are generated semi-automatically from source code and docstrings. The documentation includes:

- **API Reference**: Complete documentation of all functions, classes, and modules
- **User Guide**: Installation, quickstart, and usage examples
- **Developer Guide**: Contributing guidelines and development setup

To do so:

```bash
# Install documentation dependencies
pip install -r requirements-docs.txt

# Generate documentation
python scripts/generate_docs.py

# View the documentation
open docs/_build/html/index.html

# View documentation on a remote cluster (need to forward the port separately, this is done automatically in VScode/Cursor)
python -m http.server 8000 --directory docs/_build/html
```

## ⚡ Benchmarking: optimized vs baseline

FOVI's **KNN convolution and KNN pooling** ship with optimized CUDA kernels (selected
automatically); this is the optimization under test.
The native CUDA convolution requires CUDA 12 and an Ampere-or-newer GPU. CuPy is installed
automatically with FOVI; older NVIDIA GPUs use the portable Torch/Warp fallback rather than
attempting to compile an unsupported native kernel. Python 3.9 installations resolve to
CuPy 13, preserving compatibility with FFCV; newer Python versions may use CuPy 14.
`benchmarks/benchmark_final_comparison.py` is the single entry point that measures what
they buy you — every FOVI model variant runs in two arms (`baseline` = the reference
conv/pool kernels, `optimized` = the shipped optimized conv/pool kernels, with
output-parity columns) against two clearly-labeled dense references:

- **logpolar@64** — the matched foveated *control*: the same fixations, retina, and
  augmentation feeding a standard Conv2d/ViT (a log-polar-warped 64x64 input, with the
  necessary circular padding) instead of KNNConv, so only the backbone differs from the
  foveated model — run one warped pass per fixation (matched sample count).
- **dense@256** — the *native-resolution* pipeline a non-foveated system needs
  (ResNet18/AlexNet/ViT-S+16 on the full 256x256 image), run exactly **once per image**:
  the foveated design trades one expensive full-res pass for a few cheap glances, so
  cells are labeled `(images, n_fixations)` and per-image columns are emitted so you can
  apply either normalization.

```bash
# from the repo root (defaults: all 5 variants, 10 & 128 images, 1 & 4 fixations,
# train + inference, both dense references). Write both report formats alongside:
python benchmarks/benchmark_final_comparison.py --device 0 \
    --report-out results.md --html-out results.html

# a quick look at one model:
python benchmarks/benchmark_final_comparison.py --models resnet18_rf1 --batch 10 --repeats 5

# render reports from already-collected JSON (one file per GPU), no re-benchmarking:
python benchmarks/benchmark_final_comparison.py \
    --report-from run_ada.jsonl run_h100.jsonl --html-out results.html
```

Output: one JSON-lines record per cell (timings under both protocols — CUDA-event
median/min and wall throughput — memory, parity vs the baseline arm, per-layer backend
routing), followed by a printed summary table with `xd@64` and `xd@256` speed ratios.
`--report-out` writes a human-readable Markdown summary; `--html-out` writes a
self-contained interactive page (select batch, fixations, train/inference, scope, and the
reference — logpolar@64 tracks the fixation count, dense@256 is always one native pass —
with color-coded speedup tables across all GPUs). `--report-from` renders either format
from existing JSON without re-running.
Useful knobs: `--cache-dir` points model loading at a local Hugging Face cache (offline
friendly); env vars `FOVI_KNN_BACKEND=baseline`, `FOVI_KNN_POOL_BACKEND=baseline`, and
`FOVI_KNN_WORK_THRESHOLD` override backend selection globally. Missing optional
dependencies (cupy/warp) degrade gracefully and are annotated in the output. The harness
itself is the reproducible evidence — run the commands above to regenerate every number
on your own hardware; final published results will live in the project's PR/release
notes.

### Manual optimization test gate

GPU CI is not currently enabled. Before merging changes to the optimized kernels or retinal
sampling path, run the complete gate manually on a CUDA 12 Ampere-or-newer machine. Install
the optional Warp backend when it is part of the change; without it, its tests report as
skipped.

```bash
pip install -e ".[warp]"
python -m unittest discover -s tests -p 'test_knn*.py' -v
python -m unittest discover -s tests -p 'test_retinal_sampling.py' -v
```

Set `FOVI_TEST_DEVICE=<index>` to select a particular GPU. The gate covers baseline and
automatic routing, forward/backward parity, FP16/BF16 autocast, fused convolution and
pooling, Warp, inference tensors, graph capture, and retinal-sampling equivalence.

## 🏛️ Citation
Blauch, N. M., Alvarez, G. A., & Konkle, T. (2026). FOVI: a biologically-inspired foveated interface for deep vision models. Proceedings of the 43rd International Conference on Machine Learning (ICML). https://arxiv.org/abs/2602.03766

## 🙏 Acknowledgements
Originally developed at the Kempner Institute at Harvard University. Ongoing support provided by NVIDIA.

<p align="left">
  <a href="https://kempnerinstitute.harvard.edu/"><img src="web/foveated-player/assets/kempner-logo.png" alt="Kempner Institute at Harvard University" height="46"></a>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://www.nvidia.com/"><img src="web/foveated-player/assets/nvidia-logo.svg" alt="NVIDIA" height="40"></a>
</p>
