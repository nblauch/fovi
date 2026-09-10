# Running the tutorials

From the repository root, install the model and notebook dependencies into your
chosen environment:

```bash
pip install -e '.[models]'
pip install jupyterlab nbconvert ipykernel matplotlib cmasher pandas seaborn ipywidgets
cd notebooks
jupyter lab
```

Select a kernel from that environment and run each notebook from top to bottom.
Keep the working directory at `notebooks/`: the image examples use the bundled
`shark.png` and `streetview.jpg`, and the model construction example uses `../config`.

- `step0_sensor_manifold` and `step1_sampling` use base `fovi` sensing functions
  plus plotting dependencies; no model or training extra is needed.
- `step2_knnconv` and `step3_dinov3` use `fovi[models]`. KNN primitives remain in
  `fovi.arch`; complete architectures and inference loaders live in `fovi.models`.
  These notebooks use CPU and download published checkpoints from Hugging Face.
- `step4_get_activations` starts with model-only inference on CUDA. Its Trainer
  section additionally needs `fovi[training]`, a manual FFCV-SSL installation,
  and the ImageNet-1K validation FFCV file.
  Training utilities live in `fovi.training`.

For the Trainer section, install `pip install -e '.[training]'` and follow
[the manual FFCV instructions](../README.md#manual-ffcv-installation). Set storage
paths **before starting the notebook kernel**:

```bash
export FOVI_SAVE_DIR=/path/to/experiment-storage
export FOVI_DATASETS_DIR=/path/to/datasets
```

The example expects `ffcv/imagenet/val_compressed.ffcv` under the dataset directory.
It sets `training.eval_only=True`, so no training dataset or optimizer is created.
Edit the explicit
dataset overrides in the notebook if your layout differs. Model-only loading
does not require these paths or FFCV. `fovi[all]` installs the combined `models` and
`training` dependencies; FFCV is installed manually. Both CuPy and Warp are already
dependencies of plain `fovi`.

To execute and save outputs from the command line, use the same kernel environment:

```bash
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 step0_sensor_manifold.ipynb
```

Repeat for the other notebooks. The documentation build copies these executed
notebooks into `docs/api`; Sphinx renders their saved outputs without executing them.
