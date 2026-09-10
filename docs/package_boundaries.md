# Package boundaries and migration

`fovi` is one distribution. Install `fovi` for sensing and KNN layers, `fovi[models]` for
complete networks, and `fovi[training]` for model and training dependencies without FFCV.
`fovi[ffcv]` adds the native FFCV-SSL loader dependency. `fovi[all]`
includes every optional dependency, currently the union selected by
`fovi[models,training,ffcv]`. CuPy and Warp are base dependencies, so all
installation variants include the optimized kernel libraries.

The model and training requirement files are reused when building extras metadata, so their
dependency lists have a single source of truth. The wheel includes every namespace, even
when its optional dependencies are not installed. Missing capability dependencies raise an
error containing the appropriate installation command.

The models extra requires PyTorch 2.5 or later for the public state-dict
pre-load hook used by DINO checkpoint compatibility. This is an API minimum;
the validated full dependency environment is recorded in the parity report.

## Imports

| Capability | Public location |
| --- | --- |
| Coordinates, retinal transforms, image samplers, fixation policies | `fovi.sensing` |
| KNN convolution/pooling and CUDA/Warp kernels | Existing `fovi.arch.knn*` primitive modules |
| Shared MLP, normalization, padding, wrapper primitives | Existing `fovi.arch.mlp`, `norm`, `polar`, `wrapper` |
| AlexNet, ResNet, ConvNeXt, ViT, DINOv3 and their foveated variants | `fovi.models` modules, such as `fovi.models.knnresnet` |
| FoviNet and architecture registry | `from fovi.models import FoviNet, ARCHITECTURE_REGISTRY` |
| Model configuration, checkpoints, inference construction | `fovi.models.loading` |
| Hub downloads | `fovi.models.hub` |
| Trainer and training restoration | `fovi.training` |
| FFCV loader | `fovi.training.loader` |
| Losses, schedules, KNN probes, FLOP analysis, backup | `fovi.training.utils` modules |
| Experiment visualizer | `fovi.training.visualizer` |

Shared numerical and image transforms remain in `fovi.utils` and `fovi.utils.fastaugs`.
Importing those transforms no longer attempts to load FFCV. Plotly/video helpers import their
optional dependencies only when called.

Pure-Torch losses and schedulers can be imported without the training extra.
Each helper requires only the libraries it uses. Importing `Trainer` requires
the training dependencies and research paths, but does not import FFCV.
Its built-in `create_train_loader` and `create_val_loader` methods require
`fovi[ffcv]` and raise with an installation command when it is missing.
Training without FFCV requires external training code or a Trainer subclass
that supplies both loaders; selecting `training` does not introduce a new
automatic data-loading backend.

For validation or activation extraction, set `training.eval_only=True` and
`data.train_dataset=None`. The Trainer creates only the validation loader and
restores weights without optimizer state. Calling `train()` in this mode raises.

Complete-model imports must use `fovi.models`. The old architecture modules,
`fovi.fovinet`, `fovi.probes`, `fovi.hub`, and root model exports have been removed.
For example, replace `fovi.arch.knnvit` with `fovi.models.knnvit`, import `FoviNet`
from `fovi.models`, and import checkpoint helpers from `fovi.models.loading`.
KNN primitives such as `fovi.arch.knn` remain in their existing locations.

This is a Python import-path change, not a checkpoint-key change. Published Hub
state dictionaries and the DINO/LoRA compatibility hook do not depend on these
aliases. Old scripts, dotted configuration targets, and pickled whole-model objects
using the removed paths need migration; prefer saving model state dictionaries.

Old training utility modules, root trainer exports, and `fovi.visualizer` still
forward to their new locations. `from fovi.trainer import load_config` also remains
available; new code should import it from `fovi.models.loading`.

Prefer explicit imports. `from fovi import *` resolves the trainer and needs the
training dependencies and configured research paths. A plain `import fovi`
does not resolve those exports.

Incidental imports from the old root are not public re-exports: import
`HiddenPrints` from `fovi.utils`, storage paths from `fovi.paths`, and `OmegaConf`
from `omegaconf`. Import submodules explicitly rather than assuming that
`import fovi` populates every `fovi.arch` attribute. `fovi.trainer.get_relative_path`
was an internal loader helper and has been removed. Request `FlashLoader`
explicitly from `fovi.training.loader`; wildcard imports of image transforms
do not pull in a data loader.

## Inference configuration and checkpoints

```python
from fovi.models import get_model_from_base_fn
from fovi.models.loading import load_config

cfg, checkpoint, key = load_config("my_model", load=True, folder="/models", device="cpu")
model = get_model_from_base_fn("my_model", model_dirs=["/models"], device="cpu")
```

The loader preserves Hydra YAML, standalone Hydra configuration, and legacy `params.json`
formats; sharded checkpoints, `state_dict.pth`, `final_weights.pth`, and `model.pth` retain
their precedence and state keys. Distributed `module.` prefixes are removed during inference
restoration. The requested device applies to checkpoint tensors as well as model construction.

DINOv3 restores the known Transformers `layer`, `encoder.layer`, and `model.layer`
checkpoint prefixes to the active layout, including LoRA parametrizations. Shape errors,
missing keys, and unknown keys still fail strict loading. See the
[checkpoint compatibility details](pretrained_parity.md).

Inference does not import `fovi.paths`, build a trainer, initialize datasets, or configure
experiment tracking. Reading legacy JSON no longer creates a Hydra YAML file beside it.
An existing but broken local model now raises its actual error instead of silently trying a
different checkpoint from the Hub. If no local model exists, Hub download remains supported.

`find_config` defaults to `../models` and appends configured `FOVI_SAVE_DIR/logs` and
`FOVI_SLOW_DIR/logs` directories if present. `get_model_from_base_fn` keeps its existing
default search location of `../models`. Explicit search directories take precedence.

## Development validation

```bash
python -m pytest tests/test_package_boundaries.py tests/test_model_loading.py
uv build --wheel
python scripts/check_distribution.py dist/fovi-*.whl
```

Boundary tests start fresh processes without research environment variables and reject
imports across the sensing/model/training boundaries. Loading tests restore local CPU
checkpoints without network access or training. Run the existing sensing and model tests
after moving any of those implementations. Built-in loader runtime checks require a working FFCV-SSL
native installation; installing only the models extra does not provide that runtime.

See [pretrained output parity](pretrained_parity.md) for the cross-checkout inference
procedure, and [versions and releases](releases.md) for the 1.0 baseline and pending 2.0 release.
