# Package boundaries and migration

`fovi` is one distribution. Install `fovi` for sensing and KNN layers, `fovi[models]` for
complete networks, and `fovi[training]` for model and training dependencies. `fovi[all]`
is exactly the union selected by `fovi[models,training]`; the independent `warp` extra
retains its previous meaning and is not implicitly included in `all`.

The model and training requirement files are reused when building extras metadata, so their
dependency lists have a single source of truth. The wheel includes every namespace, even
when its optional dependencies are not installed. Missing capability dependencies raise an
error containing the appropriate installation command.

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

Old architecture modules, `fovi.fovinet`, `fovi.probes`, and `fovi.hub` forward to the new
implementations. This preserves class identity and existing pickle/configuration paths.
Old training utility modules and `fovi.visualizer` similarly forward to their new locations.
Root exports such as `fovi.FoviNet` and `fovi.get_model_from_base_fn` are lazy compatibility
exports. `from fovi.trainer import load_config` still works without importing the trainer;
new code should import it from `fovi.models.loading`.

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
after moving any of those implementations. Training runtime checks require a working FFCV-SSL
native installation; installing only the models extra does not provide that runtime.

See [pretrained output parity](pretrained_parity.md) for the cross-checkout inference
procedure, and [versions and releases](releases.md) for the 1.0 baseline and pending 2.0 release.
