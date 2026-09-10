# Pretrained inference and output parity

Install the `models` extra for pretrained inference. No trainer, FFCV installation, dataset,
or research storage environment variable is needed. See the [README example](https://github.com/nblauch/fovi#-pretrained-models)
and [package boundaries](package_boundaries.md).

## DINOv3 checkpoint compatibility

Transformers releases have placed DINOv3 blocks at `layer`, `encoder.layer`, and
`model.layer`. Fovi already supported locating these blocks for fine-tuning. Checkpoint
restoration now also translates the known block prefix to the constructed model's layout.
This happens inside the DINO backbone's PyTorch load hook, including when a parent FoviNet
loads its state dictionary.

The suffix is preserved, including LoRA `parametrizations.weight.original` and factor
`A`/`B` entries. No weights are merged, reinitialized, cast, or rewritten on disk. Saving a
model uses the active Transformers layout; loading recognizes all three layouts. This is
checkpoint-name compatibility, not a conversion of LoRA ranks, architecture configuration,
or optimizer state. The configuration must still construct the same model and adapters.

Strict loading continues to reject missing weights, unknown keys, and incompatible shapes.
A checkpoint mixing multiple block layouts is ambiguous and raises even when strict loading
is disabled. Non-DINO checkpoints are unaffected. Existing Hub revisions remain unchanged,
and the requirement remains `transformers>=4.57.6`.

## Refactor comparison

The original source at `2916774285399ea9d04026c65a828e9c1aef9923` was compared with
the refactored `fovi.models` public loader in separate processes. With Transformers 4.57.6,
all compared tensors matched exactly for these cached Hugging Face checkpoints:

| Model | Hub revision | Compared tensors, including input |
| --- | --- | --- |
| [DINOv3-S+, a=2.78](https://huggingface.co/fovi-pytorch/fovi-dinov3-splus_a-2.78_res-64_in1k) | `b9ba32e4c2ad4a8be3ce48bc358c7bbbb74e437a` | 13 |
| [DINOv3-S+, a=60.94](https://huggingface.co/fovi-pytorch/fovi-dinov3-splus_a-60.94_res-64_in1k) | `423d842bd550a3ed8c29b83d1a8f843218dfe39e` | 13 |
| [AlexNet, rfmult=1](https://huggingface.co/fovi-pytorch/fovi-alexnet_a-0.5_res-64_rfmult-1_in1k) | `4778acf941b58dc51e9822b1c451532ded1c2e5c` | 15 |
| [AlexNet, rfmult=2](https://huggingface.co/fovi-pytorch/fovi-alexnet_a-0.5_res-64_rfmult-2_in1k) | `6ec51cb04e7bbdef57244a8698348b78f3dfe7f9` | 15 |

The same comparison also passed exactly with the original source on Transformers 4.57.6
and the refactored source on Transformers 5.16.1 using the checkpoint compatibility hook.
Loaded parameter/buffer values matched exactly after normalizing the known DINO block
prefixes. Across each four-model comparison, all 56 captured tensors had zero maximum
absolute error. This includes four input tensors and 52 computed output/fixation tensors.

The inputs are the repository street-view image, its horizontal reflection, and seeded RGB
noise. Each batch is evaluated with one centered fixation and with three fixations spanning
center, off-center, and near-edge locations. Captures include retinal samples, intermediate
layers, embeddings, classifier logits, and realized fixation coordinates. Models run in
evaluation/inference mode; all captured values must be finite.

Validation used Python 3.12.3, PyTorch 2.11.0+cu128, and an NVIDIA RTX PRO 6000 Blackwell
GPU, with TF32 disabled. This is bounded inference coverage, not a training or accuracy
benchmark, and does not establish parity for every model, device, dtype, or stochastic policy.
The DINOv3-H+ and ResNet Hub models are outside this comparison.

## Reproduce

Use the same numerical dependencies in both processes. The original checkout needs its
legacy training dependencies just to import its loader; the new loader does not. Download
the four checkpoints to `~/.cache/fovi` first, pinning the revisions above when reproducing
this result. The capture command itself does not download missing checkpoints.

From the refactored checkout, with Transformers 4.57.6:

```bash
git worktree add --detach /tmp/fovi-v1 2916774285399ea9d04026c65a828e9c1aef9923
HF_HUB_OFFLINE=1 python scripts/check_pretrained_parity.py capture \
  --source-root /tmp/fovi-v1 --api legacy --image notebooks/streetview.jpg \
  --output /tmp/fovi-parity/reference
HF_HUB_OFFLINE=1 python scripts/check_pretrained_parity.py capture \
  --source-root . --api models --image notebooks/streetview.jpg \
  --output /tmp/fovi-parity/candidate
python scripts/check_pretrained_parity.py compare \
  --reference /tmp/fovi-parity/reference --candidate /tmp/fovi-parity/candidate \
  --report /tmp/fovi-parity/report.json
```

The default comparison requires exact equality. Captures record checkpoint-file hashes,
loaded-state hashes, image hashes, dependency versions, and Hub cache provenance. They also
assert that model inference did not import FFCV or the training namespace.

To test a Transformers upgrade separately, capture the candidate using the other version
and compare with `--allow-transformers-change`. This checks loaded tensors after normalizing
only the known DINO block prefixes; checkpoint and input provenance must still agree. It
does not relax numerical tolerances. Explicit `--rtol` and `--atol` options are available
for deliberate numerical comparisons and are recorded in the report.

Fast CPU regression tests need no Hub downloads or simulator:

```bash
python -m pytest tests/test_dinov3_checkpoint_compat.py tests/test_dinov3_compat.py \
  tests/test_package_boundaries.py tests/test_model_loading.py
```
