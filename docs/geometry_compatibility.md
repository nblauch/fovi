# Geometry compatibility evidence

The [full validation report](geometry_validation.md) compares six published
checkpoints on all 50,000 ImageNet validation images using the workspace uv
environment and source checkout overlays. Each comparison uses the same GPU,
checkpoint hashes, image preprocessing, batch size, fixation RNG seed, and
label ordering. This isolates the effect of the geometry correction under that
evaluation protocol; it does not establish accuracy for other datasets or
spherical inputs.

At 20 fixations, top-1 changes in percentage points are:

| Model | Corrected minus previous |
|---|---:|
| AlexNet RF multiplier 1 | -0.076 |
| AlexNet RF multiplier 2 | -0.244 |
| ResNet-18 | -0.414 |
| DINOv3-S+, a=2.78 | -0.104 |
| DINOv3-S+, a=60.94 | -0.004 |
| DINOv3-H+, a=2.78 | -0.002 |

The largest top-1 decrease across all tested counts is 0.738 points for ResNet-18
at five fixations. H+ remains within 0.1 points at every count. A second full
baseline run of AlexNet RF multiplier 2 reproduced its top-1 and top-5 results
exactly. Changes above 0.1 points are retained and reported.

## Activation diagnostics

The [machine-readable diagnostics](geometry_activation_diagnostics.json) compare
both AlexNet variants and both S+ variants using a natural image, its mirror,
and a noise image, with one and three explicit fixations. Input images,
fixations, and sampled retinal values are bitwise identical in all four cases.
Cortical features and logits change because the corrected manifold changes
discrete KNN neighborhoods and reference-kernel assignments. These checks cover
four checkpoints and a small input set; they do not prove universal input parity.

| Model | Single-fixation logit maximum absolute change | Logit RMSE |
|---|---:|---:|
| AlexNet RF multiplier 1 | 0.117721 | 0.003044 |
| AlexNet RF multiplier 2 | 0.129856 | 0.003689 |
| DINOv3-S+, a=2.78 | 0.067844 | 0.002216 |
| DINOv3-S+, a=60.94 | 0.138768 | 0.005237 |

Corrected planar geometry remains the default. Explicit `legacy` geometry
preserves the historical manifold, integration mesh, planar sampling rings, and
image-sampling behavior. A checkpoint selects it through its configuration:

```yaml
saccades:
  field_geometry: legacy
```

The loader respects that field without inferring geometry from artifact names or
hashes. Existing configs without it use the corrected planar default. Until
published configs carry the field, select legacy with an explicit override:

```python
model = get_model_from_base_fn(
    "fovi-resnet18_a-0.5_res-64_rfmult-2_in1k",
    **{"saccades.field_geometry": "legacy"},
)
```

Use `planar` explicitly to evaluate existing weights with corrected geometry.
New training configurations declare `planar`, so their saved configuration
records that choice. Updating the existing Hugging Face configs to declare
`legacy` is a separate publication step; this implementation does not modify
those remote artifacts.

A separate before/legacy comparison on the RTX 6000 Ada passed with zero
tolerance for both AlexNet variants and both S+ variants: all 56 captured tensors
(inputs, fixations, samples, intermediate layers, embeddings, and logits) were
bitwise equal for one and three fixations. Model state hashes also matched.
The capture uses the repository's street-view image, its mirror, and seeded
noise. Reproduce using `scripts/check_pretrained_parity.py capture`, supplying
`--field-geometry legacy` for the candidate, followed by `compare` with its
default zero tolerances. Full validation has not been rerun in legacy mode;
the accuracy table above remains the corrected-planar comparison.
