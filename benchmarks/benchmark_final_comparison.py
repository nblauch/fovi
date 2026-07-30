"""Definitive FOVI-vs-dense comparison harness: one entry point, six model
variants, two execution arms, two dense-reference framings.

Variants: alexnet_rf2 (shipped fovi-alexnet_a-1_res-64_rfmult-2_in1k), alexnet_rf1
(cached rf1 checkpoint), resnet18_rf1 / resnet18_rf2 (local bench configs), dinov3
(=DINOv3-S+, fovi-dinov3-splus_a-2.78_res-64_in1k), dinov3_hplus (=DINOv3-H+,
fovi-dinov3-hplus_a-2.78_res-64_in1k). Each dinov3 size is compared against a dense
reference built from the SAME backbone (S+ vs H+ ViT); reports label them s+/h+.

The optimization under test is the KNN convolution + KNN pooling CUDA kernels.
Arms (per variant):
- baseline: the reference KNN conv/pool kernels (FOVI_KNN_BACKEND=baseline,
  FOVI_KNN_POOL_BACKEND=baseline). To reconstruct the full pre-optimization timing it
  also runs the original full-image retinal sampling (fast_pre_transforms disabled) —
  a separate, bit-exact front-end reimplementation, NOT part of the kernel optimization.
- optimized: the shipped optimized conv/pool kernels (auto backend), with the bit-exact
  retinal fast path. The header records which kernel backends were importable (cupy/warp
  absent => the auto policy silently degrades; arms annotate availability).
So the baseline->optimized speedup is the conv/pool kernel optimization plus that
front-end sampling reimplementation (the latter dominates only where KNN work is tiny,
e.g. dinov3); the kernel optimization is the subject of these reports.

References (per model family, both emitted and labeled):
- logpolar@64 (matched foveated CONTROL): the SAME foveated design as the fovi arm, built
  as a FoviNet control model — identical fixation policy, retina, fused augmentation, and
  head — differing ONLY in that its griddable ``logpolar_as_grid`` glances (a 64x64 warped
  image, circular angular padding retained) feed a standard Conv2d/ViT instead of KNNConv.
  The dense backbone is matched to the fovi one: alexnet shares the 'base_lowres' kernel
  ladder; resnet uses torchvision-standard strides matched to the (cartesian-matched)
  KNNResNet node ladder (input ~4085 -> 964/230/60/16); dinov3 patch-embeds to the same 64
  tokens. It is timed in both scopes (network = backbone only; full_fovinet = the whole
  step), so the report's scope toggle stays symmetric with the fovi arms and the full-step
  comparison has no front-end residual. One warped pass per fixation (network batch =
  images x n_fix).
- dense@256 (native resolution): what a NON-foveated pipeline pays to process the full
  256x256 image — the same matched resnet18 (native, non-polar) at 256; the repo's dense
  alexnet 'base' spec (canonical k11/s4 stem) at 256; the same DINOv3 ViT-S+ trunk with
  patch 16 at 256 (16x16 = 256 tokens, same freeze/LoRA recipe as the foveated config).

PROTOCOL / NORMALIZATION (prominent by design):
- BATCH = NUMBER OF IMAGES, fixed across ALL cells; every cell is labeled
  (images, n_fixations) explicitly. At (128, 4) the low-resolution networks internally
  process 512 samples while dense@256 processes 128 — that asymmetry IS the comparison,
  not a confound: the foveated design trades one expensive full-resolution pass for
  several cheap glances, and dense@256 is by nature exactly ONE pass per image.
- n_fixations in {1, 4} for the fovi arms AND logpolar@64 (the n_fixations=1 cells are the
  cleanest per-glance statement: one foveated glance vs one downsampled pass vs one
  native full-res pass). Records carry `ms_per_image`, and the summary table both the
  per-image xdense@256 and a per-forward-sample column (xd256/sample = xdense@256 /
  n_fixations), so either convention can be applied.

Timing: both established protocols per record — CUDA-event median/min and the
wall-throughput protocol (back-to-back iterations, single sync). Train rows use the
models-harness protocol (autocast with each model's configured AMP dtype, GradScaler for
fp16 configs, pseudo-loss network scope / cross-entropy full scope, optimizer excluded);
parity between arms is loss + unscaled grad-sample max_abs via the dedicated
small-init-scale scaler (see benchmark_knn_conv_models.PARITY_INIT_SCALE).

Output: JSON-lines records (stdout) followed by a human-readable summary table.
Checkpoints resolve through the Hugging Face cache; pass --cache-dir (sets HF_HOME /
HUGGINGFACE_HUB_CACHE) or pre-set those env vars for offline use.

Run from the repo root:
    python benchmarks/benchmark_final_comparison.py --batch 10 128 --device 0
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
from functools import partial

import torch  # torch before cupy, always
from einops import rearrange
from torch.amp import GradScaler, autocast

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SEED = 20260721

VARIANTS = {
    "alexnet_rf2": ("hf", "fovi-alexnet_a-1_res-64_rfmult-2_in1k"),
    "alexnet_rf1": ("hf", "fovi-alexnet_a-1_res-64_rfmult-1_in1k"),
    "resnet18_rf1": ("local", "resnet18_rf1"),
    "resnet18_rf2": ("local", "resnet18_rf2"),
    "dinov3": ("hf", "fovi-dinov3-splus_a-2.78_res-64_in1k"),        # DINOv3-S+ (ViT-S/16+)
    "dinov3_hplus": ("hf", "fovi-dinov3-hplus_a-2.78_res-64_in1k"),  # DINOv3-H+ (ViT-H/16+)
}
DENSE_FAMILY = {
    "alexnet_rf2": "alexnet", "alexnet_rf1": "alexnet",
    "resnet18_rf1": "resnet18", "resnet18_rf2": "resnet18",
    "dinov3": "dinov3", "dinov3_hplus": "dinov3_hplus",
}
# Report row labels distinguishing the two DINOv3 sizes.
DISPLAY_NAME = {"dinov3": "dinov3-s+", "dinov3_hplus": "dinov3-h+"}
# Canonical row order for the reports (ViTs first, then resnet, then alexnet).
MODEL_ORDER = ["dinov3", "dinov3_hplus", "resnet18_rf1", "resnet18_rf2",
               "alexnet_rf1", "alexnet_rf2"]


def _model_sort_key(m):
    return MODEL_ORDER.index(m) if m in MODEL_ORDER else len(MODEL_ORDER)
# The config that carries the dense backbone (S+ vs H+ ViT) for the dinov3 dense refs.
DINOV3_DENSE_CFG = {"dinov3": "fovi-dinov3-splus_a-2.78_res-64_in1k",
                    "dinov3_hplus": "fovi-dinov3-hplus_a-2.78_res-64_in1k"}

RECORDS = []  # collected for the summary table


def emit(record, note):
    if note is not None:
        record["note"] = note
    RECORDS.append(record)
    print(json.dumps(record), flush=True)


def backend_availability():
    """Which optional kernel backends can import (graceful-degradation annotation)."""
    availability = {}
    for label, module in (("cuda", "fovi.arch.knn_cuda"), ("warp", "fovi.arch.knn_warp"),
                          ("pool_cuda", "fovi.arch.knn_pool_cuda")):
        try:
            __import__(module)
            availability[label] = True
        except Exception as exc:  # cupy/warp absent, NVRTC missing, ...
            availability[label] = False
            print(f"BACKEND-UNAVAILABLE: {module}: {exc!r}", file=sys.stderr)
    return availability


# ---------------------------------------------------------------------------
# Arm control (env escape hatches + retina fast path)
# ---------------------------------------------------------------------------


def set_arm(model, arm, models_bench, retina_cls):
    """Configure one execution arm end-to-end.

    The optimization under test is the KNN conv/pool kernels: baseline =
    FOVI_KNN_BACKEND=baseline + FOVI_KNN_POOL_BACKEND=baseline (both read at dispatch
    time; the env var overrides layer.kernel_backend); optimized = auto. The retinal
    fast path (a separate, bit-exact front-end reimplementation) is also toggled off in
    baseline only to reconstruct the full pre-optimization timing, not because it is part of
    the kernel optimization.
    """
    if arm == "baseline":
        os.environ["FOVI_KNN_BACKEND"] = "baseline"
        models_bench.set_backend(model, "baseline")  # also sets pool env to baseline
        fast = False
    else:
        os.environ.pop("FOVI_KNN_BACKEND", None)
        models_bench.set_backend(model, "auto")  # also sets pool env to auto
        fast = True
    for module in model.modules():
        if isinstance(module, retina_cls):
            module.fast_pre_transforms = fast
    fixator = getattr(model, "sup_fixator", None)
    if isinstance(fixator, retina_cls):
        fixator.fast_pre_transforms = fast


# ---------------------------------------------------------------------------
# Timing helpers (event protocol reused from the models harness; wall added here)
# ---------------------------------------------------------------------------


def time_wall(fn, repeats, prepare=None):
    """Wall-throughput protocol: back-to-back iterations, one sync."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        if prepare is not None:
            prepare()
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeats * 1000.0


# ---------------------------------------------------------------------------
# FOVI variant runner (both arms, both scopes)
# ---------------------------------------------------------------------------


def run_fovi_variant(name, args, models_bench, retina_cls, availability, device):
    mb = models_bench
    kind, spec = VARIANTS[name]
    if kind == "local":
        model = mb.build_local_model(spec, str(device))
    else:
        from fovi import get_model_from_base_fn

        model = get_model_from_base_fn(spec, device=str(device), quiet=True)
    model.eval()
    amp_dtype = model.amp_dtype
    use_amp = bool(model.cfg.training.use_amp)
    label_smoothing = float(model.cfg.training.label_smoothing)

    for batch in args.batch:
      for n_fix in args.n_fixations:
        generator = torch.Generator(device=device).manual_seed(SEED)
        inputs = torch.rand(batch, 3, 256, 256, generator=generator, device=device)
        torch.manual_seed(SEED)
        with torch.no_grad():
            fixed = model.sup_fixator(inputs, n_fixations=n_fix)
        fixed_fixations = list(fixed["fixations"].unbind(dim=1))
        fixed_inputs = rearrange(fixed["x_fixs"], "b f c n -> (f b) c n")
        full_fn = partial(mb.full_forward, model, inputs, fixed_fixations, n_fixations=n_fix)
        network_fn = partial(mb.network_forward, model, fixed_inputs)

        base = dict(
            kind="fovi", model=name, base_fn=spec, batch=batch,
            network_batch=batch * n_fix, images_per_step=batch,
            n_fixations=n_fix,
            dtype=str(amp_dtype).replace("torch.", "") + "_amp",
            backends_available=availability,
        )

        if "inference" in args.modes:
            model.eval()
            reference = {}
            for arm in ("baseline", "optimized"):
                set_arm(model, arm, mb, retina_cls)
                for scope, fn in (("network", network_fn), ("full_fovinet", full_fn)):
                    out = mb.run_once(fn, amp_dtype).float()
                    if arm == "baseline":
                        reference[scope] = out
                        max_abs = 0.0
                        cos_sim = 1.0
                    else:
                        max_abs = (reference[scope] - out).abs().max().item()
                        # Cosine similarity of the flattened final-layer outputs: a
                        # scale-invariant parity metric (1.0 = identical direction),
                        # more interpretable than an outlier-dominated max|delta|.
                        cos_sim = torch.nn.functional.cosine_similarity(
                            reference[scope].flatten(), out.flatten(), dim=0, eps=1e-12).item()
                    del out
                    median_ms, min_ms = mb.time_cuda(fn, amp_dtype, args.warmup, args.repeats)
                    wall_ms = time_wall(lambda: mb.run_once(fn, amp_dtype), args.repeats)
                    temp = mb.memory_cuda(fn, amp_dtype)
                    record = dict(base, mode="inference", arm=arm, scope=scope,
                                  median_ms=median_ms, min_ms=min_ms, wall_ms=wall_ms,
                                  ms_per_image=median_ms / batch,
                                  temporary_mib=temp, max_abs_out_vs_baseline_arm=max_abs,
                                  cos_sim_out_vs_baseline_arm=cos_sim)
                    if arm == "optimized":
                        record["knn_backends"] = {
                            n: getattr(l, "_last_knn_backend", None) for n, l in mb.knn_layers(model)
                        }
                    emit(record, args.note)
            reference.clear()

        if "train" in args.modes:
            model.train()
            labels = torch.randint(0, model.num_classes, (batch,), device=device, generator=generator)
            state_snapshot = {k: v.detach().clone() for k, v in model.state_dict().items()}
            scaler_enabled = use_amp and amp_dtype != torch.bfloat16
            scaler = GradScaler("cuda", enabled=scaler_enabled, growth_interval=100)
            parity_scaler = GradScaler("cuda", enabled=scaler_enabled, init_scale=mb.PARITY_INIT_SCALE)
            steps = {
                "network": (mb.make_network_train_step(model, fixed_inputs, amp_dtype, scaler, device),
                            mb.make_network_train_step(model, fixed_inputs, amp_dtype, parity_scaler, device)),
                "full_fovinet": (mb.make_full_train_step(model, inputs, fixed_fixations, labels,
                                                         amp_dtype, label_smoothing, scaler,
                                                         n_fixations=n_fix),
                                 mb.make_full_train_step(model, inputs, fixed_fixations, labels,
                                                         amp_dtype, label_smoothing, parity_scaler,
                                                         n_fixations=n_fix)),
            }
            reference = {}
            reset = partial(mb.zero_grads, model)
            for arm in ("baseline", "optimized"):
                set_arm(model, arm, mb, retina_cls)
                for scope, (step_fn, parity_step) in steps.items():
                    loss, grad, grad_name = mb.parity_train(model, parity_step, parity_scaler, state_snapshot)
                    if arm == "baseline":
                        reference[scope] = (loss, grad)
                        loss_delta = grad_delta = 0.0
                    else:
                        ref_loss, ref_grad = reference[scope]
                        loss_delta = abs(loss - ref_loss)
                        grad_delta = None if (grad is None or ref_grad is None) else \
                            (grad - ref_grad).abs().max().item()
                    fwd_bwd_ms, fwd_bwd_min = mb.time_train(step_fn, args.warmup, args.repeats, reset)
                    wall_ms = time_wall(step_fn, args.repeats, prepare=reset)
                    temp = mb.memory_train(step_fn, reset)
                    record = dict(base, mode="train", arm=arm, scope=scope,
                                  fwd_bwd_ms=fwd_bwd_ms, fwd_bwd_min_ms=fwd_bwd_min,
                                  fwd_bwd_wall_ms=wall_ms, ms_per_image=fwd_bwd_ms / batch,
                                  temporary_mib=temp, loss=loss,
                                  grad_sample_param=grad_name,
                                  max_abs_loss_vs_baseline_arm=loss_delta,
                                  max_abs_grad_sample_vs_baseline_arm=grad_delta)
                    if arm == "optimized":
                        record["knn_backends"] = {
                            n: getattr(l, "_last_knn_backend", None) for n, l in mb.knn_layers(model)
                        }
                    emit(record, args.note)
            model.load_state_dict(state_snapshot)
            del state_snapshot, labels, steps
        del inputs, fixed, fixed_fixations, fixed_inputs, full_fn, network_fn
        gc.collect()
        torch.cuda.empty_cache()
    # leave the process in the optimized (shipped-default) configuration
    set_arm(model, "optimized", mb, retina_cls)
    del model
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Dense references
# ---------------------------------------------------------------------------


def build_dense256(family, device):
    """Native-resolution dense counterpart (one pass per 256x256 image)."""
    if family == "resnet18":
        # Same resnet implementation as logpolar@64 (repo ResNet, torchvision-standard
        # main_block_stride=2) but native 256x256 and non-polar — so the ONLY differences
        # from the foveated control are input resolution and the circular padding.
        from fovi.arch.resnet import resnet18 as resnet18_backbone

        return resnet18_backbone(pretrained=False, polar=False, no_fc=False,
                                 num_classes=1000, main_block_stride=2).to(device), torch.float16
    if family == "alexnet":
        from fovi.arch.alexnet import baseline_alexnet_kernels, get_backbone

        # The repo's 'base' spec: canonical AlexNet k11/s4 stem (the native-resolution
        # member of the family base_lowres was derived from) — documented choice.
        return get_backbone(kernels=baseline_alexnet_kernels["base"]).to(device), torch.float16
    if family in DINOV3_DENSE_CFG:
        from fovi import find_config
        from fovi.arch.dinov3 import build_fovi_dinov3
        from fovi.arch.knn import KNNConvLayer

        cfg, _, _ = find_config(DINOV3_DENSE_CFG[family], load=False)
        cfg.saccades.mode = "grid_as_grid"
        cfg.saccades.resize_size = 256
        cfg.model.vit.patch_size = 16  # standard ViT/16+ at 256 -> 16x16 = 256 tokens
        backbone = build_fovi_dinov3(cfg, device=str(device))
        knn = [n for n, m in backbone.named_modules() if isinstance(m, KNNConvLayer)]
        assert not knn, f"dense@256 dinov3 build still contains KNN layers: {knn}"
        return backbone.to(device), torch.bfloat16
    raise ValueError(family)


# ---------------------------------------------------------------------------
# logpolar@64: the matched foveated-CONTROL design, built through FoviNet so it reuses fovi's
# EXACT front-end (fixation policy + retina + fused augmentation) and head. It is a control
# model (2D log-polar `as_grid` input -> standard Conv2d/ViT) with ZERO KNN layers; the ONLY
# difference from the fovi variant is the backbone, so the full-step comparison has no
# residual (the front-end/head cost is fovi's own code, not an approximation of it).
# ---------------------------------------------------------------------------

# The fovi config carrying each family's backbone + foveation (S+ vs H+ ViT for dinov3).
DENSE64_CFG = {
    "resnet18": ("local", "resnet18_rf1"),
    "alexnet": ("hf", "fovi-alexnet_a-1_res-64_rfmult-2_in1k"),
    "dinov3": ("hf", "fovi-dinov3-splus_a-2.78_res-64_in1k"),
    "dinov3_hplus": ("hf", "fovi-dinov3-hplus_a-2.78_res-64_in1k"),
}

_DENSE_ARCHES_REGISTERED = False


def _register_dense_arches():
    """Register benchmark-local dense (control) arches. The repo's stock dense builders are
    FoviNet-ready in interface (both return a BackboneProjectorWrapper) but need small fixes:
    alexnet2023 omits .to(device); resnet_ssl double-wraps its projector (MLPWrapper of an
    MLPWrapper) and crashes, so we wrap the bare resnet18 backbone through arch_wrapper (the
    same clean lazy-projector path the KNN and alexnet builds use)."""
    global _DENSE_ARCHES_REGISTERED
    if _DENSE_ARCHES_REGISTERED:
        return
    from fovi.arch.architectures import ARCHITECTURE_REGISTRY, arch_wrapper

    def dense_alexnet(cfg, device="cuda"):
        return ARCHITECTURE_REGISTRY.get("alexnet2023")(cfg, device=device).to(device)

    def dense_resnet18(cfg, device="cuda"):
        from fovi.arch.resnet import resnet18 as resnet18_backbone, get_repr_size
        # main_block_stride=2 matches BOTH torchvision resnet18 AND the fovi KNNResNet
        # (knnresnet.py:264-266 strides layers 2-4 by 2, ending at ~2 nodes). The repo
        # ResNet default main_block_stride=1 keeps an 11x11=121-unit tail — ~60x the KNN
        # model's spatial resolution and unmatched; stride=2 ends at 2x2=4, matched. Polar
        # (from logpolar_as_grid) keeps the necessary circular angular padding.
        polar = "polar" in cfg.saccades.mode and "comp" not in cfg.saccades.mode
        # main_block_stride=2 + pool_stride=2 match the fovi KNNResNet (in_conv_stride=2,
        # in_pool_stride=2) and torchvision, so the per-layer node ladder tracks the KNN
        # model (pool -> 256 vs the KNN's 230, etc.); polar keeps the circular padding.
        backbone = resnet18_backbone(pretrained=False, polar=polar, no_fc=True,
                                     main_block_stride=2, pool_stride=2,
                                     out_map_size=int(cfg.model.get("final_grid_size", 1) or 1),
                                     channel_mult=int(cfg.model.channel_mult))
        backbone.total_embed_dim = get_repr_size(backbone, img_size=cfg.saccades.resize_size)
        return arch_wrapper(backbone, cfg, device=device).to(device)

    for name, fn in (("dense_alexnet_bench", dense_alexnet),
                     ("dense_resnet18_bench", dense_resnet18)):
        try:
            ARCHITECTURE_REGISTRY.register(name, fn)
        except Exception:
            pass
    _DENSE_ARCHES_REGISTERED = True


def build_dense_fovinet(family, device):
    """Build the logpolar@64 baseline as a FoviNet CONTROL model — the matched foveated design
    that shares fovi's ENTIRE front-end (identical sampler machinery, fixation policy, and
    fused augmentation) and head, differing ONLY in that its griddable `logpolar_as_grid`
    output feeds a standard Conv2d/ViT instead of KNNConv. Built from the family's own fovi
    config with the mode swapped to logpolar_as_grid and the arch swapped to the dense
    counterpart, so every front-end/head cost matches the fovi variant by construction. This
    is the established control recipe (cf. config/lp-cnn-alexnet.yaml,
    config/dinov3_logpolar_control.yaml)."""
    from omegaconf import OmegaConf
    from fovi import find_config
    from fovi.fovinet import FoviNet
    import benchmark_knn_conv_models as mb
    _register_dense_arches()
    kind, spec = DENSE64_CFG[family]
    if kind == "local":
        cfg = OmegaConf.load(mb.LOCAL_MODELS[spec])
        cfg.model.arch = "dense_resnet18_bench"
        cfg.model.arch_spec = "base"
        cfg.model.final_grid_size = 1
    else:
        cfg, _, _ = find_config(spec, load=False)
        if family == "alexnet":
            cfg.model.arch = "dense_alexnet_bench"
            cfg.model.final_grid_size = 1
        # dinov3: arch stays saccadenet_dinov3; the `as_grid` mode selects its dense patch embed.
    cfg.saccades.mode = "logpolar_as_grid"
    return FoviNet(cfg, device=str(device))


def run_logpolar64(family, args, models_bench, retina_cls, device):
    """logpolar@64 (matched foveated CONTROL design) timed through the SAME machinery as the fovi
    arms: network scope = the dense backbone on the pre-warped `(f b)` grids; full_fovinet =
    the whole FoviNet step (fixator + retina + augmentation -> dense backbone -> head). Single
    arm (the shipped fast front-end path) — a control model has no KNN kernels, so there is no
    baseline/optimized distinction. Because the front-end and head are fovi's exact code, the
    only thing this measures against the fovi variant is dense-backbone vs KNN-backbone."""
    mb = models_bench
    model = build_dense_fovinet(family, device)
    model.eval()
    # shipped fast pre-transform path, matching the fovi 'optimized' front-end (no KNN kernels
    # here, so set_arm's backend toggle is a no-op; it only sets the retina fast path).
    set_arm(model, "optimized", mb, retina_cls)
    amp_dtype = model.amp_dtype
    use_amp = bool(model.cfg.training.use_amp)
    label_smoothing = float(model.cfg.training.label_smoothing)

    for batch in args.batch:
      for n_fix in args.n_fixations:
        generator = torch.Generator(device=device).manual_seed(SEED)
        inputs = torch.rand(batch, 3, 256, 256, generator=generator, device=device)
        torch.manual_seed(SEED)
        with torch.no_grad():
            fixed = model.sup_fixator(inputs, n_fixations=n_fix)
        fixed_fixations = list(fixed["fixations"].unbind(dim=1))
        # control model: fixations are 2D grids, concatenated over the batch dim
        fixed_inputs = rearrange(fixed["x_fixs"], "b f c h w -> (f b) c h w")
        full_fn = partial(mb.full_forward, model, inputs, fixed_fixations, n_fixations=n_fix)
        network_fn = partial(mb.network_forward, model, fixed_inputs)
        base = dict(kind="logpolar@64", model=family, batch=batch, network_batch=batch * n_fix,
                    images_per_step=batch, n_fixations=n_fix, resolution=64,
                    dtype=str(amp_dtype).replace("torch.", "") + "_amp")

        if "inference" in args.modes:
            model.eval()
            for scope, fn in (("network", network_fn), ("full_fovinet", full_fn)):
                median_ms, min_ms = mb.time_cuda(fn, amp_dtype, args.warmup, args.repeats)
                wall_ms = time_wall(lambda fn=fn: mb.run_once(fn, amp_dtype), args.repeats)
                temp = mb.memory_cuda(fn, amp_dtype)
                emit(dict(base, mode="inference", scope=scope, median_ms=median_ms,
                          min_ms=min_ms, wall_ms=wall_ms, ms_per_image=median_ms / batch,
                          temporary_mib=temp), args.note)

        if "train" in args.modes:
            model.train()
            labels = torch.randint(0, model.num_classes, (batch,), device=device, generator=generator)
            scaler_enabled = use_amp and amp_dtype != torch.bfloat16
            scaler = GradScaler("cuda", enabled=scaler_enabled, growth_interval=100)
            steps = {
                "network": mb.make_network_train_step(model, fixed_inputs, amp_dtype, scaler, device),
                "full_fovinet": mb.make_full_train_step(model, inputs, fixed_fixations, labels,
                                                        amp_dtype, label_smoothing, scaler,
                                                        n_fixations=n_fix),
            }
            reset = partial(mb.zero_grads, model)
            for scope, step_fn in steps.items():
                fwd_bwd_ms, fwd_bwd_min = mb.time_train(step_fn, args.warmup, args.repeats, reset)
                wall_ms = time_wall(step_fn, args.repeats, prepare=reset)
                temp = mb.memory_train(step_fn, reset)
                emit(dict(base, mode="train", scope=scope, fwd_bwd_ms=fwd_bwd_ms,
                          fwd_bwd_min_ms=fwd_bwd_min, fwd_bwd_wall_ms=wall_ms,
                          ms_per_image=fwd_bwd_ms / batch, temporary_mib=temp), args.note)
            del labels, steps
        del inputs, fixed, fixed_fixations, fixed_inputs, full_fn, network_fn
        gc.collect()
        torch.cuda.empty_cache()
    del model
    gc.collect()
    torch.cuda.empty_cache()


def run_dense(family, ref, args, models_bench, retina_cls, device):
    """dense@256 (native full-image, non-foveated) — ONE pass per 256x256 image, a single
    'native' scope with no foveated front-end. logpolar@64 is the matched foveated control
    design and is handled by run_logpolar64 (built through FoviNet)."""
    if ref == "logpolar@64":
        run_logpolar64(family, args, models_bench, retina_cls, device)
        return
    import benchmark_dense_references as dense_ref
    model, amp_dtype = build_dense256(family, device)
    scaler = GradScaler("cuda", enabled=amp_dtype == torch.float16, growth_interval=100)
    generator = torch.Generator(device=device).manual_seed(SEED)
    for batch in args.batch:
        x = torch.rand(batch, 3, 256, 256, device=device, generator=generator)
        state = {}

        def reset():
            for p in model.parameters():
                p.grad = None

        def train_step():
            with autocast("cuda", dtype=amp_dtype, enabled=True):
                y = model(x)
                while isinstance(y, (tuple, list)):
                    y = y[0]
            if "unit" not in state:
                g = torch.Generator(device=device).manual_seed(SEED)
                u = torch.randn(y.shape, device=device, dtype=torch.float32, generator=g)
                state["unit"] = u / float(u.numel()) ** 0.5
            loss = (y.float() * state["unit"]).sum()
            scaler.scale(loss).backward()

        def infer_fwd():
            with torch.no_grad(), autocast("cuda", dtype=amp_dtype, enabled=True):
                return model(x)

        base = dict(kind="dense@256", model=family, batch=batch, network_batch=batch,
                    images_per_step=batch, n_fixations=1, resolution=256, scope="native",
                    dtype=str(amp_dtype).replace("torch.", "") + "_amp")
        if "train" in args.modes:
            model.train()
            ms, min_ms = dense_ref.time_cuda(train_step, args.warmup, args.repeats, between=reset)
            wall = time_wall(train_step, args.repeats, prepare=reset)
            temp = dense_ref.peak_mib(train_step, between=reset)
            emit(dict(base, mode="train", fwd_bwd_ms=ms, fwd_bwd_min_ms=min_ms,
                      fwd_bwd_wall_ms=wall, ms_per_image=ms / batch, temporary_mib=temp), args.note)
        if "inference" in args.modes:
            model.eval()
            ms, min_ms = dense_ref.time_cuda(infer_fwd, args.warmup, args.repeats)
            wall = time_wall(infer_fwd, args.repeats)
            temp = dense_ref.peak_mib(infer_fwd)
            emit(dict(base, mode="inference", median_ms=ms, min_ms=min_ms, wall_ms=wall,
                      ms_per_image=ms / batch, temporary_mib=temp), args.note)
        state.clear()
        del x
        gc.collect()
        torch.cuda.empty_cache()
    del model
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def summary_table(records):
    def ms_of(r):
        return r.get("fwd_bwd_ms", r.get("median_ms"))

    dense = {}
    for r in records:
        # network-scope summary: keep the net-only logpolar@64 (and the native dense@256),
        # skip the full_fovinet front-end variant so keys don't collide.
        if r["kind"].startswith("dense") and (r.get("scope") or "network") != "full_fovinet":
            dense[(r["kind"], r["model"], r["mode"], r["batch"], r["n_fixations"])] = ms_of(r)
    lines = [
        "",
        "== FINAL COMPARISON SUMMARY (network scope; every cell labeled (images, n_fix)) ==",
        "   xd@64    = fovi / logpolar@64  (<1 = foveated cheaper; matched-res, per-fixation pass)",
        "   xd@256   = fovi / dense@256 (<1 = foveated cheaper; ONE native-res pass per image)",
        "   xd256/s  = xd@256 / n_fix   (per-forward-sample convention)",
        f"{'variant':13} {'arm':10} {'mode':9} {'imgs':>5} {'nfix':>4} {'net_ms':>8} "
        f"{'full_ms':>8} {'xd@64':>7} {'xd@256':>7} {'xd256/s':>8}",
    ]
    for r in records:
        if r["kind"] != "fovi" or r["scope"] != "network":
            continue
        family = DENSE_FAMILY[r["model"]]
        key64 = ("logpolar@64", family, r["mode"], r["batch"], r["n_fixations"])
        key256 = ("dense@256", family, r["mode"], r["batch"], 1)
        full = next((ms_of(q) for q in records if q["kind"] == "fovi" and q["model"] == r["model"]
                     and q["arm"] == r["arm"] and q["mode"] == r["mode"]
                     and q["batch"] == r["batch"] and q["n_fixations"] == r["n_fixations"]
                     and q["scope"] == "full_fovinet"), None)
        net = ms_of(r)
        xd64 = net / dense[key64] if key64 in dense else None
        xd256 = net / dense[key256] if key256 in dense else None
        xd256s = xd256 / r["n_fixations"] if xd256 is not None else None
        fmt = lambda v, n=2: "-" if v is None else f"{v:.{n}f}"
        lines.append(
            f"{r['model']:13} {r['arm']:10} {r['mode']:9} {r['batch']:>5} {r['n_fixations']:>4} "
            f"{fmt(net):>8} {fmt(full):>8} {fmt(xd64):>7} {fmt(xd256):>7} {fmt(xd256s):>8}"
        )
    lines.append(
        "   (footnote: earlier reports quoted the n_fix=4 cells in "
        "network-batch terms — network batch = images x n_fix; same numbers, same cells.)"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Human-readable report (postprocessing over the verbose JSON records)
# ---------------------------------------------------------------------------

_GPU_SHORT = (("RTX 6000 Ada", "RTX 6000 (Ada)"),
              ("RTX PRO 6000", "RTX 6000 Pro (Blackwell)"),
              ("L40S", "L40S"), ("H100", "H100"),
              ("Blackwell", "Blackwell"), ("L40", "L40"), ("A100", "A100"))


def _gpu_short(name):
    for needle, short in _GPU_SHORT:
        if needle in (name or ""):
            return short
    return (name or "?").replace("NVIDIA ", "")[:12]


def load_records(paths):
    """Load verbose JSON(L) records from one or more run files, stamping each data
    record with the short GPU name from its file's ``final_comparison`` meta line.
    Returns (records, envs)."""
    records, envs = [], []
    for path in paths:
        cur = None
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("kind") == "final_comparison":
                    cur = _gpu_short(r.get("gpu"))
                    envs.append({"gpu": r.get("gpu"), "short": cur,
                                 "capability": r.get("capability"), "torch": r.get("torch"),
                                 "cuda": r.get("cuda"), "source": os.path.basename(path)})
                    continue
                # Drop superseded dense@64 records (net-only or hand-rolled
                # front-end): replaced by the matched-FoviNet logpolar@64 control design.
                if r.get("kind") == "dense@64":
                    continue
                r["_gpu"] = cur
                records.append(r)
    return records, envs


def render_report(records, envs):
    """Render the verbose records into a self-contained Markdown report."""
    def ms(r):
        return None if r is None else r.get("fwd_bwd_ms", r.get("median_ms"))

    def pick(**f):
        for r in records:
            if all(r.get(k) == v for k, v in f.items()):
                return r
        return None

    def dense_ms(kind, family, mode, batch, n_fix, gpu, scope):
        # logpolar@64 is scope-symmetric with the fovi arms: network = net-only (pre-warped
        # input), full_fovinet = log-polar front-end + net. dense@256 is a single native
        # pass with no separate foveated front-end, so it is scope-independent. Legacy
        # logpolar@64 records (pre-front-end) carry no scope and count as network.
        for r in records:
            if not (r.get("kind") == kind and r.get("model") == family
                    and r.get("mode") == mode and r.get("batch") == batch
                    and r.get("n_fixations") == n_fix and r.get("_gpu") == gpu):
                continue
            if kind == "logpolar@64" and (r.get("scope") or "network") != scope:
                continue
            return ms(r)
        return None

    def fovi(model, arm, mode, batch, n_fix, gpu, scope):
        return ms(pick(kind="fovi", model=model, arm=arm, mode=mode, batch=batch,
                       n_fixations=n_fix, scope=scope, _gpu=gpu))

    # This static report uses the NETWORK scope (backbone/kernels; no front-end) for
    # every table, so the optimization speedup isolates the conv/pool kernels. The
    # interactive HTML report additionally offers a full-step scope toggle.
    SCOPE = "network"

    def xd(model, mode, batch, n_fix, gpu, which, arm="optimized"):
        # Speedup of the foveated model vs the dense reference (dense / fovi):
        # >1 = foveated faster, <1 = slower. arm selects the fovi kernels (optimized =
        # shipped; baseline = unoptimized, the pre-optimization framing).
        fv = fovi(model, arm, mode, batch, n_fix, gpu, SCOPE)
        d = dense_ms(which, DENSE_FAMILY[model], mode, batch,
                     n_fix if which == "logpolar@64" else 1, gpu, SCOPE)
        return None if (fv is None or not d or not fv) else d / fv

    def speedup(model, mode, batch, n_fix, gpu):
        base = fovi(model, "baseline", mode, batch, n_fix, gpu, SCOPE)
        opt = fovi(model, "optimized", mode, batch, n_fix, gpu, SCOPE)
        return None if (not base or not opt) else base / opt

    gpus, models = [], []
    for r in records:
        if r.get("kind") == "fovi":
            if r["_gpu"] not in gpus:
                gpus.append(r["_gpu"])
            if r["model"] not in models:
                models.append(r["model"])
    models.sort(key=_model_sort_key)

    def fmt(v, n=2):
        return "-" if v is None else f"{v:.{n}f}"

    def table(header, cell):
        rows = ["| variant | " + " | ".join(gpus) + " |",
                "|" + "---|" * (len(gpus) + 1)]
        for m in models:
            rows.append("| " + DISPLAY_NAME.get(m, m) + " | "
                        + " | ".join(cell(m, g) for g in gpus) + " |")
        return "\n".join([header, ""] + rows + [""])

    out = ["# FOVI foveated-vision benchmark: optimized vs baseline vs dense",
           "",
           "Generated from the verbose JSON records emitted by "
           "`benchmark_final_comparison.py` (one record per cell). Two arms comparing the "
           "**KNN convolution + pooling CUDA kernels** — baseline (reference kernels) vs "
           "optimized (optimized kernels) — and two references: **logpolar@64** (the matched "
           "foveated CONTROL: identical fixations/retina/augmentation feeding a standard "
           "Conv2d/ViT instead of KNNConv, one warped pass per fixation) and **dense@256** "
           "(native resolution, one pass per image). Batch = number of images; fixations are "
           "a separate dimension.",
           "",
           "All tables use the **network scope** (the backbone — KNN conv + pooling — "
           "with no retinal front-end or head), so the optimization speedup isolates the "
           "conv/pool kernels. (The interactive HTML report adds a full-step scope "
           "toggle.) Every ratio is a **speedup** (higher = better):",
           "- **Optimization speedup = baseline ÷ optimized** (how many times faster the "
           "optimized kernels make the backbone). *5.0x = 5x faster.*",
           "- **Speedup vs dense = dense ÷ fovi** (how many times faster the foveated "
           "backbone is than the dense reference). *>1 = foveated faster; <1 = slower.*",
           "- **ms** tables are absolute network-scope step time (lower = faster). "
           "`ms` = event-median.",
           ""]

    if envs:
        out += ["## Environments", "",
                "| GPU | capability | torch | cuda | source |",
                "|---|---|---|---|---|"]
        seen = set()
        for e in envs:
            k = (e["short"], e["source"])
            if k in seen:
                continue
            seen.add(k)
            cap = ".".join(str(c) for c in (e["capability"] or []))
            out.append(f"| {e['short']} ({e['gpu']}) | {cap} | {e['torch']} | "
                       f"{e['cuda']} | {e['source']} |")
        out.append("")

    # One full detail block per (images, n_fixations) cell present (batch >= 128; the
    # batch-10 cells are noise-sensitive and omitted). Ordered by EFFECTIVE batch
    # (images x fixations) so cells with the same backbone workload sit together — e.g.
    # (128 img x 4 fix) and (512 img x 1 fix) are both effective batch 512, which
    # isolates the effective-batch effect from the fixation count.
    cells = sorted({(r["batch"], r["n_fixations"]) for r in records
                    if r.get("kind") == "fovi" and r["batch"] >= 128},
                   key=lambda bf: (bf[0] * bf[1], bf[0]))
    for (B, F) in cells:
        eff = B * F
        plural = "s" if F != 1 else ""
        glance = "one foveated glance" if F == 1 else f"a full {F}-glance foveated step"
        for mode in ("Training", "Inference"):
            md = "train" if mode == "Training" else "inference"
            out.append(f"## {mode} — {B} images x {F} fixation{plural} "
                       f"(effective batch {eff}, optimized arm)\n")
            out.append(table(f"### Optimization speedup  baseline / optimized  "
                             f"(>1 = optimized kernels that many times faster)",
                             lambda m, g, md=md: fmt(speedup(m, md, B, F, g)) + "x"))
            out.append(table(f"### Speedup vs dense@256  dense / fovi (optimized)  "
                             f"({glance} vs ONE native-res pass; >1 = foveated faster)",
                             lambda m, g, md=md: fmt(xd(m, md, B, F, g, "dense@256")) + "x"))
            out.append(table(f"### Speedup vs dense@256  dense / fovi (UNOPTIMIZED baseline kernels)  "
                             f"(same comparison without the KNN kernel optimization)",
                             lambda m, g, md=md: fmt(xd(m, md, B, F, g, "dense@256", "baseline")) + "x"))
            out.append(table(f"### Speedup vs logpolar@64  dense / fovi (optimized)  "
                             f"(vs the matched foveated-control dense net; >1 = foveated faster)",
                             lambda m, g, md=md: fmt(xd(m, md, B, F, g, "logpolar@64")) + "x"))
            out.append(table(f"### Speedup vs logpolar@64  dense / fovi (UNOPTIMIZED baseline kernels)  "
                             f"(same comparison without the KNN kernel optimization)",
                             lambda m, g, md=md: fmt(xd(m, md, B, F, g, "logpolar@64", "baseline")) + "x"))
            out.append(table(f"### {mode} time — network scope, absolute (ms; lower = faster)",
                             lambda m, g, md=md: fmt(fovi(m, "optimized", md, B, F, g, SCOPE))))

    # Parity: worst (lowest) cosine similarity of the optimized-vs-baseline final-layer
    # outputs per model (across all inference cells/GPUs). 1.0 = identical direction.
    def cosfmt(v):
        if v is None:
            return "n/a"
        d = 1.0 - v
        return f"1 - {d:.1e}" if d < 5e-6 else f"{v:.5f}"

    out += ["## Output parity — cosine similarity of final-layer outputs "
            "(optimized vs baseline arm; 1.0 = identical direction, worst cell shown)", ""]
    for m in models:
        sims = [r.get("cos_sim_out_vs_baseline_arm") for r in records
                if r.get("kind") == "fovi" and r.get("model") == m
                and r.get("arm") == "optimized"
                and r.get("cos_sim_out_vs_baseline_arm") is not None]
        worst = min(sims) if sims else None
        out.append(f"- **{DISPLAY_NAME.get(m, m)}**: {cosfmt(worst)}")
    out += ["",
            "_Note: the smallest cells — batch-10, and the 1-fixation steps that fall "
            "under ~20 ms on the faster GPUs — are clock-state-sensitive (GPU idle vs "
            "boost), so read small cross-GPU / cross-variant differences there with wide "
            "bars; the 128 images x 4 fixations cells are the stable reference. dinov3's "
            "cost is its frozen ViT trunk, so it sits at dense parity by construction._",
            ""]
    return "\n".join(out)


_HTML_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FOVI benchmark — optimized vs dense</title>
<style>
:root{ color-scheme:light;
  --page:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --border:rgba(11,11,11,0.10); --good:#006300;
  --blue:37,106,191; --red:208,59,59; }
:root[data-theme="dark"]{ color-scheme:dark;
  --page:#0d0d0d; --surface:#1a1a19; --ink:#ffffff; --ink2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --border:rgba(255,255,255,0.10); --good:#0ca30c;
  --blue:57,135,229; --red:230,103,103; }
@media (prefers-color-scheme:dark){ :root:not([data-theme="light"]){ color-scheme:dark;
  --page:#0d0d0d; --surface:#1a1a19; --ink:#ffffff; --ink2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --border:rgba(255,255,255,0.10); --good:#0ca30c;
  --blue:57,135,229; --red:230,103,103; } }
*{box-sizing:border-box}
body{ margin:0; background:var(--page); color:var(--ink);
  font-family:system-ui,-apple-system,"Segoe UI",sans-serif; font-size:14px; line-height:1.5; }
.wrap{ max-width:1100px; margin:0 auto; padding:24px 20px 60px; }
h1{ font-size:20px; margin:0 0 2px; }
h3{ font-size:15px; margin:26px 0 2px; }
.sub{ color:var(--ink2); margin:0 0 10px; font-size:13px; }
.lede{ color:var(--ink2); margin:2px 0 20px; max-width:70ch; }
.controls{ display:flex; flex-wrap:wrap; gap:14px 20px; align-items:flex-end;
  background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:14px 16px; }
.controls label{ display:flex; flex-direction:column; gap:4px; font-size:12px; color:var(--ink2); }
.controls select{ font:inherit; padding:6px 8px; border:1px solid var(--border);
  border-radius:7px; background:var(--surface); color:var(--ink); }
.seg{ display:inline-flex; border:1px solid var(--border); border-radius:7px; overflow:hidden; }
.seg label{ padding:6px 14px; cursor:pointer; font-size:13px; color:var(--ink2); }
.seg input{ display:none; }
.seg input:checked + span{ background:rgba(var(--blue),0.16); color:var(--ink); font-weight:600; }
.seg span{ display:block; margin:-6px -14px; padding:6px 14px; }
.badge{ font-size:12px; color:var(--ink2); background:rgba(var(--blue),0.10);
  border-radius:20px; padding:5px 12px; }
.note{ font-size:13px; color:var(--ink2); margin:14px 0 4px; }
.spacer{ flex:1 1 auto; }
button.theme{ font:inherit; font-size:12px; padding:6px 12px; border:1px solid var(--border);
  border-radius:7px; background:var(--surface); color:var(--ink2); cursor:pointer; }
table{ border-collapse:separate; border-spacing:0; width:100%; margin:6px 0 2px;
  font-variant-numeric:tabular-nums; }
th,td{ text-align:right; padding:7px 12px; border-bottom:1px solid var(--grid); }
th{ color:var(--muted); font-weight:600; font-size:12px; }
th:first-child,td:first-child{ text-align:left; }
td.rowh{ color:var(--ink2); }
td.val{ font-weight:600; }
.legend{ display:flex; gap:18px; flex-wrap:wrap; font-size:12px; color:var(--ink2); margin:8px 0 0; }
.chip{ display:inline-block; width:34px; height:12px; border-radius:3px; vertical-align:middle;
  margin-right:6px; border:1px solid var(--border); }
details.env{ margin-top:28px; } summary{ cursor:pointer; color:var(--ink2); font-size:13px; }
.env table{ margin-top:8px; } .env td,.env th{ text-align:left; }
.pnote{ font-size:12px; color:var(--muted); margin-top:6px; }
</style></head>
<body><div class="wrap">
<div style="display:flex;align-items:baseline;gap:14px">
  <h1>FOVI — optimized vs dense</h1><div class="spacer"></div>
  <button class="theme" id="themeBtn">◐ theme</button>
</div>
<p class="lede">Two arms comparing the <b>KNN convolution + pooling CUDA kernels</b> —
<b>baseline</b> (reference kernels) vs <b>optimized</b> (optimized kernels) — against a
selectable dense reference. Pick a cell below; every table is filtered to it.</p>

<div class="controls">
  <div class="seg" id="modeSeg">
    <label><input type="radio" name="mode" value="train" checked><span>train</span></label>
    <label><input type="radio" name="mode" value="inference"><span>inference</span></label>
  </div>
  <label>batch (images)<select id="batch"></select></label>
  <label>fixations<select id="nfix"></select></label>
  <label>reference<select id="dense">
    <option value="logpolar@64">logpolar@64 (matched foveated CNN)</option>
    <option value="dense@256" selected>dense@256 (native res)</option>
  </select></label>
  <label>scope<select id="scope">
    <option value="network" selected>network (backbone / kernels)</option>
    <option value="full_fovinet">full step (incl. front-end + head)</option>
  </select></label>
  <div class="spacer"></div>
  <span class="badge" id="effbadge"></span>
</div>
<p class="note" id="densenote"></p>

<div id="tables"></div>
<div class="legend">
  <span>All ratios are speedups (higher = better):</span>
  <span><span class="chip" style="background:rgba(var(--blue),0.42)"></span>faster (&gt;1×)</span>
  <span><span class="chip" style="background:var(--surface)"></span>parity (≈1×)</span>
  <span><span class="chip" style="background:rgba(var(--red),0.42)"></span>slower (&lt;1×)</span>
</div>

<h3>Output parity — cosine similarity of final-layer outputs (optimized vs baseline arm)</h3>
<div id="parity"></div>
<p class="pnote">Cosine similarity of the optimized-vs-baseline final-layer output vectors (1.0 = identical direction; worst inference cell shown) — scale-invariant, unlike max|Δ|. dinov3 runs bf16, the CNNs fp16; all sit at &gt;0.999 alignment. The smallest timing cells (batch 10, and 1-fixation on the fast GPUs) are clock-state sensitive; the 128×4 cell is the stable reference.</p>

<details class="env"><summary>Environments &amp; provenance</summary><div id="env"></div></details>

<script>
const D = __PAYLOAD__;
const $ = id => document.getElementById(id);
const ms = r => r.fwd_bwd_ms!=null ? r.fwd_bwd_ms : r.median_ms;
function find(f){ return D.records.find(r=>{ for(const k in f) if(r[k]!==f[k]) return false; return true; }); }
function fovi(m,arm,mode,B,F,g,scope){ const r=find({kind:'fovi',model:m,arm:arm,mode:mode,batch:B,n_fixations:F,scope:scope,_gpu:g}); return r?ms(r):null; }
function densems(w,fam,mode,B,F,g,scope){ // logpolar@64 is scope-symmetric (net-only vs +front-end); dense@256 is scope-independent
  const r=D.records.find(r=>r.kind===w&&r.model===fam&&r.mode===mode&&r.batch===B&&r.n_fixations===F&&r._gpu===g
    &&(w!=='logpolar@64'||(r.scope||'network')===scope));
  return r?ms(r):null; }
function stepms(m,mode,B,F,g,scope){ return fovi(m,'optimized',mode,B,F,g,scope); }
function spdVsDense(m,mode,B,F,g,w,scope,arm){ // dense / fovi = how many times faster foveated is (arm: optimized|baseline)
  const fv=fovi(m,arm||'optimized',mode,B,F,g,scope);
  const nf = (w==='logpolar@64') ? F : 1;
  const d = densems(w, D.densefam[m], mode, B, nf, g, scope);
  return (fv==null||d==null||!d||!fv) ? null : d/fv;
}
function speedup(m,mode,B,F,g,scope){
  const b=fovi(m,'baseline',mode,B,F,g,scope);
  const o=fovi(m,'optimized',mode,B,F,g,scope);
  return (b==null||o==null||!o) ? null : b/o;
}
const cvar = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
function spdBg(x){ if(x==null) return 'transparent'; // speedup: >1 faster (blue), <1 slower (red)
  let t=Math.log(x)/Math.log(3.5); t=Math.max(-1,Math.min(1,t));
  const a=Math.min(0.5, Math.abs(t)*0.62);
  return 'rgba('+(t>0?cvar('--blue'):cvar('--red'))+','+a.toFixed(3)+')'; }
function seqBg(x,max){ if(x==null) return 'transparent';
  const t=Math.max(0,Math.min(1,(x-1)/((max-1)||1)));
  return 'rgba('+cvar('--blue')+','+(t*0.5).toFixed(3)+')'; }
function tbl(title,sub,vf,bf,fmt){
  let h='<h3>'+title+'</h3>'+(sub?'<p class="sub">'+sub+'</p>':'');
  h+='<table><thead><tr><th>variant</th>'+D.gpus.map(g=>'<th>'+g+'</th>').join('')+'</tr></thead><tbody>';
  for(const m of D.variants){ h+='<tr><td class="rowh">'+(D.display[m]||m)+'</td>';
    for(const g of D.gpus){ const v=vf(m,g);
      h+='<td class="val" style="background:'+(bf?bf(v):'transparent')+'">'+(v==null?'–':fmt(v))+'</td>'; }
    h+='</tr>'; }
  return h+'</tbody></table>'; }
function render(){
  const mode=document.querySelector('input[name=mode]:checked').value;
  const B=+$('batch').value, F=+$('nfix').value, w=$('dense').value, scope=$('scope').value;
  const sc = scope==='network' ? 'network (backbone / kernels; no front-end)' : 'full step (front-end + backbone + head)';
  $('effbadge').textContent='effective batch = '+(B*F)+'  (images × fixations)';
  const fe = (scope==='full_fovinet')
    ? ' Built as a FoviNet control, so it runs fovi\'s identical front-end (fixation policy + retina + fused augmentation) and head — the full step differs from fovi ONLY in the backbone.'
    : ' Net-only in this scope — the shared front-end is excluded on both sides, isolating the backbone.';
  $('densenote').innerHTML = 'Scope: <b>'+sc+'</b>. ' + ((w==='logpolar@64')
    ? '<b>logpolar@64</b> — the matched foveated CONTROL: the same log-polar-warped 64×64 glances feeding a standard Conv2d/ViT (with circular padding) instead of KNNConv, one pass per fixation ('+F+').' + fe
    : '<b>dense@256</b> — native-resolution network, <b>one</b> pass per image (scope-independent: no foveated front-end).');
  let smax=1; for(const m of D.variants) for(const g of D.gpus){ const s=speedup(m,mode,B,F,g,scope); if(s!=null&&s>smax) smax=s; }
  let h='';
  h+=tbl('Optimization speedup — baseline / optimized','how many times faster the optimized kernels make it (higher = better)',
        (m,g)=>speedup(m,mode,B,F,g,scope), v=>seqBg(v,smax), v=>v.toFixed(2)+'×');
  h+=tbl('Speedup vs '+w+' — dense / fovi (optimized)','>1 = foveated model is that many times faster than the dense reference (blue), <1 = slower (red)',
        (m,g)=>spdVsDense(m,mode,B,F,g,w,scope,'optimized'), spdBg, v=>v.toFixed(2)+'×');
  h+=tbl('Speedup vs '+w+' — dense / fovi (UNOPTIMIZED baseline kernels)','the same comparison run with the pre-optimization KNN kernels — the gap to the row above is what the kernel optimization buys',
        (m,g)=>spdVsDense(m,mode,B,F,g,w,scope,'baseline'), spdBg, v=>v.toFixed(2)+'×');
  h+=tbl('Step time — absolute (ms)','optimized arm; lower = faster',
        (m,g)=>stepms(m,mode,B,F,g,scope), null, v=>v.toFixed(2)+' ms');
  $('tables').innerHTML=h;
}
function fillEnv(){
  let h='<table><thead><tr><th>GPU</th><th>cap</th><th>torch</th><th>cuda</th><th>source</th></tr></thead><tbody>';
  for(const e of D.envs) h+='<tr><td>'+e.short+' ('+e.gpu+')</td><td>'+e.capability+'</td><td>'+(e.torch||'')+'</td><td>'+(e.cuda||'')+'</td><td>'+(e.source||'')+'</td></tr>';
  $('env').innerHTML=h+'</tbody></table>';
  let p='<table><thead><tr><th>variant</th><th>cosine similarity</th></tr></thead><tbody>';
  const cosfmt=v=>{ if(v==null) return 'n/a'; const d=1-v; return d<5e-6 ? '1 − '+d.toExponential(1) : v.toFixed(5); };
  for(const m of D.variants){ const v=D.parity[m]; p+='<tr><td>'+(D.display[m]||m)+'</td><td class="val">'+cosfmt(v)+'</td></tr>'; }
  $('parity').innerHTML=p+'</tbody></table>';
}
function opt(sel,vals,def){ sel.innerHTML=vals.map(v=>'<option'+(v==def?' selected':'')+'>'+v+'</option>').join(''); }
opt($('batch'), D.batches, D.batches.includes(512)?512:D.batches[D.batches.length-1]);
opt($('nfix'), D.fixations, D.fixations.includes(1)?1:D.fixations[0]);
document.querySelectorAll('input[name=mode],#batch,#nfix,#dense,#scope').forEach(e=>e.addEventListener('change',render));
$('themeBtn').addEventListener('click',()=>{
  const cur=document.documentElement.getAttribute('data-theme');
  const next = cur==='dark'?'light':(cur==='light'?'dark':(matchMedia('(prefers-color-scheme:dark)').matches?'light':'dark'));
  document.documentElement.setAttribute('data-theme',next); render(); });
matchMedia('(prefers-color-scheme:dark)').addEventListener('change',render);
fillEnv(); render();
</script>
</div></body></html>
"""


def render_html(records, envs):
    """Self-contained interactive HTML: selectors for batch, fixations, mode, and dense
    baseline; tables render client-side from the embedded records. No dependencies."""
    gpus, variants = [], []
    for r in records:
        if r.get("kind") == "fovi":
            if r["_gpu"] not in gpus:
                gpus.append(r["_gpu"])
            if r["model"] not in variants:
                variants.append(r["model"])
    variants.sort(key=_model_sort_key)
    batches = sorted({r["batch"] for r in records if r.get("kind") == "fovi"})
    fixations = sorted({r["n_fixations"] for r in records if r.get("kind") == "fovi"})
    compact = [{k: r.get(k) for k in ("kind", "model", "arm", "mode", "batch",
               "n_fixations", "scope", "_gpu", "fwd_bwd_ms", "median_ms")} for r in records]
    parity = {}
    for m in variants:
        ds = [r.get("cos_sim_out_vs_baseline_arm") for r in records
              if r.get("kind") == "fovi" and r.get("model") == m and r.get("arm") == "optimized"
              and r.get("cos_sim_out_vs_baseline_arm") is not None]
        parity[m] = min(ds) if ds else None
    seen, envlist = set(), []
    for e in envs:
        key = (e["short"], e.get("source"))
        if key in seen:
            continue
        seen.add(key)
        envlist.append({"short": e["short"], "gpu": e["gpu"],
                        "capability": ".".join(str(c) for c in (e.get("capability") or [])),
                        "torch": e.get("torch"), "cuda": e.get("cuda"), "source": e.get("source")})
    payload = json.dumps({"records": compact, "gpus": gpus, "variants": variants,
                          "batches": batches, "fixations": fixations, "densefam": DENSE_FAMILY,
                          "parity": parity, "envs": envlist,
                          "display": {m: DISPLAY_NAME.get(m, m) for m in variants}})
    return _HTML_TEMPLATE.replace("__PAYLOAD__", payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--report-from", nargs="+", default=None,
                        help="Render a human-readable report from existing verbose JSON(L) "
                             "run files (no benchmarking); pass one file per GPU.")
    parser.add_argument("--report-out", default=None,
                        help="Write the human-readable Markdown report to this path "
                             "(as a postprocessing step after a run, or with --report-from).")
    parser.add_argument("--html-out", default=None,
                        help="Write a self-contained interactive HTML report (selectors for "
                             "batch/fixations/mode/dense-baseline) to this path.")
    parser.add_argument("--models", nargs="+", choices=sorted(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--batch", type=int, nargs="+", default=[10, 128],
                        help="NUMBER OF IMAGES per step (fixed across all cells; low-res "
                             "networks internally see images x n_fixations samples)")
    parser.add_argument("--n-fixations", type=int, nargs="+", default=[1, 4],
                        help="fixation counts for the fovi arms and logpolar@64 passes "
                             "(dense@256 is always one pass per image)")
    parser.add_argument("--modes", nargs="+", choices=("train", "inference"),
                        default=["train", "inference"])
    parser.add_argument("--dense", choices=("both", "64", "256", "none"), default="both")
    parser.add_argument("--dense-only", action="store_true",
                        help="Run only the dense references (skip the fovi arms); use to "
                             "regenerate logpolar@64/dense@256 records without re-timing fovi.")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--note", default=None)
    parser.add_argument("--cache-dir", default=None,
                        help="Model cache root: sets HF_HOME and HUGGINGFACE_HUB_CACHE "
                             "(pre-set env is honored when omitted; no network needed "
                             "when checkpoints are cached)")
    args = parser.parse_args()

    # Postprocessing-only path: render report(s) from already-collected JSON and exit.
    if args.report_from is not None:
        records, envs = load_records(args.report_from)
        if args.html_out:
            with open(args.html_out, "w") as fh:
                fh.write(render_html(records, envs))
            print(f"wrote html: {args.html_out}", file=sys.stderr)
        report = render_report(records, envs)
        if args.report_out:
            with open(args.report_out, "w") as fh:
                fh.write(report + "\n")
            print(f"wrote report: {args.report_out}", file=sys.stderr)
        elif not args.html_out:
            print(report)
        return

    if args.cache_dir is not None:
        os.environ["HF_HOME"] = args.cache_dir
        os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(args.cache_dir, "hub")

    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)

    # Heavy imports AFTER the cache env is set.
    import benchmark_knn_conv_models as models_bench
    from fovi.sensing.retina import RetinalTransform

    availability = backend_availability()
    print(json.dumps({
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "capability": torch.cuda.get_device_capability(device),
        "kind": "final_comparison", "batch": args.batch, "modes": args.modes,
        "backends_available": availability, "note": args.note,
    }), flush=True)

    families = []
    for name in args.models:
        family = DENSE_FAMILY[name]
        if family not in families:
            families.append(family)
    if args.dense in ("both", "64"):
        for family in families:
            run_dense(family, "logpolar@64", args, models_bench, RetinalTransform, device)
    if args.dense in ("both", "256"):
        for family in families:
            run_dense(family, "dense@256", args, models_bench, RetinalTransform, device)
    if not args.dense_only:
        for name in args.models:
            run_fovi_variant(name, args, models_bench, RetinalTransform, availability, device)

    print(summary_table(RECORDS), flush=True)

    # Postprocessing: render the human-readable report from the records just collected.
    if args.report_out:
        envs = [{"gpu": torch.cuda.get_device_name(device), "short": _gpu_short(
                    torch.cuda.get_device_name(device)),
                 "capability": list(torch.cuda.get_device_capability(device)),
                 "torch": torch.__version__, "cuda": torch.version.cuda, "source": "(this run)"}]
        for r in RECORDS:
            r.setdefault("_gpu", envs[0]["short"])
        data = [r for r in RECORDS if r.get("kind") != "final_comparison"]
        with open(args.report_out, "w") as fh:
            fh.write(render_report(data, envs) + "\n")
        print(f"wrote report: {args.report_out}", file=sys.stderr)
        if args.html_out:
            with open(args.html_out, "w") as fh:
                fh.write(render_html(data, envs))
            print(f"wrote html: {args.html_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
