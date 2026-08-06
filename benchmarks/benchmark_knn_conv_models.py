"""Benchmark baseline versus optimized KNN convolution in the two notebook models.

``--mode inference`` (default) preserves the original inference behavior. ``--mode train`` replicates the
real cached training configuration as closely as reasonable: ``model.train()``, autocast with the
model's configured AMP dtype (alexnet float16 with ``torch.amp.GradScaler(growth_interval=100)``
as in ``fovi/trainer.py``; dinov3 bfloat16 without a scaler), synthetic integer labels, and a
timed region of full supervised forward + cross-entropy loss + ``scaler.scale(loss).backward()``.
``optimizer.step()`` is excluded from the timed region and measured once, reported separately.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
from functools import partial

import torch
import torch.nn.functional as F
from einops import rearrange
from torch.amp import GradScaler, autocast

from fovi import get_model_from_base_fn
from fovi.arch.knn import KNNConvLayer
from fovi.arch.knn_optimization import VALID_BACKENDS


MODELS = {
    "alexnet": "fovi-alexnet_a-0.5_res-64_rfmult-2_in1k",
    # 1x reference-frame variant: identical KNN geometry, V == K on every conv
    # (ref frame side = ceil(sqrt(K)) instead of 2*ceil(sqrt(K))). First-class alongside
    # the rf2 default in all reporting (user directive).
    "alexnet_rf1": "fovi-alexnet_a-0.5_res-64_rfmult-1_in1k",
    "dinov3": "fovi-dinov3-splus_a-2.78_res-64_in1k",
}

# Models built from a LOCAL config with random init (no trained checkpoint exists).
# Used for speed and backend-vs-backend parity, never accuracy.
LOCAL_MODELS = {
    "resnet18": "benchmarks/configs/fovi-resnet18_bench.yaml",
    # HISTORY: KNNResNet previously ignored cfg.model.ref_frame_mult, so "resnet18" and
    # "resnet18_rf1" built identical V==K models. ref_frame_mult was later made REAL in
    # KNNResNet (default 1 = historical builds byte-identical; both yamls now pin 1
    # explicitly and the builder below passes it through), so all earlier resnet18 numbers
    # remain valid as rf1 numbers.
    "resnet18_rf1": "benchmarks/configs/fovi-resnet18_rf1_bench.yaml",
    # 2x reference-frame variant (first rf2 resnet18 benchmark): stem V=196,
    # k=9 blocks V=36, k=1 downsamples V=1 (ref frame side = 2*ceil(sqrt(K)) for K>1).
    "resnet18_rf2": "benchmarks/configs/fovi-resnet18_rf2_bench.yaml",
}


def build_local_model(name, device):
    """Construct a random-init model from a local benchmark config.

    resnet18: fovi_resnet18 has never been trained. `saccades.auto_match_cart_resources=1`
    (the alexnet template value, now the bench-config value) previously crashed KNNResNet's
    coordinate builder (`cart_res=None` reached `auto_match_num_coords` -> `None**2`); that
    is FIXED (knnresnet.py now seeds+threads cart_res like KNNAlexNet), so the resnet is
    cartesian-matched to the 64^2 dense ladder (input ~4085 nodes, then 964/230/60/16/2). A
    second latent issue remains worked around HERE: `KNNResNet.__init__`'s avgpool branch
    hits an UnboundLocalError (`in_coords` referenced before assignment, default out_res=1
    path), so we build with `out_res=None` (no KNN avg-pool; the head consumes the un-pooled
    512 x N_last features) — reported to the arch owners. Training config mirrors the alexnet
    recipe (fp16 AMP + GradScaler, batch 128, 4 fixations, adamw) as a documented assumption.
    """
    from omegaconf import OmegaConf
    from fovi.fovinet import FoviNet
    from fovi.arch import ARCHITECTURE_REGISTRY
    from fovi.arch.architectures import arch_wrapper, rescale_fov
    from fovi.arch.knnresnet import KNNResNet

    if name not in ("resnet18", "resnet18_rf1", "resnet18_rf2"):
        raise ValueError(f"unknown local model {name!r}")

    def bench_resnet18(cfg, device="cuda"):
        cfg = rescale_fov(cfg)
        knn = KNNResNet(
            layers=[2, 2, 2, 2], in_conv_stride=2, in_pool_stride=2,
            fov=cfg.saccades.fov, cmf_a=cfg.saccades.cmf_a, in_res=cfg.saccades.resize_size,
            style=cfg.saccades.mode, norm_type=cfg.model.norm, arch_flag=cfg.model.arch_flag,
            sample_cortex=cfg.saccades.sample_cortex, device=device,
            auto_match_cart_resources=cfg.saccades.auto_match_cart_resources,
            num_classes=None,
            out_res=None,  # avoids the knnresnet.py:248 UnboundLocalError (see docstring)
            # Now honored (default 1 = historical builds); pass the config value.
            ref_frame_mult=cfg.model.get("ref_frame_mult", 1),
        )
        return arch_wrapper(knn, cfg, device=device)

    try:
        ARCHITECTURE_REGISTRY.register("fovi_resnet18_bench", bench_resnet18)
    except Exception:
        pass  # already registered in this process
    cfg = OmegaConf.load(LOCAL_MODELS[name])
    cfg.model.arch = "fovi_resnet18_bench"
    return FoviNet(cfg, device=device)


SEED = 20260721

# Free-text tag (e.g. GPU co-tenancy) added to every emitted record; set from --note in main().
NOTE = None


def emit(record):
    if NOTE is not None:
        record["note"] = NOTE
    print(json.dumps(record), flush=True)


def knn_layers(model):
    return [(name, module) for name, module in model.named_modules() if isinstance(module, KNNConvLayer)]


def set_backend(model, backend):
    # KNNPoolingLayer.forward carries its own optimized hook, gated by the
    # FOVI_KNN_POOL_BACKEND env var which it reads PER CALL — tie the pooling arm to the
    # conv arm so 'baseline' rows stay honest end-to-end and every optimized arm includes
    # the pooling kernel (auto is the production default).
    os.environ["FOVI_KNN_POOL_BACKEND"] = "baseline" if backend == "baseline" else "auto"
    for _, layer in knn_layers(model):
        layer.kernel_backend = backend
        layer.clear_optimized_cache()


def output_tensor(value):
    while isinstance(value, (tuple, list)):
        value = value[0]
    return value


def full_forward(model, inputs, fixed_fixations, n_fixations=None):
    """n_fixations=None preserves the default behavior (the model's configured count)."""
    return model(
        inputs,
        setting="supervised",
        n_fixations=model.n_fixations if n_fixations is None else n_fixations,
        fixations=fixed_fixations,
    )


def network_forward(model, fixed_inputs):
    return model.network(fixed_inputs, return_layer_outputs=True)


def run_once(fn, amp_dtype):
    with torch.no_grad(), torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
        return output_tensor(fn())


def time_cuda(fn, amp_dtype, warmup, repeats):
    for _ in range(warmup):
        result = run_once(fn, amp_dtype)
        del result
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = run_once(fn, amp_dtype)
        end.record()
        end.synchronize()
        del result
        samples.append(start.elapsed_time(end))
    return statistics.median(samples), min(samples)


def memory_cuda(fn, amp_dtype):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    result = run_once(fn, amp_dtype)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    del result
    return (peak - before) / 2**20


def cache_mib(model):
    total = 0
    for _, layer in knn_layers(model):
        cached = getattr(layer, "_compact_effective_weight_cache", None)
        if cached is not None:
            total += cached[1].numel() * cached[1].element_size()
    return total / 2**20


def benchmark_scope(model, scope_name, fn, amp_dtype, warmup, repeats, optimized_backend):
    set_backend(model, "baseline")
    reference = run_once(fn, amp_dtype).float()
    baseline_ms, baseline_min = time_cuda(fn, amp_dtype, warmup, repeats)
    baseline_temp = memory_cuda(fn, amp_dtype)

    set_backend(model, optimized_backend)
    actual = run_once(fn, amp_dtype).float()
    max_abs = (reference - actual).abs().max().item()
    optimized_ms, optimized_min = time_cuda(fn, amp_dtype, warmup, repeats)
    optimized_temp = memory_cuda(fn, amp_dtype)
    backends = {name: getattr(layer, "_last_knn_backend", None) for name, layer in knn_layers(model)}
    cached_mib = cache_mib(model)
    record = {
        "scope": scope_name,
        "optimized_backend": optimized_backend,
        "dtype": "float32" if amp_dtype is None else str(amp_dtype).removeprefix("torch."),
        "baseline_median_ms": baseline_ms,
        "baseline_min_ms": baseline_min,
        "optimized_median_ms": optimized_ms,
        "optimized_min_ms": optimized_min,
        "speedup": baseline_ms / optimized_ms,
        "baseline_temporary_mib": baseline_temp,
        "optimized_temporary_mib": optimized_temp,
        "optimized_cache_mib": cached_mib,
        "optimized_cache_plus_temporary_mib": cached_mib + optimized_temp,
        "max_abs": max_abs,
        "backends": backends,
    }
    emit(record)


# ---------------------------------------------------------------------------
# Training-mode (forward+backward) benchmarking
# ---------------------------------------------------------------------------


def zero_grads(model):
    for parameter in model.parameters():
        parameter.grad = None


def pick_grad_sample(model):
    """Prefer the first KNN layer with a populated grad; fall back to any trainable parameter."""
    for name, module in model.named_modules():
        if isinstance(module, KNNConvLayer):
            for param_name, parameter in module.named_parameters():
                if parameter.requires_grad and parameter.grad is not None:
                    return f"{name}.{param_name}", parameter.grad
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            return name, parameter.grad
    return None, None


# Small scale for the dedicated parity scaler: at the timing scaler's default 65536,
# borderline fp16 grads overflow to inf BEFORE unscaling and poison the parity delta with
# nan (observed once in testing; real training's scaler.update() would back off, but parity
# passes never call update()). 2**8 preserves fp16 denormal resolution without overflow.
PARITY_INIT_SCALE = 2 ** 8


def parity_train(model, step_fn, scaler, state_snapshot):
    """One seeded fwd+bwd from a restored model state; returns loss, unscaled grad sample, name.

    ``step_fn``/``scaler`` must be the dedicated parity pair built with PARITY_INIT_SCALE.
    """
    model.load_state_dict(state_snapshot)
    zero_grads(model)
    torch.manual_seed(SEED)
    loss = step_fn()
    scale = scaler.get_scale() if scaler.is_enabled() else 1.0
    grad_name, grad = pick_grad_sample(model)
    grad = None if grad is None else grad.detach().float() / scale
    return loss.detach().float().item(), grad, grad_name


def time_train(step_fn, warmup, repeats, reset):
    for _ in range(warmup):
        reset()
        step_fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        reset()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        step_fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples), min(samples)


def memory_train(step_fn, reset):
    """Peak allocation of one fwd+bwd above the pre-step footprint (includes new grad storage)."""
    reset()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    step_fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - before) / 2**20


def make_full_train_step(model, inputs, fixed_fixations, labels, amp_dtype, label_smoothing,
                         scaler, n_fixations=None):
    n_fix = model.n_fixations if n_fixations is None else n_fixations

    def step():
        with autocast("cuda", dtype=amp_dtype or torch.float16, enabled=amp_dtype is not None):
            embeddings, _, _ = model(
                inputs,
                setting="supervised",
                n_fixations=n_fix,
                fixations=fixed_fixations,
            )
            loss = F.cross_entropy(embeddings, labels, label_smoothing=label_smoothing)
        scaler.scale(loss).backward()
        return loss

    return step


def make_network_train_step(model, fixed_inputs, amp_dtype, scaler, device):
    """Network-only scope: pseudo-loss is a dot with a fixed random unit grad_output."""
    state = {}

    def step():
        with autocast("cuda", dtype=amp_dtype or torch.float16, enabled=amp_dtype is not None):
            embeddings = output_tensor(model.network(fixed_inputs, return_layer_outputs=True))
        if "unit_grad" not in state:
            generator = torch.Generator(device=device).manual_seed(SEED)
            unit = torch.randn(
                embeddings.shape, device=device, dtype=torch.float32, generator=generator
            )
            state["unit_grad"] = unit / float(unit.numel()) ** 0.5
        loss = (embeddings.float() * state["unit_grad"]).sum()
        scaler.scale(loss).backward()
        return loss

    return step


def benchmark_scope_train(
    model, scope_name, step_fn, amp_dtype, warmup, repeats, optimized_backend, scaler,
    state_snapshot, parity_step_fn=None, parity_scaler=None
):
    reset = partial(zero_grads, model)
    # Timing uses the real training scaler (default init_scale); parity uses the dedicated
    # small-scale pair when provided so unscaled grad comparisons cannot hit fp16 inf.
    if parity_step_fn is None:
        parity_step_fn, parity_scaler = step_fn, scaler

    set_backend(model, "baseline")
    baseline_loss, baseline_grad, grad_name = parity_train(
        model, parity_step_fn, parity_scaler, state_snapshot
    )
    baseline_ms, baseline_min = time_train(step_fn, warmup, repeats, reset)
    baseline_temp = memory_train(step_fn, reset)

    set_backend(model, optimized_backend)
    optimized_loss, optimized_grad, _ = parity_train(
        model, parity_step_fn, parity_scaler, state_snapshot
    )
    optimized_ms, optimized_min = time_train(step_fn, warmup, repeats, reset)
    optimized_temp = memory_train(step_fn, reset)

    backends = {name: getattr(layer, "_last_knn_backend", None) for name, layer in knn_layers(model)}
    max_abs_grad = None
    grad_nonfinite = None
    if baseline_grad is not None and optimized_grad is not None:
        max_abs_grad = (baseline_grad - optimized_grad).abs().max().item()
        grad_nonfinite = bool(
            (~torch.isfinite(baseline_grad)).any() or (~torch.isfinite(optimized_grad)).any()
        )
    record = {
        "scope": scope_name,
        "mode": "train",
        "optimized_backend": optimized_backend,
        "dtype": "float32" if amp_dtype is None else str(amp_dtype).removeprefix("torch."),
        "grad_scaler_enabled": scaler.is_enabled(),
        "grad_scaler_scale": scaler.get_scale() if scaler.is_enabled() else None,
        "parity_scaler_scale": parity_scaler.get_scale() if parity_scaler.is_enabled() else None,
        "baseline_fwd_bwd_ms": baseline_ms,
        "baseline_fwd_bwd_min_ms": baseline_min,
        "optimized_fwd_bwd_ms": optimized_ms,
        "optimized_fwd_bwd_min_ms": optimized_min,
        "speedup": baseline_ms / optimized_ms,
        "baseline_temporary_mib": baseline_temp,
        "optimized_temporary_mib": optimized_temp,
        "optimized_cache_mib": cache_mib(model),
        "baseline_loss": baseline_loss,
        "optimized_loss": optimized_loss,
        "max_abs_loss": abs(baseline_loss - optimized_loss),
        "grad_sample_param": grad_name,
        "max_abs_grad_sample": max_abs_grad,
        "grad_sample_nonfinite": grad_nonfinite,
        "backends": backends,
    }
    emit(record)


def measure_optimizer_step(model, model_name, step_fn, amp_dtype):
    """Time optimizer.step() separately; run last because it mutates the weights."""
    cfg = model.cfg
    optimizer_name = cfg.training.optimizer
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    lr = float(cfg.training.base_lr)
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=lr, eps=float(cfg.training.eps))
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(parameters, lr=lr, momentum=float(cfg.training.momentum))
    else:
        optimizer = torch.optim.AdamW(parameters, lr=lr)
    zero_grads(model)
    step_fn()
    torch.cuda.synchronize()

    def one_step_ms():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        optimizer.step()
        end.record()
        end.synchronize()
        return start.elapsed_time(end)

    first_step_ms = one_step_ms()  # includes lazy optimizer-state allocation
    steady_step_ms = one_step_ms()
    record = {
        "scope": "optimizer_step",
        "mode": "train",
        "model": model_name,
        "optimizer": optimizer_name,
        "dtype": "float32" if amp_dtype is None else str(amp_dtype).removeprefix("torch."),
        "num_trainable_params": sum(parameter.numel() for parameter in parameters),
        "first_step_ms": first_step_ms,
        "steady_step_ms": steady_step_ms,
    }
    emit(record)


def main():
    parser = argparse.ArgumentParser()
    all_models = list(MODELS) + list(LOCAL_MODELS)
    parser.add_argument("--models", nargs="+", choices=all_models, default=list(MODELS))
    parser.add_argument("--batch", type=int, default=10)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--dtype", choices=("float32", "amp", "both"), default="both")
    parser.add_argument(
        "--mode",
        choices=("inference", "train"),
        default="inference",
        help="train benchmarks the full supervised fwd+bwd training step (see module docstring)",
    )
    parser.add_argument(
        "--optimized-backend",
        choices=sorted(VALID_BACKENDS),
        default="auto",
    )
    parser.add_argument(
        "--note",
        default=None,
        help="Free-text tag added to every record (e.g. GPU co-tenancy during the run)",
    )
    args = parser.parse_args()
    global NOTE
    NOTE = args.note
    if args.batch == 2:
        parser.error(
            "--batch 2 is ambiguous in the current FOVI fixation API; use batch 1 or batch >=3"
        )
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)

    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(device),
                "capability": torch.cuda.get_device_capability(device),
                "batch": args.batch,
                "mode": args.mode,
            }
        )
    )
    for model_name in args.models:
        if model_name in LOCAL_MODELS:
            model = build_local_model(model_name, str(device))
        else:
            model = get_model_from_base_fn(MODELS[model_name], device=str(device), quiet=True)
        model.eval()
        generator = torch.Generator(device=device).manual_seed(SEED)
        inputs = torch.rand(
            args.batch, 3, 256, 256, generator=generator, device=device, dtype=torch.float32
        )
        torch.manual_seed(SEED)  # sup_fixator draws random fixations from the global RNG
        with torch.no_grad():
            fixed = model.sup_fixator(inputs, n_fixations=model.n_fixations)
        fixed_fixations = list(fixed["fixations"].unbind(dim=1))
        fixed_inputs = rearrange(fixed["x_fixs"], "b f c n -> (f b) c n")

        full_fn = partial(full_forward, model, inputs, fixed_fixations)
        network_fn = partial(network_forward, model, fixed_inputs)
        dtypes = []
        if args.dtype in ("float32", "both"):
            dtypes.append(None)
        if args.dtype in ("amp", "both"):
            dtypes.append(model.amp_dtype)
        base_fn = MODELS.get(model_name, LOCAL_MODELS.get(model_name))
        print(json.dumps({"model": model_name, "base_fn": base_fn, "fixations": model.n_fixations}))
        if args.mode == "inference":
            for amp_dtype in dtypes:
                benchmark_scope(
                    model, "network", network_fn, amp_dtype, args.warmup, args.repeats,
                    args.optimized_backend,
                )
                benchmark_scope(
                    model, "full_fovinet", full_fn, amp_dtype, args.warmup, args.repeats,
                    args.optimized_backend,
                )
        else:
            model.train()
            use_amp = bool(model.cfg.training.use_amp)
            label_smoothing = float(model.cfg.training.label_smoothing)
            labels = torch.randint(
                0, model.num_classes, (args.batch,), device=device, generator=generator
            )
            print(
                json.dumps(
                    {
                        "model": model_name,
                        "mode": "train",
                        "config_amp_dtype": str(model.amp_dtype).removeprefix("torch."),
                        "config_use_amp": use_amp,
                        "config_batch_size": int(model.cfg.training.batch_size),
                        "label_smoothing": label_smoothing,
                        "total_params": sum(p.numel() for p in model.parameters()),
                        "trainable_params": sum(
                            p.numel() for p in model.parameters() if p.requires_grad
                        ),
                    }
                ),
                flush=True,
            )
            state_snapshot = {
                key: value.detach().clone() for key, value in model.state_dict().items()
            }
            last_step_fn = None
            last_amp_dtype = None
            for amp_dtype in dtypes:
                # Replicate fovi/trainer.py: the scaler is enabled only for non-bf16 AMP.
                scaler_enabled = use_amp and amp_dtype is not None and amp_dtype != torch.bfloat16
                scaler = GradScaler("cuda", enabled=scaler_enabled, growth_interval=100)
                # Dedicated small-scale scaler for parity passes (see parity_train).
                parity_scaler = GradScaler(
                    "cuda", enabled=scaler_enabled, init_scale=PARITY_INIT_SCALE
                )
                network_step = make_network_train_step(model, fixed_inputs, amp_dtype, scaler, device)
                full_step = make_full_train_step(
                    model, inputs, fixed_fixations, labels, amp_dtype, label_smoothing, scaler
                )
                network_parity_step = make_network_train_step(
                    model, fixed_inputs, amp_dtype, parity_scaler, device
                )
                full_parity_step = make_full_train_step(
                    model, inputs, fixed_fixations, labels, amp_dtype, label_smoothing, parity_scaler
                )
                benchmark_scope_train(
                    model, "network", network_step, amp_dtype, args.warmup, args.repeats,
                    args.optimized_backend, scaler, state_snapshot,
                    parity_step_fn=network_parity_step, parity_scaler=parity_scaler,
                )
                benchmark_scope_train(
                    model, "full_fovinet", full_step, amp_dtype, args.warmup, args.repeats,
                    args.optimized_backend, scaler, state_snapshot,
                    parity_step_fn=full_parity_step, parity_scaler=parity_scaler,
                )
                last_step_fn = full_step
                last_amp_dtype = amp_dtype
            # Run last: optimizer.step() mutates the weights.
            measure_optimizer_step(model, model_name, last_step_fn, last_amp_dtype)
            del state_snapshot, labels, last_step_fn

        del model, inputs, fixed, fixed_fixations, fixed_inputs, full_fn, network_fn
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
