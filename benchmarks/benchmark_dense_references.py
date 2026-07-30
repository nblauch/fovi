"""Practical dense CNN/ViT references for the three FOVI architectures.

Framing (user directive): these measure the efficiency of the ALTERNATIVE design — a
downsampled/warped-2D input feeding standard Conv2d — versus the 3D manifold that requires
KNNConv. Protocol rules: same initial resolution (64x64-equivalent) and same effective
sample count (network batch = model batch x n_fixations, since a warped input is
per-fixation too). All comparisons are NETWORK-scope with the models benchmark's
pseudo-loss protocol (fixed seeded random unit grad_output as a dot-product loss), so
head differences do not contaminate the trunk comparison; the retinal front-end share of
the foveated models is quantified separately (full_fovinet minus network scope).

Per-model dense variants:
- resnet18: torchvision.models.resnet18 (canonical dense ladder at 64x64: 32->16->16->8->
  4->2 vs foveated nodes 1469->356->356->83->16->2). The repo's polar-capable subclass
  (fovi/arch/resnet.py) is NOT used: its __init__ calls the torchvision parent without
  block/layers (line ~262) and raises TypeError — reported to the arch owners.
- alexnet: the repo's own dense spec the KNN kernels were derived from:
  fovi.arch.alexnet.get_backbone(kernels=baseline_alexnet_kernels['base_lowres']) —
  the 64x64-adapted kernel/stride ladder (k11 s2 stem).
- dinov3: the SAME HF checkpoint config built with dense patch embedding
  (get_model_from_base_fn(..., load=False, model.vit.partitioning_patches=None)): the
  pretrained Conv2d patch embed resampled to kernel=8, stride=8 on the 64x64 input ->
  8x8 = 64 tokens, standard DINOv3ViTRopePositionEmbedding instead of the foveated RoPE,
  same 12-layer trunk with the same freeze/LoRA setup from the cached config. Differences
  beyond the patch embed: RoPE coordinate source (grid vs foveated coords); everything
  else identical. Random/pretrained-free weights (load=False) — speed only.

dtypes mirror the real training configs: fp16 AMP + GradScaler (resnet18, alexnet),
bf16 AMP without scaler (dinov3).

Run from the repo root:
    python benchmarks/benchmark_dense_references.py --models resnet18 alexnet dinov3 \
        --batch 40 512 --device 0
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics

import torch
from torch.amp import GradScaler, autocast

SEED = 20260721


def build(name, device):
    if name == "resnet18":
        import torchvision.models

        return torchvision.models.resnet18(num_classes=1000).to(device), torch.float16
    if name == "alexnet":
        from fovi.arch.alexnet import baseline_alexnet_kernels, get_backbone

        return get_backbone(kernels=baseline_alexnet_kernels["base_lowres"]).to(device), torch.float16
    if name in ("dinov3", "dinov3_hplus"):
        # Build the DENSE backbone directly via build_fovi_dinov3 (the dense patch-embed
        # branch is gated on 'as_grid' in saccades.mode; FoviNet's RetinalTransform rejects
        # such modes, so we bypass FoviNet entirely — trunk-only reference, no projector).
        from fovi import find_config
        from fovi.arch.dinov3 import build_fovi_dinov3
        from fovi.arch.knn import KNNConvLayer

        cfgname = {"dinov3": "fovi-dinov3-splus_a-2.78_res-64_in1k",
                   "dinov3_hplus": "fovi-dinov3-hplus_a-2.78_res-64_in1k"}[name]
        cfg, _, _ = find_config(cfgname, load=False)
        cfg.saccades.mode = "grid_as_grid"
        backbone = build_fovi_dinov3(cfg, device=str(device))
        knn = [n for n, m in backbone.named_modules() if isinstance(m, KNNConvLayer)]
        assert not knn, f"dense dinov3 build still contains KNN layers: {knn}"

        class DenseViT(torch.nn.Module):
            def __init__(self, net):
                super().__init__()
                self.net = net

            def forward(self, x):
                return self.net(x)

        return DenseViT(backbone).to(device), torch.bfloat16
    raise ValueError(name)


def time_cuda(fn, warmup, repeats, between=None):
    for _ in range(warmup):
        if between:
            between()
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        if between:
            between()
            torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples), min(samples)


def peak_mib(fn, between=None):
    if between:
        between()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - before) / 2**20


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+",
                        choices=("resnet18", "alexnet", "dinov3", "dinov3_hplus"),
                        default=["resnet18", "alexnet", "dinov3"])
    parser.add_argument("--batch", type=int, nargs="+", default=[40, 512],
                        help="NETWORK batch (= model batch x 4 fixations)")
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--note", default=None)
    args = parser.parse_args()
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)

    header = {
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device), "kind": "dense_reference",
        "resolution": args.resolution,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
    }
    if args.note is not None:
        header["note"] = args.note
    print(json.dumps(header), flush=True)

    for name in args.models:
        model, amp_dtype = build(name, device)
        scaler = GradScaler("cuda", enabled=amp_dtype == torch.float16, growth_interval=100)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        generator = torch.Generator(device=device).manual_seed(SEED)
        for batch in args.batch:
            x = torch.rand(batch, 3, args.resolution, args.resolution, device=device,
                           generator=generator)
            state = {}

            def reset():
                for p in model.parameters():
                    p.grad = None

            def train_step():
                with autocast("cuda", dtype=amp_dtype, enabled=True):
                    y = model(x)
                if "unit" not in state:
                    g = torch.Generator(device=device).manual_seed(SEED)
                    u = torch.randn(y.shape, device=device, dtype=torch.float32, generator=g)
                    state["unit"] = u / float(u.numel()) ** 0.5
                loss = (y.float() * state["unit"]).sum()
                scaler.scale(loss).backward()

            def infer_fwd():
                with torch.no_grad(), autocast("cuda", dtype=amp_dtype, enabled=True):
                    return model(x)

            model.train()
            train_ms, train_min = time_cuda(train_step, args.warmup, args.repeats, between=reset)
            train_temp = peak_mib(train_step, between=reset)
            model.eval()
            infer_ms, infer_min = time_cuda(infer_fwd, args.warmup, args.repeats)
            infer_temp = peak_mib(infer_fwd)
            record = {
                "kind": "dense_reference", "model": name, "network_batch": batch,
                "dtype": str(amp_dtype).replace("torch.", "") + "_amp",
                "trainable_params": trainable,
                "train_fwd_bwd_ms": train_ms, "train_fwd_bwd_min_ms": train_min,
                "train_temporary_mib": train_temp,
                "infer_ms": infer_ms, "infer_min_ms": infer_min,
                "infer_temporary_mib": infer_temp,
            }
            if args.note is not None:
                record["note"] = args.note
            print(json.dumps(record), flush=True)
            state.clear()
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
