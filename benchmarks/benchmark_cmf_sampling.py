"""Bounded synthetic benchmark of mixed CMF sampling and a DINOv3 training step.

Uses randomly initialized DINOv3-S weights locally; downloads no weights or data.
Reports the scalar per-batch baseline and per-image bank selection with identical
model, image, fixation, and optimizer workloads.
"""

import argparse
import json
from collections.abc import Callable
from functools import partial
from statistics import median
from time import perf_counter

import torch
from fovi.sensing.projection import CameraModel
from fovi.sensing.retina import RetinalTransform
from fovi.utils.fastaugs import transforms as fastT
from torch import nn
from transformers import DINOv3ViTConfig, DINOv3ViTModel


def measure(
    operation: Callable[[], None], warmup: int, iterations: int
) -> dict[str, float]:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    samples = []
    for _ in range(iterations):
        start = perf_counter()
        operation()
        torch.cuda.synchronize()
        samples.append((perf_counter() - start) * 1000)
    return {
        "median_ms": median(samples),
        "peak_memory_mib": torch.cuda.max_memory_allocated() / 1024**2,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--fixations", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--geometry", choices=["planar", "spherical"], default="planar")
    parser.add_argument("--tuning", choices=["full", "lora"], default="lora")
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.backends.cuda.enable_cudnn_sdp(False)
    device = torch.device("cuda")
    levels = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    kwargs = {
        "resolution": 128,
        "start_res": 256,
        "fixation_size": 256,
        "fov": 16.0,
        "style": "warped_cartesian_as_grid",
        "sampler": "grid_nn",
        "sampler_backend": "cuda",
        "device": device,
        "field_geometry": args.geometry,
        "pre_transforms": fastT.Compose(
            [fastT.NormalizeGPU([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        ),
    }
    if args.geometry == "spherical":
        kwargs["camera_model"] = CameraModel(
            "pinhole", (256, 256), (180, 180, 127.5, 127.5)
        )
    bank = RetinalTransform(cmf_a=levels, **kwargs)
    scalars = [RetinalTransform(cmf_a=value, **kwargs) for value in levels]
    images = torch.randint(
        0, 256, (args.batch_size, 3, 256, 256), device=device, dtype=torch.uint8
    )
    fixations = (
        torch.rand(args.fixations, args.batch_size, 2, device=device) * 0.5 + 0.25
    )
    model = (
        DINOv3ViTModel(
            DINOv3ViTConfig(
                image_size=128,
                patch_size=16,
                hidden_size=384,
                num_hidden_layers=12,
                num_attention_heads=6,
                intermediate_size=1536,
                num_register_tokens=4,
            )
        )
        .to(device)
        .train()
    )
    if args.tuning == "lora":
        from pathlib import Path

        from fovi.models.dinov3 import prep_fovi_dinov3_finetuning
        from fovi.models.loading import load_config

        cfg, _, _ = load_config(
            "dinov3_warped_grid_multi_cmf",
            False,
            Path(__file__).parents[1] / "config",
            device="cpu",
        )
        model = prep_fovi_dinov3_finetuning(model, cfg, device=device)
    head = nn.Linear(384, 1000).to(device)
    parameters = [
        p for p in (*model.parameters(), *head.parameters()) if p.requires_grad
    ]
    optimizer = torch.optim.AdamW(parameters, lr=1e-4)
    labels = torch.arange(args.batch_size, device=device) % 1000
    step = 0

    def sample(mixed: bool) -> torch.Tensor:
        nonlocal step
        if mixed:
            retina = bank
            selection = {
                "cmf_indices": torch.randint(
                    len(levels), (args.batch_size,), device=device
                )
            }
        else:
            retina = scalars[step % len(levels)]
            selection = {}
        step += 1
        return torch.cat(
            [retina(images, fixation, **selection) for fixation in fixations]
        )

    def train_step(mixed: bool) -> None:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pixels = sample(mixed)
            features = (
                model(pixels)
                .pooler_output.reshape(args.fixations, args.batch_size, -1)
                .mean(0)
            )
            loss = nn.functional.cross_entropy(head(features), labels)
        loss.backward()
        optimizer.step()

    report = {"gpu": torch.cuda.get_device_name(), **vars(args)}
    # Alternating repeats reduce warmup/order bias in the comparison.
    for label, operation in [("sampler", sample), ("training_step", train_step)]:
        results = {"per_batch": [], "per_image": []}
        for order in ((False, True), (True, False)):
            for mixed in order:
                name = "per_image" if mixed else "per_batch"
                results[name].append(
                    measure(partial(operation, mixed), args.warmup, args.iterations)
                )
        report[label] = {}
        for name, runs in results.items():
            report[label][name] = {
                key: median(run[key] for run in runs) for key in runs[0]
            }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
