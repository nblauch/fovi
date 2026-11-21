"""Compare fused and Torch inference pooling in separate, reproducible processes.

Run each backend separately for isolated allocator peaks, for example:
    python benchmarks/benchmark_grid_pooling.py --backend warp --output /tmp/pool-warp.json
    python benchmarks/benchmark_grid_pooling.py --backend torch --output /tmp/pool-torch.json

The Torch reference reproduces the public layer's eager visibility/dtype path.
Timings include synchronized pooling and exclude setup, rendering and physics.
Memory is PyTorch's peak allocated bytes during measurement, not total device memory.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from fovi.arch.pooling import GridPoolingLayer
from fovi.sensing.coords import SamplingCoords


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("warp", "torch"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--seconds", type=float, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--mode", choices=("avg", "max"), default="avg")
    args = parser.parse_args()
    if args.batch < 1 or args.resolution < 1 or args.seconds <= 0 or args.repeats < 1:
        parser.error("batch, resolution, seconds and repeats must be positive")
    torch.cuda.set_device(0)
    low = SamplingCoords(
        90,
        0.591,
        args.resolution,
        style="warped_cartesian_as_grid",
        field_geometry="spherical",
        device="cuda",
    )
    high = low.clone(resolution=2 * args.resolution)
    pool = GridPoolingLayer(None, high, low, mode=args.mode)
    generator = torch.Generator(device="cuda").manual_seed(42)
    storage = torch.randint(
        0,
        256,
        (args.batch, len(high), 4),
        dtype=torch.uint8,
        device="cuda",
        generator=generator,
    )
    values = storage[..., :3].transpose(1, 2)
    visible = (
        torch.rand((args.batch, len(high)), device="cuda", generator=generator) > 0.1
    )

    def reference() -> torch.Tensor:
        work = values.float()
        seen = visible[:, None, :]
        masked = torch.where(seen, work, -torch.inf if args.mode == "max" else 0.0)
        pooled = pool._forward_torch(
            torch.cat((masked, seen.to(work.dtype)), dim=1), normalize=False
        )
        result, coverage = pooled[:, :3], pooled[:, 3:]
        if args.mode != "max":
            result = result / coverage.clamp_min(torch.finfo(work.dtype).tiny)
        result = torch.where(coverage > 0, result, 0)
        return result.round().clamp(0, 255).to(torch.uint8)

    expected = reference()
    actual = pool(values, visible)
    torch.testing.assert_close(actual, expected)
    del expected, actual
    operation = reference if args.backend == "torch" else lambda: pool(values, visible)
    for _ in range(20):
        operation()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    windows = []
    for _ in range(args.repeats):
        seconds, calls = 0.0, 0
        while seconds < args.seconds:
            torch.cuda.synchronize()
            start = time.perf_counter()
            operation()
            torch.cuda.synchronize()
            seconds += time.perf_counter() - start
            calls += 1
        windows.append(
            {"updates": calls, "seconds": seconds, "mean_ms": seconds * 1000 / calls}
        )
    summary = {
        "gpu": torch.cuda.get_device_name(),
        "backend": args.backend,
        "batch": args.batch,
        "mode": args.mode,
        "input_shape": list(values.shape),
        "input_stride": list(values.stride()),
        "output_samples": len(low),
        "warmup_updates": 20,
        "windows": windows,
        "median_ms": statistics.median(w["mean_ms"] for w in windows),
        "torch_baseline_allocated_mib": baseline / 2**20,
        "torch_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
