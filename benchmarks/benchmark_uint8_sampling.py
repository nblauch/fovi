"""Benchmark moving-grid uint8 retinal sampling.

Run from the repository root::

    python benchmarks/benchmark_uint8_sampling.py --device cuda:0

The float baseline includes the full-frame uint8-to-unit-float conversion that a camera
stream otherwise pays before ``torch.grid_sample``.  ``native + compact unit`` includes the
small conversion performed by ``RetinalTransform`` after native nearest sampling.
"""

import argparse
import statistics

import torch

from fovi.sensing.samplers import GridSampler


def _measure(fn, warmup=30, iterations=300, repeats=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    timings = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end) / iterations)
    return statistics.median(timings)


def benchmark_case(height, width, output_resolution, device):
    image = torch.randint(
        0, 256, (1, 3, height, width), dtype=torch.uint8, device=device)
    fix_loc = torch.tensor([[0.47, 0.53]], device=device)
    fix_size = torch.tensor([[min(height, width), min(height, width)]], device=device)

    eager = GridSampler(
        16.0, 0.5, output_resolution, device=device, mode="nearest", backend="torch")
    native = GridSampler(
        16.0, 0.5, output_resolution, device=device, mode="nearest", backend="cuda",
        coords=eager.coords)
    float_sampler = GridSampler(
        16.0, 0.5, output_resolution, device=device, mode="nearest", backend="torch",
        coords=eager.coords)

    def eager_sample():
        return eager(image, fix_loc, fix_size)

    def native_sample():
        return native(image, fix_loc, fix_size)

    def native_unit():
        return native(image, fix_loc, fix_size).float().div_(255.0)

    def float_grid_sample():
        return float_sampler(image.float().div_(255.0), fix_loc, fix_size)

    # Compile the native kernel outside measurements.
    native_sample()
    torch.cuda.synchronize()
    results = {
        "eager gather": _measure(eager_sample),
        "native": _measure(native_sample),
        "native + compact unit": _measure(native_unit),
        "full float + grid_sample": _measure(float_grid_sample),
    }
    return eager.sampling_grid.shape[2], results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-resolution", type=int, default=64)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires CUDA")

    print(f"device: {torch.cuda.get_device_name(torch.device(args.device))}")
    for label, height, width in (
        ("720p", 720, 1280),
        ("1080p", 1080, 1920),
        ("4K", 2160, 3840),
    ):
        points, results = benchmark_case(
            height, width, args.output_resolution, args.device)
        print(f"\n{label} ({height}x{width}), sampled points={points}")
        for name, milliseconds in results.items():
            print(f"  {name:26s} {milliseconds:8.4f} ms")


if __name__ == "__main__":
    main()
