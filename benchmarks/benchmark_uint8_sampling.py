"""Benchmark moving-grid uint8 retinal sampling.

Run from the repository root::

    python benchmarks/benchmark_uint8_sampling.py --device cuda:0

Every measured path returns unit-range float output.  The PyTorch direct-index and native
paths convert only the compact sampled tensor; the ``grid_sample`` baseline converts the
full camera frame before sampling.  Two regimes are reported: a fixed foveated sampling
resolution across every input size, and a scale-matched regime targeting one sampled point
per 4x4 input region (16x fewer points than the input image).
"""

import argparse
import statistics

import torch

from fovi.sensing.samplers import GridSampler
from fovi.sensing.coords import find_desired_res


INPUT_CASES = (
    ("256p", 256, 456),
    ("1080p", 1080, 1920),
    ("4K", 2160, 3840),
)


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

    def native_sample():
        return native(image, fix_loc, fix_size)

    def eager_unit():
        return eager(image, fix_loc, fix_size).float().div_(255.0)

    def native_unit():
        return native(image, fix_loc, fix_size).float().div_(255.0)

    def float_grid_sample():
        return float_sampler(image.float().div_(255.0), fix_loc, fix_size)

    # Compile the native kernel outside measurements.
    native_sample()
    torch.cuda.synchronize()
    results = {
        "PyTorch direct + compact conversion": _measure(eager_unit),
        "native + compact conversion": _measure(native_unit),
        "full-frame conversion + grid_sample": _measure(float_grid_sample),
    }
    return eager.sampling_grid.shape[2], results


def _print_case(label, height, width, output_resolution, device, target_points=None):
    points, results = benchmark_case(height, width, output_resolution, device)
    target = "" if target_points is None else f", target points={target_points}"
    print(
        f"\n{label} ({height}x{width}), sampling resolution={output_resolution}, "
        f"sampled points={points}{target}")
    for name, milliseconds in results.items():
        print(f"  {name:38s} {milliseconds:8.4f} ms")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-resolution", type=int, default=64)
    parser.add_argument("--downsample-per-side", type=int, default=4)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires CUDA")
    if args.downsample_per_side < 1:
        raise ValueError("--downsample-per-side must be positive")

    print(f"device: {torch.cuda.get_device_name(torch.device(args.device))}")
    print(
        f"\nFixed foveated sampling resolution: {args.output_resolution} "
        "(sample count is independent of input resolution)")
    for label, height, width in INPUT_CASES:
        _print_case(
            label, height, width, args.output_resolution, args.device)

    ratio = args.downsample_per_side
    print(
        f"\nScale-matched sampling: {ratio}x downsampling per side "
        f"({ratio * ratio}x fewer sampled points)")
    for label, height, width in INPUT_CASES:
        target_points = (height // ratio) * (width // ratio)
        output_resolution, _ = find_desired_res(
            16.0, 0.5, target_points, "isotropic", device=args.device,
            bounds=(1, 2048), force_less_than=True, quiet=True)
        _print_case(
            label, height, width, output_resolution, args.device,
            target_points=target_points)


if __name__ == "__main__":
    main()
