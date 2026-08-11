"""Benchmark moving-grid retinal sampling for uint8 and float32 inputs.

Run from the repository root::

    python benchmarks/benchmark_uint8_sampling.py --device cuda:0 --batch-size 512

Every measured path returns unit-range float output.  The PyTorch direct-index and native
paths convert only the compact sampled tensor; the ``grid_sample`` baseline converts the
full camera frame before sampling.  Additional tables start with already-floating inputs
and preserve float32 coordinate math for float16, native math for float32/float64, and the
input dtype at output.  PyTorch grid_sample requires input and grid dtypes to match, so its
correctness-preserving float16 path includes full-frame float32 promotion and a compact
output cast.  Two regimes are reported: a fixed foveated sampling resolution across every
input size, and a scale-matched regime targeting one sampled point per 4x4 input region
(16x fewer points than the input image).
"""

import argparse
import statistics
from types import SimpleNamespace

import torch

from fovi.sensing.coords import _compute_isotropic_r_and_num_theta, find_desired_res
from fovi.sensing.samplers import GridSampler


INPUT_CASES = (
    ("256p", 256, 456),
    ("1080p", 1080, 1920),
    ("4K", 2160, 3840),
    ("16K", 8640, 15360),
)

FLOAT_CASES = (
    ("float16 (float32 math; grid_sample promotes input)",
     torch.float16, torch.float32),
    ("float32", torch.float32, torch.float32),
    ("float64", torch.float64, torch.float64),
)


def _measure(fn, iterations, warmup=30, repeats=5):
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


def _sampling_coords(output_resolution, device):
    """Build the sampling-relevant coordinates without plotting metadata.

    SamplingCoords also prepares plotting and cortical representations and builds each
    polar sample in Python.  Those one-time facilities are outside this benchmark and
    become prohibitively slow for the roughly 8.3M-point 16K case.
    """
    if output_resolution == 1:
        origin = torch.zeros((1, 2), device=device)
        return SimpleNamespace(cartesian=origin, polar=origin.clone())
    radius, n_angles = _compute_isotropic_r_and_num_theta(
        16.0, 0.5, output_resolution, device=device)
    n_angles = n_angles.to(device=device)
    radii = torch.repeat_interleave(radius, n_angles)
    ring_starts = torch.cumsum(n_angles, dim=0) - n_angles
    point_starts = torch.repeat_interleave(ring_starts, n_angles)
    point_counts = torch.repeat_interleave(n_angles, n_angles)
    point_in_ring = torch.arange(radii.numel(), device=device) - point_starts
    angles = point_in_ring.to(radii.dtype) * (2.0 * torch.pi) / point_counts
    polar = torch.stack((radii, angles), dim=1)
    cartesian = torch.stack(
        (radii * torch.cos(angles), radii * torch.sin(angles)), dim=1)
    return SimpleNamespace(cartesian=cartesian, polar=polar)


def benchmark_case(
        height, width, output_resolution, device, iterations, batch_size=1):
    # Each case can have a very different allocation profile. Release cached blocks from
    # the previous untimed case so a large dtype conversion is not defeated by allocator
    # fragmentation even when its live tensors fit on the device.
    torch.cuda.empty_cache()
    image = torch.randint(
        0, 256, (batch_size, 3, height, width), dtype=torch.uint8, device=device)
    fix_loc = torch.tensor([[0.47, 0.53]], device=device).expand(batch_size, -1)
    fix_size = torch.tensor(
        [[min(height, width), min(height, width)]], device=device
    ).expand(batch_size, -1)

    coords = _sampling_coords(output_resolution, device)
    eager = GridSampler(
        16.0, 0.5, output_resolution, device=device, mode="nearest", backend="torch",
        coords=coords)
    native = GridSampler(
        16.0, 0.5, output_resolution, device=device, mode="nearest", backend="cuda",
        coords=eager.coords)
    converted_sampler = GridSampler(
        16.0, 0.5, output_resolution, device=device, mode="nearest", backend="torch",
        coords=eager.coords)

    def native_sample():
        return native(image, fix_loc, fix_size)

    def eager_unit():
        return eager(image, fix_loc, fix_size).float().div_(255.0)

    def native_unit():
        return native(image, fix_loc, fix_size).float().div_(255.0)

    def converted_grid_sample():
        return converted_sampler(
            image.float().div_(255.0), fix_loc, fix_size)

    # Compile the uint8 kernel outside measurements.
    native_sample()
    torch.cuda.synchronize()
    uint8_results = {
        "PyTorch direct + compact conversion": _measure(eager_unit, iterations),
        "native + compact conversion": _measure(native_unit, iterations),
        "full-frame conversion + grid_sample": _measure(
            converted_grid_sample, iterations),
    }
    float_results = {}
    for dtype_label, image_dtype, coordinate_dtype in FLOAT_CASES:
        float_image = image.to(image_dtype).div_(255.0)
        direct_sampler = GridSampler(
            16.0, 0.5, output_resolution, device=device, dtype=coordinate_dtype,
            mode="nearest", backend="torch", coords=coords)
        native_sampler = GridSampler(
            16.0, 0.5, output_resolution, device=device, dtype=coordinate_dtype,
            mode="nearest", backend="cuda", coords=coords)
        grid_sampler = GridSampler(
            16.0, 0.5, output_resolution, device=device, dtype=coordinate_dtype,
            mode="nearest", backend="torch", coords=coords)

        def float_direct():
            return direct_sampler(
                float_image, fix_loc, fix_size, direct=True)

        def float_native():
            return native_sampler(float_image, fix_loc, fix_size)

        def float_grid_sample():
            return grid_sampler(float_image, fix_loc, fix_size)

        # Compile and validate each public native path outside timing.
        native_float = float_native()
        float_direct_reference = float_direct()
        torch.testing.assert_close(
            native_float, float_direct_reference, rtol=0.0, atol=0.0)
        if native_sampler._last_backend != "cuda":
            raise RuntimeError("native benchmark did not use the CUDA backend")
        grid_sample_reference = float_grid_sample()
        grid_mismatch_percent = (
            torch.count_nonzero(
                grid_sample_reference != float_direct_reference).item()
            * 100.0 / float_direct_reference.numel())
        del native_float, float_direct_reference, grid_sample_reference
        torch.cuda.synchronize()
        float_results[dtype_label] = {
            "timings": {
                "PyTorch direct": _measure(float_direct, iterations),
                "native fused": _measure(float_native, iterations),
                "grid_sample": _measure(float_grid_sample, iterations),
            },
            "grid_mismatch_percent": grid_mismatch_percent,
        }
        del (
            float_direct, float_native, float_grid_sample, direct_sampler,
            native_sampler, grid_sampler, float_image)
        torch.cuda.empty_cache()
    return eager.sampling_grid.shape[2], uint8_results, float_results


def _print_case(
        label, height, width, output_resolution, device, iterations,
        batch_size=1, target_points=None):
    points, uint8_results, float_results = benchmark_case(
        height, width, output_resolution, device, iterations, batch_size)
    target = "" if target_points is None else f", target points={target_points}"
    print(
        f"\n{label} ({height}x{width}), sampling resolution={output_resolution}, "
        f"sampled points={points}{target}")
    print("  uint8 input -> unit float output")
    for name, milliseconds in uint8_results.items():
        print(f"    {name:38s} {milliseconds:8.4f} ms")
    print("  pre-existing floating input")
    for dtype_label, result in float_results.items():
        print(f"    {dtype_label}")
        for name, milliseconds in result["timings"].items():
            print(f"      {name:36s} {milliseconds:8.4f} ms")
        print(
            "      grid_sample/direct value mismatch "
            f"{result['grid_mismatch_percent']:8.4f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-resolution", type=int, default=64)
    parser.add_argument("--downsample-per-side", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--inputs", nargs="+", choices=[case[0] for case in INPUT_CASES],
        default=[case[0] for case in INPUT_CASES],
        help="input resolutions to benchmark (default: all)")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires CUDA")
    if args.downsample_per_side < 1:
        raise ValueError("--downsample-per-side must be positive")
    if args.iterations < 1:
        raise ValueError("--iterations must be positive")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")

    print(f"device: {torch.cuda.get_device_name(torch.device(args.device))}")
    print(f"batch size: {args.batch_size}")
    print(f"timing: median of 5 repeats, {args.iterations} iterations per repeat")
    print(
        f"\nFixed foveated sampling resolution: {args.output_resolution} "
        "(sample count is independent of input resolution)")
    for label, height, width in INPUT_CASES:
        if label not in args.inputs:
            continue
        _print_case(
            label, height, width, args.output_resolution, args.device,
            args.iterations, batch_size=args.batch_size)

    ratio = args.downsample_per_side
    print(
        f"\nScale-matched sampling: {ratio}x downsampling per side "
        f"({ratio * ratio}x fewer sampled points)")
    for label, height, width in INPUT_CASES:
        if label not in args.inputs:
            continue
        target_points = (height // ratio) * (width // ratio)
        output_resolution, _ = find_desired_res(
            16.0, 0.5, target_points, "isotropic", device=args.device,
            bounds=(1, 4096), force_less_than=True, quiet=True)
        _print_case(
            label, height, width, output_resolution, args.device, args.iterations,
            batch_size=args.batch_size, target_points=target_points)


if __name__ == "__main__":
    main()
