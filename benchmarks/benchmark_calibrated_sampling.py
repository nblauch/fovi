"""Compare native/compiled calibrated gaze and canonical-grid image translation."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict
from functools import partial
from pathlib import Path

import torch

from fovi.sensing.projection import CameraModel
from fovi.sensing.samplers import GridSampler


def measure(operation: Callable[[], torch.Tensor | None], iterations: int) -> float:
    for _ in range(10):
        operation()
    torch.cuda.synchronize()
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    start.record()
    for _ in range(iterations):
        operation()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def measure_graph(operation: Callable[[], torch.Tensor], iterations: int) -> float:
    """Measure replay after capture of a warmed, fixed-shape sampling operation."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            operation()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = operation()
    milliseconds = measure(graph.replay, iterations)
    # Keep captured output storage alive through the last replay.
    assert output.is_cuda
    return milliseconds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 32, 256])
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--dtypes",
        nargs="+",
        choices=["uint8", "float16", "bfloat16", "float32", "float64"],
        default=["uint8"],
    )
    parser.add_argument(
        "--lenses",
        nargs="+",
        choices=["pinhole", "fisheye"],
        default=["pinhole", "fisheye"],
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["nearest", "bilinear"],
        default=["nearest", "bilinear"],
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument(
        "--convention", choices=["camera_xyz", "pan_tilt"], default="camera_xyz"
    )
    args = parser.parse_args()
    torch.cuda.set_device(args.device)
    torch.manual_seed(2026)
    results = []
    lenses = [
        ("pinhole", ()),
        ("pinhole", (-0.08, 0.01, 0.001, -0.001, 0.0)),
        ("fisheye", ()),
        ("fisheye", (0.02, -0.003, 0.0002, 0.0)),
    ]
    for model, distortion in lenses:
        if model not in args.lenses:
            continue
        camera = CameraModel(model, (480, 640), (300, 300, 319.5, 239.5), distortion)
        for mode in args.modes:
            calibrated = GridSampler(
                60,
                1.875,
                40,
                device=args.device,
                mode=mode,
                field_geometry="spherical",
                camera_model=camera,
                gaze_convention=args.convention,
            )
            compiled = GridSampler(
                60,
                1.875,
                40,
                device=args.device,
                mode=mode,
                field_geometry="spherical",
                camera_model=camera,
                backend="compiled",
                gaze_convention=args.convention,
            )
            # Identical nodes isolate projection/gaze overhead from sample-count differences.
            canonical = GridSampler(
                60, 1.875, 40, device=args.device, mode=mode, coords=calibrated.coords
            )
            central_pixels, central_valid = camera.project(
                calibrated.canonical_directions
            )
            if not bool(central_valid.all()):
                raise ValueError(
                    "Benchmark retinal window must fit the central camera view"
                )
            # Calibrate once at central gaze; the native moving-grid sampler then
            # translates these pixels. Its window scale cancels this normalization.
            origin = central_pixels.new_tensor((319.5, 239.5))
            canonical.sampling_grid = ((central_pixels - origin) / 160).reshape_as(
                canonical.sampling_grid
            )
            canonical.out_sampling_grid = canonical.sampling_grid
            for dtype_name, batch in (
                (d, b) for d in args.dtypes for b in args.batches
            ):
                if batch == args.batches[0]:
                    # Each lens/mode/dtype is a separate deployment configuration.
                    # Do not exhaust Dynamo's per-function specialization limit
                    # by benchmarking all of them in one process.
                    torch.compiler.reset()
                dtype = getattr(torch, dtype_name)
                image = torch.randint(
                    0,
                    256,
                    (batch, args.channels, 480, 640),
                    dtype=torch.uint8,
                    device=args.device,
                )
                if dtype != torch.uint8:
                    image = image.to(dtype) / 256
                # Avoid exact nearest-neighbor ties when comparing fused arithmetic.
                for gaze_name, center in [
                    ("center", (0.5013, 0.5017)),
                    ("periphery", (0.3013, 0.8017)),
                ]:
                    fixation = (
                        torch.tensor(center, device=args.device)
                        .expand(batch, -1)
                        .contiguous()
                    )
                    size = torch.full((batch, 2), 320, device=args.device)
                    started = time.perf_counter()
                    expected, expected_grid = calibrated(
                        image, fixation, direct=True, return_coords=True
                    )
                    actual, actual_grid = calibrated(
                        image, fixation, return_coords=True
                    )
                    previous = compiled(image, fixation)
                    torch.cuda.synchronize()
                    tolerance = {
                        torch.uint8: 0.02,
                        torch.float16: 0.001,
                        torch.bfloat16: 0.008,
                        # FP32 projection rounding is amplified by high-contrast
                        # bilinear samples; independently bound source pixels below.
                        torch.float32: 1e-4,
                        torch.float64: 1e-10,
                    }[dtype]
                    torch.testing.assert_close(
                        actual,
                        expected,
                        atol=0 if mode == "nearest" else tolerance,
                        rtol=1e-4 if mode == "bilinear" else 0,
                        msg=f"Native parity: {model}, distortion={distortion}, {mode}, {batch=}, {dtype_name}, {gaze_name}",
                    )
                    pixel_delta = (
                        actual_grid - expected_grid
                    ) * actual_grid.new_tensor(
                        (camera.image_size[1] / 2, camera.image_size[0] / 2)
                    )
                    torch.testing.assert_close(
                        pixel_delta,
                        torch.zeros_like(pixel_delta),
                        atol=1e-9 if dtype == torch.float64 else 2e-4,
                        rtol=0,
                    )
                    initialization_s = time.perf_counter() - started
                    operations = {
                        "native": partial(calibrated, image, fixation),
                        "compiled": partial(compiled, image, fixation),
                        "canonical": partial(canonical, image, fixation, size),
                    }
                    timings = {name: [] for name in operations}
                    graphs = {name: [] for name in operations}
                    # Alternate order to limit systematic clock/temperature bias.
                    for repeat in range(args.repeats):
                        names = list(operations)
                        if repeat % 2:
                            names.reverse()
                        for name in names:
                            timings[name].append(
                                measure(operations[name], args.iterations)
                            )
                            graphs[name].append(
                                measure_graph(operations[name], args.iterations)
                            )
                    results.append(
                        {
                            "camera": asdict(camera),
                            "mode": mode,
                            "batch": batch,
                            "dtype": dtype_name,
                            "channels": args.channels,
                            "convention": args.convention,
                            "max_abs_error": {
                                "native": (actual.double() - expected.double())
                                .abs()
                                .max()
                                .item(),
                                "compiled": (previous.double() - expected.double())
                                .abs()
                                .max()
                                .item(),
                            },
                            "gaze": gaze_name,
                            "max_source_pixel_error": pixel_delta.abs().max().item(),
                            "samples": len(calibrated.coords),
                            "initialization_and_parity_seconds": initialization_s,
                            "ms": {
                                name: statistics.median(values)
                                for name, values in timings.items()
                            },
                            "graph_ms": {
                                name: statistics.median(values)
                                for name, values in graphs.items()
                            },
                            "measurements_ms": timings,
                            "graph_measurements_ms": graphs,
                            "canonical_backend": canonical.last_backend,
                            "calibrated_backend": calibrated.last_backend,
                            "compiled_backend": compiled.last_backend,
                        }
                    )
                    print(json.dumps(results[-1]), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(),
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )
    regressions = [
        row
        for row in results
        if any(row[key]["native"] > row[key]["compiled"] for key in ("ms", "graph_ms"))
    ]
    if regressions:
        raise RuntimeError(
            f"Native sampling was slower in {len(regressions)} cases; see {args.output}"
        )


if __name__ == "__main__":
    main()
