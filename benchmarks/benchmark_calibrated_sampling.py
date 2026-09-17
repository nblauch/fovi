"""Compare calibrated dynamic gaze against canonical-grid image translation."""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.cuda.set_device(args.device)
    torch.manual_seed(2026)
    results = []
    for model, distortion in [
        ("pinhole", (-0.08, 0.01, 0.001, -0.001, 0.0)),
        ("fisheye", (0.02, -0.003, 0.0002, 0.0)),
    ]:
        camera = CameraModel(model, (480, 640), (300, 300, 319.5, 239.5), distortion)
        for mode in ("nearest", "bilinear"):
            calibrated = GridSampler(
                60,
                1.875,
                40,
                device=args.device,
                mode=mode,
                field_geometry="spherical",
                camera_model=camera,
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
            for batch in args.batches:
                image = torch.randint(
                    0, 256, (batch, 3, 480, 640), dtype=torch.uint8, device=args.device
                )
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
                    expected = calibrated(image, fixation, direct=True)
                    actual = calibrated(image, fixation)
                    torch.cuda.synchronize()
                    torch.testing.assert_close(
                        actual.float(), expected.float(), atol=0.02, rtol=1e-4
                    )
                    first_call_s = time.perf_counter() - started
                    dynamic_ms = measure(
                        partial(calibrated, image, fixation), args.iterations
                    )
                    canonical_ms = measure(
                        partial(canonical, image, fixation, size), args.iterations
                    )
                    calibrated_graph_ms = measure_graph(
                        partial(calibrated, image, fixation), args.iterations
                    )
                    canonical_graph_ms = measure_graph(
                        partial(canonical, image, fixation, size), args.iterations
                    )
                    results.append(
                        {
                            "camera": asdict(camera),
                            "mode": mode,
                            "batch": batch,
                            "gaze": gaze_name,
                            "samples": len(calibrated.coords),
                            "first_call_seconds": first_call_s,
                            "calibrated_ms": dynamic_ms,
                            "canonical_ms": canonical_ms,
                            "ratio": dynamic_ms / canonical_ms,
                            "calibrated_graph_ms": calibrated_graph_ms,
                            "canonical_graph_ms": canonical_graph_ms,
                            "canonical_backend": canonical.last_backend,
                            "calibrated_backend": calibrated.last_backend,
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


if __name__ == "__main__":
    main()
