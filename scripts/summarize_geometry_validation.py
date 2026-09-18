"""Summarize full, same-device checkpoint validation without hiding accuracy changes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, nargs="+", required=True)
    parser.add_argument("--candidate", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidates = {}
    for directory in args.candidate:
        for path in directory.glob("*.json"):
            metadata = json.loads(path.read_text())
            key = metadata["model"], metadata["gpu"]
            if key in candidates:
                raise ValueError(f"Duplicate candidate: {key}")
            candidates[key] = path, metadata

    rows = []
    provenance = []
    for directory in args.reference:
        for path in sorted(directory.glob("*.json")):
            before = json.loads(path.read_text())
            after_path, after = candidates[before["model"], before["gpu"]]
            for key in (
                "model",
                "gpu",
                "torch",
                "examples",
                "batch_size",
                "seed",
                "checkpoint_sha256",
            ):
                if before[key] != after[key]:
                    raise ValueError(f"{before['model']}: mismatched {key}")
            if (
                before["examples"] != 50000
                or not before["full_dataset"]
                or not after["full_dataset"]
            ):
                raise ValueError("Only complete ImageNet validation runs are accepted")
            baseline_outputs = torch.load(path.with_suffix(".pt"), weights_only=True)
            candidate_outputs = torch.load(
                after_path.with_suffix(".pt"), weights_only=True
            )
            torch.testing.assert_close(
                baseline_outputs["labels"], candidate_outputs["labels"], rtol=0, atol=0
            )
            if (
                baseline_outputs["fixation_counts"]
                != candidate_outputs["fixation_counts"]
            ):
                raise ValueError("Fixation counts differ")
            provenance.append(
                {
                    key: before[key]
                    for key in (
                        "model",
                        "gpu",
                        "torch",
                        "examples",
                        "batch_size",
                        "seed",
                        "checkpoint_sha256",
                    )
                }
            )
            for count, metrics in before["metrics"].items():
                result = {"model": before["model"], "fixations": int(count)}
                for metric in ("top1", "top5"):
                    result[f"reference_{metric}"] = metrics[metric]
                    result[f"corrected_{metric}"] = after["metrics"][count][metric]
                    result[f"delta_{metric}_pp"] = (
                        after["metrics"][count][metric] - metrics[metric]
                    )
                rows.append(result)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(
        json.dumps({"provenance": provenance, "results": rows}, indent=2) + "\n"
    )
    lines = [
        "# Geometry validation",
        "",
        "Full ImageNet validation comparisons against the pre-correction source (1f5fd27).",
        "Each pair uses the same GPU, checkpoint files, preprocessing, batch size, and",
        "per-batch fixation RNG seed. Checkpoint hashes and label ordering are checked.",
        "Inputs use centered square crops decoded to each model's configured resolution,",
        "ImageNet normalization, the configured AMP mode, and the configured fixation policy.",
        "These comparisons measure a geometry change; they are not a new training run.",
        "",
        "The corrected planar geometry is used for every candidate below. Differences",
        "above 0.1 percentage points are shown explicitly; checkpoint neighborhoods have",
        "not been restored to their historical values. New models default to planar",
        "geometry. The explicit legacy option is available for older checkpoints;",
        "the measurements below use corrected planar geometry.",
        "",
        "| Model | Fixations | Top-1 before | Top-1 corrected | Δ pp | Top-5 before | Top-5 corrected | Δ pp |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['fixations']} | {row['reference_top1']:.3f} | {row['corrected_top1']:.3f} | {row['delta_top1_pp']:+.3f} | {row['reference_top5']:.3f} | {row['corrected_top5']:.3f} | {row['delta_top5_pp']:+.3f} |"
        )
    lines += ["", "## Execution", ""]
    for item in provenance:
        lines.append(
            f"- {item['model']}: {item['gpu']}; PyTorch {item['torch']}; {item['examples']} images; batch {item['batch_size']}; seed {item['seed']}."
        )
    lines += [
        "",
        "Use `scripts/validate_pretrained_geometry.py` in each source checkout, then",
        "`scripts/summarize_geometry_validation.py` to reproduce this comparison.",
        "",
        "The accompanying JSON contains checkpoint hashes and unrounded metrics.",
        "",
    ]
    args.output.with_suffix(".md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
