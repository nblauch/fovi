"""Reproducible full FFCV validation for comparisons between source checkouts."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--field-geometry", choices=("planar", "spherical", "legacy"))
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Diagnostic subset only; zero evaluates the full dataset.",
    )
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_root.resolve()))
    from check_pretrained_parity import digest_file
    from ffcv.fields.basics import IntDecoder
    from ffcv.fields.rgb_image import CenterCropRGBImageDecoder
    from ffcv.loader import Loader, OrderOption
    from ffcv.transforms import Squeeze, ToTensor
    from fovi.models import get_model_from_base_fn

    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    for name in args.models:
        random.seed(2026)
        np.random.seed(2026)
        torch.manual_seed(2026)
        model = get_model_from_base_fn(
            name,
            model_dirs=[str(args.model_dir)],
            device="cuda:0",
            quiet=True,
            **(
                {"saccades.field_geometry": args.field_geometry}
                if args.field_geometry
                else {}
            ),
        ).eval()
        res = int(model.cfg.training.resolution)
        configured = model.cfg.saccades.n_fixations_val
        counts = [configured] if isinstance(configured, int) else list(configured)
        loader = Loader(
            str(args.dataset),
            batch_size=args.batch_size,
            num_workers=4,
            order=OrderOption.SEQUENTIAL,
            drop_last=False,
            seed=2026,
            pipelines={
                "image": [CenterCropRGBImageDecoder((res, res), ratio=1.0), ToTensor()],
                "label": [IntDecoder(), ToTensor(), Squeeze()],
            },
        )
        records = []
        labels = []
        seen = 0
        started = time.perf_counter()
        mean = torch.tensor([0.485, 0.456, 0.406], device="cuda")[None, :, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], device="cuda")[None, :, None, None]
        with torch.inference_mode():
            for batch_index, (image, label) in enumerate(loader):
                image = image.to("cuda").permute(0, 3, 1, 2).float().div_(255)
                image = (image - mean) / std
                # The original validation policy owns fixation sizes and placement.
                # Reset all RNGs per batch so geometry construction cannot shift it.
                random.seed(2026 + batch_index)
                np.random.seed(2026 + batch_index)
                torch.manual_seed(2026 + batch_index)
                with torch.autocast(
                    "cuda",
                    dtype=getattr(torch, model.cfg.training.amp_dtype),
                    enabled=bool(model.cfg.training.use_amp),
                ):
                    embeddings, _, _ = model(
                        image,
                        setting="supervised",
                        n_fixations=max(counts),
                        do_postproc=False,
                    )
                    predictions = torch.stack(
                        [
                            model.head(embeddings[:, :n]).topk(5, dim=-1).indices
                            for n in counts
                        ],
                        1,
                    )
                records.append(predictions.cpu())
                # FFCV recycles host buffers between batches.
                labels.append(label.reshape(-1).cpu().clone())
                seen += image.shape[0]
                if batch_index % 50 == 0:
                    print(
                        f"{name}: {seen} examples, {time.perf_counter() - started:.1f}s",
                        flush=True,
                    )
                if args.limit and seen >= args.limit:
                    break
        predictions = torch.cat(records)
        labels = torch.cat(labels)
        correct = predictions == labels[:, None, None]
        report = {
            "model": name,
            "field_geometry": getattr(model.cfg.saccades, "field_geometry", None),
            "source": str(args.source_root.resolve()),
            "dataset": str(args.dataset),
            "examples": seen,
            "full_dataset": args.limit == 0,
            "batch_size": args.batch_size,
            "seed": 2026,
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(),
            "seconds": time.perf_counter() - started,
            "metrics": {
                str(n): {
                    "top1": float(correct[:, i, 0].float().mean() * 100),
                    "top5": float(correct[:, i].any(-1).float().mean() * 100),
                }
                for i, n in enumerate(counts)
            },
            "checkpoint_sha256": {
                p.name: digest_file(p)
                for p in sorted((args.model_dir / name).glob("*.pth"))
            },
        }
        torch.save(
            {"predictions": predictions, "labels": labels, "fixation_counts": counts},
            args.output / f"{name}.pt",
        )
        (args.output / f"{name}.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report["metrics"]), flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
