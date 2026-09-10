"""Capture and compare pretrained inference across separate fovi checkouts.

Run ``capture`` once per checkout in separate processes, then ``compare``.
Checkpoints must already be downloaded; this command never trains a model.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
from PIL import Image
import torch

MODELS = (
    "fovi-dinov3-splus_a-2.78_res-64_in1k",
    "fovi-dinov3-splus_a-60.94_res-64_in1k",
    "fovi-alexnet_a-0.5_res-64_rfmult-1_in1k",
    "fovi-alexnet_a-0.5_res-64_rfmult-2_in1k",
)


def digest_file(path: Path) -> str:
    """Hash an input artifact without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_state(state: dict[str, torch.Tensor]) -> str:
    """Hash state keys, types, dimensions, and values in a stable order."""
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(f"{name}:{tensor.dtype}:{tuple(tensor.shape)}".encode())
        digest.update(
            tensor.detach()
            .cpu()
            .contiguous()
            .reshape(-1)
            .view(torch.uint8)
            .numpy()
            .tobytes()
        )
    return digest.hexdigest()


def canonical_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Normalize only the known DINO block paths when comparing library versions."""
    result = {}
    for key, value in state.items():
        for prefix in (
            "network.backbone.model.layer.",
            "network.backbone.encoder.layer.",
        ):
            if key.startswith(prefix):
                key = "network.backbone.layer." + key[len(prefix) :]
                break
        if key in result:
            raise ValueError(f"Duplicate canonical checkpoint key: {key}")
        result[key] = value
    return result


def seed_everything() -> None:
    """Reset construction and fixation randomness in both processes."""
    random.seed(2026)
    np.random.seed(2026)
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)


def capture(args: argparse.Namespace) -> None:
    """Run the public loader and save retinal samples, features, and logits."""
    source = args.source_root.resolve()
    sys.path.insert(0, str(source))
    if args.api == "legacy":
        os.environ["FOVI_SAVE_DIR"] = str(args.output / "unused_research_storage")
        os.environ["FOVI_DATASETS_DIR"] = str(args.output / "unused_datasets")
        loader_module = "fovi"
    elif args.api == "models":
        for name in tuple(os.environ):
            if name.startswith("FOVI_") and name.endswith("_DIR"):
                del os.environ[name]
        loader_module = "fovi.models"
    else:
        raise ValueError(f"Unsupported API {args.api!r}")
    fovi = importlib.import_module("fovi")
    if fovi.__file__ is None or Path(fovi.__file__).resolve().parent.parent != source:
        raise RuntimeError(f"Wrong fovi checkout imported: {fovi.__file__}")
    loader = importlib.import_module(loader_module).get_model_from_base_fn
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    args.output.mkdir(parents=True, exist_ok=True)
    for name in args.models:
        seed_everything()
        checkpoint = args.model_dir / name
        if not (checkpoint / "config.yaml").is_file():
            raise FileNotFoundError(f"Download {name} before running: {checkpoint}")
        model = loader(
            name, model_dirs=[str(args.model_dir)], device=args.device, quiet=True
        ).eval()
        resolution = int(model.cfg.training.resolution)
        street = Image.open(args.image).convert("RGB").resize((resolution, resolution))
        natural = torch.from_numpy(np.array(street)).permute(2, 0, 1)
        noise = torch.randint(
            0,
            256,
            natural.shape,
            dtype=torch.uint8,
            generator=torch.Generator().manual_seed(2026),
        )
        images = torch.stack((natural, natural.flip(-1), noise)).to(args.device)
        # Batch size three avoids ambiguity between a coordinate pair and a batch.
        centers = torch.tensor([[0.5, 0.5]] * 3, device=args.device)
        off_center = torch.tensor(
            [[0.3, 0.7], [0.7, 0.3], [0.2, 0.2]], device=args.device
        )
        near_edge = torch.tensor(
            [[0.05, 0.5], [0.95, 0.8], [0.5, 0.95]], device=args.device
        )
        tensors = {"inputs": images.cpu()}
        with torch.inference_mode():
            for label, fixations in (
                ("single", [centers]),
                ("multiple", [centers, off_center, near_edge]),
            ):
                seed_everything()
                embeddings, layers, samples = model(
                    images.clone(),
                    setting="supervised",
                    fixations=fixations,
                    n_fixations=len(fixations),
                    do_postproc=False,
                )
                tensors[f"{label}/embeddings"] = embeddings.cpu()
                tensors[f"{label}/samples"] = samples.cpu()
                tensors[f"{label}/logits"] = model.head(embeddings).cpu()
                tensors[f"{label}/fixations"] = model.last_fixations.cpu()
                for index, layer in enumerate(layers):
                    tensors[f"{label}/layer_{index}"] = layer.cpu()
        for key, tensor in tensors.items():
            if not torch.isfinite(tensor).all():
                raise AssertionError(f"{name}: non-finite values in {key}")
        if args.api == "models":
            forbidden = [
                module
                for module in sys.modules
                if module == "ffcv"
                or module == "fovi.trainer"
                or module.startswith("fovi.training")
            ]
            if forbidden:
                raise AssertionError(
                    f"Inference imported training modules: {forbidden}"
                )
        files = sorted(checkpoint.glob("*.pth")) + sorted(checkpoint.glob("*.json"))
        files.append(checkpoint / "config.yaml")
        metadata = {
            "model": name,
            "api": args.api,
            "source": str(source),
            "torch": torch.__version__,
            "device": args.device,
            "transformers": version("transformers"),
            "state_sha256": digest_state(model.state_dict()),
            "canonical_state_sha256": digest_state(canonical_state(model.state_dict())),
            "checkpoint_sha256": {path.name: digest_file(path) for path in files},
            "image_sha256": digest_file(args.image),
            "hub_config_metadata": (
                checkpoint / ".cache/huggingface/download/config.yaml.metadata"
            ).read_text()
            if (
                checkpoint / ".cache/huggingface/download/config.yaml.metadata"
            ).is_file()
            else None,
        }
        torch.save(tensors, args.output / f"{name}.pt")
        (args.output / f"{name}.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Captured {args.api}: {name} ({len(tensors)} tensors)", flush=True)
        del model, embeddings, layers, samples, tensors
        torch.cuda.empty_cache()


def compare(args: argparse.Namespace) -> None:
    """Require identical provenance and compare every captured output tensor."""
    results = []
    for name in args.models:
        before_meta = json.loads((args.reference / f"{name}.json").read_text())
        after_meta = json.loads((args.candidate / f"{name}.json").read_text())
        fields = [
            "model",
            "torch",
            "device",
            "canonical_state_sha256",
            "checkpoint_sha256",
            "image_sha256",
        ]
        if not args.allow_transformers_change:
            fields.extend(("transformers", "state_sha256"))
        for field in fields:
            if before_meta[field] != after_meta[field]:
                raise AssertionError(f"{name}: {field} differs between captures")
        before = torch.load(
            args.reference / f"{name}.pt", map_location="cpu", weights_only=True
        )
        after = torch.load(
            args.candidate / f"{name}.pt", map_location="cpu", weights_only=True
        )
        if before.keys() != after.keys():
            raise AssertionError(f"{name}: different output keys")
        comparisons = {}
        for key in before:
            torch.testing.assert_close(
                after[key], before[key], rtol=args.rtol, atol=args.atol
            )
            comparisons[key] = {
                "shape": list(before[key].shape),
                "exact": torch.equal(before[key], after[key]),
                "max_abs_error": float(
                    (before[key].double() - after[key].double()).abs().max()
                ),
            }
        results.append(
            {
                "model": name,
                "tensors": comparisons,
                "reference": before_meta,
                "candidate": after_meta,
                "rtol": args.rtol,
                "atol": args.atol,
            }
        )
        print(f"PASS {name}: {len(comparisons)} tensors", flush=True)
    args.report.write_text(json.dumps(results, indent=2) + "\n")


def main() -> None:
    """Parse capture or comparison arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    record = commands.add_parser("capture")
    record.add_argument("--source-root", type=Path, required=True)
    record.add_argument("--api", choices=("legacy", "models"), required=True)
    record.add_argument("--model-dir", type=Path, default=Path.home() / ".cache/fovi")
    record.add_argument("--device", default="cuda:0")
    record.add_argument("--image", type=Path, required=True)
    record.add_argument("--output", type=Path, required=True)
    record.add_argument("--models", nargs="+", default=MODELS)
    check = commands.add_parser("compare")
    check.add_argument("--reference", type=Path, required=True)
    check.add_argument("--candidate", type=Path, required=True)
    check.add_argument("--report", type=Path, required=True)
    check.add_argument("--models", nargs="+", default=MODELS)
    check.add_argument("--rtol", type=float, default=0.0)
    check.add_argument("--atol", type=float, default=0.0)
    check.add_argument("--allow-transformers-change", action="store_true")
    args = parser.parse_args()
    if args.command == "capture":
        capture(args)
    elif args.command == "compare":
        compare(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
