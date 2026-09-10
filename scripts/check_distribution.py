"""Validate dependency boundaries and source contents in a built fovi wheel."""

from __future__ import annotations

import argparse
from email import message_from_bytes
from pathlib import Path
from zipfile import ZipFile

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def check_wheel(path: Path) -> None:
    """Check the built artifact rather than only its source requirement lists."""
    with ZipFile(path) as wheel:
        metadata_path = next(
            name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")
        )
        metadata = message_from_bytes(wheel.read(metadata_path))
        requirements = [
            Requirement(value) for value in metadata.get_all("Requires-Dist", [])
        ]
        expected_files = {
            "fovi/sensing/retina.py",
            "fovi/arch/knn.py",
            "fovi/models/fovinet.py",
            "fovi/models/loading.py",
            "fovi/training/trainer.py",
            "fovi/training/loader.py",
        }
        assert expected_files <= set(wheel.namelist()), (
            "Missing package source in wheel"
        )
    extras = set(metadata.get_all("Provides-Extra", []))
    assert extras == {
        "models",
        "training",
        "ffcv",
        "all",
    }
    for python_version in ("3.9", "3.12"):
        selected: dict[str, set[tuple[str, str, str | None]]] = {}
        for extra in ("", *sorted(extras)):
            environment = {"python_version": python_version, "extra": extra}
            selected[extra] = {
                (canonicalize_name(req.name), str(req.specifier), req.url)
                for req in requirements
                if req.marker is None or req.marker.evaluate(environment)
            }
        all_dependencies = set(selected[""])
        for extra in extras - {"all"}:
            all_dependencies.update(selected[extra])
        assert selected["all"] == all_dependencies, "all must include every extra"
        assert selected["models"] <= selected["training"]
        base_names = {name for name, _, _ in selected[""]}
        assert not base_names & {
            "ffcv",
            "ffcv-ssl",
            "timm",
            "transformers",
            "wandb",
            "torchmetrics",
            "hydra-core",
        }
        assert {
            "torch",
            "torchvision",
            "scipy",
            "trimesh",
            "scikit-image",
            "cupy-cuda12x",
            "warp-lang",
        } <= base_names
        assert "ffcv-ssl" not in {name for name, _, _ in selected["training"]}
        assert "ffcv-ssl" in {name for name, _, _ in selected["ffcv"]}
    print(
        f"Validated source contents and base/models/training/ffcv/all metadata: {path}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    check_wheel(parser.parse_args().wheel)
