"""Validate dependency boundaries and source contents in a built fovi wheel."""

from __future__ import annotations

import argparse
from email import message_from_bytes
from pathlib import Path
from zipfile import ZipFile

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version


def check_release_tag(version: str, tag: str) -> None:
    """Require a stable release tag matching the built distribution's version."""
    parsed = Version(version)
    if parsed.is_prerelease or parsed.is_devrelease or parsed.local is not None:
        raise ValueError(f"Release publishing requires a stable version, got {version}")
    if tag != f"v{version}":
        raise ValueError(
            f"Release tag {tag!r} does not match package version {version!r}"
        )


def check_wheel(path: Path, release_tag: str | None = None) -> None:
    """Check the built artifact rather than only its source requirement lists."""
    with ZipFile(path) as wheel:
        metadata_path = next(
            name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")
        )
        metadata = message_from_bytes(wheel.read(metadata_path))
        if release_tag is not None:
            check_release_tag(metadata["Version"], release_tag)
        assert metadata["Description-Content-Type"] == "text/markdown"
        assert metadata["License-Expression"] == "MIT"
        assert metadata.get_payload().strip(), "Missing package README"
        requirements = [
            Requirement(value) for value in metadata.get_all("Requires-Dist", [])
        ]
        assert all(req.url is None for req in requirements), (
            "PyPI distributions must not contain direct-URL dependencies"
        )
        assert not {canonicalize_name(req.name) for req in requirements} & {
            "ffcv",
            "ffcv-ssl",
        }, "FFCV is an externally installed prerequisite"
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
    print(f"Validated source contents and base/models/training/all metadata: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument(
        "--release-tag", help="Require this tag to match a stable version"
    )
    args = parser.parse_args()
    check_wheel(args.wheel, args.release_tag)
