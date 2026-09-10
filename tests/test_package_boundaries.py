"""Exercise optional dependency boundaries in fresh Python processes."""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest


class RejectImports(importlib.abc.MetaPathFinder):
    def __init__(self, names: tuple[str, ...]) -> None:
        self.names = names

    def find_spec(
        self, fullname: str, path: list[str] | None, target: None = None
    ) -> importlib.machinery.ModuleSpec | None:
        if any(
            fullname == name or fullname.startswith(name + ".") for name in self.names
        ):
            raise AssertionError(f"Forbidden import: {fullname}")
        return None


def probe_core() -> None:
    sys.meta_path.insert(
        0,
        RejectImports(
            (
                "fovi.models",
                "fovi.training",
                "fovi.trainer",
                "fovi.paths",
                "ffcv",
                "wandb",
                "torchmetrics",
                "timm",
                "transformers",
                "accelerate",
                "hydra",
            )
        ),
    )
    import torch

    import fovi
    from fovi.arch.knn import KNNPoolingLayer
    from fovi.sensing.coords import SamplingCoords
    from fovi.sensing.retina import RetinalTransform
    from fovi.sensing.samplers import GridSampler
    from fovi.utils.image import crop_frame

    assert fovi.__name__ == "fovi"
    assert KNNPoolingLayer.__module__ == "fovi.arch.knn"
    assert SamplingCoords.__module__ == "fovi.sensing.coords"
    assert RetinalTransform.__module__ == "fovi.sensing.retina"
    assert crop_frame(torch.ones(4, 4), left=0, right=1, top=0, bottom=1).shape == (
        4,
        4,
    )
    sampler = GridSampler(fov=20, cmf_a=1, resolution=8, device="cpu")
    output = sampler(
        torch.full((1, 1, 5, 5), 7.0), fix_loc=(0.5, 0.5), fixation_size=(4, 4)
    )
    torch.testing.assert_close(output, torch.full_like(output, 7.0))


def probe_models() -> None:
    original_find_spec = importlib.util.find_spec

    def find_spec(
        name: str, package: str | None = None
    ) -> importlib.machinery.ModuleSpec | None:
        # Installed optional trackers can be auto-imported by Accelerate; model-only
        # installations must also work when these packages are absent.
        if name in {"ffcv", "wandb", "torchmetrics"}:
            return None
        return original_find_spec(name, package)

    importlib.util.find_spec = find_spec
    sys.meta_path.insert(
        0,
        RejectImports(
            (
                "fovi.training",
                "fovi.paths",
                "ffcv",
                "wandb",
                "torchmetrics",
            )
        ),
    )
    from fovi import FoviNet, get_model_from_base_fn
    from fovi.arch.knnresnet import KNNResNet as LegacyResNet
    from fovi.models import FoviNet as NewFoviNet
    from fovi.models.knnresnet import KNNResNet
    from fovi.models.loading import load_config
    from fovi.trainer import load_config as legacy_load_config

    assert FoviNet is NewFoviNet
    assert LegacyResNet is KNNResNet
    assert legacy_load_config is load_config
    assert get_model_from_base_fn.__module__ == "fovi.models.loading"


def probe_missing_extra(extra: str) -> None:
    from fovi import _optional

    original_find_spec = _optional.find_spec
    missing_module = {"models": "timm", "training": "ffcv"}[extra]

    def find_spec(name: str) -> importlib.machinery.ModuleSpec | None:
        if name == missing_module:
            return None
        return original_find_spec(name)

    _optional.find_spec = find_spec
    with pytest.raises(ModuleNotFoundError, match=f"pip install 'fovi\\[{extra}\\]'"):
        __import__(f"fovi.{extra}")


@pytest.mark.parametrize(
    "capability", ["core", "models", "missing-models", "missing-training"]
)
def test_import_boundary(capability: str, tmp_path: Path) -> None:
    env = os.environ.copy()
    for name in tuple(env):
        if name.startswith("FOVI_"):
            del env[name]
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    subprocess.run(
        [sys.executable, __file__, capability], cwd=tmp_path, env=env, check=True
    )


def test_missing_extra_has_install_command(monkeypatch: pytest.MonkeyPatch) -> None:
    from fovi import _optional

    monkeypatch.setattr(_optional, "find_spec", lambda name: None)
    with pytest.raises(ModuleNotFoundError, match=r"pip install 'fovi\[models\]'"):
        _optional.require_dependencies("models", ("transformers",))


if __name__ == "__main__":
    if sys.argv[1].startswith("missing-"):
        probe_missing_extra(sys.argv[1].removeprefix("missing-"))
    else:
        {"core": probe_core, "models": probe_models}[sys.argv[1]]()
