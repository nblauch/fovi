"""W&B snapshots retain matching model/config revisions while training advances."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import DictConfig
from torch import nn

from fovi.models import get_model_from_base_fn, resolve_model_path
from fovi.models.loading import load_config
from fovi.models.wandb import download_wandb_model


def checkpoint_bytes(value: float, epoch: int = 1, key: str = "model") -> bytes:
    stream = io.BytesIO()
    torch.save(
        {
            key: {"weight": torch.tensor([value])},
            "params": {"width": 1, "epoch_config": epoch},
            "epoch": epoch,
        },
        stream,
    )
    return stream.getvalue()


@dataclass
class RemoteFile:
    content: bytes
    name: str = "model.pth"
    downloads: int = 0
    download_error: Exception | None = None
    replacement: bytes | None = None

    @property
    def md5(self) -> str:
        return base64.b64encode(hashlib.md5(self.content).digest()).decode()

    @property
    def size(self) -> int:
        return len(self.content)

    def download(self, root: str, api: FakeApi) -> io.TextIOWrapper:
        self.downloads += 1
        if self.download_error is not None:
            raise self.download_error
        destination = Path(root) / self.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(
            self.content if self.replacement is None else self.replacement
        )
        return destination.open()


@dataclass
class RemoteRun:
    checkpoint: RemoteFile | None
    id: str = "run12345"
    name: str = "training-example"

    def files(self, names: list[str]) -> list[RemoteFile]:
        if self.checkpoint is not None and self.checkpoint.name in names:
            return [self.checkpoint]
        return []


@dataclass
class FakeApi:
    available_runs: list[RemoteRun]
    lookups: list[str] = field(default_factory=list)

    def runs(
        self, project: str, filters: dict[str, list[dict[str, str]]]
    ) -> list[RemoteRun]:
        assert project == "team/project"
        reference = filters["$or"][0]["name"]
        assert filters["$or"][1] == {"display_name": reference}
        self.lookups.append(reference)
        return [run for run in self.available_runs if reference in (run.id, run.name)]


@pytest.fixture
def remote(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[FakeApi, RemoteFile]:
    checkpoint = RemoteFile(checkpoint_bytes(1.0))
    api = FakeApi([RemoteRun(checkpoint)])
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(Api=lambda: api))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return api, checkpoint


class Model(nn.Module):
    def __init__(self, cfg: DictConfig, device: str) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(cfg.width, device=device))
        self.cfg = cfg


@pytest.mark.parametrize("reference", ["run12345", "training-example"])
@pytest.mark.parametrize("key", ["model", "state_dict"])
def test_model_loads_matching_embedded_config(
    remote: tuple[FakeApi, RemoteFile], reference: str, key: str
) -> None:
    api, checkpoint = remote
    checkpoint.content = checkpoint_bytes(4.0, epoch=7, key=key)
    model = get_model_from_base_fn(
        f"wandb://team/project/{reference}", device="cpu", fovinet_cls=Model
    )
    torch.testing.assert_close(model.weight, torch.tensor([4.0]))
    assert model.cfg.epoch_config == 7
    assert api.lookups == [reference]


def test_mutable_checkpoint_refresh_keeps_original_snapshot(
    remote: tuple[FakeApi, RemoteFile],
) -> None:
    _, checkpoint = remote
    uri = "wandb://team/project/training-example"
    first = resolve_model_path(uri)
    assert resolve_model_path(uri) == first
    assert checkpoint.downloads == 1
    checkpoint.content = checkpoint_bytes(2.0, epoch=2)
    second = resolve_model_path(uri)
    assert first != second
    assert checkpoint.downloads == 2
    for path, value, epoch in ((first, 1.0, 1), (second, 2.0, 2)):
        cfg, weights, key = load_config(path.name, True, path.parent, device="cpu")
        assert cfg.epoch_config == epoch == weights["epoch"]
        torch.testing.assert_close(weights[key]["weight"], torch.tensor([value]))
        assert (
            json.loads((path / "source.json").read_text())["run_path"]
            == "team/project/run12345"
        )


def test_checkpoint_pin_and_relative_filename(
    remote: tuple[FakeApi, RemoteFile],
    tmp_path: Path,
) -> None:
    api, checkpoint = remote
    checkpoint.name = "checkpoints/epoch-000005.pth"
    uri = "wandb://team/project/run12345#checkpoints/epoch-000005.pth"
    first = resolve_model_path(uri, expected_checkpoint_md5=checkpoint.md5)
    assert first.is_dir()
    checksum = checkpoint.md5
    checkpoint.content = checkpoint_bytes(3.0)
    lookups = len(api.lookups)
    assert resolve_model_path(uri, expected_checkpoint_md5=checksum) == first
    assert len(api.lookups) == lookups
    with pytest.raises(ValueError, match="no longer matches"):
        download_wandb_model(uri, tmp_path / "empty", expected_checkpoint_md5=checksum)
    with pytest.raises(ValueError, match="requires a wandb"):
        resolve_model_path(str(first), expected_checkpoint_md5=checksum)


def test_parallel_resolvers_publish_one_complete_bundle(
    remote: tuple[FakeApi, RemoteFile],
) -> None:
    _, checkpoint = remote
    with ThreadPoolExecutor(max_workers=2) as executor:
        paths = list(
            executor.map(resolve_model_path, ["wandb://team/project/run12345"] * 2)
        )
    assert paths[0] == paths[1]
    assert checkpoint.downloads == 1
    assert (paths[0] / "state_dict.pth").is_file()
    assert (paths[0] / "resolved_config.yaml").is_file()


@pytest.mark.parametrize("checksum", ["", "wrong", "a" * 32, "YWJjZA=="])
def test_invalid_checkpoint_pin_fails_before_network(
    remote: tuple[FakeApi, RemoteFile],
    checksum: str,
) -> None:
    api, _ = remote
    with pytest.raises(ValueError, match="base64-encoded 16-byte"):
        resolve_model_path(
            "wandb://team/project/run12345", expected_checkpoint_md5=checksum
        )
    assert not api.lookups


@pytest.mark.parametrize("damage", ["missing", "truncated", "identity", "symlink"])
def test_pinned_cache_requires_complete_matching_bundle(
    remote: tuple[FakeApi, RemoteFile],
    damage: str,
    tmp_path: Path,
) -> None:
    _, checkpoint = remote
    uri = "wandb://team/project/run12345"
    path = resolve_model_path(uri)
    weights = path / "state_dict.pth"
    if damage == "missing":
        weights.unlink()
    elif damage == "truncated":
        weights.write_bytes(b"truncated")
    elif damage == "identity":
        manifest = path / "source.json"
        source = json.loads(manifest.read_text())
        source["run_path"] = "team/project/other"
        manifest.write_text(json.dumps(source))
    elif damage == "symlink":
        outside = tmp_path / "external.pth"
        weights.replace(outside)
        weights.symlink_to(outside)
    with pytest.raises(RuntimeError, match="cache"):
        resolve_model_path(uri, expected_checkpoint_md5=checkpoint.md5)
    assert checkpoint.downloads == 1


@pytest.mark.parametrize(
    "uri",
    [
        "wandb:team/project/run",
        "wandb:///project/run",
        "wandb://team/project",
        "wandb://team/project/run/extra",
        "wandb://team/project/run?file=model.pth",
        "wandb://team/project/run#../model.pth",
        "wandb://team/project/run#%2Fmodel.pth",
        "wandb://team/project/run#folder/%2e%2e/model.pth",
        "wandb://team/project/run#folder\\model.pth",
        "wandb://../project/run",
        "wandb://team/project/run#config.yaml",
    ],
)
def test_malformed_uri_rejected_before_network(
    remote: tuple[FakeApi, RemoteFile], uri: str
) -> None:
    api, _ = remote
    with pytest.raises(ValueError):
        download_wandb_model(uri)
    assert not api.lookups


def test_missing_and_ambiguous_runs_fail(remote: tuple[FakeApi, RemoteFile]) -> None:
    api, checkpoint = remote
    with pytest.raises(FileNotFoundError, match="No W&B run"):
        resolve_model_path("wandb://team/project/missing")
    api.available_runs.append(RemoteRun(checkpoint, id="anotherid"))
    with pytest.raises(ValueError, match="Ambiguous.*anotherid.*run12345"):
        resolve_model_path("wandb://team/project/training-example")


def test_missing_published_checkpoint_fails(remote: tuple[FakeApi, RemoteFile]) -> None:
    api, _ = remote
    api.available_runs[0].checkpoint = None
    with pytest.raises(FileNotFoundError, match="no published 'model.pth'"):
        resolve_model_path("wandb://team/project/run12345")


@pytest.mark.parametrize("failure", ["network", "checksum"])
def test_download_failure_cannot_poison_cache(
    remote: tuple[FakeApi, RemoteFile], failure: str
) -> None:
    _, checkpoint = remote
    if failure == "network":
        checkpoint.download_error = ConnectionError("offline")
    else:
        checkpoint.replacement = checkpoint_bytes(9.0)
    with pytest.raises((ConnectionError, RuntimeError)):
        resolve_model_path("wandb://team/project/run12345")
    checkpoint.download_error = None
    checkpoint.replacement = None
    path = resolve_model_path("wandb://team/project/run12345")
    assert (path / "state_dict.pth").is_file()
    assert checkpoint.downloads == 2


@pytest.mark.parametrize(
    "payload",
    [
        {"model": {"weight": torch.ones(1)}},
        {"params": [], "model": {"weight": torch.ones(1)}},
        {"params": {"width": 1}, "probes": {"weight": torch.ones(1)}},
        {"params": {"width": 1}, "model": {"weight": "wrong"}},
    ],
)
def test_invalid_checkpoint_fails(
    remote: tuple[FakeApi, RemoteFile],
    payload: dict[str, dict[str, int | str | torch.Tensor] | list[None]],
) -> None:
    _, checkpoint = remote
    stream = io.BytesIO()
    torch.save(payload, stream)
    checkpoint.content = stream.getvalue()
    with pytest.raises((TypeError, ValueError)):
        resolve_model_path("wandb://team/project/run12345")
