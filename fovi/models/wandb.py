"""Resolve published W&B training checkpoints into immutable inference bundles."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import unquote, urlsplit

import torch
from filelock import FileLock
from omegaconf import DictConfig, OmegaConf


def _parse_model_uri(uri: str) -> tuple[str, str, str, str]:
    parsed = urlsplit(uri)
    parts = [parsed.netloc, *parsed.path.removeprefix("/").split("/")]
    if (
        parsed.scheme != "wandb"
        or parsed.query
        or len(parts) != 3
        or any(
            not re.fullmatch(r"[A-Za-z0-9_.-]+", part) or part in (".", "..")
            for part in parts[:2]
        )
        or not parts[2]
    ):
        raise ValueError(
            "Expected wandb://entity/project/run-name-or-id[#checkpoint.pth]"
        )
    checkpoint = unquote(parsed.fragment) if parsed.fragment else "model.pth"
    if (
        not checkpoint.endswith(".pth")
        or "\\" in checkpoint
        or any(not part or part in (".", "..") for part in checkpoint.split("/"))
    ):
        raise ValueError(
            "W&B checkpoint must be a relative .pth path without traversal"
        )
    return parts[0], parts[1], unquote(parts[2]), checkpoint


def _file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return base64.b64encode(digest.digest()).decode("ascii")


def _revision_key(filename: str, checksum: str) -> str:
    return hashlib.sha256(f"{filename}:{checksum}".encode()).hexdigest()


def _validate_checksum(checksum: str) -> None:
    try:
        decoded = base64.b64decode(checksum, validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError(
            "Checkpoint MD5 must be a base64-encoded 16-byte digest"
        ) from error
    if len(decoded) != 16 or base64.b64encode(decoded).decode("ascii") != checksum:
        raise ValueError("Checkpoint MD5 must be a base64-encoded 16-byte digest")


def _validate_cached_bundle(
    path: Path, run_path: str, filename: str, checksum: str
) -> Path:
    if path.is_symlink():
        raise RuntimeError(f"W&B checkpoint cache must not be a symlink: {path}")
    for name in ("state_dict.pth", "resolved_config.yaml", "source.json"):
        file = path / name
        if not file.is_file() or file.is_symlink():
            raise RuntimeError(
                f"Incomplete W&B checkpoint cache at {path}; remove it and retry"
            )
    source = json.loads((path / "source.json").read_text())
    if any(
        source[key] != value
        for key, value in (
            ("run_path", run_path),
            ("filename", filename),
            ("md5", checksum),
            ("state_dict_size", (path / "state_dict.pth").stat().st_size),
            ("config_size", (path / "resolved_config.yaml").stat().st_size),
        )
    ):
        raise RuntimeError(
            f"W&B checkpoint cache identity mismatch at {path}; remove it and retry"
        )
    return path


def download_wandb_model(
    uri: str,
    cache_dir: str | Path | None = None,
    *,
    expected_checkpoint_md5: str | None = None,
) -> Path:
    """Download an exact run's latest published checkpoint, without a Trainer.

    Args:
        uri: ``wandb://entity/project/run-name-or-id[#filename.pth]``. The
            default file is ``model.pth``; names must match exactly and uniquely.
        cache_dir: Optional cache root; defaults to ``~/.cache/fovi/wandb``.
        expected_checkpoint_md5: Optional base64 W&B checksum from a previous
            snapshot's ``source.json``; a different published revision raises.

    Returns:
        Immutable local bundle containing ``state_dict.pth`` and its matched
        ``resolved_config.yaml``. Embedded checkpoint parameters are authoritative.

    Raises:
        ValueError: Invalid URI, ambiguous run name, or malformed checkpoint.
        FileNotFoundError: No matching run or published checkpoint.
        RuntimeError: Downloaded bytes do not match the published checksum.

    Unpinned calls check the remote revision, including while a run is training.
    A run-ID URI with an expected checksum can reuse its exact cached snapshot
    offline. Cached metadata is in ``source.json`` (run_path, filename, md5, size).
    W&B authentication uses the installed SDK's normal environment/login settings.
    """
    entity, project, reference, filename = _parse_model_uri(uri)
    root = (
        Path(cache_dir) if cache_dir is not None else Path.home() / ".cache/fovi/wandb"
    )
    if expected_checkpoint_md5 is not None:
        _validate_checksum(expected_checkpoint_md5)
    if expected_checkpoint_md5 is not None and re.fullmatch(
        r"[A-Za-z0-9_-]+", reference
    ):
        # Saved run-ID references can reuse their exact snapshot offline. Names
        # are resolved remotely before any new cache entry is created.
        pinned = (
            root
            / entity
            / project
            / reference
            / _revision_key(filename, expected_checkpoint_md5)
        )
        if pinned.is_dir():
            with FileLock(str(pinned.with_suffix(".lock"))):
                return _validate_cached_bundle(
                    pinned,
                    f"{entity}/{project}/{reference}",
                    filename,
                    expected_checkpoint_md5,
                )
    try:
        import wandb
    except ModuleNotFoundError as error:
        if error.name != "wandb":
            raise
        raise ModuleNotFoundError(
            "Loading a W&B checkpoint requires `pip install wandb`."
        ) from error

    api = wandb.Api()
    runs = list(
        api.runs(
            f"{entity}/{project}",
            filters={"$or": [{"name": reference}, {"display_name": reference}]},
        )
    )
    if not runs:
        raise FileNotFoundError(f"No W&B run matches {entity}/{project}/{reference!s}")
    if len(runs) != 1:
        ids = ", ".join(sorted(run.id for run in runs))
        raise ValueError(f"Ambiguous W&B run name {reference!r}; use a run ID: {ids}")
    run = runs[0]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run.id):
        raise ValueError("W&B returned an invalid run ID")
    files = list(run.files(names=[filename]))
    if not files:
        raise FileNotFoundError(
            f"W&B run {entity}/{project}/{run.id} has no published {filename!r}. "
            "Publish a Fovi checkpoint from the run's training output first."
        )
    remote = files[0]
    if remote.name != filename or not remote.md5 or remote.size <= 0:
        raise ValueError(f"W&B checkpoint {filename!r} has no complete file metadata")
    _validate_checksum(remote.md5)
    if expected_checkpoint_md5 is not None and remote.md5 != expected_checkpoint_md5:
        raise ValueError(
            f"W&B checkpoint {filename!r} no longer matches the saved checkpoint "
            "checksum. Select its immutable checkpoint filename to replay it."
        )
    revision = _revision_key(filename, remote.md5)
    run_dir = root / entity / project / run.id
    run_dir.mkdir(parents=True, exist_ok=True)
    destination = run_dir / revision
    with FileLock(str(run_dir / f"{revision}.lock")):
        if destination.is_dir():
            return _validate_cached_bundle(
                destination,
                f"{entity}/{project}/{run.id}",
                filename,
                remote.md5,
            )
        with TemporaryDirectory(prefix=f".{revision}-", dir=run_dir) as temporary:
            staging = Path(temporary)
            with remote.download(root=str(staging), api=api):
                pass
            downloaded = staging / filename
            if (
                downloaded.stat().st_size != remote.size
                or _file_md5(downloaded) != remote.md5
            ):
                raise RuntimeError(
                    f"W&B checkpoint {filename!r} changed during download or failed "
                    "checksum verification; retry to resolve its current revision."
                )
            checkpoint = torch.load(downloaded, map_location="cpu", weights_only=True)
            if not isinstance(checkpoint, dict) or "params" not in checkpoint:
                raise ValueError("A W&B Fovi checkpoint must contain its saved params")
            cfg = OmegaConf.create(checkpoint["params"])
            if not isinstance(cfg, DictConfig):
                raise TypeError("Checkpoint params must be a configuration mapping")
            keys = [key for key in ("model", "state_dict") if key in checkpoint]
            if len(keys) != 1 or not isinstance(checkpoint[keys[0]], dict):
                raise ValueError(
                    "Checkpoint must contain exactly one model or state_dict mapping"
                )
            weights = checkpoint[keys[0]]
            if not weights or any(
                not isinstance(key, str) or not isinstance(value, torch.Tensor)
                for key, value in weights.items()
            ):
                raise ValueError(
                    "Checkpoint model weights must be a nonempty tensor mapping"
                )
            bundle = staging / "bundle"
            bundle.mkdir()
            normalized = {"state_dict": weights, "params": checkpoint["params"]}
            if "epoch" in checkpoint:
                normalized["epoch"] = checkpoint["epoch"]
            torch.save(normalized, bundle / "state_dict.pth")
            OmegaConf.save(cfg, bundle / "resolved_config.yaml")
            (bundle / "source.json").write_text(
                json.dumps(
                    {
                        "uri": uri,
                        "run_path": f"{entity}/{project}/{run.id}",
                        "run_name": run.name,
                        "filename": filename,
                        "md5": remote.md5,
                        "size": remote.size,
                        "state_dict_size": (bundle / "state_dict.pth").stat().st_size,
                        "config_size": (bundle / "resolved_config.yaml").stat().st_size,
                    },
                    indent=2,
                )
                + "\n"
            )
            bundle.rename(destination)
    return destination
