"""Resuming archives configuration overrides without destroying provenance."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from hydra.core.hydra_config import HydraConfig


def test_resume_preserves_original_and_each_launch_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("FOVI_SAVE_DIR", str(tmp_path))
    monkeypatch.setenv("FOVI_DATASETS_DIR", str(tmp_path))
    from fovi.training.trainer import Trainer

    source = tmp_path / "launch"
    (source / ".hydra").mkdir(parents=True)
    config = source / ".hydra" / "config.yaml"
    config.write_text("training:\n  epochs: 1\n")
    trainer = Trainer.__new__(Trainer)
    trainer.rank = 0
    trainer.log_folder = tmp_path / "model"
    trainer.cfg = SimpleNamespace(
        training=SimpleNamespace(from_checkpoint=False, eval_only=False)
    )
    monkeypatch.setattr(HydraConfig, "initialized", lambda: True)
    monkeypatch.setattr(
        HydraConfig,
        "get",
        lambda: SimpleNamespace(run=SimpleNamespace(dir=str(source))),
    )
    trainer.copy_hydra_outputs()
    original = (trainer.log_folder / "hydra" / "config.yaml").read_text()
    trainer.cfg.training.from_checkpoint = True
    for epochs in (2, 3):
        config.write_text(f"training:\n  epochs: {epochs}\n")
        trainer.copy_hydra_outputs()
    assert (trainer.log_folder / "hydra" / "config.yaml").read_text() == original
    archived = sorted(
        path.read_text()
        for path in trainer.log_folder.glob("hydra-resumes/*/config.yaml")
    )
    assert archived == ["training:\n  epochs: 2\n", "training:\n  epochs: 3\n"]
