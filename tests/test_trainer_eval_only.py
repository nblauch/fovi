"""Evaluation must not require a training dataset or optimizer state."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def test_eval_only_restores_weights_without_training_resources(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("FOVI_SAVE_DIR", str(tmp_path))
    monkeypatch.setenv("FOVI_DATASETS_DIR", str(tmp_path))
    from fovi.models.loading import load_config
    from fovi.training.trainer import Trainer

    cfg, _, _ = load_config(
        "fovi_alexnet", load=False, folder=Path(__file__).parents[1] / "config"
    )
    cfg.training.eval_only = True
    cfg.training.use_amp = False
    cfg.data.train_dataset = None
    cfg.data.val_dataset = str(tmp_path / "validation.ffcv")
    cfg.logging.use_wandb = False
    cfg.logging.folder = str(tmp_path)

    def forbidden_train_loader(
        self: Trainer, dataset: str | None, subset: float | None
    ) -> None:
        pytest.fail("Evaluation attempted to open the training dataset")

    def forbidden_optimizer(self: Trainer) -> None:
        pytest.fail("Evaluation attempted to create training resources")

    validation_loader = SimpleNamespace(indices=torch.arange(2))
    monkeypatch.setattr(Trainer, "create_train_loader", forbidden_train_loader)
    monkeypatch.setattr(Trainer, "create_optimizer", forbidden_optimizer)
    monkeypatch.setattr(
        Trainer, "create_val_loader", lambda *args, **kwargs: validation_loader
    )
    trainer = Trainer(None, cfg, load_checkpoint=False)
    assert trainer.val_loader is validation_loader
    assert trainer.train_loader is None

    checkpoint = {
        "epoch": 3,
        "model": trainer.model_.state_dict(),
        "probes": trainer.probes.state_dict(),
        "optimizer": {"invalid": "must not be read during evaluation"},
        "optimizer_probes": {},
        "lr_scheduler": {},
    }
    trainer.load_checkpoint(checkpoint)
    assert trainer.start_epoch == 3
    with pytest.raises(RuntimeError, match="eval_only"):
        trainer.train()
