"""A checkpoint becomes visible locally and remotely only after a complete save."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn


@pytest.mark.parametrize("publish", [False, True])
def test_checkpoint_publication_sees_complete_replacement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    publish: bool,
) -> None:
    monkeypatch.setenv("FOVI_SAVE_DIR", str(tmp_path))
    monkeypatch.setenv("FOVI_DATASETS_DIR", str(tmp_path))
    from fovi.training.trainer import Trainer

    trainer = Trainer.__new__(Trainer)
    trainer.rank = 0
    trainer.cfg = SimpleNamespace(
        training=SimpleNamespace(train_probes_only=False),
        logging=SimpleNamespace(use_wandb=publish),
    )
    trainer.cfg_dict = {"width": 2}
    trainer.log_folder = tmp_path
    trainer.model_ = nn.Linear(2, 1)
    trainer.optimizer = torch.optim.SGD(trainer.model_.parameters(), lr=0.1)
    trainer.probes = nn.Linear(2, 1)
    trainer.optimizer_probes = torch.optim.SGD(trainer.probes.parameters(), lr=0.1)
    trainer.lr_schedule = False
    published = []

    def save(path: str, *, base_path: str, policy: str) -> None:
        if Path(path).parent == tmp_path / "checkpoints":
            assert Path(path).name.startswith("epoch-")
            assert Path(path).stat().st_ino == (tmp_path / "model.pth").stat().st_ino
        else:
            assert Path(path) == tmp_path / "model.pth"
        assert base_path == str(tmp_path)
        assert policy == "now"
        assert not (tmp_path / "model.pth.tmp").exists()
        published.append(torch.load(path, weights_only=True))

    monkeypatch.setattr("fovi.training.trainer.wandb.save", save)
    trainer.save_checkpoint(5)
    original = (tmp_path / "model.pth").read_bytes()
    loaded = torch.load(tmp_path / "model.pth", weights_only=True)
    assert loaded["epoch"] == 5
    assert loaded["params"] == {"width": 2}
    torch.testing.assert_close(loaded["model"]["weight"], trainer.model_.weight)
    assert len(published) == 2 * int(publish)
    if publish:
        assert published[0]["epoch"] == 5
        torch.testing.assert_close(
            published[0]["model"]["weight"], trainer.model_.weight
        )

    with torch.no_grad():
        trainer.model_.weight.add_(1)
    trainer.save_checkpoint(10)
    latest = torch.load(tmp_path / "model.pth", weights_only=True)
    assert latest["epoch"] == 10
    torch.testing.assert_close(latest["model"]["weight"], trainer.model_.weight)
    if publish:
        snapshot = next((tmp_path / "checkpoints").glob("epoch-000005-model-*.pth"))
        assert snapshot.read_bytes() == original
        saved = torch.load(snapshot, weights_only=True)
        torch.testing.assert_close(saved["model"]["weight"], loaded["model"]["weight"])
        assert not torch.equal(saved["model"]["weight"], latest["model"]["weight"])
    original = (tmp_path / "model.pth").read_bytes()

    def fail_during_save(payload: dict[str, torch.Tensor], path: Path) -> None:
        path.write_bytes(b"incomplete write")
        raise OSError("disk write failed")

    monkeypatch.setattr("fovi.training.trainer.torch.save", fail_during_save)
    with pytest.raises(OSError, match="disk write failed"):
        trainer.save_checkpoint(15)
    assert (tmp_path / "model.pth").read_bytes() == original
    assert len(published) == 4 * int(publish)
