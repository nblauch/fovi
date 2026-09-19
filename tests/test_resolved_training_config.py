"""Training artifacts persist the settings selected while building the model."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from fovi.models import dinov3
from fovi.models.loading import load_config
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from transformers import DINOv3ViTConfig, DINOv3ViTModel


@pytest.mark.parametrize("space", [None, "cortical", "cartesian"])
def test_trainer_saves_resolved_positions_and_preserves_launch_config(
    space: str | None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("FOVI_SAVE_DIR", str(tmp_path))
    monkeypatch.setenv("FOVI_DATASETS_DIR", str(tmp_path))
    from fovi.training.trainer import Trainer

    cfg, _, _ = load_config(
        "dinov3_warped_cartesian_control", False, Path(__file__).parents[1] / "config"
    )
    cfg.model.vit.position_coordinate_space = space
    cfg.model.mlp = "16"
    cfg.pretrained_model.lora = None
    cfg.saccades.resize_size = 32
    cfg.saccades.fixation_size = 32
    cfg.saccades.n_fixations = 1
    cfg.saccades.n_fixations_val = [1]
    cfg.training.resolution = 32
    cfg.training.batch_size = 2
    cfg.training.use_amp = False
    cfg.logging.folder = str(tmp_path / "model")
    cfg.logging.use_wandb = True
    cfg.data.num_classes = 10
    launch = tmp_path / "launch"
    (launch / ".hydra").mkdir(parents=True)
    OmegaConf.save(cfg, launch / ".hydra" / "config.yaml")
    original = (launch / ".hydra" / "config.yaml").read_text()
    monkeypatch.setattr(HydraConfig, "initialized", lambda: True)
    monkeypatch.setattr(
        HydraConfig,
        "get",
        lambda: SimpleNamespace(run=SimpleNamespace(dir=str(launch))),
    )

    native = DINOv3ViTModel(
        DINOv3ViTConfig(
            image_size=32,
            patch_size=16,
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=64,
            num_register_tokens=0,
        )
    )
    monkeypatch.setattr(
        dinov3, "load_dinov3", lambda *args, **kwargs: (copy.deepcopy(native), None)
    )
    loader = SimpleNamespace(indices=torch.arange(8))
    monkeypatch.setattr(Trainer, "create_train_loader", lambda *args, **kwargs: loader)
    monkeypatch.setattr(Trainer, "create_val_loader", lambda *args, **kwargs: loader)
    remote_configs = []
    monkeypatch.setattr(
        "fovi.training.trainer.wandb.init",
        lambda **kwargs: remote_configs.append(copy.deepcopy(kwargs["config"])),
    )
    trainer = Trainer(None, cfg, load_checkpoint=False)
    expected = "cortical" if space is None else space
    assert trainer.cfg_dict["model"]["vit"]["position_coordinate_space"] == expected
    assert remote_configs[0]["model"]["vit"]["position_coordinate_space"] == expected
    trainer.save_checkpoint(1)
    saved, checkpoint, key = load_config("model", True, tmp_path, device="cpu")
    assert saved.model.vit.position_coordinate_space == expected
    assert checkpoint["params"]["model"]["vit"]["position_coordinate_space"] == expected
    assert key == "model"
    params = json.loads((tmp_path / "model" / "params.json").read_text())
    assert params["model"]["vit"]["position_coordinate_space"] == expected
    assert (tmp_path / "model" / "hydra" / "config.yaml").read_text() == original

    saved.training.from_checkpoint = True
    saved.training.epochs = 2
    resumed = Trainer(None, saved)
    assert resumed.start_epoch == 1
    restored, _, _ = load_config("model", False, tmp_path)
    assert restored.training.epochs == 2
    assert restored.model.vit.position_coordinate_space == expected
    assert (tmp_path / "model" / "hydra" / "config.yaml").read_text() == original
    assert len(list((tmp_path / "model" / "hydra-resumes").iterdir())) == 1

    resolved_path = tmp_path / "model" / "resolved_config.yaml"
    resolved_before_failure = resolved_path.read_bytes()
    del checkpoint["model"][next(iter(checkpoint["model"]))]
    torch.save(checkpoint, tmp_path / "model" / "model.pth")
    saved.training.epochs = 99
    with pytest.raises(RuntimeError, match="Missing key"):
        Trainer(None, saved)
    assert resolved_path.read_bytes() == resolved_before_failure


def test_historical_config_precedence_is_unchanged(tmp_path: Path) -> None:
    directory = tmp_path / "model"
    (directory / "hydra").mkdir(parents=True)
    (directory / "hydra" / "config.yaml").write_text("source: hydra\n")
    (directory / "config.yaml").write_text("source: root\n")
    assert load_config("model", False, tmp_path)[0].source == "hydra"
