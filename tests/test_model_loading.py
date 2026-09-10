"""Local inference restores checkpoints without invoking a trainer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

from fovi.models.loading import find_config, get_model_from_base_fn, load_config


class LocalModel(nn.Module):
    def __init__(self, cfg: DictConfig, device: str | torch.device) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(cfg.width, device=device))


@pytest.mark.parametrize(
    "filename,model_key",
    [
        ("state_dict.pth", "state_dict"),
        ("final_weights.pth", "state_dict"),
        ("model.pth", "model"),
    ],
)
def test_inference_loads_cpu_checkpoint(
    tmp_path: Path, filename: str, model_key: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("FOVI_SAVE_DIR", raising=False)
    monkeypatch.delenv("FOVI_DATASETS_DIR", raising=False)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    OmegaConf.save(OmegaConf.create({"width": 3}), model_dir / "config.yaml")
    torch.save(
        {model_key: {"module.weight": torch.tensor([1.0, 2.0, 3.0])}},
        model_dir / filename,
    )
    model = get_model_from_base_fn(
        "model", device="cpu", model_dirs=[tmp_path], fovinet_cls=LocalModel
    )
    torch.testing.assert_close(model.weight, torch.tensor([1.0, 2.0, 3.0]))


def test_sharded_checkpoint_and_json_config(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "params.json").write_text(json.dumps({"width": 2}))
    (model_dir / "state_dict.index.json").write_text(
        json.dumps(
            {
                "weight_map": {"left": "first.pth", "right": "second.pth"},
            }
        )
    )
    torch.save({"left": torch.tensor([1.0])}, model_dir / "first.pth")
    torch.save({"right": torch.tensor([2.0])}, model_dir / "second.pth")
    cfg, checkpoint, key = load_config("model", True, tmp_path, device="cpu")
    assert cfg.width == 2
    assert key == "state_dict"
    assert set(checkpoint[key]) == {"left", "right"}
    assert not (model_dir / "hydra").exists()


def test_standalone_hydra_config(tmp_path: Path) -> None:
    (tmp_path / "model.yaml").write_text("width: 4\n")
    cfg, checkpoint, key = load_config("model", False, tmp_path)
    assert cfg.width == 4
    assert checkpoint is None
    assert key is None


@pytest.mark.parametrize("broken", [False, True])
def test_nested_hydra_composition_preserves_application(
    tmp_path: Path, broken: bool
) -> None:
    import hydra
    from hydra.core.global_hydra import GlobalHydra
    from hydra.errors import MissingConfigException

    application = tmp_path / "application"
    model = tmp_path / "model"
    application.mkdir()
    model.mkdir()
    (application / "app.yaml").write_text("owner: application\n")
    (model / "width.yaml").write_text("value: 4\n")
    (model / "network.yaml").write_text(
        "defaults: [missing]\n" if broken else "defaults: [width]\n"
    )
    with hydra.initialize_config_dir(version_base=None, config_dir=str(application)):
        original = GlobalHydra.instance()
        if broken:
            with pytest.raises(MissingConfigException):
                load_config("network", False, model)
        else:
            cfg, _, _ = load_config("network", False, model)
            assert cfg.value == 4
        assert GlobalHydra.instance() is original
        assert hydra.compose(config_name="app").owner == "application"


def test_logging_override_explains_missing_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("FOVI_SAVE_DIR", raising=False)
    (tmp_path / "model.yaml").write_text(
        "width: 2\nlogging: {base_fn: old, use_wandb: false}\n"
    )
    with pytest.raises(ValueError, match="logging.folder.*FOVI_SAVE_DIR"):
        get_model_from_base_fn(
            "model",
            load=False,
            device="cpu",
            model_dirs=[tmp_path],
            fovinet_cls=LocalModel,
            **{"logging.base_fn": "new"},
        )
    model = get_model_from_base_fn(
        "model",
        load=False,
        device="cpu",
        model_dirs=[tmp_path],
        fovinet_cls=LocalModel,
        **{"logging.base_fn": "new", "logging.folder": str(tmp_path / "logs")},
    )
    assert model.weight.shape == (2,)


def test_existing_broken_local_model_does_not_download(tmp_path: Path) -> None:
    (tmp_path / "model").mkdir()
    with pytest.raises(FileNotFoundError, match="params.json"):
        find_config("model", True, [tmp_path], device="cpu")


def test_local_model_without_weights_fails(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.yaml").write_text("width: 4\n")
    with pytest.raises(ValueError, match="state_dict not found"):
        find_config("model", True, [tmp_path], device="cpu")
