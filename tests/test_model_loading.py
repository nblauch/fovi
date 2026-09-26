"""Local inference restores checkpoints without invoking a trainer."""

from __future__ import annotations

import json
import warnings
from collections.abc import Iterator
from functools import partial
from pathlib import Path

import pytest
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from torch import nn

from fovi.models.loading import find_config, get_model_from_base_fn, load_config


class LocalModel(nn.Module):
    def __init__(self, cfg: DictConfig, device: str | torch.device) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(cfg.width, device=device))
        self.cfg = cfg


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


@pytest.mark.parametrize("geometry", ["legacy", "planar", "spherical"])
def test_checkpoint_preserves_explicit_geometry(tmp_path: Path, geometry: str) -> None:
    directory = tmp_path / "model"
    directory.mkdir()
    OmegaConf.save(
        OmegaConf.create({"width": 1, "saccades": {"field_geometry": geometry}}),
        directory / "config.yaml",
    )
    torch.save({"state_dict": {"weight": torch.ones(1)}}, directory / "state_dict.pth")
    model = get_model_from_base_fn(
        "model", device="cpu", model_dirs=[tmp_path], fovinet_cls=LocalModel
    )
    assert model.cfg.saccades.field_geometry == geometry
    overridden = get_model_from_base_fn(
        "model",
        device="cpu",
        model_dirs=[tmp_path],
        fovinet_cls=LocalModel,
        **{"saccades.field_geometry": "planar"},
    )
    assert overridden.cfg.saccades.field_geometry == "planar"


@pytest.fixture
def small_fovi_config() -> DictConfig:
    """Use the real CNN builders with fewer channels for CPU compatibility checks."""
    path = (
        Path(__file__).resolve().parents[1]
        / "config/pretrained/fovi-alexnet_a-0.5_res-64_rfmult-1_in1k.yaml"
    )
    cfg = OmegaConf.load(path)
    cfg.model.channel_mult = 0.125
    cfg.model.mlp = "16-16"
    cfg.training.load_cpu = 1
    with open_dict(cfg.saccades):
        cfg.saccades.field_geometry = "legacy"
    return cfg


@pytest.fixture
def cpu_model_threads() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.set_num_threads(previous)


@pytest.mark.usefixtures("cpu_model_threads")
@pytest.mark.parametrize("geometry", ["planar", "legacy"])
def test_pooling_model_constructs_and_samples(
    small_fovi_config: DictConfig, geometry: str
) -> None:
    from fovi.models import FoviNet

    cfg = small_fovi_config
    cfg.saccades.field_geometry = geometry
    cfg.saccades.sampler = "pooling"
    model = FoviNet(cfg, device="cpu").eval()
    image = torch.ones(1, 3, cfg.training.resolution, cfg.training.resolution)
    with torch.no_grad():
        samples = model.retinal_transform(image, torch.tensor([[0.5, 0.5]]))
    assert samples.shape[-1] == len(model.retinal_transform.sampler.coords)
    assert torch.isfinite(samples).all()
    assert samples.abs().max() > 0
    assert model.retinal_transform.sampler.coords.field_geometry == geometry


@pytest.mark.usefixtures("cpu_model_threads")
def test_spherical_model_shares_calibrated_crop_geometry(
    small_fovi_config: DictConfig,
) -> None:
    from dataclasses import asdict

    from fovi.arch.knn import KNNConvLayer
    from fovi.models import FoviNet
    from fovi.sensing.projection import CameraModel, field_of_view_pair

    camera = CameraModel("fisheye", (480, 640), (300, 300, 319.5, 239.5))
    cfg = small_fovi_config
    with open_dict(cfg.saccades):
        cfg.saccades.field_geometry = "spherical"
        cfg.saccades.camera_model = asdict(camera)
        cfg.saccades.fov_reference_side = "long"
        cfg.saccades.rescale_fov = 1
        cfg.saccades.fixation_size = 320
        cfg.saccades.fixation_size_min_frac = 1
        cfg.saccades.fixation_size_max_frac = 1
        cfg.saccades.cmf_a = "auto"
    model = FoviNet(cfg, device="cpu").eval()
    assert tuple(model.retinal_transform.fov) == pytest.approx(
        field_of_view_pair(camera, 0.5)
    )
    assert model.retinal_transform.cmf_a == cfg.saccades.cmf_a
    assert model.retinal_transform.camera_model == camera
    convolutions = [m for m in model.modules() if isinstance(m, KNNConvLayer)]
    assert convolutions
    for layer in convolutions:
        assert layer.in_coords.field_geometry == "spherical"
        assert tuple(layer.in_coords.fov) == pytest.approx(model.retinal_transform.fov)
        assert layer.in_coords.cmf_a == model.retinal_transform.cmf_a


@pytest.mark.usefixtures("cpu_model_threads")
@pytest.mark.parametrize("via_loader", [False, True])
def test_missing_geometry_warns_and_preserves_legacy_outputs(
    tmp_path: Path, small_fovi_config: DictConfig, via_loader: bool
) -> None:
    from fovi.arch.knn import KNNConvLayer, KNNPoolingLayer
    from fovi.models import FoviNet

    cfg = small_fovi_config
    torch.manual_seed(123)
    reference = FoviNet(cfg, device="cpu").eval()
    checkpoint_cfg = OmegaConf.create(OmegaConf.to_container(cfg))
    del checkpoint_cfg.saccades.field_geometry
    OmegaConf.set_struct(checkpoint_cfg, True)
    directory = tmp_path / "custom-checkpoint"
    directory.mkdir()
    OmegaConf.save(checkpoint_cfg, directory / "config.yaml")
    torch.save({"state_dict": reference.state_dict()}, directory / "state_dict.pth")

    if via_loader:
        build = partial(
            get_model_from_base_fn,
            "custom-checkpoint",
            device="cpu",
            model_dirs=[tmp_path],
            quiet=True,
        )
    else:
        build = partial(FoviNet, checkpoint_cfg, device="cpu")
    with pytest.warns(
        UserWarning, match="saccades.field_geometry.*missing.*legacy"
    ) as caught:
        restored = build()
    restored.eval()
    if not via_loader:
        restored.load_state_dict(reference.state_dict())
    geometry_warnings = [
        item for item in caught if "saccades.field_geometry" in str(item.message)
    ]
    assert len(geometry_warnings) == 1
    assert "planar" in str(geometry_warnings[0].message)
    assert restored.cfg.saccades.field_geometry == "legacy"
    assert restored.retinal_transform.field_geometry == "legacy"
    old_layers = [
        module
        for module in reference.modules()
        if isinstance(module, (KNNConvLayer, KNNPoolingLayer))
    ]
    new_layers = [
        module
        for module in restored.modules()
        if isinstance(module, (KNNConvLayer, KNNPoolingLayer))
    ]
    assert len(old_layers) == len(new_layers) > 0
    for old, new in zip(old_layers, new_layers):
        assert new.in_coords.field_geometry == new.out_coords.field_geometry == "legacy"
        assert torch.equal(old.knn_indices_pad_token, new.knn_indices_pad_token)
        if isinstance(old, KNNConvLayer):
            assert torch.equal(old.local_rf, new.local_rf)
    images = torch.rand(4, 3, 256, 256)
    gaze = torch.tensor([[0.4, 0.6]]).expand(4, -1)
    with torch.inference_mode():
        expected = reference(images.clone(), fixations=[gaze], n_fixations=1)
        actual = restored(images.clone(), fixations=[gaze], n_fixations=1)
    assert torch.equal(expected[0], actual[0])
    assert torch.equal(expected[2], actual[2])


@pytest.mark.usefixtures("cpu_model_threads")
def test_missing_geometry_explicit_override_is_planar_without_warning(
    tmp_path: Path, small_fovi_config: DictConfig
) -> None:
    from fovi.arch.knn import KNNConvLayer

    del small_fovi_config.saccades.field_geometry
    OmegaConf.save(small_fovi_config, tmp_path / "recipe.yaml")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = get_model_from_base_fn(
            "recipe",
            load=False,
            device="cpu",
            model_dirs=[tmp_path],
            quiet=True,
            **{"saccades.field_geometry": "planar"},
        )
    assert not [
        item for item in caught if "saccades.field_geometry" in str(item.message)
    ]
    assert model.cfg.saccades.field_geometry == "planar"
    assert model.retinal_transform.field_geometry == "planar"
    assert all(
        module.in_coords.field_geometry == "planar"
        for module in model.modules()
        if isinstance(module, KNNConvLayer)
    )


MODEL_CONFIG_ROOT = Path(__file__).resolve().parents[1]
TRAINING_CONFIGS = sorted(
    list((MODEL_CONFIG_ROOT / "config").glob("*.yaml"))
    + list((MODEL_CONFIG_ROOT / "config/pretrained").glob("*.yaml"))
    + list((MODEL_CONFIG_ROOT / "benchmarks/configs").glob("*.yaml"))
)


@pytest.mark.parametrize("path", TRAINING_CONFIGS, ids=lambda path: path.stem)
def test_training_and_benchmark_recipes_select_planar(path: Path) -> None:
    cfg, _, _ = load_config(path.stem, False, path.parent)
    assert cfg.saccades.field_geometry == "planar"


@pytest.mark.usefixtures("cpu_model_threads")
@pytest.mark.parametrize("value", [None, "???", "invalid"])
def test_explicit_invalid_geometry_does_not_select_legacy(
    small_fovi_config: DictConfig, value: str | None
) -> None:
    from omegaconf.errors import MissingMandatoryValue

    from fovi.models import FoviNet

    small_fovi_config.saccades.field_geometry = value
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises((ValueError, MissingMandatoryValue), match="field_geometry"):
            FoviNet(small_fovi_config, device="cpu")
    assert not [
        item for item in caught if "saccades.field_geometry" in str(item.message)
    ]
