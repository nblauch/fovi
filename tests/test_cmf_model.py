"""Shared encoder construction, image-level selection, and checkpoint compatibility."""

from copy import deepcopy
from pathlib import Path

import pytest
import torch
from fovi.models import dinov3
from fovi.models.architectures import rescale_fov
from fovi.models.fovinet import FoviNet
from fovi.models.loading import load_config
from fovi.sensing.coords import SamplingCoords
from fovi.sensing.policies import MultiRandomSaccadePolicy
from fovi.training.cmf_metrics import CmfValidationMetrics, validation_cmf_indices
from omegaconf import DictConfig
from transformers import DINOv3ViTConfig, DINOv3ViTModel


@pytest.fixture
def local_config(monkeypatch: pytest.MonkeyPatch) -> DictConfig:
    native = DINOv3ViTModel(
        DINOv3ViTConfig(
            image_size=32,
            patch_size=8,
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=64,
            num_register_tokens=0,
        )
    )
    monkeypatch.setattr(
        dinov3,
        "load_dinov3",
        lambda path, device, pretrained: (deepcopy(native).to(device), None),
    )
    cfg, _, _ = load_config(
        "dinov3_warped_grid_multi_cmf",
        False,
        Path(__file__).parents[1] / "config",
        device="cpu",
    )
    cfg.pretrained_model.lora = None
    cfg.pretrained_model.freeze_backbone = False
    cfg.model.vit.patch_size = 8
    cfg.model.mlp = "16"
    cfg.saccades.resize_size = 32
    cfg.saccades.fixation_size = 32
    cfg.training.resolution = 32
    cfg.saccades.n_fixations = 2
    cfg.model.dropout = 0
    cfg.transforms.color_jitter = 0
    cfg.transforms.gray = 0
    cfg.transforms.flip = 0
    return cfg


def test_model_reuses_level_across_fixations_and_roundtrips(
    local_config: DictConfig,
) -> None:
    model = FoviNet(local_config, device="cpu")
    images = torch.rand(3, 3, 32, 32)
    indices = torch.tensor([0, 3, 6])
    fixations = [torch.full((3, 2), 0.5), torch.full((3, 2), 0.5)]
    model.eval()
    with pytest.raises(ValueError, match="explicit cmf_indices"):
        model(images)
    actual, _, views = model(images, fixations=fixations, cmf_indices=indices)
    torch.testing.assert_close(views[:, 0], views[:, 1], atol=0, rtol=0)
    for i, level in enumerate(indices.tolist()):
        cfg = deepcopy(local_config)
        cfg.saccades.cmf_a = cfg.saccades.cmf_a[level]
        scalar = FoviNet(cfg, device="cpu").eval()
        scalar.load_state_dict(model.state_dict(), strict=True)
        _, _, expected = scalar(images, fixations=fixations)
        torch.testing.assert_close(views[i], expected[i], atol=0, rtol=0)
    restored = FoviNet(deepcopy(local_config), device="cpu").eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(
        actual,
        restored(images, fixations=fixations, cmf_indices=indices)[0],
        atol=0,
        rtol=0,
    )
    assert restored.network.backbone.config.fovi_sensor["cmf_a"] == list(
        local_config.saccades.cmf_a
    )
    model.train()
    torch.manual_seed(12)
    selected = model.select_cmf_indices(images)
    torch.manual_seed(12)
    assert torch.equal(selected, model.select_cmf_indices(images))
    # Fixed image selections also survive the training policy's fixation loop.
    policy = MultiRandomSaccadePolicy(
        model.retinal_transform, n_fixations=2, crop_area_range=[1, 1]
    )
    training_views = policy(images, fixations=fixations, cmf_indices=indices)["x_fixs"]
    torch.testing.assert_close(
        training_views[:, 0], training_views[:, 1], atol=0, rtol=0
    )


def test_cartesian_rejected_before_loading(
    local_config: DictConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    local_config.model.vit.position_coordinate_space = "cartesian"
    monkeypatch.setattr(
        dinov3,
        "load_dinov3",
        lambda *args, **kwargs: pytest.fail("Weights loaded before validation"),
    )
    for levels in ([1.0], [0.1, 1.0]):
        local_config.saccades.cmf_a = levels
        with pytest.raises(ValueError, match="cortical"):
            dinov3.build_fovi_dinov3(local_config, device="cpu")


def test_multi_level_checkpoint_keeps_cortical_positions(
    local_config: DictConfig,
) -> None:
    backbone = FoviNet(local_config, device="cpu").network.backbone
    sensor = backbone.config.fovi_sensor
    scalar = SamplingCoords(
        sensor["fov"], sensor["cmf_a"][3], sensor["resolution"],
        style=sensor["style"], fov_type=sensor["fov_type"], device="cpu",
    )
    patch_size = backbone.config.patch_size
    for space in ("cartesian", None):
        if space is None:
            # A config already switched to Cartesian must not resurrect it.
            backbone.config.position_coordinate_space = "cartesian"
        with pytest.raises(ValueError, match="cortical"):
            dinov3.configure_dinov3_positions(
                backbone, sensor_coords=scalar, patch_size=patch_size,
                position_coordinate_space=space,
            )
    assert backbone.config.fovi_sensor["cmf_a"] == list(local_config.saccades.cmf_a)
    dinov3.configure_dinov3_positions(
        backbone, sensor_coords=scalar, patch_size=patch_size,
        position_coordinate_space="cortical",
    )
    assert backbone.config.fovi_sensor["cmf_a"] == sensor["cmf_a"][3]


def test_rescale_preserves_levels(local_config: DictConfig) -> None:
    levels = list(local_config.saccades.cmf_a)
    assert list(rescale_fov(local_config).saccades.cmf_a) == levels


def test_validation_partition_and_metrics() -> None:
    device = torch.device("cpu")
    labels = torch.tensor([0, 1, 2, 3, 4])
    logits = torch.eye(6)[labels] * 5
    logits[1] = torch.tensor([5.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    loss = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    metrics = CmfValidationMetrics((0.1, 1.0, 10.0), loss, device)
    offset = 0
    for size in (2, 3):
        indices = validation_cmf_indices(size, offset, 3, device)
        part = slice(offset, offset + size)
        metrics.count(indices)
        metrics.update_accuracy("top_1_val", logits[part], labels[part], indices)
        metrics.update_loss("loss_val", logits[part], labels[part], indices)
        offset += size
    stats = metrics.compute()
    assignment = validation_cmf_indices(5, 0, 3, device)
    for i, level in enumerate(metrics.values):
        selected = assignment == i
        assert stats[f"samples_val_cmf_a-{level:g}"] == int(selected.sum())
        assert stats[f"top_1_val_cmf_a-{level:g}"] == pytest.approx(
            float(logits[selected].argmax(-1).eq(labels[selected]).float().mean())
        )
        assert stats[f"loss_val_cmf_a-{level:g}"] == pytest.approx(
            float(loss(logits[selected], labels[selected]))
        )
    interleaved = torch.stack(
        [
            validation_cmf_indices(5, 0, 3, device, rank=rank, world_size=2)
            for rank in range(2)
        ],
        -1,
    ).flatten()
    assert torch.equal(interleaved, validation_cmf_indices(10, 0, 3, device))
    empty = CmfValidationMetrics((0.1, 1.0), loss, device)
    empty.count(torch.tensor([0]))
    empty.update_accuracy("top_1_val", logits[:1], labels[:1], torch.tensor([0]))
    assert "top_1_val_cmf_a-1" not in empty.compute()


def test_validation_loop_keeps_original_image_workload(
    local_config: DictConfig, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("FOVI_SAVE_DIR", str(tmp_path))
    from fovi.training.trainer import Trainer
    from torchmetrics import Accuracy, MeanMetric

    cfg = local_config
    cfg.training.distributed = False
    cfg.training.use_amp = False
    cfg.training.no_probes = True
    cfg.validation.do_roc = False
    cfg.validation.repeats = 1
    model = FoviNet(cfg, device="cpu").eval()
    trainer = Trainer.__new__(Trainer)
    trainer.cfg = cfg
    trainer.model = model
    trainer.model_head = model.head
    trainer.probes = torch.nn.Identity()
    trainer.gpu = "cpu"
    trainer.device = torch.device("cpu")
    trainer.amp_dtype = torch.float32
    trainer.rank = 0
    trainer.world_size = 1
    trainer.n_fixations_val = [1, 2]
    trainer.supervised_loss = True
    trainer.do_network_training = True
    trainer.loss_name = "supervised"
    trainer.sup_loss = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    trainer.val_loader = [
        (torch.rand(size, 3, 32, 32), torch.arange(size)) for size in (5, 4)
    ]
    trainer.val_meters = {"loss_classif_val": MeanMetric()}
    for suffix in ("", "_nfix-1", "_nfix-2"):
        for k in (1, 5):
            trainer.val_meters[f"top_{k}_val{suffix}"] = Accuracy(
                "multiclass", num_classes=cfg.data.num_classes, top_k=k
            )
    observed = []

    def capture(
        module: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
        kwargs: dict[str, object],
        output: tuple[torch.Tensor, ...],
    ) -> None:
        observed.append(kwargs["cmf_indices"].clone())

    model.register_forward_hook(capture, with_kwargs=True)
    stats, predictions, targets = trainer.val_loop(return_preds=True)
    assert [len(indices) for indices in observed] == [5, 4]
    indices = torch.cat(observed)
    assert torch.equal(indices, torch.arange(9) % 7)
    for i, value in enumerate(cfg.saccades.cmf_a):
        selected = indices == i
        assert stats[f"samples_val_cmf_a-{value:g}"] == int(selected.sum())
        expected = trainer.sup_loss(
            torch.tensor(predictions[selected]), torch.tensor(targets[selected]).long()
        )
        assert stats[f"loss_classif_val_cmf_a-{value:g}"] == pytest.approx(
            float(expected)
        )


def _distributed_metrics_worker(rank: int, directory: str) -> None:
    from torch import distributed as dist

    dist.init_process_group(
        "gloo", init_method=f"file://{directory}/rendezvous", rank=rank, world_size=2
    )
    try:
        indices = validation_cmf_indices(2, 0, 3, torch.device("cpu"), rank, 2)
        metrics = CmfValidationMetrics(
            (0.1, 1.0, 10.0), torch.nn.CrossEntropyLoss(), torch.device("cpu")
        )
        labels = torch.tensor([0, 1])
        logits = torch.eye(6)[labels] * 4
        if rank == 1:
            logits = logits.roll(1, -1)
        metrics.count(indices)
        metrics.update_accuracy("top_1_val", logits, labels, indices)
        torch.save(metrics.compute(), Path(directory) / f"rank-{rank}.pth")
    finally:
        dist.destroy_process_group()


def test_distributed_metrics_reduce_sums_and_counts(tmp_path: Path) -> None:
    torch.multiprocessing.spawn(
        _distributed_metrics_worker, args=(str(tmp_path),), nprocs=2, join=True
    )
    first = torch.load(tmp_path / "rank-0.pth", weights_only=True)
    assert first == torch.load(tmp_path / "rank-1.pth", weights_only=True)
    assert first["samples_val_cmf_a-0.1"] == 2
    assert first["top_1_val_cmf_a-0.1"] == 0.5
    assert first["top_1_val_cmf_a-1"] == 0
    assert first["top_1_val_cmf_a-10"] == 1


def test_activation_extraction_requires_explicit_level(
    local_config: DictConfig, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("FOVI_SAVE_DIR", str(tmp_path))
    from fovi.training.trainer import Trainer

    cfg = local_config
    cfg.training.use_amp = False
    model = FoviNet(cfg, device="cpu")
    trainer = Trainer.__new__(Trainer)
    trainer.cfg = cfg
    trainer.model = trainer.model_ = model
    trainer.gpu = "cpu"
    trainer.amp_dtype = torch.float32
    trainer.n_fixations_val = [1, 2]
    loader = [(torch.rand(size, 3, 32, 32), torch.arange(size)) for size in (5, 4)]
    observed = []
    select = model.select_cmf_indices

    def capture(
        inputs: torch.Tensor, cmf_indices: torch.Tensor | None = None
    ) -> torch.Tensor | None:
        # get_activations calls forward directly, bypassing module hooks.
        observed.append(cmf_indices.clone())
        return select(inputs, cmf_indices)

    monkeypatch.setattr(model, "select_cmf_indices", capture)
    with pytest.raises(ValueError, match="require cmf_level"):
        trainer.compute_activations(loader, layer_names=["projector"])
    with pytest.raises(ValueError, match="out of range"):
        trainer.compute_activations(loader, layer_names=["projector"], cmf_level=7)
    outputs, _, _ = trainer.compute_activations(
        loader, layer_names=["projector"], cmf_level=3
    )
    assert len(outputs) == 9
    assert [indices.tolist() for indices in observed] == [[3] * 5, [3] * 4]

    cfg = deepcopy(cfg)
    cfg.saccades.cmf_a = cfg.saccades.cmf_a[3]
    with pytest.raises(ValueError, match="requires list-valued"):
        FoviNet(cfg, device="cpu").fixed_cmf_kwargs(0, 2, torch.device("cpu"))
