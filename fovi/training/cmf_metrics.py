"""Deterministic CMF assignment and sample-weighted validation statistics."""

from copy import deepcopy

import torch
from torch import distributed as dist
from torch import nn


def validation_cmf_indices(
    batch_size: int,
    offset: int,
    levels: int,
    device: torch.device,
    rank: int = 0,
    world_size: int = 1,
) -> torch.Tensor:
    """Assign the strided distributed validation traversal to levels in order."""
    return (
        (torch.arange(batch_size, device=device) + offset) * world_size + rank
    ) % levels


class CmfValidationMetrics:
    """Accumulate per-level sums/counts on device, reducing once after validation."""

    def __init__(
        self, values: tuple[float, ...], loss: nn.Module, device: torch.device
    ) -> None:
        self.values = values
        self.device = device
        self.loss = deepcopy(loss)
        self.loss.reduction = "none"
        self.counts = torch.zeros(len(values), device=device, dtype=torch.float64)
        self.sums: dict[str, torch.Tensor] = {}
        self.denominators: dict[str, torch.Tensor] = {}

    def count(self, indices: torch.Tensor) -> None:
        """Record images once, independently of probe and fixation metrics."""
        self.counts.scatter_add_(
            0, indices, torch.ones_like(indices, dtype=self.counts.dtype)
        )

    def _update(
        self, name: str, per_image: torch.Tensor, indices: torch.Tensor
    ) -> None:
        if name not in self.sums:
            self.sums[name] = torch.zeros_like(self.counts)
            self.denominators[name] = torch.zeros_like(self.counts)
        self.sums[name].scatter_add_(
            0, indices, per_image.detach().to(self.counts.dtype)
        )
        self.denominators[name].scatter_add_(
            0, indices, torch.ones_like(indices, dtype=self.counts.dtype)
        )

    def update_loss(
        self,
        name: str,
        logits: torch.Tensor,
        target: torch.Tensor,
        indices: torch.Tensor,
    ) -> None:
        """Reduce class/label losses per image before grouping by CMF level."""
        loss = self.loss(logits, target).reshape(logits.shape[0], -1).mean(-1)
        self._update(name, loss, indices)

    def update_accuracy(
        self,
        name: str,
        logits: torch.Tensor,
        target: torch.Tensor,
        indices: torch.Tensor,
    ) -> None:
        """Match the trainer's multiclass top-k and micro multilabel accuracy."""
        if name.startswith("top_1"):
            correct = logits.argmax(-1).eq(target).float()
        elif name.startswith("top_5"):
            correct = logits.topk(5, dim=-1).indices.eq(target[:, None]).any(-1).float()
        elif name.startswith("multilabel_acc"):
            correct = logits.sigmoid().gt(0.5).eq(target.bool()).float().mean(-1)
        else:
            raise ValueError(f"Unsupported CMF accuracy metric {name!r}")
        self._update(name, correct, indices)

    def compute(self) -> dict[str, float]:
        """Reduce sums and counts across ranks; omit empty level metrics."""
        names = list(self.sums)
        packed = torch.stack(
            [
                self.counts,
                *[self.sums[name] for name in names],
                *[self.denominators[name] for name in names],
            ]
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(packed)
        packed = packed.cpu()
        stats = {}
        for level, value in enumerate(self.values):
            suffix = f"_cmf_a-{value:g}"
            stats["samples_val" + suffix] = float(packed[0, level])
            for i, name in enumerate(names):
                denominator = packed[1 + len(names) + i, level]
                if denominator > 0:
                    stats[name + suffix] = float(packed[1 + i, level] / denominator)
        return stats
