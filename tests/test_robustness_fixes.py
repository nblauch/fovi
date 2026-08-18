"""CPU regression tests for the local LoRA and KNN robustness fixes."""

import unittest

import torch
from torch import nn

from fovi.arch.knnvit import KNNPartitioningPatchEmbedding
from fovi.utils.lora import apply_lora


class TestLoRAParameterPlacement(unittest.TestCase):
    def test_apply_lora_matches_base_parameter_device_and_dtype(self):
        layer = nn.Linear(4, 3, bias=False, device="cpu", dtype=torch.float64)

        lora = apply_lora(layer, r=2, device="cuda")

        self.assertEqual(lora.A.device, layer.weight.device)
        self.assertEqual(lora.B.device, layer.weight.device)
        self.assertEqual(lora.A.dtype, layer.weight.dtype)
        self.assertEqual(lora.B.dtype, layer.weight.dtype)
        output = layer(torch.ones(1, 4, dtype=torch.float64))
        self.assertEqual(output.dtype, torch.float64)


class TestMinimumFullCoverageNeighborhood(unittest.TestCase):
    def test_returns_smallest_k_covering_every_nonpadding_input(self):
        distances = torch.tensor([
            [0.0, 10.0],
            [1.0, 10.0],
            [2.0, 2.0],
            [10.0, 0.0],
            [10.0, 1.0],
            [0.5, 0.5],  # Padding row; index equals num_inputs.
        ])

        k = KNNPartitioningPatchEmbedding._minimum_k_covering_inputs(
            distances,
            num_inputs=5,
        )

        self.assertEqual(k, 4)
        selected = torch.topk(
            distances,
            k,
            dim=0,
            largest=False,
        ).indices
        covered = torch.unique(selected[selected < 5])
        torch.testing.assert_close(covered, torch.arange(5))

    def test_raises_when_supported_k_cannot_cover_all_inputs(self):
        distances = torch.tensor([
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [100.0, 100.0],
            [0.5, 0.5],  # Padding displaces the last input at maximum k.
        ])

        with self.assertRaisesRegex(RuntimeError, "covers every non-padding"):
            KNNPartitioningPatchEmbedding._minimum_k_covering_inputs(
                distances,
                num_inputs=5,
            )


if __name__ == "__main__":
    unittest.main()
