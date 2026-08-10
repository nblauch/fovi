import unittest
from unittest import mock

import torch
from torch import nn

import fovi.arch.knn as knn_module
from fovi.arch.knn import KNNPoolingLayer
from fovi.sensing.samplers import KNNGridSampler


def _stub_pooler() -> KNNPoolingLayer:
    layer = KNNPoolingLayer.__new__(KNNPoolingLayer)
    nn.Module.__init__(layer)
    layer.mode = "avg"
    layer._k = 2
    layer.knn_pad_token_val = 3
    layer.knn_indices_pad_token = torch.tensor([[0, 1], [2, 3]])
    return layer


class TestKNNPoolingSampler(unittest.TestCase):
    def test_fractional_resolution_multiplier_is_rounded_once(self):
        sampler = KNNGridSampler(
            fov=1.0,
            cmf_a=0.1,
            resolution=5,
            res_mult=1.5,
            cmf_a_mult=0.75,
            k=1,
            sample_cortex=False,
            device="cpu",
        )

        self.assertEqual(sampler.highres_resolution, 8)
        self.assertEqual(sampler.res_mult, 1.5)
        self.assertEqual(sampler.cmf_a_mult, 0.75)

    def test_reference_pooling_reports_baseline_backend(self):
        layer = _stub_pooler()
        values = torch.tensor([[[1.0, 2.0, 3.0]]])

        output = layer(values)

        torch.testing.assert_close(output, torch.tensor([[[2.0, 2.0]]]))
        self.assertEqual(layer._last_knn_pool_backend, "baseline")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device required")
    def test_fused_pooling_reports_cuda_backend(self):
        layer = _stub_pooler().cuda()
        values = torch.ones(1, 1, 3, device="cuda")
        fused_output = torch.full((1, 1, 2), 7.0, device="cuda")

        with mock.patch.object(
            knn_module,
            "_optimized_pool_forward",
            return_value=fused_output,
        ):
            output = layer(values)

        self.assertIs(output, fused_output)
        self.assertEqual(layer._last_knn_pool_backend, "cuda")


if __name__ == "__main__":
    unittest.main()
