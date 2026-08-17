import unittest
from unittest import mock

import torch
from torch import nn

import fovi.arch.knn as knn_module
from fovi.arch.knn import KNNPoolingLayer
from fovi.sensing.samplers import BaseGridSampler, KNNGridSampler


def _stub_pooler(*, invalid_output=False) -> KNNPoolingLayer:
    layer = KNNPoolingLayer.__new__(KNNPoolingLayer)
    nn.Module.__init__(layer)
    layer.mode = "avg"
    layer._k = 2
    layer.knn_pad_token_val = 3
    if invalid_output:
        layer.knn_indices_pad_token = torch.tensor([[0, 3], [2, 3]])
        output_valid_mask = torch.tensor([True, False])
    else:
        layer.knn_indices_pad_token = torch.tensor([[0, 1], [2, 3]])
        output_valid_mask = torch.tensor([True, True])
    layer.register_buffer(
        "_knn_output_valid_mask", output_valid_mask, persistent=False)
    layer._knn_all_outputs_valid = bool(output_valid_mask.all().item())
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

    def test_all_valid_mask_preserves_output_identity(self):
        layer = _stub_pooler()
        output = torch.ones(1, 1, 2)

        self.assertIs(layer._mask_invalid_outputs(output), output)
        self.assertIs(
            BaseGridSampler._mask_invalid_samples(
                output, layer._knn_output_valid_mask, all_valid=True),
            output,
        )

    def test_invalid_pooling_outputs_are_still_zeroed(self):
        layer = _stub_pooler(invalid_output=True)
        values = torch.tensor([[[1.0, 2.0, 3.0]]])

        output = layer(values)

        torch.testing.assert_close(output, torch.tensor([[[2.0, 0.0]]]))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device required")
    def test_invalid_fov_masks_are_cuda_graph_capturable(self):
        layer = _stub_pooler(invalid_output=True).cuda()
        values = torch.ones(1, 1, 2, device="cuda")
        sample_mask = torch.tensor([True, False], device="cuda")

        side_stream = torch.cuda.Stream()
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            for _ in range(3):
                layer._mask_invalid_outputs(values)
                BaseGridSampler._mask_invalid_samples(
                    values, sample_mask, all_valid=False)
        torch.cuda.current_stream().wait_stream(side_stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            knn_output = layer._mask_invalid_outputs(values)
            sampler_output = BaseGridSampler._mask_invalid_samples(
                values, sample_mask, all_valid=False)
        graph.replay()

        expected = torch.tensor([[[1.0, 0.0]]], device="cuda")
        torch.testing.assert_close(knn_output, expected)
        torch.testing.assert_close(sampler_output, expected)

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
