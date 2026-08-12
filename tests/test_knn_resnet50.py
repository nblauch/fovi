import unittest

import torch

from fovi.arch.architectures import ARCHITECTURE_REGISTRY
from fovi.arch.knnresnet import KNNResNet, KNNResNetBottleneck


class TestKNNResNet50(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = KNNResNet(
            block=KNNResNetBottleneck,
            layers=[3, 4, 6, 3],
            in_res=64,
            out_res=1,
            fov=16,
            cmf_a=0.5,
            style="isotropic",
            norm_type="batch",
            sample_cortex=False,
            device="cpu",
            auto_match_cart_resources=1,
            ref_frame_mult=2,
        ).eval()

    def test_registered_resnet50_topology(self):
        self.assertTrue(ARCHITECTURE_REGISTRY.has("fovi_resnet50"))
        self.assertEqual(
            [
                len(self.model.layer1),
                len(self.model.layer2),
                len(self.model.layer3),
                len(self.model.layer4),
            ],
            [3, 4, 6, 3],
        )

        first = self.model.layer1[0]
        self.assertIsInstance(first, KNNResNetBottleneck)
        self.assertEqual(
            (first.conv1.in_channels, first.conv1.out_channels), (64, 64)
        )
        self.assertEqual(
            (first.conv3.in_channels, first.conv3.out_channels), (64, 256)
        )
        self.assertEqual(
            (first.downsample[0].in_channels, first.downsample[0].out_channels),
            (64, 256),
        )
        self.assertEqual(first.conv1.ref_grid_size, 1)
        self.assertEqual(first.conv2.ref_grid_size, 6)
        self.assertEqual(first.conv3.ref_grid_size, 1)
        self.assertEqual(self.model.out_channels, 2048)
        self.assertEqual(
            self.model.total_embed_dim,
            self.model.out_channels * len(self.model.out_coords),
        )

    def test_forward_shape(self):
        inputs = torch.randn(2, 3, len(self.model.in_coords))
        with torch.no_grad():
            outputs = self.model(inputs)
        self.assertEqual(
            tuple(outputs.shape),
            (2, self.model.out_channels, len(self.model.out_coords)),
        )


if __name__ == "__main__":
    unittest.main()
