"""Regression tests for the dense log-polar image-axis contract."""

import unittest

import torch
from transformers import DINOv3ViTConfig, DINOv3ViTModel

from fovi.arch.alexnet import alexnet2023_baseline
from fovi.arch.polar import PolarPadder
from fovi.sensing.coords import SamplingCoords
from fovi.sensing.retina import RetinalTransform


class TestLogPolarGridCoordinates(unittest.TestCase):
    resolution = 4

    def test_dense_grid_is_angle_by_eccentricity(self):
        coords = SamplingCoords(
            16.0, 0.5, self.resolution, style="logpolar_as_grid"
        )
        polar = coords.polar.reshape(self.resolution, self.resolution, 2)
        eccentricity = polar[..., 0]
        angle = polar[..., 1]

        torch.testing.assert_close(
            eccentricity, eccentricity[:1].expand_as(eccentricity)
        )
        torch.testing.assert_close(angle, angle[:, :1].expand_as(angle))
        self.assertTrue(torch.all(eccentricity[:, 1:] > eccentricity[:, :-1]))
        self.assertTrue(coords.cartesian.is_contiguous())
        self.assertTrue(coords.polar.is_contiguous())

    def test_flat_and_dense_logpolar_share_canonical_order(self):
        flat = SamplingCoords(
            16.0, 0.5, self.resolution, style="logpolar"
        )
        grid = SamplingCoords(
            16.0, 0.5, self.resolution, style="logpolar_as_grid"
        )

        for name in ("cartesian", "polar", "plotting", "cortical", "valid_mask"):
            with self.subTest(name=name):
                torch.testing.assert_close(
                    getattr(grid, name), getattr(flat, name)
                )


class TestLogPolarRetinalTransform(unittest.TestCase):
    def test_dense_output_is_angle_by_eccentricity(self):
        resolution = 4
        common = dict(
            resolution=resolution,
            start_res=32,
            fov=16.0,
            cmf_a=0.5,
            sampler="grid_nn",
            fixation_size=32,
            auto_match_cart_resources=False,
            sampler_backend="torch",
            device="cpu",
        )
        flat_retina = RetinalTransform(style="logpolar", **common).eval()
        grid_retina = RetinalTransform(style="logpolar_as_grid", **common).eval()
        image = torch.arange(3 * 32 * 32, dtype=torch.float32).reshape(
            1, 3, 32, 32
        )
        fixation = torch.tensor([[0.5, 0.5]])

        flat = flat_retina(image, fixation, 32)
        grid = grid_retina(image, fixation, 32)
        expected = flat.reshape(1, 3, resolution, resolution)

        self.assertEqual(tuple(grid.shape), (1, 3, resolution, resolution))
        self.assertTrue(grid.is_contiguous())
        torch.testing.assert_close(grid, expected)


class TestPolarPadder(unittest.TestCase):
    def test_wraps_angle_and_zero_pads_eccentricity(self):
        inputs = torch.arange(12, dtype=torch.float32).reshape(1, 1, 3, 4)
        padded = PolarPadder(1)(inputs)

        self.assertEqual(tuple(padded.shape), (1, 1, 5, 6))
        torch.testing.assert_close(padded[:, :, 1:-1, 1:-1], inputs)
        torch.testing.assert_close(padded[:, :, 0, 1:-1], inputs[:, :, -1])
        torch.testing.assert_close(padded[:, :, -1, 1:-1], inputs[:, :, 0])
        self.assertTrue(torch.all(padded[:, :, :, 0] == 0))
        self.assertTrue(torch.all(padded[:, :, :, -1] == 0))


class TestLogPolarDenseConsumers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.resolution = 64
        retina = RetinalTransform(
            resolution=cls.resolution,
            start_res=cls.resolution,
            fov=16.0,
            cmf_a=0.5,
            style="logpolar_as_grid",
            sampler="grid_nn",
            fixation_size=cls.resolution,
            auto_match_cart_resources=False,
            sampler_backend="torch",
            device="cpu",
        ).eval()
        image = torch.linspace(
            0.0, 1.0, steps=3 * cls.resolution**2, dtype=torch.float32
        ).reshape(1, 3, cls.resolution, cls.resolution)
        cls.samples = retina(
            image, torch.tensor([[0.5, 0.5]]), cls.resolution
        )

    def test_grid_runs_through_polar_alexnet(self):
        model = alexnet2023_baseline(
            polar=True, img_size=self.resolution
        ).eval()

        with torch.no_grad():
            output = model(self.samples, apply_mlp=False)

        self.assertEqual(tuple(output.shape), (1, 256, 6, 6))

    def test_grid_runs_through_dinov3_patch_embedding_and_rope(self):
        config = DINOv3ViTConfig(
            image_size=self.resolution,
            patch_size=8,
            num_channels=3,
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=64,
            num_register_tokens=0,
        )
        model = DINOv3ViTModel(config).eval()

        with torch.no_grad():
            output = model(self.samples)

        self.assertEqual(tuple(output.last_hidden_state.shape), (1, 65, 32))
        self.assertEqual(tuple(output.pooler_output.shape), (1, 32))


if __name__ == "__main__":
    unittest.main()
