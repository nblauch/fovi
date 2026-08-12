import copy
import unittest
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from fovi.arch.architectures import ARCHITECTURE_REGISTRY
from fovi.arch.knn import KNNBaseLayer


CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "config")


def compose_config(name):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(config_name=name)


class TestWarpedCartesianControls(unittest.TestCase):
    def test_warped_configs_only_change_sensing_mode(self):
        pairs = [
            ("resnet18_logpolar_control", "resnet18_warped_cartesian_control"),
            ("resnet50_logpolar_control", "resnet50_warped_cartesian_control"),
            ("dinov3_logpolar_control", "dinov3_warped_cartesian_control"),
        ]
        for logpolar_name, warped_name in pairs:
            with self.subTest(config=warped_name):
                expected = copy.deepcopy(compose_config(logpolar_name))
                expected.saccades.mode = "warped_cartesian_as_grid"
                actual = compose_config(warped_name)
                self.assertEqual(actual.saccades.fov_type, "circular")
                self.assertEqual(
                    OmegaConf.to_container(actual, resolve=False),
                    OmegaConf.to_container(expected, resolve=False),
                )

    def test_matched_resnets_use_dense_backbones(self):
        expected_widths = {
            "resnet18_warped_cartesian_control": 512,
            "resnet50_warped_cartesian_control": 2048,
        }
        for config_name, output_width in expected_widths.items():
            with self.subTest(config=config_name):
                cfg = compose_config(config_name)
                network = ARCHITECTURE_REGISTRY.get(cfg.model.arch)(
                    cfg, device="cpu"
                ).eval()
                self.assertFalse(
                    any(isinstance(module, KNNBaseLayer) for module in network.modules())
                )
                inputs = torch.randn(2, 3, cfg.saccades.resize_size, cfg.saccades.resize_size)
                with torch.no_grad():
                    outputs = network(inputs, apply_mlp=False)
                self.assertEqual(tuple(outputs.shape), (2, output_width))
                self.assertEqual(network.backbone.total_embed_dim, output_width)


if __name__ == "__main__":
    unittest.main()
