import copy
import unittest
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "config")


def compose_config(name):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(config_name=name)


class TestDINOv3Configs(unittest.TestCase):
    def test_configs_compose(self):
        for name in (
            "fovi-dinov3-splus",
            "fovi-dinov3-hplus",
            "dinov3_logpolar_control",
            "dinov3_warped_cartesian_control",
            "dinov3_uniform_control",
            "fovi-dinov3_weak_control",
        ):
            with self.subTest(config=name):
                cfg = compose_config(name)
                self.assertEqual(cfg.model.arch, "fovi_dinov3")

    def test_hplus_only_overrides_model_and_batch_size(self):
        expected = copy.deepcopy(compose_config("fovi-dinov3-splus"))
        expected.training.batch_size = 128
        expected.pretrained_model.variant = "vith16plus"
        expected.pretrained_model.path = (
            "facebook/dinov3-vith16plus-pretrain-lvd1689m"
        )

        actual = compose_config("fovi-dinov3-hplus")
        self.assertEqual(
            OmegaConf.to_container(actual, resolve=False),
            OmegaConf.to_container(expected, resolve=False),
        )

    def test_controls_only_override_sensing(self):
        cases = {
            "dinov3_logpolar_control": {
                "saccades.mode": "logpolar_as_grid",
            },
            "dinov3_uniform_control": {
                "saccades.cmf_a": None,
                "saccades.mode": "uniform_as_grid",
            },
            "fovi-dinov3_weak_control": {
                "saccades.cmf_a": 60.936638,
            },
        }
        base = compose_config("fovi-dinov3-splus")
        for name, overrides in cases.items():
            with self.subTest(config=name):
                expected = copy.deepcopy(base)
                for key, value in overrides.items():
                    OmegaConf.update(expected, key, value)
                self.assertEqual(
                    OmegaConf.to_container(
                        compose_config(name), resolve=False
                    ),
                    OmegaConf.to_container(expected, resolve=False),
                )


if __name__ == "__main__":
    unittest.main()
