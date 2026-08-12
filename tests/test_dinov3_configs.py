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
    def assert_config_contains(self, actual, expected):
        if OmegaConf.is_dict(expected):
            for key in expected:
                self.assertIn(key, actual)
                self.assert_config_contains(actual[key], expected[key])
        elif OmegaConf.is_list(expected):
            self.assertEqual(
                OmegaConf.to_container(actual, resolve=False),
                OmegaConf.to_container(expected, resolve=False),
            )
        else:
            self.assertEqual(actual, expected)

    def test_splus_matches_published_training_recipe(self):
        cfg = compose_config("fovi-dinov3-splus")
        data = OmegaConf.to_container(cfg.data, resolve=False)

        self.assertTrue(data["train_dataset"].endswith("train_256_raw.ffcv"))
        self.assertTrue(data["val_dataset"].endswith("val_256_raw.ffcv"))
        self.assertEqual(cfg.data.num_workers, 6)
        self.assertEqual(cfg.data.in_memory, 1)
        self.assertIsNone(cfg.data.subset)

        self.assertEqual(cfg.pretrained_model.variant, "vits16plus")
        self.assertEqual(
            cfg.pretrained_model.path,
            "facebook/dinov3-vits16plus-pretrain-lvd1689m",
        )
        self.assertEqual(cfg.model.vit.patch_size, 8)
        self.assertIsNone(cfg.model.vit.patch_overlap_factor)
        self.assertEqual(cfg.training.batch_size, 256)
        self.assertEqual(cfg.saccades.n_fixations_val, [1, 2, 3, 5, 10, 20])
        self.assertEqual(cfg.saccades.nonrandom_first, 1)

        published = OmegaConf.load(
            Path(CONFIG_DIR)
            / "pretrained"
            / "fovi-dinov3-splus_a-2.78_res-64_in1k.yaml"
        )
        for section in (
            "model",
            "saccades",
            "transforms",
            "validation",
            "training",
            "dist",
            "pretrained_model",
        ):
            self.assert_config_contains(cfg[section], published[section])

    def test_hplus_only_overrides_published_model_differences(self):
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

        published = OmegaConf.load(
            Path(CONFIG_DIR)
            / "pretrained"
            / "fovi-dinov3-hplus_a-2.78_res-64_in1k.yaml"
        )
        for section in (
            "model",
            "saccades",
            "transforms",
            "validation",
            "training",
            "dist",
            "pretrained_model",
        ):
            self.assert_config_contains(actual[section], published[section])

    def test_controls_inherit_splus(self):
        for name in (
            "dinov3_logpolar_control",
            "dinov3_uniform_control",
            "fovi-dinov3_weak_control",
        ):
            with self.subTest(config=name):
                cfg = compose_config(name)
                self.assertEqual(cfg.pretrained_model.variant, "vits16plus")
                self.assertIn("num_workers", cfg.data)
                self.assertIn("in_memory", cfg.data)
                self.assertIn("subset", cfg.data)


if __name__ == "__main__":
    unittest.main()
