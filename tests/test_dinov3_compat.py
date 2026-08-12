import unittest
from types import SimpleNamespace

from torch import nn

from fovi.arch.dinov3 import _get_dinov3_layers


class TestDINOv3LayerCompatibility(unittest.TestCase):
    def test_supported_transformers_layouts(self):
        direct_layers = nn.ModuleList([nn.Identity()])
        nested_layers = nn.ModuleList([nn.Identity(), nn.Identity()])
        encoder_layers = nn.ModuleList([nn.Identity(), nn.Identity(), nn.Identity()])

        cases = [
            (SimpleNamespace(layer=direct_layers), direct_layers),
            (
                SimpleNamespace(model=SimpleNamespace(layer=nested_layers)),
                nested_layers,
            ),
            (
                SimpleNamespace(encoder=SimpleNamespace(layer=encoder_layers)),
                encoder_layers,
            ),
        ]
        for model, expected in cases:
            with self.subTest(layout=vars(model).keys()):
                self.assertIs(_get_dinov3_layers(model), expected)

    def test_missing_layers_raise_clear_error(self):
        with self.assertRaisesRegex(AttributeError, "Cannot locate transformer layers"):
            _get_dinov3_layers(SimpleNamespace())


if __name__ == "__main__":
    unittest.main()
