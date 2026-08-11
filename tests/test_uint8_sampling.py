"""Correctness tests for native-scale uint8 foveated sampling."""

import os
import unittest
from unittest import mock

import torch

from fovi.sensing.retina import RetinalTransform
from fovi.sensing.samplers import GridSampler
from fovi.utils.fastaugs import transforms as fastT


def _sampler(mode="nearest", device="cpu", backend="torch", output_dtype=None):
    return GridSampler(
        16.0, 0.5, 8, device=device, mode=mode, backend=backend,
        output_dtype=output_dtype)


def _retina(mode="grid_nn", device="cpu", backend="torch", pre=None, post=None,
            **kwargs):
    return RetinalTransform(
        resolution=8, start_res=32, fov=16.0, cmf_a=0.5,
        style="isotropic", sampler=mode, fixation_size=32, device=device,
        dtype=torch.float32, auto_match_cart_resources=0,
        sampler_backend=backend, pre_transforms=pre, post_transforms=post,
        **kwargs)


def _inputs(device="cpu", batch=2):
    generator = torch.Generator(device=device).manual_seed(123)
    image = torch.randint(
        0, 256, (batch, 3, 32, 40), dtype=torch.uint8,
        device=device, generator=generator)
    fix_loc = torch.tensor([[0.5, 0.5], [0.05, 0.95]], device=device)[:batch]
    fix_size = torch.tensor([[32, 32], [36, 30]], device=device)[:batch]
    return image, fix_loc, fix_size


class TestUint8SamplerContract(unittest.TestCase):
    def test_nearest_preserves_uint8_and_native_scale(self):
        image, fix_loc, fix_size = _inputs()
        sampler = _sampler()
        output = sampler(image, fix_loc, fix_size)
        self.assertEqual(output.dtype, torch.uint8)
        self.assertGreater(int(output.max()), 1)
        self.assertEqual(sampler._last_backend, "torch_gather")

    def test_nearest_output_dtype_only_casts(self):
        image, fix_loc, fix_size = _inputs()
        native = _sampler()(image, fix_loc, fix_size)
        floating = _sampler(output_dtype=torch.float32)(image, fix_loc, fix_size)
        self.assertEqual(floating.dtype, torch.float32)
        torch.testing.assert_close(floating, native.float(), rtol=0, atol=0)
        self.assertGreater(float(floating.max()), 1.0)

    def test_bilinear_promotes_but_preserves_native_scale(self):
        image, fix_loc, fix_size = _inputs()
        output = _sampler("bilinear")(image, fix_loc, fix_size)
        self.assertEqual(output.dtype, torch.float32)
        self.assertGreater(float(output.max()), 1.0)

    def test_bilinear_rejects_integer_output(self):
        with self.assertRaisesRegex(ValueError, "floating output"):
            _sampler("bilinear", output_dtype=torch.uint8)

    def test_return_coords_matches_direct_reference(self):
        image, fix_loc, fix_size = _inputs()
        sampler = _sampler()
        output, grid = sampler(image, fix_loc, fix_size, return_coords=True)
        self.assertEqual(tuple(grid.shape), (2, 1, output.shape[-1], 2))
        expected = sampler._direct_grid(image.shape[-2:], fix_loc, fix_size)
        torch.testing.assert_close(grid, expected, rtol=0, atol=0)

    def test_auto_falls_back_when_native_dependency_is_unavailable(self):
        if not torch.cuda.is_available():
            self.skipTest("requires CUDA to enter native auto-selection")
        device = "cuda:0"
        image, fix_loc, fix_size = _inputs(device)
        sampler = _sampler(device=device, backend="auto")
        with mock.patch("fovi.sensing.samplers.warnings.warn") as warn:
            with mock.patch.object(
                    sampler, "_native_uint8_sample", side_effect=ImportError("missing")):
                output = sampler(image, fix_loc, fix_size)
        self.assertIn("using Torch gather", warn.call_args.args[0])
        self.assertEqual(output.dtype, torch.uint8)
        self.assertEqual(sampler._last_backend, "torch_gather")

    def test_environment_override(self):
        image, fix_loc, fix_size = _inputs()
        sampler = _sampler(backend="auto")
        with mock.patch.dict(os.environ, {"FOVI_GRID_SAMPLER_BACKEND": "torch"}):
            sampler(image, fix_loc, fix_size)
        self.assertEqual(sampler._last_backend, "torch_gather")


class TestUint8RetinalTransform(unittest.TestCase):
    def test_no_transforms_returns_unit_float(self):
        image, fix_loc, fix_size = _inputs()
        retina = _retina()
        retina.eval()
        output = retina(image, fix_loc, fix_size)
        reference = retina.sampler(
            image.float().div(255.0), fix_loc, fix_size, direct=True)
        self.assertEqual(output.dtype, torch.float32)
        self.assertGreaterEqual(float(output.min()), 0.0)
        self.assertLessEqual(float(output.max()), 1.0)
        torch.testing.assert_close(output, reference, rtol=0, atol=0)

    def test_bilinear_unit_conversion(self):
        image, fix_loc, fix_size = _inputs()
        retina = _retina("grid_bilinear")
        retina.eval()
        output = retina(image, fix_loc, fix_size)
        reference = retina.sampler(
            image.float().div(255.0), fix_loc, fix_size, direct=True)
        torch.testing.assert_close(output, reference, rtol=1e-6, atol=1e-6)

    def test_post_normalization_receives_unit_values(self):
        image, fix_loc, fix_size = _inputs()
        normalize = fastT.NormalizeGPU(
            [0.5, 0.5, 0.5], [0.25, 0.25, 0.25], device="cpu", inplace=False)
        retina = _retina(post=fastT.Compose([normalize]))
        retina.train()
        output = retina(image, fix_loc, fix_size)
        sampled_unit = retina.sampler(
            image.float().div(255.0), fix_loc, fix_size, direct=True)
        reference = normalize(sampled_unit.unsqueeze(3)).squeeze(3)
        torch.testing.assert_close(output, reference, rtol=0, atol=0)
        self.assertLess(float(output.abs().max()), 4.1)

    def test_pre_normalization_fast_path_matches_reference(self):
        image, fix_loc, fix_size = _inputs()
        normalize = fastT.NormalizeGPU(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225],
            device="cpu", inplace=True)
        retina = _retina(pre=fastT.Compose([normalize]))
        retina.train()
        retina.fast_pre_transforms = False
        reference = retina(image, fix_loc, fix_size)
        retina.fast_pre_transforms = True
        optimized = retina(image, fix_loc, fix_size)
        torch.testing.assert_close(optimized, reference, rtol=0, atol=0)

    def test_float_input_keeps_existing_grid_sample_path(self):
        image, fix_loc, fix_size = _inputs()
        image = image.float().div(255.0)
        retina = _retina()
        retina.eval()
        output = retina(image, fix_loc, fix_size)
        grid = retina.sampler._transform_fix_grid(image.shape[-2:], fix_loc, fix_size)
        reference = torch.nn.functional.grid_sample(
            image, grid, mode="nearest", align_corners=False).squeeze(2)
        torch.testing.assert_close(output, reference, rtol=0, atol=0)
        self.assertEqual(retina.sampler._last_backend, "torch_grid_sample")


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestNativeUint8Sampler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import cupy  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("CuPy is unavailable")
        cls.device = "cuda:0"

    def _compare(self, image, mode):
        _, fix_loc, fix_size = _inputs(self.device, batch=image.shape[0])
        reference_sampler = _sampler(mode, self.device, "torch")
        native_sampler = _sampler(mode, self.device, "cuda")
        native_sampler.coords = reference_sampler.coords
        native_sampler.sampling_grid = reference_sampler.sampling_grid
        reference = reference_sampler(image, fix_loc, fix_size)
        native = native_sampler(image, fix_loc, fix_size)
        self.assertEqual(native_sampler._last_backend, "cuda")
        if mode == "nearest":
            self.assertTrue(torch.equal(native, reference))
        else:
            torch.testing.assert_close(native, reference, rtol=1e-5, atol=2e-3)

    def test_contiguous_nearest_and_bilinear(self):
        image, _, _ = _inputs(self.device)
        self._compare(image, "nearest")
        self._compare(image, "bilinear")

    def test_nhwc_camera_view_and_strided_slice(self):
        generator = torch.Generator(device=self.device).manual_seed(22)
        nhwc = torch.randint(
            0, 256, (2, 32, 40, 3), device=self.device,
            dtype=torch.uint8, generator=generator)
        camera_view = nhwc.permute(0, 3, 1, 2)
        larger = torch.randint(
            0, 256, (2, 3, 64, 80), device=self.device,
            dtype=torch.uint8, generator=generator)
        for image in (camera_view, larger[:, :, ::2, ::2]):
            self.assertFalse(image.is_contiguous())
            self._compare(image, "nearest")
            self._compare(image, "bilinear")

    def test_native_pre_normalization_matches_reference_path(self):
        image, fix_loc, fix_size = _inputs(self.device)
        pre = fastT.Compose([fastT.NormalizeGPU(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225],
            device=self.device, inplace=True)])
        retina = _retina(device=self.device, backend="cuda", pre=pre)
        retina.train()
        retina.fast_pre_transforms = False
        reference = retina(image, fix_loc, fix_size)
        retina.fast_pre_transforms = True
        optimized = retina(image, fix_loc, fix_size)
        self.assertEqual(retina.sampler._last_backend, "cuda")
        torch.testing.assert_close(optimized, reference, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
