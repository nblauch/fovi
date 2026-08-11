"""Correctness tests for native-scale uint8 foveated sampling."""

import gc
import os
import unittest
import weakref
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

    def test_auto_fallback_cache_does_not_retain_inputs(self):
        if not torch.cuda.is_available():
            self.skipTest("requires CUDA to enter native auto-selection")

        def exercise(image_dtype):
            image, fix_loc, fix_size = _inputs("cuda:0")
            method = "_native_uint8_sample"
            if image_dtype != torch.uint8:
                image = image.to(image_dtype).div_(255.0)
                method = "_native_float_sample"
            image_ref = weakref.ref(image)
            sampler = _sampler(device="cuda:0", backend="auto")

            def fail(*_args):
                raise ImportError("missing")

            with mock.patch.object(sampler, method, new=fail):
                with mock.patch("fovi.sensing.samplers.warnings.warn"):
                    output = sampler(image, fix_loc, fix_size)
            del image, fix_loc, fix_size, output
            return sampler, image_ref

        for image_dtype in (torch.uint8, torch.float32):
            with self.subTest(dtype=image_dtype):
                sampler, image_ref = exercise(image_dtype)
                gc.collect()
                self.assertIsNone(image_ref())
                self.assertTrue(all(
                    isinstance(message, str)
                    for message in sampler._native_errors.values()))


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

    def test_direct_true_forces_torch_oracle(self):
        image, fix_loc, fix_size = _inputs(self.device)
        for mode in ("nearest", "bilinear"):
            with self.subTest(mode=mode):
                reference_sampler = _sampler(
                    mode, self.device, backend="torch")
                sampler = _sampler(mode, self.device, backend="cuda")
                sampler.coords = reference_sampler.coords
                sampler.sampling_grid = reference_sampler.sampling_grid
                reference = reference_sampler(
                    image, fix_loc, fix_size, direct=True)
                with mock.patch.object(
                        sampler, "_native_uint8_sample",
                        side_effect=AssertionError("fused path must not run")):
                    output = sampler(
                        image, fix_loc, fix_size, direct=True)
                self.assertEqual(sampler._last_backend, "torch_direct")
                torch.testing.assert_close(output, reference, rtol=0, atol=0)

    def test_16k_nearest_coordinate_parity(self):
        height, width = 8640, 15360
        reference_sampler = GridSampler(
            16.0, 0.5, 256, device=self.device, mode="nearest",
            backend="torch")
        sampler = GridSampler(
            16.0, 0.5, 256, device=self.device, mode="nearest",
            backend="cuda", coords=reference_sampler.coords)
        fix_loc = torch.tensor([[0.47, 0.53]], device=self.device)
        fix_size = torch.tensor([[height, height]], device=self.device)
        x_pattern = (
            torch.arange(width, device=self.device).remainder_(251)
            .to(torch.uint8).view(1, 1, 1, width)
            .expand(1, 1, height, width))
        y_pattern = (
            torch.arange(height, device=self.device).remainder_(251)
            .to(torch.uint8).view(1, 1, height, 1)
            .expand(1, 1, height, width))

        for axis, image in (("x", x_pattern), ("y", y_pattern)):
            with self.subTest(axis=axis):
                reference = reference_sampler(
                    image, fix_loc, fix_size, direct=True)
                output = sampler(image, fix_loc, fix_size)
                self.assertTrue(torch.equal(output, reference))

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


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestNativeFloatingSampler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import cupy  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("CuPy is unavailable")
        cls.device = "cuda:0"

    def _compare(self, image_dtype, mode, noncontiguous=False):
        image, fix_loc, fix_size = _inputs(self.device)
        image = image.to(image_dtype).div_(255.0)
        if noncontiguous:
            image = image.repeat_interleave(2, dim=-1)[..., ::2]
            self.assertFalse(image.is_contiguous())
        reference_sampler = _sampler(mode, self.device, "torch")
        native_sampler = _sampler(mode, self.device, "cuda")
        native_sampler.coords = reference_sampler.coords
        native_sampler.sampling_grid = reference_sampler.sampling_grid
        reference = reference_sampler(
            image, fix_loc, fix_size, direct=True)
        native = native_sampler(image, fix_loc, fix_size)
        self.assertEqual(native.dtype, image_dtype)
        self.assertEqual(native_sampler._last_backend, "cuda")
        if mode == "nearest":
            self.assertTrue(torch.equal(native, reference))
        elif image_dtype == torch.float16:
            torch.testing.assert_close(native, reference, rtol=0, atol=5e-4)
        elif image_dtype == torch.float32:
            torch.testing.assert_close(native, reference, rtol=1e-6, atol=1e-6)
        else:
            torch.testing.assert_close(native, reference, rtol=1e-12, atol=1e-12)

    def test_nearest_and_bilinear_all_supported_dtypes(self):
        for image_dtype in (torch.float16, torch.float32, torch.float64):
            with self.subTest(dtype=image_dtype, mode="nearest"):
                self._compare(image_dtype, "nearest")
            with self.subTest(dtype=image_dtype, mode="bilinear"):
                self._compare(image_dtype, "bilinear")

    def test_noncontiguous_inputs(self):
        for image_dtype in (torch.float16, torch.float32, torch.float64):
            with self.subTest(dtype=image_dtype):
                self._compare(image_dtype, "nearest", noncontiguous=True)

    def test_auto_selects_native_and_torch_forces_grid_sample(self):
        image, fix_loc, fix_size = _inputs(self.device)
        image = image.float().div_(255.0)
        automatic = _sampler(device=self.device, backend="auto")
        forced_torch = _sampler(device=self.device, backend="torch")
        automatic(image, fix_loc, fix_size)
        forced_torch(image, fix_loc, fix_size)
        self.assertEqual(automatic._last_backend, "cuda")
        self.assertEqual(forced_torch._last_backend, "torch_grid_sample")

    def test_environment_override_forces_float_grid_sample(self):
        image, fix_loc, fix_size = _inputs(self.device)
        image = image.float().div_(255.0)
        sampler = _sampler(device=self.device, backend="auto")
        with mock.patch.dict(
                os.environ, {"FOVI_GRID_SAMPLER_BACKEND": "torch"}):
            sampler(image, fix_loc, fix_size)
        self.assertEqual(sampler._last_backend, "torch_grid_sample")

    def test_auto_falls_back_for_gradients(self):
        image, fix_loc, fix_size = _inputs(self.device)
        image = image.float().div_(255.0).requires_grad_()
        sampler = _sampler(device=self.device, backend="auto")
        output = sampler(image, fix_loc, fix_size)
        self.assertEqual(sampler._last_backend, "torch_grid_sample")
        output.sum().backward()
        self.assertIsNotNone(image.grad)

    def test_cuda_rejects_gradients(self):
        image, fix_loc, fix_size = _inputs(self.device)
        image = image.float().div_(255.0).requires_grad_()
        sampler = _sampler(device=self.device, backend="cuda")
        with self.assertRaisesRegex(RuntimeError, "no required gradients"):
            sampler(image, fix_loc, fix_size)

    def test_auto_falls_back_when_native_dependency_is_unavailable(self):
        image, fix_loc, fix_size = _inputs(self.device)
        image = image.float().div_(255.0)
        sampler = _sampler(device=self.device, backend="auto")
        with mock.patch("fovi.sensing.samplers.warnings.warn") as warn:
            with mock.patch.object(
                    sampler, "_native_float_sample",
                    side_effect=ImportError("missing")):
                output = sampler(image, fix_loc, fix_size)
        self.assertIn("using grid_sample", warn.call_args.args[0])
        self.assertEqual(output.dtype, torch.float32)
        self.assertEqual(sampler._last_backend, "torch_grid_sample")

    def test_bfloat16_auto_uses_float32_grid_sample_fallback(self):
        image, fix_loc, fix_size = _inputs(self.device)
        image = image.to(torch.bfloat16).div_(255.0)
        sampler = _sampler(device=self.device, backend="auto")
        output = sampler(image, fix_loc, fix_size)
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertEqual(sampler._last_backend, "torch_grid_sample")

    def test_float64_return_coords_preserves_compute_dtype(self):
        image, fix_loc, fix_size = _inputs(self.device)
        image = image.double().div_(255.0)
        sampler = _sampler(device=self.device, backend="cuda")
        _, grid = sampler(image, fix_loc, fix_size, return_coords=True)
        self.assertEqual(grid.dtype, torch.float64)


if __name__ == "__main__":
    unittest.main()
