import copy
import importlib.util
import os
import unittest
from unittest import mock

import torch
import torch.nn as nn
import torch.nn.functional as F

from fovi.arch.knn import KNNConvLayer
from fovi.arch.knn_autograd import KNNConvFunction, ensure_training_metadata


class _Coords:
    def __init__(self, count):
        self.shape = (count,)


def _make_layer(
    *,
    cin=3,
    cout=7,
    nin=19,
    nout=11,
    k=5,
    reference_points=13,
    bias=True,
    backend="baseline",
    device="cpu",
    dtype=torch.float32,
):
    generator = torch.Generator(device=device).manual_seed(1729)
    layer = KNNConvLayer.__new__(KNNConvLayer)
    nn.Module.__init__(layer)
    layer.in_channels = cin
    layer.out_channels = cout
    layer._k = k
    layer.k = torch.tensor(k)
    layer.in_coords = _Coords(nin)
    layer.kernel_backend = backend
    indices = torch.randint(nin, (k, nout), generator=generator, device=device)
    indices.reshape(-1)[:: max(1, indices.numel() // 3)] = nin
    layer.knn_indices_pad_token = indices
    layer.knn_pad_token_val = nin
    output_valid_mask = ~torch.all(indices == nin, dim=0)
    layer.register_buffer(
        "_knn_output_valid_mask", output_valid_mask, persistent=False
    )
    layer._knn_all_outputs_valid = bool(output_valid_mask.all().item())
    rf_index = torch.randint(
        reference_points, (nout, k), generator=generator, device=device
    )
    layer.local_rf = F.one_hot(rf_index, num_classes=reference_points).to(dtype)
    layer.weight = nn.Parameter(
        torch.randn(cout, cin * reference_points, generator=generator, device=device, dtype=dtype)
        / (cin * reference_points) ** 0.5
    )
    if bias:
        layer.bias = nn.Parameter(
            torch.randn(cout, generator=generator, device=device, dtype=dtype)
        )
    else:
        layer.register_parameter("bias", None)
    return layer


_BACKWARD_ATOL = {torch.float32: 1e-5, torch.float16: 2e-3, torch.bfloat16: 2e-2}


def _run_fwd_bwd(layer, x, grad_output, autocast_dtype=None):
    x = x.detach().clone().requires_grad_(True)
    for parameter in layer.parameters():
        parameter.grad = None
    if autocast_dtype is not None:
        with torch.autocast("cuda", dtype=autocast_dtype):
            y = layer(x)
    else:
        y = layer(x)
    (y * grad_output.to(y.dtype)).sum().backward()
    return (
        y.detach(),
        x.grad,
        layer.weight.grad,
        layer.bias.grad if layer.bias is not None else None,
    )


def assert_backward_parity(
    test,
    *,
    backend,
    batch=4,
    dtype=torch.float32,
    device="cpu",
    autocast_dtype=None,
    **layer_kwargs,
):
    """Relative-to-baseline-error criterion: the optimized backend's deviation from the
    fp32 oracle must not exceed 3x the same-dtype baseline's own deviation (plus a small
    dtype floor). Robust where fixed rtol/atol is brittle for large accumulations."""
    oracle_layer = _make_layer(backend="baseline", device=device, dtype=torch.float32, **layer_kwargs)
    reference_layer = copy.deepcopy(oracle_layer).to(dtype)
    optimized_layer = copy.deepcopy(reference_layer)
    optimized_layer.kernel_backend = backend

    generator = torch.Generator(device="cpu").manual_seed(4321)
    x32 = torch.randn(batch, oracle_layer.in_channels, oracle_layer.in_coords.shape[0], generator=generator).to(device)
    g32 = torch.randn(batch, oracle_layer.out_channels, oracle_layer.local_rf.shape[0], generator=generator).to(device)

    oracle = _run_fwd_bwd(oracle_layer, x32, g32)
    reference = _run_fwd_bwd(reference_layer, x32.to(dtype), g32, autocast_dtype=autocast_dtype)
    actual = _run_fwd_bwd(optimized_layer, x32.to(dtype), g32, autocast_dtype=autocast_dtype)

    test.assertEqual(optimized_layer._last_knn_backend, backend)
    atol = _BACKWARD_ATOL[autocast_dtype if autocast_dtype is not None else dtype]
    for name, oracle_t, reference_t, actual_t in zip(
        ("output", "grad_input", "grad_weight", "grad_bias"), oracle, reference, actual
    ):
        if oracle_t is None:
            continue
        reference_error = (reference_t.float() - oracle_t.float()).abs().max().item()
        actual_error = (actual_t.float() - oracle_t.float()).abs().max().item()
        test.assertLessEqual(
            actual_error,
            3.0 * reference_error + atol,
            msg=f"{backend}/{name}: {actual_error:.3e} vs baseline {reference_error:.3e}",
        )
    return reference, actual


class TestCompactTorchBackend(unittest.TestCase):
    def test_forward_parity_with_padding_bias_and_collisions(self):
        baseline = _make_layer(backend="baseline")
        optimized = copy.deepcopy(baseline)
        optimized.kernel_backend = "torch_cached"
        x = torch.randn(4, baseline.in_channels, baseline.in_coords.shape[0])
        with torch.no_grad():
            expected = baseline(x)
            actual = optimized(x)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
        self.assertEqual(optimized._last_knn_backend, "torch_cached")

    def test_noncontiguous_input_and_no_bias(self):
        baseline = _make_layer(bias=False, backend="baseline")
        optimized = copy.deepcopy(baseline)
        optimized.kernel_backend = "torch_cached"
        x = torch.randn(3, baseline.in_coords.shape[0], baseline.in_channels).transpose(1, 2)
        self.assertFalse(x.is_contiguous())
        with torch.no_grad():
            torch.testing.assert_close(optimized(x), baseline(x), rtol=2e-5, atol=2e-5)

    def test_training_falls_back_and_all_gradients_match(self):
        baseline = _make_layer(backend="baseline")
        optimized = copy.deepcopy(baseline)
        optimized.kernel_backend = "auto"
        x0 = torch.randn(2, baseline.in_channels, baseline.in_coords.shape[0], requires_grad=True)
        x1 = x0.detach().clone().requires_grad_(True)
        grad_output = torch.randn(2, baseline.out_channels, baseline.local_rf.shape[0])
        (baseline(x0) * grad_output).sum().backward()
        (optimized(x1) * grad_output).sum().backward()
        self.assertEqual(optimized._last_knn_backend, "baseline")
        torch.testing.assert_close(x1.grad, x0.grad)
        torch.testing.assert_close(optimized.weight.grad, baseline.weight.grad)
        torch.testing.assert_close(optimized.bias.grad, baseline.bias.grad)

    def test_cache_invalidates_after_weight_update(self):
        baseline = _make_layer(backend="baseline")
        optimized = copy.deepcopy(baseline)
        optimized.kernel_backend = "torch_cached"
        x = torch.randn(2, baseline.in_channels, baseline.in_coords.shape[0])
        with torch.no_grad():
            optimized(x)
            first_signature = optimized._compact_effective_weight_cache[0]
            optimized.weight.add_(0.25)
            baseline.weight.copy_(optimized.weight)
            actual = optimized(x)
            expected = baseline(x)
        self.assertNotEqual(first_signature, optimized._compact_effective_weight_cache[0])
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)

    def test_auto_keeps_high_channel_layer_on_baseline(self):
        layer = _make_layer(cin=32, cout=32, backend="auto")
        x = torch.randn(2, 32, layer.in_coords.shape[0])
        with torch.no_grad():
            layer(x)
        self.assertEqual(layer._last_knn_backend, "baseline")

    def test_auto_keeps_sparse_patch_like_layer_on_baseline(self):
        layer = _make_layer(cin=3, nout=64, backend="auto")
        x = torch.randn(2, 3, layer.in_coords.shape[0])
        with torch.no_grad():
            layer(x)
        self.assertEqual(layer._last_knn_backend, "baseline")

    def test_derived_cache_does_not_change_state_dict(self):
        layer = _make_layer(backend="torch_cached")
        keys_before = tuple(layer.state_dict())
        with torch.no_grad():
            layer(torch.randn(2, layer.in_channels, layer.in_coords.shape[0]))
        self.assertEqual(tuple(layer.state_dict()), keys_before)
        layer.clear_optimized_cache()
        self.assertIsNone(layer._compact_effective_weight_cache)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestCudaBackends(unittest.TestCase):
    def test_torch_cached_float16_and_bfloat16(self):
        for dtype, rtol, atol in (
            (torch.float16, 2e-3, 2e-3),
            (torch.bfloat16, 2e-2, 3e-2),
        ):
            with self.subTest(dtype=dtype):
                baseline = _make_layer(backend="baseline", device="cuda", dtype=dtype)
                optimized = copy.deepcopy(baseline)
                optimized.kernel_backend = "torch_cached"
                x = torch.randn(
                    3,
                    baseline.in_channels,
                    baseline.in_coords.shape[0],
                    device="cuda",
                    dtype=dtype,
                )
                with torch.no_grad():
                    expected = baseline(x)
                    actual = optimized(x)
                torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)

    @unittest.skipUnless(importlib.util.find_spec("warp") is not None, "warp-lang is required")
    def test_cached_and_uncached_warp_parity(self):
        for cout, bias in ((32, True), (7, False)):
            with self.subTest(cout=cout, bias=bias):
                baseline = _make_layer(
                    nout=257,
                    nin=67,
                    k=7,
                    reference_points=31,
                    cout=cout,
                    bias=bias,
                    backend="baseline",
                    device="cuda",
                    dtype=torch.float16,
                )
                x = torch.randn(3, 3, 67, device="cuda", dtype=torch.float16)
                with torch.no_grad():
                    expected = baseline(x)
                    for backend in ("warp_cached", "warp_memory"):
                        optimized = copy.deepcopy(baseline)
                        optimized.kernel_backend = backend
                        actual = optimized(x)
                        torch.testing.assert_close(actual, expected, rtol=3e-3, atol=3e-3)
                        self.assertEqual(optimized._last_knn_backend, backend)


class TestTrainingBackendsCPU(unittest.TestCase):
    def test_backward_parity_fp32(self):
        for backend in ("torch_scatter", "torch_compact"):
            for bias in (True, False):
                with self.subTest(backend=backend, bias=bias):
                    assert_backward_parity(self, backend=backend, bias=bias)

    def test_gradcheck_compact_function_fp64(self):
        layer = _make_layer(
            backend="baseline",
            dtype=torch.float64,
            cin=2,
            cout=3,
            nin=9,
            nout=5,
            k=3,
            reference_points=6,
        )
        meta = ensure_training_metadata(layer, torch.device("cpu"))
        x = torch.randn(2, 2, 9, dtype=torch.float64, requires_grad=True)
        weight = layer.weight.detach().clone().requires_grad_(True)
        bias = layer.bias.detach().clone().requires_grad_(True)
        self.assertTrue(
            torch.autograd.gradcheck(
                lambda xx, ww, bb: KNNConvFunction.apply(xx, ww, bb, meta, "torch_compact"),
                (x, weight, bias),
            )
        )

    def test_noncontiguous_input_training(self):
        layer = _make_layer(backend="torch_compact")
        baseline = copy.deepcopy(layer)
        baseline.kernel_backend = "baseline"
        x = torch.randn(3, layer.in_coords.shape[0], layer.in_channels).transpose(1, 2)
        self.assertFalse(x.is_contiguous())
        grad_output = torch.randn(3, layer.out_channels, layer.local_rf.shape[0])
        expected = _run_fwd_bwd(baseline, x.contiguous(), grad_output)
        actual = _run_fwd_bwd(layer, x, grad_output)
        for expected_t, actual_t in zip(expected, actual):
            torch.testing.assert_close(actual_t, expected_t, rtol=1e-4, atol=1e-5)

    def test_partial_requires_grad_paths(self):
        layer = _make_layer(backend="torch_compact")
        x = torch.randn(2, layer.in_channels, layer.in_coords.shape[0])
        layer(x).sum().backward()
        self.assertIsNotNone(layer.weight.grad)

        layer = _make_layer(backend="torch_compact")
        layer.weight.requires_grad_(False)
        layer.bias.requires_grad_(False)
        x = torch.randn(2, layer.in_channels, layer.in_coords.shape[0], requires_grad=True)
        layer(x).sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNone(layer.weight.grad)

    def test_training_metadata_state_dict_invariance_and_clear(self):
        layer = _make_layer(backend="torch_compact")
        keys_before = tuple(layer.state_dict())
        x = torch.randn(2, layer.in_channels, layer.in_coords.shape[0], requires_grad=True)
        layer(x).sum().backward()
        self.assertEqual(tuple(layer.state_dict()), keys_before)
        self.assertIsNotNone(layer._knn_training_metadata)
        layer.clear_optimized_cache()
        self.assertIsNone(layer._knn_training_metadata)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestCudaTrainingBackends(unittest.TestCase):
    def test_backward_parity_dtypes(self):
        for backend in ("torch_scatter", "torch_compact"):
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                with self.subTest(backend=backend, dtype=dtype):
                    assert_backward_parity(self, backend=backend, dtype=dtype, device="cuda")

    def test_backward_parity_autocast_grad_dtypes(self):
        for backend in ("torch_scatter", "torch_compact"):
            for autocast_dtype in (torch.float16, torch.bfloat16):
                with self.subTest(backend=backend, autocast_dtype=autocast_dtype):
                    _, actual = assert_backward_parity(
                        self,
                        backend=backend,
                        device="cuda",
                        autocast_dtype=autocast_dtype,
                    )
                    _, grad_x, grad_w, grad_b = actual
                    # fp32 master params keep fp32 grads under AMP; input grad follows x.
                    self.assertEqual(grad_x.dtype, torch.float32)
                    self.assertEqual(grad_w.dtype, torch.float32)
                    self.assertEqual(grad_b.dtype, torch.float32)

    def test_auto_routing_under_grad(self):
        from fovi.arch import knn_optimization as ko

        def run_auto(layer_kwargs, batch, dtype=torch.float32):
            layer = _make_layer(backend="auto", device="cuda", dtype=dtype, **layer_kwargs)
            x = torch.randn(
                batch,
                layer.in_channels,
                layer.in_coords.shape[0],
                device="cuda",
                dtype=dtype,
                requires_grad=True,
            )
            layer(x).sum().backward()
            return layer._last_knn_backend

        # Tiny fixture shapes sit far below the real work threshold -> baseline.
        self.assertEqual(run_auto(dict(cin=32), 2), "baseline")
        self.assertEqual(run_auto(dict(), 128), "baseline")
        # K=1/V=1 layers route to gather_gemm at any batch/dtype.
        self.assertEqual(run_auto(dict(k=1, reference_points=1), 4), "gather_gemm")
        # Above the threshold: fp32 -> torch_compact; fp16 -> cuda when cupy exists.
        original = ko.WORK_VOLUME_THRESHOLD
        ko.WORK_VOLUME_THRESHOLD = 1
        try:
            self.assertEqual(run_auto(dict(cin=32), 128), "torch_compact")
            expected = (
                "cuda"
                if importlib.util.find_spec("cupy") is not None
                and ko._native_cuda_supported(torch.device("cuda"))
                else "torch_compact"
            )
            self.assertEqual(
                run_auto(dict(cin=3, nout=257, nin=67, k=7, reference_points=31), 4, torch.float16),
                expected,
            )
        finally:
            ko.WORK_VOLUME_THRESHOLD = original

    @unittest.skipUnless(importlib.util.find_spec("cupy") is not None, "cupy is required")
    def test_auto_inference_dense_low_cin_prefers_cuda_at_any_batch(self):
        from fovi.arch import knn_optimization as ko

        if not ko._native_cuda_supported(torch.device("cuda")):
            self.skipTest("native CUDA convolution requires Ampere or newer")
        # The cuda any-batch exception requires a large-K stem (alex0-like, K>=100);
        # smaller-K stems take the cached-GEMM cell (later measurement reverted
        # the earlier baseline gate: warp_cached on small-K stems is an in-model
        # win at small batch).
        for k, nin, expected in ((101, 150, "cuda"), (49, 150, "warp_cached")):
            with self.subTest(k=k):
                layer = _make_layer(
                    backend="auto",
                    device="cuda",
                    dtype=torch.float16,
                    cin=3,
                    nout=257,
                    nin=nin,
                    k=k,
                    reference_points=31,
                )
                x = torch.randn(2, 3, nin, device="cuda", dtype=torch.float16)
                with torch.no_grad():
                    layer(x)
                if expected == "warp_cached" and layer._last_knn_backend == "torch_cached":
                    expected = "torch_cached"  # warp-lang absent fallback
                self.assertEqual(layer._last_knn_backend, expected)

    def test_native_cuda_capability_gate(self):
        from fovi.arch import knn_optimization as ko

        layer = _make_layer(
            backend="auto",
            device="cuda",
            dtype=torch.float16,
            cin=3,
            nout=257,
            nin=67,
            k=7,
            reference_points=31,
        )
        x = torch.randn(4, 3, 67, device="cuda", dtype=torch.float16)
        original = ko.WORK_VOLUME_THRESHOLD
        ko.WORK_VOLUME_THRESHOLD = 1
        try:
            with mock.patch.object(torch.cuda, "get_device_capability", return_value=(7, 5)):
                self.assertEqual(ko.select_backend(layer, x), "torch_compact")
                layer.kernel_backend = "cuda"
                with self.assertRaisesRegex(RuntimeError, "Ampere-or-newer"):
                    ko.select_backend(layer, x)
        finally:
            ko.WORK_VOLUME_THRESHOLD = original

    def test_chunked_compact_parity(self):
        os.environ["FOVI_KNN_STAGE_MIB"] = "0"
        try:
            assert_backward_parity(self, backend="torch_compact", device="cuda")
        finally:
            del os.environ["FOVI_KNN_STAGE_MIB"]


if __name__ == "__main__":
    unittest.main()
