"""Tests for the Warp KNN convolution kernels (fovi/arch/knn_warp.py).

Forward: batch-size dispatch, the multi-b-tile weight-reuse kernels,
non-multiple-of-tile batch and output-channel edges, and the canary validation that guards
against Warp's silent kernel-launch failures.

Backward: the warp_train adjoint kernels (dA staging + reverse-CSR grad_input, dWeff
staging + index_add_ grad_weight) against a fp32 autograd oracle and CompactTorchOps,
including chunked staging, fp32 staging, pad-heavy neighborhoods, no-bias, and NaN-prefill
discipline (plain tile_load overreads).
"""

import contextlib
import importlib.util
import os
import unittest

import torch
import torch.nn.functional as F


def _warp_ready():
    return torch.cuda.is_available() and importlib.util.find_spec("warp") is not None


def make_case(cin, cout, nin, nout, k, v, pads, batch, device, seed=20260721):
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(batch, cin, nin, dtype=torch.float16, device=device, generator=generator)
    weight = torch.randn(
        cout, cin * v, dtype=torch.float16, device=device, generator=generator
    ) / (cin * v) ** 0.5
    bias = torch.randn(cout, dtype=torch.float16, device=device, generator=generator)
    indices = torch.randint(nin, (k, nout), dtype=torch.int64, device=device, generator=generator)
    if pads:
        flat = indices.flatten()
        flat[: min(pads, flat.numel())] = nin
        indices = flat[
            torch.randperm(flat.numel(), device=device, generator=generator)
        ].reshape_as(indices)
    rf_index = torch.randint(v, (nout, k), dtype=torch.int64, device=device, generator=generator)
    channels = torch.arange(cin, device=device).reshape(1, cin, 1)
    neighbors = indices.transpose(0, 1).reshape(nout, 1, k)
    input_linear = torch.where(
        neighbors < nin,
        channels * nin + neighbors,
        torch.full_like(neighbors, cin * nin),
    ).reshape(nout, cin * k)
    weight_linear = (channels * v + rf_index.reshape(nout, 1, k)).reshape(nout, cin * k)
    pad_p = (-input_linear.shape[1]) % 64
    input_linear = F.pad(input_linear, (0, pad_p), value=cin * nin).to(torch.int32).contiguous()
    weight_linear = F.pad(weight_linear, (0, pad_p), value=0).to(torch.int32).contiguous()
    return x, weight, bias, indices, rf_index, input_linear, weight_linear


def oracle_fp32(x, weight, bias, indices, rf_index):
    """y[b,o,n] = bias[o] + sum_{c,k} x[b,c,knn[k,n]] * W[o, c*V + rf_index[n,k]] in fp32."""
    cin = x.shape[1]
    cout = weight.shape[0]
    v = weight.shape[1] // cin
    x_padded = F.pad(x.to(torch.float32), (0, 1))
    batch = x.shape[0]
    feats = torch.gather(
        x_padded, 2, indices.reshape(1, 1, -1).expand(batch, cin, -1)
    ).reshape(batch, cin, indices.shape[0], indices.shape[1])
    w = weight.to(torch.float32).reshape(cout, cin, v)[:, :, rf_index]
    return torch.einsum("bckn,ocnk->bon", feats, w) + bias.to(torch.float32).reshape(1, -1, 1)


# Small enough to keep the test fast, with multiple P tiles (cin * k = 1200 -> 19 tiles) and a
# padded-neighbor count, plus a non-multiple Cout edge shape.
MAIN_SHAPE = dict(cin=48, cout=96, nin=128, nout=40, k=25, v=100, pads=37)
ODD_SHAPE = dict(cin=17, cout=7, nin=300, nout=33, k=11, v=25, pads=13)

TOLERANCE = 3e-3


@unittest.skipUnless(_warp_ready(), "requires CUDA and warp-lang")
class TestLargeBatchUncachedForward(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda", torch.cuda.current_device())

    def _check(self, shape, batch, config=None):
        from fovi.arch import knn_warp

        x, weight, bias, indices, rf_index, input_linear, weight_linear = make_case(
            batch=batch, device=self.device, **shape
        )
        reference = oracle_fp32(x, weight, bias, indices, rf_index)
        output = knn_warp.run_uncached(
            x, weight, bias, input_linear, weight_linear, config=config
        )
        torch.cuda.synchronize()
        self.assertEqual(output.dtype, torch.float16)
        self.assertEqual(tuple(output.shape), tuple(reference.shape))
        self.assertFalse(torch.isnan(output).any().item(), "kernel launch silently failed")
        max_abs = (output.to(torch.float32) - reference).abs().max().item()
        self.assertLess(max_abs, TOLERANCE, f"shape={shape} batch={batch} config={config}")

    def test_dispatch_batches(self):
        for batch in (64, 100, 128, 512):
            self._check(MAIN_SHAPE, batch)

    def test_non_multiple_cout(self):
        for batch in (64, 100, 512):
            self._check(ODD_SHAPE, batch)

    def test_reuse_configs_explicit(self):
        # The configs the dispatch heuristic can select, including fully out-of-bounds
        # batch subtiles (B=64 under 128/256-row blocks).
        for config in ("m64n64r1", "m64n64r2b256", "m64n128r2"):
            for batch in (64, 129):
                self._check(MAIN_SHAPE, batch, config=config)

    def test_zero_bias(self):
        from fovi.arch import knn_warp

        x, weight, _, indices, rf_index, input_linear, weight_linear = make_case(
            batch=128, device=self.device, **MAIN_SHAPE
        )
        bias = torch.zeros(MAIN_SHAPE["cout"], dtype=torch.float16, device=self.device)
        reference = oracle_fp32(x, weight, bias, indices, rf_index)
        output = knn_warp.run_uncached(x, weight, bias, input_linear, weight_linear)
        max_abs = (output.to(torch.float32) - reference).abs().max().item()
        self.assertLess(max_abs, TOLERANCE)

    def test_small_batch_path_unchanged(self):
        # B < 64 must keep using the small-batch kernels.
        for batch in (1, 10, 32):
            self._check(MAIN_SHAPE, batch)

    def test_canary_rejects_unlaunchable_config(self):
        from fovi.arch import knn_warp

        # m64n64r4 exceeds the sm_89 CTA budget; Warp only prints the launch failure, so the
        # canary must catch it and run_uncached must raise for an explicit config.
        knn_warp._UNCACHED_BATCH_CONFIG_SPECS["_test_bad"] = (
            knn_warp._make_uncached_r4_kernel, 64, 64, 4, 256
        )
        try:
            x, weight, bias, _, _, input_linear, weight_linear = make_case(
                batch=64, device=self.device, **ODD_SHAPE
            )
            with self.assertRaises(RuntimeError):
                knn_warp.run_uncached(
                    x, weight, bias, input_linear, weight_linear, config="_test_bad"
                )
        finally:
            knn_warp._UNCACHED_BATCH_CONFIG_SPECS.pop("_test_bad", None)
            knn_warp._UNCACHED_BATCH_KERNELS.pop("_test_bad", None)
            knn_warp._UNCACHED_BATCH_VALIDATED.pop(("_test_bad", self.device.index), None)

    def test_dispatch_falls_back_when_config_invalid(self):
        from fovi.arch import knn_warp

        x, weight, bias, indices, rf_index, input_linear, weight_linear = make_case(
            batch=128, device=self.device, **MAIN_SHAPE
        )
        selected = knn_warp._select_uncached_batch_config(
            128, weight.shape[0], input_linear.shape[0], input_linear.shape[1]
        )
        validated_key = (selected, self.device.index)
        previous = knn_warp._UNCACHED_BATCH_VALIDATED.get(validated_key)
        knn_warp._UNCACHED_BATCH_VALIDATED[validated_key] = False
        try:
            output = knn_warp.run_uncached(x, weight, bias, input_linear, weight_linear)
            reference = oracle_fp32(x, weight, bias, indices, rf_index)
            max_abs = (output.to(torch.float32) - reference).abs().max().item()
            self.assertLess(max_abs, TOLERANCE)
        finally:
            if previous is None:
                knn_warp._UNCACHED_BATCH_VALIDATED.pop(validated_key, None)
            else:
                knn_warp._UNCACHED_BATCH_VALIDATED[validated_key] = previous


def build_meta(shape, indices, input_linear, weight_linear, device):
    """Construct a TrainingMeta directly from synthetic case tables (no layer needed)."""
    from fovi.arch.knn_autograd import TrainingMeta

    cin, cout, nin, nout, k, v = (
        shape["cin"], shape["cout"], shape["nin"], shape["nout"], shape["k"], shape["v"]
    )
    knn_flat = indices.reshape(-1).to(torch.int64)  # [K*Nout], j = k*Nout + n
    valid = knn_flat < nin
    j = torch.nonzero(valid, as_tuple=False).squeeze(1)
    m = knn_flat[valid]
    order = torch.argsort(m, stable=True)
    rev_col = j[order].to(torch.int32).contiguous()
    counts = torch.bincount(m, minlength=nin)
    rev_rowptr = torch.zeros(nin + 1, dtype=torch.int64, device=device)
    torch.cumsum(counts, dim=0, out=rev_rowptr[1:])
    return TrainingMeta(
        cin=cin, cout=cout, nin=nin, nout=nout, k=k, v=v,
        p=cin * k, p64=input_linear.shape[1], q=cin * v,
        input_linear=input_linear, weight_linear=weight_linear,
        input_linear_flat=input_linear.reshape(-1).to(torch.int64).contiguous(),
        weight_linear_flat=weight_linear.reshape(-1).to(torch.int64).contiguous(),
        rev_rowptr=rev_rowptr.to(torch.int32).contiguous(),
        rev_col=rev_col, device=device,
    )


def autograd_grads(x, weight, indices, rf_index, grad_y, dtype):
    """Direct-math forward in ``dtype`` with torch autograd; returns (dx, dW)."""
    cin = x.shape[1]
    cout = weight.shape[0]
    v = weight.shape[1] // cin
    xg = x.detach().to(dtype).requires_grad_(True)
    wg = weight.detach().to(dtype).requires_grad_(True)
    x_padded = F.pad(xg, (0, 1))
    batch = x.shape[0]
    feats = torch.gather(
        x_padded, 2, indices.reshape(1, 1, -1).expand(batch, cin, -1)
    ).reshape(batch, cin, indices.shape[0], indices.shape[1])
    w = wg.reshape(cout, cin, v)[:, :, rf_index]
    y = torch.einsum("bckn,ocnk->bon", feats, w)
    (y * grad_y.to(dtype)).sum().backward()
    return xg.grad.detach(), wg.grad.detach()


PAD_HEAVY_SHAPE = dict(cin=9, cout=33, nin=90, nout=45, k=13, v=30, pads=200)


@contextlib.contextmanager
def _env(**overrides):
    saved = {key: os.environ.get(key) for key in overrides}
    os.environ.update({key: value for key, value in overrides.items()})
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@unittest.skipUnless(_warp_ready(), "requires CUDA and warp-lang")
class TestWarpTrainBackward(unittest.TestCase):
    """Relative-to-baseline criterion as in tests/test_knn_optimized.assert_backward_parity:
    warp error vs the fp32 oracle must be <= 3x the fp16 direct-math autograd error + 2e-3."""

    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda", torch.cuda.current_device())

    def _case(self, shape, batch):
        x, weight, bias, indices, rf_index, input_linear, weight_linear = make_case(
            batch=batch, device=self.device, **shape
        )
        meta = build_meta(shape, indices, input_linear, weight_linear, self.device)
        g = torch.randn(
            batch, meta.cout, meta.nout, device=self.device,
            generator=torch.Generator(device=self.device).manual_seed(7),
        ).to(torch.float16)
        return x, weight, bias, indices, rf_index, meta, g

    def _check_grads(self, shape, batch, nan_fill=False):
        from fovi.arch import knn_warp

        x, weight, bias, indices, rf_index, meta, g = self._case(shape, batch)
        dx32, dw32 = autograd_grads(x, weight, indices, rf_index, g, torch.float32)
        dx16, dw16 = autograd_grads(x, weight, indices, rf_index, g, torch.float16)

        original_empty = torch.empty
        if nan_fill:
            def nan_empty(*args, **kwargs):
                tensor = original_empty(*args, **kwargs)
                if tensor.is_floating_point() and tensor.is_cuda:
                    tensor.fill_(float("nan"))
                return tensor

            torch.empty = nan_empty
        try:
            dx = knn_warp.WarpTrainOps.grad_input(meta, g, weight)
            dw = knn_warp.WarpTrainOps.grad_weight(meta, g, x)
            torch.cuda.synchronize()
        finally:
            torch.empty = original_empty

        for label, actual, ref16, ref32 in (("dx", dx, dx16, dx32), ("dw", dw, dw16, dw32)):
            self.assertFalse(torch.isnan(actual).any().item(), f"{label} has NaN")
            error = (actual.float() - ref32.float()).abs().max().item()
            reference_error = (ref16.float() - ref32.float()).abs().max().item()
            self.assertLessEqual(
                error,
                3.0 * reference_error + 2e-3,
                msg=f"{label} shape={shape} batch={batch}: {error:.3e} vs ref {reference_error:.3e}",
            )

    def test_backward_parity_batches(self):
        for batch in (64, 100, 512):
            self._check_grads(MAIN_SHAPE, batch)

    def test_backward_parity_odd_cout_nan_discipline(self):
        # Cout=7 < one tile: the plain tile_load overread regression case. NaN-prefill
        # all allocations so any unpadded contraction operand poisons the result.
        for batch in (64, 100, 512):
            self._check_grads(ODD_SHAPE, batch, nan_fill=True)

    def test_backward_parity_pad_heavy(self):
        self._check_grads(PAD_HEAVY_SHAPE, 128, nan_fill=True)

    def test_backward_chunked_staging(self):
        with _env(FOVI_KNN_STAGE_MIB="1"):
            self._check_grads(MAIN_SHAPE, 200)

    def test_backward_fp32_staging(self):
        with _env(FOVI_KNN_GRAD_STAGE_FP32="1"):
            self._check_grads(MAIN_SHAPE, 100)
            self._check_grads(ODD_SHAPE, 64, nan_fill=True)

    def test_matches_compact_torch_ops(self):
        from fovi.arch import knn_warp
        from fovi.arch.knn_autograd import CompactTorchOps

        x, weight, bias, indices, rf_index, meta, g = self._case(MAIN_SHAPE, 128)
        dx_torch = CompactTorchOps.grad_input(meta, g, weight)
        dw_torch = CompactTorchOps.grad_weight(meta, g, x)
        dx_warp = knn_warp.WarpTrainOps.grad_input(meta, g, weight)
        dw_warp = knn_warp.WarpTrainOps.grad_weight(meta, g, x)
        self.assertLess((dx_warp - dx_torch).abs().max().item(), TOLERANCE)
        self.assertLess((dw_warp - dw_torch).abs().max().item(), 0.1 * dw_torch.abs().max().item() + TOLERANCE)

    def test_end_to_end_function_warp_train(self):
        from fovi.arch.knn_autograd import KNNConvFunction

        x, weight, bias, indices, rf_index, meta, g = self._case(MAIN_SHAPE, 128)
        results = {}
        for ops_name in ("torch_compact", "warp_train"):
            xg = x.detach().requires_grad_(True)
            wg = weight.detach().requires_grad_(True)
            bg = bias.detach().requires_grad_(True)
            y = KNNConvFunction.apply(xg, wg, bg, meta, ops_name)
            y.backward(gradient=g)
            results[ops_name] = (y.detach(), xg.grad, wg.grad, bg.grad)
        for warp_t, torch_t in zip(results["warp_train"], results["torch_compact"]):
            relative = warp_t.float() - torch_t.float()
            scale = torch_t.float().abs().max().item()
            self.assertLess(relative.abs().max().item(), 0.05 * scale + TOLERANCE)

    def test_registered_in_ops_registry(self):
        import fovi.arch.knn_warp  # noqa: F401
        from fovi.arch.knn_autograd import OPS_REGISTRY

        self.assertIn("warp_train", OPS_REGISTRY)

    def test_no_bias_through_function(self):
        from fovi.arch.knn_autograd import KNNConvFunction

        x, weight, bias, indices, rf_index, meta, g = self._case(ODD_SHAPE, 64)
        outputs = {}
        for ops_name in ("torch_compact", "warp_train"):
            xg = x.detach().requires_grad_(True)
            wg = weight.detach().requires_grad_(True)
            y = KNNConvFunction.apply(xg, wg, None, meta, ops_name)
            y.backward(gradient=g)
            outputs[ops_name] = (y.detach(), xg.grad, wg.grad)
        for warp_t, torch_t in zip(outputs["warp_train"], outputs["torch_compact"]):
            scale = torch_t.float().abs().max().item()
            self.assertLess((warp_t.float() - torch_t.float()).abs().max().item(), 0.05 * scale + TOLERANCE)


if __name__ == "__main__":
    unittest.main()
