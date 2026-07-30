"""Tests for the native CUDA (CuPy NVRTC) KNN pooling kernels.

Run with: python -m unittest tests.test_knn_pool_cuda -v
Device selection: first CUDA device by default; override with FOVI_TEST_DEVICE=<index>.

Parity doctrine:
- max outputs: bit-exact vs the same-dtype baseline (both return an input element; the
  fp16<->fp32 conversion round-trip is exact).
- max gradients: judged vs the same-dtype baseline (the fp32 oracle is tie-brittle:
  reduced-precision rounding swaps near-tie argmax winners and reroutes whole gradient
  entries — a property of reduced-precision max pooling, not a kernel bug). The kernel's
  strict-> / ascending-k tie rule matches torch.max on CUDA, so the residual difference
  is only the baseline's own fp16/bf16 scatter-add accumulation noise; a deterministic
  fp32-routing reference (baseline's own argmax indices + fp32 scatter) pins the kernel
  to final-store rounding.
- avg: full fp32-oracle criterion (native error <= 3x the same-dtype baseline's error).
"""

import importlib.util
import os
import unittest

import torch

_HAVE_CUDA = torch.cuda.is_available()
_HAVE_CUPY = importlib.util.find_spec("cupy") is not None

if _HAVE_CUDA and _HAVE_CUPY:
    from fovi.arch import knn_pool_cuda as kpc


def _device():
    index = int(os.environ.get("FOVI_TEST_DEVICE", "0"))
    return torch.device("cuda", index)


def _baseline_pool(x, indices, mode):
    """Exact replica of ``KNNPoolingLayer.forward`` (fovi/arch/knn.py:346-395) for the
    modes the benchmark models instantiate; differentiable through native autograd."""
    x = torch.concatenate(
        [x, torch.nan * torch.zeros(x.shape[0], x.shape[1], 1, device=x.device, dtype=x.dtype)],
        dim=2,
    )
    batch, channels, _ = x.shape
    nout = indices.shape[1]
    knn = torch.gather(x, 2, indices.reshape(1, 1, -1).expand(batch, channels, -1))
    knn = knn.reshape(batch, channels, indices.shape[0], nout)
    if mode == "avg":
        return torch.nanmean(knn, dim=2)
    if mode == "max":
        knn[torch.isnan(knn)] = -float("inf")
        return torch.max(knn, dim=2)[0]
    raise NotImplementedError(mode)


def _make_case(batch, channels, nin, nout, k, *, pads=0, dtype=torch.float16, seed=20260721,
               device=None):
    """Harness-style synthetic pooling case (non-symmetric data; pad token == nin);
    keeps the layer invariant of >= 1 valid neighbor per output node."""
    device = device or _device()
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(batch, channels, nin, dtype=torch.float32, device=device,
                    generator=generator).to(dtype)
    indices = torch.randint(nin, (k, nout), dtype=torch.int64, device=device,
                            generator=generator)
    if pads:
        flat = indices.flatten()
        flat[: min(pads, flat.numel())] = nin
        indices = flat[
            torch.randperm(flat.numel(), device=device, generator=generator)
        ].reshape_as(indices)
        indices[0, (indices < nin).sum(dim=0) == 0] = 0  # restore the >= 1 valid invariant
    return x, indices


# Real model shapes: (channels, nin, nout, k, mode, pads)
_REAL_SHAPES = {
    "alexp0": (96, 964, 230, 9, "max", 75),
    "alexp1": (256, 230, 60, 9, "max", 50),
    "alexp5": (256, 16, 1, 16, "avg", 0),
    "res18p": (64, 1469, 356, 9, "max", 96),
}

# Same-dtype gradient judge tolerances: the baseline's own fp16/bf16 scatter-add noise
# bands (measured 8e-3 / 6e-2 and reproduced here) with margin; fp32 covers
# the baseline's atomic-order noise.
_GRAD_TOL = {torch.float16: 3e-2, torch.bfloat16: 2.5e-1, torch.float32: 1e-4}
_OUT_ATOL = {torch.float16: 2e-3, torch.bfloat16: 2e-2, torch.float32: 1e-6}


def _train_step(fn, x, indices, mode, grad_out):
    leaf = x.detach().clone().requires_grad_(True)
    y = fn(leaf, indices, mode)
    y.backward(grad_out.to(y.dtype))
    return y.detach(), leaf.grad.detach()


def _native_fn(x, indices, mode):
    return kpc.pool_function(x, indices, mode)


@unittest.skipUnless(_HAVE_CUDA, "CUDA device required")
@unittest.skipUnless(_HAVE_CUPY, "CuPy required")
class TestKNNPoolForward(unittest.TestCase):
    def test_max_output_bitexact_real_shapes(self):
        for name in ("alexp0", "alexp1", "res18p"):
            channels, nin, nout, k, mode, pads = _REAL_SHAPES[name]
            for dtype in (torch.float16, torch.bfloat16, torch.float32):
                x, idx = _make_case(10, channels, nin, nout, k, pads=pads, dtype=dtype)
                meta = kpc.pool_meta_from_indices(idx, nin, _device())
                y, _ = kpc.pool_forward(meta, x, "max")
                ref = _baseline_pool(x, idx, "max")
                self.assertTrue(torch.equal(y, ref), msg=f"{name}/{dtype}")

    def test_max_output_bitexact_large_batch(self):
        channels, nin, nout, k, mode, pads = _REAL_SHAPES["alexp0"]
        x, idx = _make_case(128, channels, nin, nout, k, pads=pads)
        meta = kpc.pool_meta_from_indices(idx, nin, _device())
        y, _ = kpc.pool_forward(meta, x, "max")
        self.assertTrue(torch.equal(y, _baseline_pool(x, idx, "max")))

    def test_avg_output_oracle_band(self):
        for shape in ((256, 16, 1, 16, 0), (24, 200, 77, 7, 30)):
            channels, nin, nout, k, pads = shape
            for dtype in (torch.float16, torch.bfloat16, torch.float32):
                x, idx = _make_case(33, channels, nin, nout, k, pads=pads, dtype=dtype)
                meta = kpc.pool_meta_from_indices(idx, nin, _device())
                y, aux = kpc.pool_forward(meta, x, "avg")
                self.assertIsNone(aux)
                oracle = _baseline_pool(x.float(), idx, "avg")
                reference = _baseline_pool(x, idx, "avg").float()
                ref_err = (reference - oracle).abs().max().item()
                native_err = (y.float() - oracle).abs().max().item()
                self.assertLessEqual(
                    native_err, 3.0 * ref_err + _OUT_ATOL[dtype],
                    msg=f"{shape}/{dtype}: {native_err:.3e} vs ref {ref_err:.3e}",
                )

    def test_aux_matches_torch_max_indices(self):
        # includes a tie-heavy case: quantized values force many exact ties, pinning the
        # kernel's strict-> / ascending-k rule to torch.max's first-occurrence tie rule
        device = _device()
        generator = torch.Generator(device=device).manual_seed(3)
        cases = []
        x, idx = _make_case(10, 32, 300, 90, 9, pads=25)
        cases.append((x, idx, 300))
        x_tie = torch.randint(0, 3, (8, 16, 400), device=device, generator=generator).half()
        idx_tie = torch.randint(400, (9, 111), device=device, generator=generator)
        cases.append((x_tie, idx_tie, 400))
        for x, idx, nin in cases:
            meta = kpc.pool_meta_from_indices(idx, nin, device)
            _, aux = kpc.pool_forward(meta, x, "max")
            xp = torch.concatenate(
                [x, torch.nan * torch.zeros(x.shape[0], x.shape[1], 1, device=device,
                                            dtype=x.dtype)], dim=2)
            knn = torch.gather(
                xp, 2, idx.reshape(1, 1, -1).expand(x.shape[0], x.shape[1], -1)
            ).reshape(x.shape[0], x.shape[1], idx.shape[0], idx.shape[1])
            knn[torch.isnan(knn)] = -float("inf")
            _, kidx = torch.max(knn, dim=2)
            self.assertTrue(bool((aux.long() == kidx).all()))

    def test_all_pad_output_node(self):
        # a fully padded neighborhood must reproduce the baseline exactly: -inf for max,
        # NaN (empty nanmean) for avg
        device = _device()
        x, idx = _make_case(6, 4, 50, 11, 5, dtype=torch.float16)
        idx = idx.clone()
        idx[:, 3] = 50
        for mode in ("max", "avg"):
            # consume the first-use canary on a clean case first (deterministic ordering)
            xc, idxc = _make_case(2, 2, 20, 4, 3)
            metac = kpc.pool_meta_from_indices(idxc, 20, device)
            kpc.pool_forward(metac, xc, mode)
            meta = kpc.pool_meta_from_indices(idx, 50, device)
            y, _ = kpc.pool_forward(meta, x, mode)
            ref = _baseline_pool(x, idx, mode)
            if mode == "max":
                self.assertTrue(bool(torch.isinf(y[:, :, 3]).all()))
                self.assertTrue(torch.equal(y, ref))
            else:
                # avg is judged vs the fp32 oracle: the baseline's nanmean rounds its
                # sum to fp16 BEFORE dividing, so the kernel (fp32 sum, IEEE divide,
                # one final rounding) is strictly closer to the oracle, not bit-equal.
                self.assertTrue(bool(torch.isnan(y[:, :, 3]).all()))
                self.assertTrue(torch.equal(y.isnan(), ref.isnan()))
                oracle = _baseline_pool(x.float(), idx, mode)
                valid = ~torch.isnan(oracle)
                ref_err = (ref.float() - oracle)[valid].abs().max().item()
                native_err = (y.float() - oracle)[valid].abs().max().item()
                self.assertLessEqual(native_err, ref_err + 1e-6)

    def test_k1_permutation_identity_exact(self):
        # K=1 permutation table: the index-orientation canary (a transposed or misread
        # table cannot reproduce an exact permutation and its exact inverse gradient)
        device = _device()
        generator = torch.Generator(device=device).manual_seed(11)
        perm = torch.randperm(64, device=device, generator=generator)
        x = torch.randn(4, 8, 64, device=device, generator=generator).half()
        meta = kpc.pool_meta_from_indices(perm.reshape(1, 64), 64, device)
        y, aux = kpc.pool_forward(meta, x, "max")
        self.assertTrue(torch.equal(y, x[:, :, perm]))
        self.assertTrue(bool((aux == 0).all()))
        gy = torch.randn(4, 8, 64, device=device, generator=generator).half()
        dx = kpc.pool_backward(meta, gy, aux, "max")
        inverse = torch.empty_like(perm)
        inverse[perm] = torch.arange(64, device=device)
        self.assertTrue(torch.equal(dx, gy[:, :, inverse]))

    def test_rs_override_and_row_tail(self):
        # rows = 3 * 5 = 15 exercises the rcount tail for every swept RS
        x, idx = _make_case(3, 5, 120, 40, 9, pads=12)
        meta = kpc.pool_meta_from_indices(idx, 120, _device())
        ref = _baseline_pool(x, idx, "max")
        for rs in (1, 2, 4):
            y, _ = kpc.pool_forward(meta, x, "max", rs=rs)
            self.assertTrue(torch.equal(y, ref), msg=f"rs={rs}")

    def test_need_aux_false(self):
        x, idx = _make_case(5, 7, 90, 30, 9, pads=8)
        meta = kpc.pool_meta_from_indices(idx, 90, _device())
        y, aux = kpc.pool_forward(meta, x, "max", need_aux=False)
        self.assertIsNone(aux)
        self.assertTrue(torch.equal(y, _baseline_pool(x, idx, "max")))

    def test_non_contiguous_input(self):
        x, idx = _make_case(6, 4, 64, 20, 9)
        meta = kpc.pool_meta_from_indices(idx, 64, _device())
        x_nc = x.transpose(1, 2).contiguous().transpose(1, 2)
        self.assertFalse(x_nc.is_contiguous())
        y, _ = kpc.pool_forward(meta, x_nc, "max")
        self.assertTrue(torch.equal(y, _baseline_pool(x, idx, "max")))

    def test_invalid_arguments(self):
        x, idx = _make_case(4, 4, 40, 10, 5)
        meta = kpc.pool_meta_from_indices(idx, 40, _device())
        with self.assertRaises(ValueError):
            kpc.pool_forward(meta, x, "sum")
        with self.assertRaises(ValueError):
            kpc.pool_forward(meta, x[:, :, :30], "max")  # Nin mismatch
        with self.assertRaises(ValueError):
            kpc.pool_meta_from_indices(torch.zeros(256, 4, dtype=torch.int64,
                                                   device=_device()), 40)  # K > 255


@unittest.skipUnless(_HAVE_CUDA, "CUDA device required")
@unittest.skipUnless(_HAVE_CUPY, "CuPy required")
class TestKNNPoolBackward(unittest.TestCase):
    def _routing_reference(self, meta, x, idx, gy):
        """Deterministic fp32 reference: the baseline's own torch.max argmax indices with
        an fp32 scatter (isolates accumulation precision from tie routing)."""
        xp = torch.concatenate(
            [x, torch.nan * torch.zeros(x.shape[0], x.shape[1], 1, device=x.device,
                                        dtype=x.dtype)], dim=2)
        knn = torch.gather(
            xp, 2, idx.reshape(1, 1, -1).expand(x.shape[0], x.shape[1], -1)
        ).reshape(x.shape[0], x.shape[1], idx.shape[0], idx.shape[1])
        knn[torch.isnan(knn)] = -float("inf")
        _, kidx = torch.max(knn, dim=2)
        nout = meta.nout
        pos = torch.gather(meta.indices_i64, 0, kidx.reshape(-1, nout))
        dxf = torch.zeros(x.shape[0] * x.shape[1], meta.nin + 1, dtype=torch.float32,
                          device=x.device)
        dxf.scatter_add_(1, pos, gy.reshape(-1, nout).float())
        return dxf[:, : meta.nin].reshape(x.shape[0], x.shape[1], meta.nin)

    def test_max_grad_matches_fp32_routing_reference(self):
        for dtype, band in ((torch.float16, 2 ** -10), (torch.bfloat16, 2 ** -7),
                            (torch.float32, 2 ** -22)):
            x, idx = _make_case(33, 24, 300, 90, 9, pads=40, dtype=dtype)
            meta = kpc.pool_meta_from_indices(idx, 300, _device())
            gy = torch.randn(33, 24, 90, device=_device()).to(dtype)
            _, aux = kpc.pool_forward(meta, x, "max")
            dx = kpc.pool_backward(meta, gy, aux, "max")
            ref = self._routing_reference(meta, x, idx, gy)
            tol = band * max(ref.abs().max().item(), 1.0) + 1e-6
            self.assertLess((dx.float() - ref).abs().max().item(), tol, msg=str(dtype))

    def test_max_grad_same_dtype_judge_real_shapes(self):
        for name in ("alexp0", "res18p"):
            channels, nin, nout, k, mode, pads = _REAL_SHAPES[name]
            for dtype in (torch.float16, torch.bfloat16):
                x, idx = _make_case(10, channels, nin, nout, k, pads=pads, dtype=dtype)
                gy = torch.randn(10, channels, nout, device=_device())
                y_n, dx_n = _train_step(_native_fn, x, idx, mode, gy)
                y_b, dx_b = _train_step(_baseline_pool, x, idx, mode, gy)
                self.assertTrue(torch.equal(y_n, y_b), msg=f"{name}/{dtype}")
                self.assertLess(
                    (dx_n.float() - dx_b.float()).abs().max().item(), _GRAD_TOL[dtype],
                    msg=f"{name}/{dtype}",
                )

    def test_batches(self):
        channels, nin, nout, k, mode, pads = _REAL_SHAPES["alexp1"]
        for batch in (10, 128, 512):
            x, idx = _make_case(batch, channels, nin, nout, k, pads=pads)
            gy = torch.randn(batch, channels, nout, device=_device())
            y_n, dx_n = _train_step(_native_fn, x, idx, mode, gy)
            y_b, dx_b = _train_step(_baseline_pool, x, idx, mode, gy)
            self.assertTrue(torch.equal(y_n, y_b), msg=f"B={batch}")
            self.assertLess(
                (dx_n.float() - dx_b.float()).abs().max().item(),
                _GRAD_TOL[torch.float16], msg=f"B={batch}",
            )

    def test_tie_heavy_max_deterministic(self):
        device = _device()
        generator = torch.Generator(device=device).manual_seed(5)
        x = torch.randint(0, 3, (8, 16, 400), device=device, generator=generator).half()
        idx = torch.randint(400, (9, 111), device=device, generator=generator)
        meta = kpc.pool_meta_from_indices(idx, 400, device)
        gy = torch.randn(8, 16, 111, device=device).half()
        y, aux = kpc.pool_forward(meta, x, "max")
        self.assertTrue(torch.equal(y, _baseline_pool(x, idx, "max")))
        dx1 = kpc.pool_backward(meta, gy, aux, "max")
        dx2 = kpc.pool_backward(meta, gy, aux, "max")
        self.assertTrue(torch.equal(dx1, dx2))
        ref = self._routing_reference(meta, x, idx, gy)
        self.assertLess((dx1.float() - ref).abs().max().item(),
                        2 ** -10 * max(ref.abs().max().item(), 1.0) + 1e-6)

    def test_avg_grad_oracle(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for shape in ((256, 16, 1, 16, 0), (24, 200, 77, 7, 30)):
                channels, nin, nout, k, pads = shape
                x, idx = _make_case(33, channels, nin, nout, k, pads=pads, dtype=dtype)
                gy = torch.randn(33, channels, nout, device=_device())
                _, dx_n = _train_step(_native_fn, x, idx, "avg", gy)
                _, dx_b = _train_step(_baseline_pool, x, idx, "avg", gy)
                _, dx_o = _train_step(_baseline_pool, x.float(), idx, "avg", gy)
                ref_err = (dx_b.float() - dx_o.float()).abs().max().item()
                native_err = (dx_n.float() - dx_o.float()).abs().max().item()
                self.assertLessEqual(
                    native_err, 3.0 * ref_err + _OUT_ATOL[dtype],
                    msg=f"{shape}/{dtype}: {native_err:.3e} vs ref {ref_err:.3e}",
                )

    def test_all_pad_node_contributes_nothing(self):
        device = _device()
        x, idx = _make_case(6, 4, 50, 11, 5, dtype=torch.float16)
        idx = idx.clone()
        idx[:, 3] = 50
        gy = torch.randn(6, 4, 11, device=device)
        for mode in ("max", "avg"):
            y_n, dx_n = _train_step(_native_fn, x, idx, mode, gy)
            # zeroing the all-pad node's grad_out must not change dx (no CSR entries)
            gy_zero = gy.clone()
            gy_zero[:, :, 3] = 0
            _, dx_z = _train_step(_native_fn, x, idx, mode, gy_zero)
            self.assertTrue(torch.equal(dx_n, dx_z), msg=mode)

    def test_scaled_gradients_finite(self):
        # GradScaler reality: grad_y arrives pre-scaled; fp32 register accumulation must
        # stay finite and match the routing/oracle reference of the scaled problem.
        # (dx is stored in grad_y's dtype like the baseline's, so per-node sums beyond
        # the fp16 max overflow identically in both paths — the scale here keeps the
        # true sums representable, which is the regime GradScaler maintains.)
        x, idx = _make_case(64, 16, 200, 60, 9, pads=20)
        meta = kpc.pool_meta_from_indices(idx, 200, _device())
        gy = (torch.randn(64, 16, 60, device=_device()).clamp(-1, 1) * 2.0 ** 11).half()
        _, aux = kpc.pool_forward(meta, x, "max")
        dx = kpc.pool_backward(meta, gy, aux, "max")
        self.assertTrue(bool(torch.isfinite(dx).all()))
        ref = self._routing_reference(meta, x, idx, gy)
        # exact up to the single final-store rounding (fp16 ulp of the largest sum)
        self.assertLess(
            (dx.float() - ref).abs().max().item(),
            2.0 ** -10 * max(ref.abs().max().item(), 1.0) + 1e-6,
        )

    def test_backward_rs_override(self):
        x, idx = _make_case(3, 5, 120, 40, 9, pads=12)
        meta = kpc.pool_meta_from_indices(idx, 120, _device())
        gy = torch.randn(3, 5, 40, device=_device()).half()
        _, aux = kpc.pool_forward(meta, x, "max")
        ref = kpc.pool_backward(meta, gy, aux, "max", rs=1)
        for rs in (2, 4):
            self.assertTrue(torch.equal(kpc.pool_backward(meta, gy, aux, "max", rs=rs), ref))

    def test_max_backward_requires_aux(self):
        x, idx = _make_case(4, 4, 40, 10, 5)
        meta = kpc.pool_meta_from_indices(idx, 40, _device())
        gy = torch.randn(4, 4, 10, device=_device()).half()
        with self.assertRaises(ValueError):
            kpc.pool_backward(meta, gy, None, "max")


class _StubCoords:
    def __init__(self, nin):
        self.shape = (nin,)


class _StubPoolLayer:
    """Minimal stand-in exposing exactly the attribute surface optimized_pool_forward
    reads (mode, knn_indices_pad_token, knn_pad_token_val; metadata cache slot)."""

    def __init__(self, indices, nin, mode):
        self.mode = mode
        self.knn_indices_pad_token = indices
        self.knn_pad_token_val = nin
        self.in_coords = _StubCoords(nin)


@unittest.skipUnless(_HAVE_CUDA, "CUDA device required")
@unittest.skipUnless(_HAVE_CUPY, "CuPy required")
class TestKNNPoolDispatch(unittest.TestCase):
    def test_layer_dispatch_parity(self):
        for mode, shape in (("max", (16, 150, 44, 9, 15)), ("avg", (16, 150, 44, 9, 15))):
            channels, nin, nout, k, pads = shape
            x, idx = _make_case(12, channels, nin, nout, k, pads=pads)
            layer = _StubPoolLayer(idx, nin, mode)
            leaf = x.detach().clone().requires_grad_(True)
            y = kpc.optimized_pool_forward(layer, leaf)
            self.assertIsNotNone(y)
            gy = torch.randn_like(y, dtype=torch.float32)
            y.backward(gy.to(y.dtype))
            y_b, dx_b = _train_step(_baseline_pool, x, idx, mode, gy)
            if mode == "max":
                self.assertTrue(torch.equal(y.detach(), y_b))
            else:
                self.assertLess((y.detach().float() - y_b.float()).abs().max().item(), 2e-3)
            self.assertLess(
                (leaf.grad.float() - dx_b.float()).abs().max().item(),
                _GRAD_TOL[torch.float16], msg=mode,
            )
            # metadata is cached on the layer
            self.assertIsNotNone(layer._knn_pool_cuda_meta)
            meta_first = layer._knn_pool_cuda_meta
            kpc.optimized_pool_forward(layer, x)
            self.assertIs(layer._knn_pool_cuda_meta, meta_first)

    def test_dispatch_fallbacks(self):
        x, idx = _make_case(4, 4, 40, 10, 5)
        # unsupported mode
        self.assertIsNone(kpc.optimized_pool_forward(_StubPoolLayer(idx, 40, "sum"), x))
        self.assertIsNone(kpc.optimized_pool_forward(_StubPoolLayer(idx, 40, "gaussian"), x))
        # CPU tensor
        self.assertIsNone(
            kpc.optimized_pool_forward(_StubPoolLayer(idx.cpu(), 40, "max"), x.cpu())
        )
        # Nin mismatch
        self.assertIsNone(kpc.optimized_pool_forward(_StubPoolLayer(idx, 40, "max"),
                                                     x[:, :, :30]))
        # index table outside the uint8-argmax envelope (K > 255)
        big_k = torch.zeros(256, 10, dtype=torch.int64, device=_device())
        layer = _StubPoolLayer(big_k, 40, "max")
        self.assertIsNone(kpc.optimized_pool_forward(layer, x))

    def test_dispatch_smem_envelope(self):
        # K * Nout far beyond shared memory: dispatch must decline, not crash
        device = _device()
        generator = torch.Generator(device=device).manual_seed(9)
        idx = torch.randint(64, (128, 60000), dtype=torch.int64, device=device,
                            generator=generator)
        x = torch.randn(1, 2, 64, device=device).half()
        self.assertIsNone(kpc.optimized_pool_forward(_StubPoolLayer(idx, 64, "max"), x))

    def test_inference_no_graph(self):
        x, idx = _make_case(4, 4, 40, 10, 5)
        layer = _StubPoolLayer(idx, 40, "max")
        with torch.no_grad():
            y = kpc.optimized_pool_forward(layer, x)
        self.assertIsNotNone(y)
        self.assertFalse(y.requires_grad)
        self.assertTrue(torch.equal(y, _baseline_pool(x, idx, "max")))

    def test_autocast_context(self):
        # pooling is not autocast-eligible: under AMP it consumes the previous layer's
        # half output unchanged; the Function must behave identically inside autocast
        x, idx = _make_case(4, 4, 40, 10, 5, dtype=torch.float16)
        layer = _StubPoolLayer(idx, 40, "max")
        with torch.autocast("cuda", dtype=torch.float16):
            y = kpc.optimized_pool_forward(layer, x)
        self.assertEqual(y.dtype, torch.float16)
        self.assertTrue(torch.equal(y, _baseline_pool(x, idx, "max")))


if __name__ == "__main__":
    unittest.main()
