"""Tests for the native CUDA (CuPy NVRTC) KNN convolution kernels.

Run with: python -m unittest tests.test_knn_cuda -v
Device selection: first CUDA device by default; override with FOVI_TEST_DEVICE=<index>
(useful on multi-GPU hosts where a specific architecture should be exercised).
"""

import importlib.util
import os
import unittest

import torch
import torch.nn.functional as F

_HAVE_CUDA = torch.cuda.is_available()
_HAVE_CUPY = importlib.util.find_spec("cupy") is not None

if _HAVE_CUDA and _HAVE_CUPY:
    from fovi.arch import knn_cuda
    from fovi.arch.knn_cuda import KernelConfig


def _device():
    index = int(os.environ.get("FOVI_TEST_DEVICE", "0"))
    return torch.device("cuda", index)


def _make_case(batch, cin, nin, k, nout, cout, v, *, pads=0, dtype=torch.float16,
               device=None, bias=True, seed=0):
    device = device or _device()
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(batch, cin, nin, device=device, generator=generator).to(dtype)
    weight = (
        torch.randn(cout, cin * v, device=device, generator=generator) / (cin * v) ** 0.5
    ).to(dtype)
    bias_t = (
        torch.randn(cout, device=device, generator=generator).to(dtype) if bias else None
    )
    indices = torch.randint(nin, (nout, k), device=device, generator=generator)
    if pads:
        indices.reshape(-1)[::max(1, indices.numel() // pads)] = nin
    rf_index = torch.randint(v, (nout, k), device=device, generator=generator)
    channels = torch.arange(cin, device=device).reshape(1, cin, 1)
    neighbors = indices.reshape(nout, 1, k)
    iw = cin * nin
    input_linear = torch.where(
        neighbors < nin, channels * nin + neighbors, torch.full_like(neighbors, iw)
    ).reshape(nout, cin * k)
    weight_linear = (channels * v + rf_index.reshape(nout, 1, k)).reshape(nout, cin * k)
    pad_p = (-input_linear.shape[1]) % 64
    input_linear = F.pad(input_linear, (0, pad_p), value=iw).to(torch.int32).contiguous()
    weight_linear = F.pad(weight_linear, (0, pad_p), value=0).to(torch.int32).contiguous()
    return x, weight, bias_t, input_linear, weight_linear


def _reference(x, weight, bias, input_linear, weight_linear):
    batch, cin, nin = x.shape
    xflat = F.pad(x.float().reshape(batch, cin * nin), (0, 1))
    wt = weight.float().reshape(weight.shape[0], -1).t().contiguous()
    a = xflat[:, input_linear.long()]  # [B, Nout, P64]
    w = wt[weight_linear.long()]       # [Nout, P64, Cout]
    y = torch.einsum("bnp,npc->bcn", a, w)
    if bias is not None:
        y += bias.float().reshape(1, -1, 1)
    return y


_TOL = {torch.float16: 6e-3, torch.bfloat16: 5e-2}


@unittest.skipUnless(_HAVE_CUDA, "CUDA device required")
@unittest.skipUnless(_HAVE_CUPY, "CuPy required")
class TestKNNCudaForward(unittest.TestCase):
    def _check(self, case, config=None, dtype=torch.float16):
        x, weight, bias, il, wl = case
        y = knn_cuda.forward(x, weight, bias, il, wl, config)
        ref = _reference(x, weight, bias, il, wl)
        self.assertEqual(tuple(y.shape), tuple(ref.shape))
        max_abs = (y.float() - ref).abs().max().item()
        self.assertLess(max_abs, _TOL[dtype])
        self.assertFalse(bool(torch.isnan(y.float()).any()))
        return y

    def test_forward_parity_small_batch(self):
        for dtype in (torch.float16, torch.bfloat16):
            case = _make_case(10, 6, 50, 11, 23, 40, 20, pads=9, dtype=dtype)
            self._check(case, dtype=dtype)

    def test_forward_accepts_inference_tensors(self):
        with torch.inference_mode():
            for dtype in (torch.float16, torch.bfloat16):
                case = _make_case(10, 6, 50, 11, 23, 40, 20, pads=9, dtype=dtype)
                self._check(case, dtype=dtype)

    def test_forward_parity_large_batch_all_tiles(self):
        # exercises the bm128/bn128 config with multiple k-steps, o-tiles, and b-groups
        for dtype in (torch.float16, torch.bfloat16):
            case = _make_case(300, 8, 60, 24, 17, 200, 36, pads=15, dtype=dtype)
            self._check(case, dtype=dtype)

    def test_batch_not_multiple_of_tile(self):
        for batch in (1, 5, 33):
            case = _make_case(batch, 4, 40, 16, 9, 72, 30, pads=4)
            self._check(case)

    def test_cout_not_multiple_of_tile(self):
        case = _make_case(12, 4, 40, 16, 9, 7, 30)
        self._check(case)

    def test_no_bias(self):
        case = _make_case(10, 4, 40, 16, 9, 48, 30, bias=False)
        self._check(case)

    def test_non_contiguous_input(self):
        x, weight, bias, il, wl = _make_case(10, 4, 40, 16, 9, 48, 30)
        x_nc = x.transpose(1, 2).contiguous().transpose(1, 2)
        self.assertFalse(x_nc.is_contiguous())
        y = knn_cuda.forward(x_nc, weight, bias, il, wl)
        ref = _reference(x, weight, bias, il, wl)
        self.assertLess((y.float() - ref).abs().max().item(), _TOL[torch.float16])

    def test_all_padding_node(self):
        # one output node whose neighbors are entirely padding must equal plain bias
        x, weight, bias, il, wl = _make_case(10, 4, 40, 16, 9, 48, 30)
        il = il.clone()
        il[3, :] = 4 * 40
        y = knn_cuda.forward(x, weight, bias, il, wl)
        expected = bias.float().reshape(1, -1).expand(10, -1)
        self.assertLess((y[:, :, 3].float() - expected).abs().max().item(), 1e-3)

    def test_smem_limit_config_rejected(self):
        case = _make_case(300, 8, 60, 24, 17, 200, 36)
        huge = KernelConfig(128, 128, 64, 4, 4, 4, 0, True, 8)
        with self.assertRaises(ValueError):
            knn_cuda.forward(*case, huge)

    def test_async_requires_vectorized(self):
        case = _make_case(10, 4, 40, 16, 9, 48, 30)
        bad = KernelConfig(16, 64, 64, 1, 1, 4, 0, True, 1)
        with self.assertRaises(ValueError):
            knn_cuda.forward(*case, bad)


def _make_meta(*, cin, nin, k, nout, cout, v, il, wl, device):
    from fovi.arch.knn_autograd import TrainingMeta

    return TrainingMeta(
        cin=cin, cout=cout, nin=nin, nout=nout, k=k, v=v, p=cin * k, p64=il.shape[1],
        q=cin * v, input_linear=il, weight_linear=wl,
        input_linear_flat=il.reshape(-1).to(torch.int64).contiguous(),
        weight_linear_flat=wl.reshape(-1).to(torch.int64).contiguous(),
        rev_rowptr=torch.zeros(nin + 1, dtype=torch.int32, device=device),
        rev_col=torch.zeros(0, dtype=torch.int32, device=device),
        device=device,
    )


@unittest.skipUnless(_HAVE_CUDA, "CUDA device required")
@unittest.skipUnless(_HAVE_CUPY, "CuPy required")
class TestKNNCudaBackward(unittest.TestCase):
    """Gradient parity with the relative-to-fp32-oracle criterion.

    The fp32 oracle is CompactTorchOps on fp32 tensors; the same-dtype CompactTorchOps run
    provides the reference error. The native kernels must not exceed 3x that reference error
    plus a small dtype floor (in practice they sit far below it: the fused path keeps fp32
    from the WMMA accumulators onward while the torch path rounds dA/dWeff to fp16/bf16).
    """

    _ATOL = {torch.float16: 2e-3, torch.bfloat16: 2e-2}

    def _assert_grad_parity(self, dims, batch, dtype, pads=0):
        from fovi.arch.knn_autograd import CompactTorchOps

        cin, nin, k, nout, cout, v = dims
        device = _device()
        x, weight, bias, il, wl = _make_case(
            batch, cin, nin, k, nout, cout, v, pads=pads, dtype=dtype
        )
        meta = _make_meta(cin=cin, nin=nin, k=k, nout=nout, cout=cout, v=v,
                          il=il, wl=wl, device=device)
        g = torch.randn(batch, cout, nout, device=device).to(dtype).contiguous()
        atol = self._ATOL[dtype]
        checks = (
            ("grad_input",
             knn_cuda.CudaOps.grad_input(meta, g, weight),
             CompactTorchOps.grad_input(meta, g.float(), weight.float()),
             CompactTorchOps.grad_input(meta, g, weight)),
            ("grad_weight",
             knn_cuda.CudaOps.grad_weight(meta, g, x),
             CompactTorchOps.grad_weight(meta, g.float(), x.float()),
             CompactTorchOps.grad_weight(meta, g, x)),
        )
        for kind, native, oracle, reference in checks:
            self.assertEqual(native.dtype, torch.float32)
            self.assertEqual(tuple(native.shape), tuple(oracle.shape))
            reference_error = (reference.float() - oracle).abs().max().item()
            native_error = (native - oracle).abs().max().item()
            self.assertLessEqual(
                native_error, 3.0 * reference_error + atol,
                msg=f"{kind}/{dtype}/B={batch}: {native_error:.3e} vs ref {reference_error:.3e}",
            )

    def test_grad_parity_pad_heavy(self):
        for dtype in (torch.float16, torch.bfloat16):
            self._assert_grad_parity((6, 50, 11, 23, 40, 20), 10, dtype, pads=20)

    def test_grad_parity_batches(self):
        for batch in (10, 128, 512):
            self._assert_grad_parity((4, 40, 16, 9, 72, 30), batch, torch.float16, pads=4)

    def test_grad_parity_bf16_large_batch(self):
        self._assert_grad_parity((8, 60, 24, 17, 200, 36), 512, torch.bfloat16, pads=15)

    def test_grad_parity_cout7(self):
        for dtype in (torch.float16, torch.bfloat16):
            self._assert_grad_parity((4, 40, 16, 9, 7, 30), 33, dtype)

    def test_grad_parity_res18_l1_shape(self):
        # high-Cin k=9 mid-Nout regime (fovi-resnet18 layer1 family)
        for dtype in (torch.float16, torch.bfloat16):
            self._assert_grad_parity((64, 356, 9, 356, 64, 9), 128, dtype, pads=113)

    def test_grad_parity_res18_l3_shape(self):
        # high-Cin k=9 tiny-Nout regime (16 nodes): exercises the small per-node grid
        self._assert_grad_parity((256, 16, 9, 16, 256, 9), 512, torch.float16, pads=32)

    def test_grad_parity_res18_stem_shape(self):
        # pad-heavy k=49 stem-like regime with P64 padding (147 -> 192)
        self._assert_grad_parity((3, 512, 49, 96, 64, 49), 128, torch.float16, pads=180)

    def test_permutation_weight_grad_input_exact(self):
        # Non-symmetric permutation weight: the transpose canary for the backward W gather.
        # With wl identity and W[o, q] = delta(q, (o+1) % cout), dA[b, p] = g[b, (p-1) % cout].
        device = _device()
        batch, cout = 16, 64
        cin, nin, v = 1, 128, 64
        il = torch.arange(64, device=device).reshape(1, -1).to(torch.int32).contiguous()
        wl = torch.arange(64, device=device).reshape(1, -1).to(torch.int32).contiguous()
        weight = torch.zeros(cout, cin * v, device=device)
        for o in range(cout):
            weight[o, (o + 1) % cout] = 1.0
        g = torch.randn(batch, cout, 1, device=device).half().contiguous()
        dx = knn_cuda.grad_input(g, weight.half(), il, wl, cin, nin)
        expected = g[:, torch.arange(-1, cout - 1) % cout, 0].float()
        self.assertLess((dx[:, 0, :cout] - expected).abs().max().item(), 1e-6)

    def test_grad_weight_splitk_parity(self):
        # split-K-over-batch dW. Any split factor must match the fp32 oracle to
        # fp32-reassociation accuracy (the dwt atomic scatter is the second-stage
        # reduction), including non-dividing splits (3 over 8 ksteps -> one short slice).
        from fovi.arch.knn_autograd import CompactTorchOps

        device = _device()
        for dims, batch in (
            ((128, 83, 9, 83, 128, 9), 512),   # res18_l2-like mid tier
            ((256, 16, 9, 16, 256, 9), 512),   # res18_l3-like tiny Nout
        ):
            cin, nin, k, nout, cout, v = dims
            x, weight, bias, il, wl = _make_case(
                batch, cin, nin, k, nout, cout, v, pads=32, dtype=torch.float16
            )
            meta = _make_meta(cin=cin, nin=nin, k=k, nout=nout, cout=cout, v=v,
                              il=il, wl=wl, device=device)
            g = torch.randn(batch, cout, nout, device=device).half().contiguous()
            oracle = CompactTorchOps.grad_weight(meta, g.float(), x.float())
            scale = max(oracle.abs().max().item(), 1.0)
            for ksplit in (1, 2, 3, 8, None):  # None = heuristic
                dw = knn_cuda.grad_weight(g, x, il, wl, meta.q, ksplit=ksplit)
                err = (dw - oracle).abs().max().item()
                self.assertLess(err / scale, 1e-4,
                                msg=f"ksplit={ksplit}: {err:.3e} vs scale {scale:.3e}")
            with self.assertRaises(ValueError):
                knn_cuda.grad_weight(g, x, il, wl, meta.q, ksplit=64)  # > ksteps

    def test_backward_combined_matches_split(self):
        # Same kernels, same configs, same math: combined must equal the split entries up
        # to fp32 atomic reassociation (scale-aware criterion).
        device = _device()
        for dtype in (torch.float16, torch.bfloat16):
            for dims, batch, pads in (
                ((6, 50, 11, 23, 40, 20), 10, 20),
                ((64, 356, 9, 356, 64, 9), 128, 113),   # res18_l1
                ((256, 16, 9, 16, 256, 9), 512, 32),    # res18_l3
            ):
                cin, nin, k, nout, cout, v = dims
                x, weight, bias, il, wl = _make_case(
                    batch, cin, nin, k, nout, cout, v, pads=pads, dtype=dtype
                )
                meta = _make_meta(cin=cin, nin=nin, k=k, nout=nout, cout=cout, v=v,
                                  il=il, wl=wl, device=device)
                g = torch.randn(batch, cout, nout, device=device).to(dtype).contiguous()
                dx_s = knn_cuda.CudaOps.grad_input(meta, g, weight)
                dw_s = knn_cuda.CudaOps.grad_weight(meta, g, x)
                dx_c, dw_c = knn_cuda.CudaOps.backward_combined(meta, g, x, weight)
                for name, split_t, comb_t in (("dx", dx_s, dx_c), ("dW", dw_s, dw_c)):
                    scale = max(split_t.abs().max().item(), 1.0)
                    err = (comb_t - split_t).abs().max().item()
                    self.assertLess(
                        err / scale, 1e-4,
                        msg=f"{name}/{dtype}/B={batch}: {err:.3e} vs scale {scale:.3e}",
                    )
        # fp32 delegates to the torch oracle (no fused kernel), matching the split entries.
        x, weight, bias, il, wl = _make_case(10, 6, 50, 11, 23, 40, 20, pads=9,
                                             dtype=torch.float32)
        meta = _make_meta(cin=6, nin=50, k=11, nout=23, cout=40, v=20,
                          il=il, wl=wl, device=device)
        g32 = torch.randn(10, 40, 23, device=device)
        dx32, dw32 = knn_cuda.CudaOps.backward_combined(meta, g32, x, weight)
        self.assertEqual(dx32.dtype, torch.float32)
        self.assertEqual(tuple(dw32.shape), (40, 6 * 20))

    def test_needs_input_grad_routing(self):
        # The Function must use backward_combined only when BOTH grads are needed and the
        # split entries otherwise (weight-frozen / input-only edge cases).
        from fovi.arch.knn_autograd import KNNConvFunction, OPS_REGISTRY

        device = _device()
        x, weight, bias, il, wl = _make_case(10, 6, 50, 11, 23, 40, 20, pads=9)
        meta = _make_meta(cin=6, nin=50, k=11, nout=23, cout=40, v=20,
                          il=il, wl=wl, device=device)
        g = torch.randn(10, 40, 23, device=device, dtype=torch.float16)
        calls = []

        class CountingOps:
            name = "cuda_counting"
            forward = staticmethod(knn_cuda.CudaOps.forward)

            @staticmethod
            def grad_input(meta_, gy, w):
                calls.append("grad_input")
                return knn_cuda.CudaOps.grad_input(meta_, gy, w)

            @staticmethod
            def grad_weight(meta_, gy, x_):
                calls.append("grad_weight")
                return knn_cuda.CudaOps.grad_weight(meta_, gy, x_)

            @staticmethod
            def backward_combined(meta_, gy, x_, w):
                calls.append("combined")
                return knn_cuda.CudaOps.backward_combined(meta_, gy, x_, w)

        OPS_REGISTRY[CountingOps.name] = CountingOps
        try:
            for x_grad, w_grad, expected in (
                (True, True, ["combined"]),
                (True, False, ["grad_input"]),
                (False, True, ["grad_weight"]),
            ):
                del calls[:]
                xg = x.detach().clone().requires_grad_(x_grad)
                wg = weight.detach().clone().requires_grad_(w_grad)
                bg = bias.detach().clone().requires_grad_(w_grad)
                y = KNNConvFunction.apply(xg, wg, bg, meta, CountingOps.name)
                y.backward(g)
                self.assertEqual(calls, expected)
                self.assertEqual(xg.grad is not None, x_grad)
                self.assertEqual(wg.grad is not None, w_grad)
        finally:
            del OPS_REGISTRY[CountingOps.name]

    def test_graph_capture_layer_step(self):
        # CUDA-graph smoke: capture one KNNConvFunction fwd+bwd (CuPy raw
        # launches, in-graph torch.zeros accumulators, cache rebuilds), replay it, and
        # compare against eager. Replays must be self-contained (grads re-accumulate from
        # the zeroed buffers each time). Warmup runs first-use canaries BEFORE capture --
        # the canary host syncs are capture-illegal by design.
        # NOTE: the tensor device must be CURRENT during capture (torch.cuda.graph
        # captures the current device's stream); with a mismatched current device the
        # work runs eagerly and the graph captures EMPTY -- replays then silently produce
        # zero grads (caught by exactly this test's replay-parity assertions).
        device = _device()
        with torch.cuda.device(device):
            self._graph_capture_layer_step(device)

    def _graph_capture_layer_step(self, device):
        from fovi.arch.knn_autograd import KNNConvFunction

        batch, (cin, nin, k, nout, cout, v) = 64, (32, 60, 9, 40, 48, 9)
        x, weight, bias, il, wl = _make_case(batch, cin, nin, k, nout, cout, v, pads=16)
        meta = _make_meta(cin=cin, nin=nin, k=k, nout=nout, cout=cout, v=v,
                          il=il, wl=wl, device=device)
        g = torch.randn(batch, cout, nout, device=device).half().contiguous()
        static_x = x.detach().clone().requires_grad_(True)
        static_w = weight.detach().clone().requires_grad_(True)
        static_b = bias.detach().clone().requires_grad_(True)

        def step():
            y = KNNConvFunction.apply(static_x, static_w, static_b, meta, "cuda")
            y.backward(g)
            return y

        side = torch.cuda.Stream(device)
        side.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(side):
            for _ in range(3):
                static_x.grad = static_w.grad = static_b.grad = None
                step()
        torch.cuda.current_stream(device).wait_stream(side)
        for t in (static_x, static_w, static_b):
            t.grad.zero_()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_y = step()

        ex = x.detach().clone().requires_grad_(True)
        ew = weight.detach().clone().requires_grad_(True)
        eb = bias.detach().clone().requires_grad_(True)
        ey = KNNConvFunction.apply(ex, ew, eb, meta, "cuda")
        ey.backward(g)

        for _ in range(2):  # two replays: each must reproduce eager from zeroed grads
            for t in (static_x, static_w, static_b):
                t.grad.zero_()
            graph.replay()
            torch.cuda.synchronize()
            self.assertEqual((static_y.float() - ey.float()).abs().max().item(), 0.0)
            for name, got, ref in (
                ("dx", static_x.grad, ex.grad),
                ("dW", static_w.grad, ew.grad),
                ("db", static_b.grad, eb.grad),
            ):
                scale = max(ref.abs().max().item(), 1.0)
                err = (got.float() - ref.float()).abs().max().item()
                # fp16 output rounding of atomically-reassociated fp32 sums
                self.assertLess(err / scale, 2e-3, msg=f"{name}: {err:.3e} @ {scale:.3e}")

    def test_scaled_gradients_finite(self):
        # GradScaler reality: grad_y arrives pre-scaled (up to ~2^16). The fully fused path
        # (fp16 operands -> fp32 accumulators -> fp32 atomics) must stay finite and match the
        # fp32 oracle of the scaled problem.
        from fovi.arch.knn_autograd import CompactTorchOps

        device = _device()
        batch = 128
        cin, nin, k, nout, cout, v = 4, 40, 16, 9, 72, 30
        x, weight, bias, il, wl = _make_case(batch, cin, nin, k, nout, cout, v, pads=4)
        meta = _make_meta(cin=cin, nin=nin, k=k, nout=nout, cout=cout, v=v,
                          il=il, wl=wl, device=device)
        g = (torch.randn(batch, cout, nout, device=device).clamp(-1, 1)
             * 2.0 ** 14).half().contiguous()
        dx = knn_cuda.CudaOps.grad_input(meta, g, weight)
        dw = knn_cuda.CudaOps.grad_weight(meta, g, x)
        self.assertTrue(bool(torch.isfinite(dx).all()))
        self.assertTrue(bool(torch.isfinite(dw).all()))
        dx_oracle = CompactTorchOps.grad_input(meta, g.float(), weight.float())
        dw_oracle = CompactTorchOps.grad_weight(meta, g.float(), x.float())
        self.assertLess((dx - dx_oracle).abs().max().item() / 2.0 ** 14, 1e-3)
        self.assertLess((dw - dw_oracle).abs().max().item() / 2.0 ** 14, 1e-3)


@unittest.skipUnless(_HAVE_CUDA, "CUDA device required")
@unittest.skipUnless(_HAVE_CUPY, "CuPy required")
class TestKNNCudaRegistry(unittest.TestCase):
    def test_registered(self):
        from fovi.arch.knn_autograd import OPS_REGISTRY

        self.assertIn("cuda", OPS_REGISTRY)
        self.assertIs(OPS_REGISTRY["cuda"], knn_cuda.CudaOps)

    def test_ops_forward_matches_torch_compact(self):
        from fovi.arch.knn_autograd import CompactTorchOps

        device = _device()
        x, weight, bias, il, wl = _make_case(10, 6, 50, 11, 23, 40, 20, pads=9)
        meta = _make_meta(cin=6, nin=50, k=11, nout=23, cout=40, v=20,
                          il=il, wl=wl, device=device)
        y_cuda = knn_cuda.CudaOps.forward(meta, x, weight, bias)
        y_ref = CompactTorchOps.forward(meta, x, weight, bias)
        self.assertLess((y_cuda.float() - y_ref.float()).abs().max().item(), 6e-3)
        # fp32 input delegates to the torch oracle rather than degrading precision
        y32 = knn_cuda.CudaOps.forward(meta, x.float(), weight.float(),
                                       bias.float() if bias is not None else None)
        self.assertEqual(y32.dtype, torch.float32)

    def test_autograd_function_forward_backward(self):
        from fovi.arch.knn_autograd import KNNConvFunction

        device = _device()
        x, weight, bias, il, wl = _make_case(10, 6, 50, 11, 23, 40, 20, pads=9)
        meta = _make_meta(cin=6, nin=50, k=11, nout=23, cout=40, v=20,
                          il=il, wl=wl, device=device)
        grad_out = torch.randn(10, 40, 23, device=device, dtype=torch.float16)
        results = {}
        for ops_name in ("torch_compact", "cuda"):
            xg = x.detach().clone().requires_grad_(True)
            wg = weight.detach().clone().requires_grad_(True)
            bg = bias.detach().clone().requires_grad_(True)
            y = KNNConvFunction.apply(xg, wg, bg, meta, ops_name)
            (y * grad_out).sum().backward()
            results[ops_name] = (y.detach(), xg.grad, wg.grad, bg.grad)
        # Relative-to-fp32-oracle criterion: the native path keeps fp32 from the accumulators
        # onward, so it can be (and is) MORE accurate than torch_compact's rounded
        # intermediates; a direct backend-to-backend tolerance would be wrong.
        from fovi.arch.knn_autograd import CompactTorchOps

        g32 = grad_out.float()
        oracle = (
            CompactTorchOps.forward(meta, x.float(), weight.float(), bias.float()),
            CompactTorchOps.grad_input(meta, g32, weight.float()),
            CompactTorchOps.grad_weight(meta, g32, x.float()),
            g32.sum(dim=(0, 2)),
        )
        for name, cuda_t, compact_t, oracle_t in zip(
            ("output", "grad_input", "grad_weight", "grad_bias"),
            results["cuda"], results["torch_compact"], oracle,
        ):
            compact_error = (compact_t.float() - oracle_t).abs().max().item()
            cuda_error = (cuda_t.float() - oracle_t).abs().max().item()
            self.assertLessEqual(
                cuda_error, 3.0 * compact_error + 2e-3,
                msg=f"{name}: {cuda_error:.3e} vs torch_compact {compact_error:.3e}",
            )


if __name__ == "__main__":
    unittest.main()
