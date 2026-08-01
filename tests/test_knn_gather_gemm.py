"""Tests for the K=1/V=1 gather+GEMM ops (fovi/arch/knn_gather_gemm.py, "gather_gemm").

Torch-only backend (no Warp dependency): fwd+bwd parity vs a fp32 autograd oracle and a
same-dtype autograd reference at the fovi-resnet18 downsample shapes plus a pad-containing
synthetic, across fp32/fp16/bf16 and B in {10, 128, 512}, on CPU and CUDA.
"""

import unittest

import torch

from tests.test_knn_warp_train import autograd_grads, build_meta, make_case

# fovi-resnet18 downsample cluster (K=1, V=1, zero padding) + a pad-containing synthetic.
DS_SHAPES = {
    "res18_ds2": dict(cin=64, cout=128, nin=356, nout=83, k=1, v=1, pads=0),
    "res18_ds3": dict(cin=128, cout=256, nin=83, nout=16, k=1, v=1, pads=0),
    "res18_ds4": dict(cin=256, cout=512, nin=16, nout=2, k=1, v=1, pads=0),
    "ds_padded": dict(cin=48, cout=96, nin=120, nout=40, k=1, v=1, pads=9),
}

_ATOL = {torch.float32: 1e-5, torch.float16: 2e-3, torch.bfloat16: 2e-2}


def _oracle_forward(x, weight, bias, indices, rf_index, dtype):
    import torch.nn.functional as F

    cin = x.shape[1]
    cout = weight.shape[0]
    v = weight.shape[1] // cin
    x_padded = F.pad(x.to(dtype), (0, 1))
    batch = x.shape[0]
    feats = torch.gather(
        x_padded, 2, indices.reshape(1, 1, -1).expand(batch, cin, -1)
    ).reshape(batch, cin, indices.shape[0], indices.shape[1])
    w = weight.to(dtype).reshape(cout, cin, v)[:, :, rf_index]
    y = torch.einsum("bckn,ocnk->bon", feats, w)
    if bias is not None:
        y = y + bias.to(dtype).reshape(1, -1, 1)
    return y


class GatherGemmParityMixin:
    device = None  # set by subclasses
    dtypes = ()

    def _check(self, shape, batch, dtype, bias=True):
        from fovi.arch.knn_autograd import KNNConvFunction
        import fovi.arch.knn_gather_gemm  # noqa: F401  (registers gather_gemm)

        device = torch.device(self.device)
        x16, weight16, bias16, indices, rf_index, il, wl = make_case(
            batch=batch, device=device, **shape
        )
        # make_case emits fp16; hold fp32 masters so every dtype derives from one source
        x0, w0 = x16.float(), weight16.float()
        b0 = bias16.float() if bias else None
        meta = build_meta(shape, indices, il, wl, device)
        g32 = torch.randn(
            batch, meta.cout, meta.nout, device=device,
            generator=torch.Generator(device=device.type).manual_seed(7),
        )
        y32 = _oracle_forward(x0, w0, b0, indices, rf_index, torch.float32)
        dx32, dw32 = autograd_grads(x0, w0, indices, rf_index, g32, torch.float32)
        yref = _oracle_forward(x0, w0, b0, indices, rf_index, dtype)
        dxref, dwref = autograd_grads(x0, w0, indices, rf_index, g32, dtype)

        xg = x0.to(dtype).requires_grad_(True)
        wg = w0.to(dtype).requires_grad_(True)
        bg = b0.to(dtype).requires_grad_(True) if bias else None
        y = KNNConvFunction.apply(xg, wg, bg, meta, "gather_gemm")
        y.backward(gradient=g32.to(y.dtype))

        for label, actual, reference, oracle in (
            ("y", y.detach(), yref, y32),
            ("dx", xg.grad, dxref, dx32),
            ("dw", wg.grad, dwref, dw32),
        ):
            reference_error = (reference.float() - oracle.float()).abs().max().item()
            actual_error = (actual.float() - oracle.float()).abs().max().item()
            scale = max(1.0, oracle.float().abs().max().item())
            self.assertLessEqual(
                actual_error,
                3.0 * reference_error + _ATOL[dtype] * scale,
                msg=f"{label} shape={shape} B={batch} {dtype}: "
                    f"{actual_error:.3e} vs reference {reference_error:.3e}",
            )
        if bias:
            db32 = g32.sum(dim=(0, 2))
            db_scale = max(1.0, db32.abs().max().item())
            self.assertLessEqual(
                (bg.grad.float() - db32).abs().max().item(), _ATOL[dtype] * db_scale
            )

    def test_all_shapes_batches_dtypes(self):
        for shape in DS_SHAPES.values():
            for batch in self.batches:
                for dtype in self.dtypes:
                    self._check(shape, batch, dtype)

    def test_no_bias(self):
        self._check(DS_SHAPES["ds_padded"], 10, self.dtypes[0], bias=False)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestGatherGemmCUDA(GatherGemmParityMixin, unittest.TestCase):
    device = "cuda"
    dtypes = (torch.float32, torch.float16, torch.bfloat16)
    batches = (10, 128, 512)


class TestGatherGemmCPU(GatherGemmParityMixin, unittest.TestCase):
    device = "cpu"
    dtypes = (torch.float32,)
    batches = (10,)


class TestGatherGemmContract(unittest.TestCase):
    def test_registered_in_ops_registry(self):
        import fovi.arch.knn_gather_gemm  # noqa: F401
        from fovi.arch.knn_autograd import OPS_REGISTRY

        self.assertIn("gather_gemm", OPS_REGISTRY)

    def test_rejects_general_shapes(self):
        from fovi.arch.knn_gather_gemm import GatherGemmOps

        shape = dict(cin=4, cout=5, nin=30, nout=8, k=3, v=9, pads=0)
        device = torch.device("cpu")
        x, weight, bias, indices, rf_index, il, wl = make_case(batch=2, device=device, **shape)
        meta = build_meta(shape, indices, il, wl, device)
        with self.assertRaises(RuntimeError):
            GatherGemmOps.forward(meta, x.float(), weight.float(), bias.float())

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_matches_compact_torch_ops(self):
        import fovi.arch.knn_gather_gemm  # noqa: F401
        from fovi.arch.knn_autograd import CompactTorchOps
        from fovi.arch.knn_gather_gemm import GatherGemmOps

        device = torch.device("cuda")
        shape = DS_SHAPES["ds_padded"]
        x, weight, bias, indices, rf_index, il, wl = make_case(batch=128, device=device, **shape)
        meta = build_meta(shape, indices, il, wl, device)
        g = torch.randn(128, meta.cout, meta.nout, device=device).to(torch.float16)
        # Backend-vs-backend fp16 comparisons must be scale-aware: CompactTorchOps'
        # index_add_ is atomic-nondeterministic and both sides round fp16 at magnitude, so a
        # hard absolute atol false-fails intermittently on the largest-magnitude entries.
        for label, actual, reference in (
            ("y", GatherGemmOps.forward(meta, x, weight, bias),
             CompactTorchOps.forward(meta, x, weight, bias)),
            ("dx", GatherGemmOps.grad_input(meta, g, weight),
             CompactTorchOps.grad_input(meta, g, weight)),
            ("dw", GatherGemmOps.grad_weight(meta, g, x),
             CompactTorchOps.grad_weight(meta, g, x)),
        ):
            scale = max(1.0, reference.float().abs().max().item())
            self.assertLess(
                (actual.float() - reference.float()).abs().max().item(),
                2e-3 * scale,
                msg=label,
            )


if __name__ == "__main__":
    unittest.main()
