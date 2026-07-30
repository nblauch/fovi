"""Gather+GEMM ops for the degenerate K=1 / V=1 KNN convolution class.

fovi-resnet18's downsample convolutions (res18_ds2/ds3/ds4) have exactly one neighbor per
output node (K=1) and a single reference-grid cell (V=1), so the general operator

    y[b, o, n] = bias[o] + sum_{c,k} x[b, c, knn[k, n]] * weight[o, c*V + rf_index[n, k]]

degenerates to a pure indexed gather followed by ONE shared dense GEMM:

    y[b, o, n] = bias[o] + sum_c x[b, c, idx[n]] * weight[o, c]

with no per-node weight indexing at all (weight is [Cout, Cin]). Forward and backward are
each an ``index_select``/``index_add_`` plus a single (batched) matmul, which works for every
dtype (fp32 / fp16 / bf16) and is within launch-overhead distance of the dense Conv2d floor.

Registered as ``"gather_gemm"`` in the :mod:`fovi.arch.knn_autograd` ops registry. Routing
predicate for the dispatcher: ``meta.k == 1 and meta.v == 1`` (layer-level:
``layer._k == 1 and layer.local_rf.shape[2] == 1``).

Padding semantics match the baseline: padding neighbors (``input_linear`` entries equal to
``Cin*Nin``) contribute exact zeros forward and receive no grad_input scatter.
"""

from __future__ import annotations

import torch


def _node_index(meta):
    """Derive the [Nout] gather index and (optional) valid mask from meta.input_linear.

    For K=1 the c=0 column of ``input_linear`` is ``0*Nin + idx[n]`` for real neighbors and
    ``Cin*Nin`` (out of range) for padding neighbors. Returns ``(index, valid)`` with
    ``valid is None`` when there is no padding (the common case for the res18 downsamples).
    """
    cached = getattr(meta, "_gather_gemm_index", None)
    if cached is not None:
        return cached
    first = meta.input_linear[:, 0].to(torch.int64)
    valid = first < meta.nin
    if bool(valid.all().item()):  # one-time host sync; result cached on the meta object
        cached = (first, None)
    else:
        cached = (torch.where(valid, first, torch.zeros_like(first)), valid)
    meta._gather_gemm_index = cached
    return cached


def _scatter_matrix(meta, dtype):
    """Cached one-hot [Nout, Nin] scatter operand: S[n, idx[n]] = 1 for real neighbors.

    grad_input then becomes a single GEMM ``d_selected @ S`` — collisions (several output
    nodes sharing one input node) sum inside cuBLAS's fp32 accumulators, padding rows are
    all-zero, and the fp32 zeros/index_add/cast round-trips disappear. Dense S is only
    sensible because this class is tiny (Nout*Nin <= a few 1e4 for the res18 downsamples).
    """
    cached = getattr(meta, "_gather_gemm_scatter", None)
    if cached is not None and cached[0] == dtype:
        return cached[1]
    index, valid = _node_index(meta)
    scatter = torch.zeros(meta.nout, meta.nin, device=index.device, dtype=dtype)
    source = torch.ones(meta.nout, 1, device=index.device, dtype=dtype)
    if valid is not None:
        source = source * valid.reshape(-1, 1).to(dtype)
    scatter.scatter_(1, index.reshape(-1, 1), source)
    meta._gather_gemm_scatter = (dtype, scatter)
    return scatter


class GatherGemmOps:
    """Ops-registry backend for the K=1/V=1 class: index_select + one dense GEMM."""

    name = "gather_gemm"

    @staticmethod
    def forward(meta, x, weight, bias):
        if meta.k != 1 or meta.v != 1:
            raise RuntimeError("gather_gemm requires K == 1 and V == 1")
        index, valid = _node_index(meta)
        selected = x.index_select(2, index)  # [B, Cin, Nout]
        if valid is not None:
            selected = selected * valid.to(selected.dtype)
        # [Cout, Cin] @ [B, Cin, Nout] -> [B, Cout, Nout] (one strided-batched GEMM)
        output = torch.matmul(weight.reshape(meta.cout, meta.cin), selected)
        if bias is not None:
            output += bias.reshape(1, -1, 1)
        return output

    @staticmethod
    def grad_input(meta, grad_y, weight):
        # [Cin, Cout] @ [B, Cout, Nout] -> [B, Cin, Nout] in compute dtype
        d_selected = torch.matmul(weight.reshape(meta.cout, meta.cin).t(), grad_y)
        # One scatter GEMM: [B, Cin, Nout] @ [Nout, Nin]. Accumulation over colliding
        # output nodes happens in cuBLAS fp32 internally; padding rows of S are zero.
        dx = torch.matmul(d_selected, _scatter_matrix(meta, d_selected.dtype))
        return dx  # compute dtype; KNNConvFunction casts to x.dtype

    @staticmethod
    def grad_weight(meta, grad_y, x):
        index, valid = _node_index(meta)
        # Gather directly in [Cin, B, Nout] layout (index_select output is contiguous), so
        # only grad_y needs a flattening copy for the single [Cout, B*Nout] x [B*Nout, Cin]
        # reduction GEMM (fp32 accumulation inside cuBLAS for half dtypes).
        selected_t = x.permute(1, 0, 2).index_select(2, index)
        if valid is not None:
            selected_t = selected_t * valid.to(selected_t.dtype)
        g_flat = grad_y.permute(1, 0, 2).reshape(meta.cout, -1)
        dw = torch.mm(g_flat, selected_t.reshape(meta.cin, -1).t())
        return dw.to(torch.float32).reshape(meta.cout, meta.q)


try:
    from .knn_autograd import register_ops as _register_ops

    _register_ops(GatherGemmOps)
except ImportError:  # pragma: no cover - standalone use outside the package
    pass
