"""Training-capable compact backends for :class:`KNNConvLayer`.

The baseline layer computes

    y[b, o, n] = bias[o] + sum_{c,k} x[b, c, knn[k, n]] * weight[o, c*V + rf_index[n, k]]

through a dense one-hot einsum followed by ``F.linear``. The backends here use the
compact contraction width ``P = Cin*K`` instead of ``Q = Cin*V`` and provide a custom
autograd path so training no longer falls back to the baseline.

Structure:

- :class:`TrainingMeta` bundles the derived index tables every backend consumes,
  including a reverse-CSR structure mapping each input node to the (k, n) pairs that
  reference it (needed for deterministic grad-input kernels).
- :class:`KNNConvFunction` is the single ``torch.autograd.Function``; per-backend
  compute is looked up in :data:`OPS_REGISTRY` so kernel backends (Warp, CUDA) only
  need to register an ops object with ``forward`` / ``grad_input`` / ``grad_weight``.
- ``torch_scatter`` replaces the dense one-hot einsum with ``scatter_add_`` and keeps
  native autograd; ``torch_compact`` is the compact bmm formulation with the custom
  backward.

Gradient accumulation is always fp32; scattered writes go through ``index_add_`` on
fp32 buffers (never fp16 atomics). Under autocast, inputs are cast to the autocast
dtype inside the Function, grad_input is returned in ``x.dtype`` and grad_weight /
grad_bias in the parameter dtype (fp32 master weights under AMP).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------------------
# Metadata
# --------------------------------------------------------------------------------------


@dataclass
class TrainingMeta:
    """Derived, device-resident index tables shared by all training backends."""

    cin: int
    cout: int
    nin: int
    nout: int
    k: int
    v: int
    p: int
    p64: int
    q: int
    input_linear: torch.Tensor  # [Nout, P64] int32, pad entries == cin*nin
    weight_linear: torch.Tensor  # [Nout, P64] int32, pad entries == 0 (exact-zero contributions)
    input_linear_flat: torch.Tensor  # [Nout*P64] int64 (for index_select / index_add_)
    weight_linear_flat: torch.Tensor  # [Nout*P64] int64
    rev_rowptr: torch.Tensor  # [Nin+1] int32 reverse-CSR row pointers
    rev_col: torch.Tensor  # [nnz] int32, packed j = k*Nout + n
    device: torch.device


def ensure_training_metadata(layer, device: torch.device) -> TrainingMeta:
    """Build (or fetch cached) :class:`TrainingMeta` for ``layer`` on ``device``.

    Cached on ``layer._knn_training_metadata``; released by
    :func:`fovi.arch.knn_optimization.clear_cache`. Never enters ``state_dict``.
    """
    cached = getattr(layer, "_knn_training_metadata", None)
    if cached is not None and cached.device == device:
        return cached

    from .knn_optimization import _ensure_metadata

    _, input_linear, weight_linear = _ensure_metadata(layer, device)
    cin = layer.in_channels
    cout = layer.out_channels
    nin = layer.in_coords.shape[0]
    nout = input_linear.shape[0]
    k = layer._k
    v = layer.local_rf.shape[2]

    # Reverse CSR over input nodes: for each m in [0, Nin), the flattened (k, n) pairs
    # j = k*Nout + n whose neighbor index is m. Padding neighbors (>= Nin) are dropped.
    knn_flat = layer.knn_indices_pad_token.to(device=device).reshape(-1).to(torch.int64)
    valid = knn_flat < nin
    j = torch.nonzero(valid, as_tuple=False).squeeze(1)
    m = knn_flat[valid]
    order = torch.argsort(m, stable=True)
    rev_col = j[order].to(torch.int32).contiguous()
    counts = torch.bincount(m, minlength=nin)
    rev_rowptr = torch.zeros(nin + 1, dtype=torch.int64, device=device)
    torch.cumsum(counts, dim=0, out=rev_rowptr[1:])
    rev_rowptr = rev_rowptr.to(torch.int32).contiguous()

    meta = TrainingMeta(
        cin=cin,
        cout=cout,
        nin=nin,
        nout=nout,
        k=k,
        v=v,
        p=cin * k,
        p64=input_linear.shape[1],
        q=cin * v,
        input_linear=input_linear,
        weight_linear=weight_linear,
        input_linear_flat=input_linear.reshape(-1).to(torch.int64).contiguous(),
        weight_linear_flat=weight_linear.reshape(-1).to(torch.int64).contiguous(),
        rev_rowptr=rev_rowptr,
        rev_col=rev_col,
        device=device,
    )
    layer._knn_training_metadata = meta
    return meta


def _nout_chunk(meta: TrainingMeta, batch: int, itemsize: int) -> int:
    """Largest Nout slice keeping per-chunk staging (W_eff + A + dA) under budget."""
    budget_mib = int(os.environ.get("FOVI_KNN_STAGE_MIB", "512"))
    budget = budget_mib * 1024 * 1024
    per_node = (meta.p64 * meta.cout + 2 * batch * meta.p64) * itemsize
    return max(1, min(meta.nout, budget // max(per_node, 1)))


def _pad_flat_input(meta: TrainingMeta, x: torch.Tensor) -> torch.Tensor:
    """[B, Cin, Nin] -> [B, Cin*Nin + 1] with a trailing zero pad column."""
    x_flat = x.contiguous().reshape(x.shape[0], meta.cin * meta.nin)
    return F.pad(x_flat, (0, 1))


def _effective_weight_t(meta: TrainingMeta, weight: torch.Tensor) -> torch.Tensor:
    """weight [Cout, Q] -> transposed compact operand source [Q, Cout], contiguous."""
    return weight.reshape(meta.cout, meta.q).t().contiguous()


# --------------------------------------------------------------------------------------
# Ops registry: kernel backends (Warp / CUDA) register objects with this interface.
# --------------------------------------------------------------------------------------


class CompactTorchOps:
    """Pure-torch compact ops: gather -> bmm forward, bmm -> fp32 index_add_ backward.

    Serves as the correctness oracle and the structural skeleton for kernel backends.
    All three entry points take the compute-dtype tensors prepared by
    :class:`KNNConvFunction` (casting/AMP policy handled there, not here).
    """

    name = "torch_compact"

    @staticmethod
    def forward(meta: TrainingMeta, x: torch.Tensor, weight: torch.Tensor, bias) -> torch.Tensor:
        batch = x.shape[0]
        x_flat = _pad_flat_input(meta, x)
        weight_t = _effective_weight_t(meta, weight)
        y = torch.empty(batch, meta.cout, meta.nout, device=x.device, dtype=x.dtype)
        chunk = _nout_chunk(meta, batch, x.element_size())
        for n0 in range(0, meta.nout, chunk):
            n1 = min(n0 + chunk, meta.nout)
            il = meta.input_linear_flat[n0 * meta.p64 : n1 * meta.p64]
            wl = meta.weight_linear_flat[n0 * meta.p64 : n1 * meta.p64]
            # [B, nc*P64] -> [nc, B, P64] (strided batch view; cuBLAS-compatible layout)
            a = x_flat.index_select(1, il).view(batch, n1 - n0, meta.p64).transpose(0, 1)
            w_eff = weight_t.index_select(0, wl).view(n1 - n0, meta.p64, meta.cout)
            y[:, :, n0:n1] = torch.bmm(a, w_eff).permute(1, 2, 0)
        if bias is not None:
            y += bias.reshape(1, -1, 1)
        return y

    @staticmethod
    def grad_input(meta: TrainingMeta, grad_y: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        batch = grad_y.shape[0]
        weight_t = _effective_weight_t(meta, weight)
        dx_flat = torch.zeros(
            batch, meta.cin * meta.nin + 1, device=grad_y.device, dtype=torch.float32
        )
        chunk = _nout_chunk(meta, batch, grad_y.element_size())
        for n0 in range(0, meta.nout, chunk):
            n1 = min(n0 + chunk, meta.nout)
            il = meta.input_linear_flat[n0 * meta.p64 : n1 * meta.p64]
            wl = meta.weight_linear_flat[n0 * meta.p64 : n1 * meta.p64]
            g = grad_y[:, :, n0:n1].permute(2, 0, 1).contiguous()  # [nc, B, Cout]
            w_eff = weight_t.index_select(0, wl).view(n1 - n0, meta.p64, meta.cout)
            da = torch.bmm(g, w_eff.transpose(1, 2))  # [nc, B, P64]
            dx_flat.index_add_(
                1, il, da.to(torch.float32).permute(1, 0, 2).reshape(batch, -1)
            )
        return dx_flat[:, :-1].reshape(batch, meta.cin, meta.nin)

    @staticmethod
    def grad_weight(meta: TrainingMeta, grad_y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        batch = grad_y.shape[0]
        x_flat = _pad_flat_input(meta, x)
        dw_t = torch.zeros(meta.q, meta.cout, device=grad_y.device, dtype=torch.float32)
        chunk = _nout_chunk(meta, batch, grad_y.element_size())
        for n0 in range(0, meta.nout, chunk):
            n1 = min(n0 + chunk, meta.nout)
            il = meta.input_linear_flat[n0 * meta.p64 : n1 * meta.p64]
            wl = meta.weight_linear_flat[n0 * meta.p64 : n1 * meta.p64]
            g = grad_y[:, :, n0:n1].permute(2, 0, 1).contiguous()  # [nc, B, Cout]
            # [B, nc, P64] -> [nc, P64, B] (transposed strided view; cuBLAS handles it)
            a_t = x_flat.index_select(1, il).view(batch, n1 - n0, meta.p64).permute(1, 2, 0)
            dweff = torch.bmm(a_t, g)  # [nc, P64, Cout]
            dw_t.index_add_(0, wl, dweff.reshape(-1, meta.cout).to(torch.float32))
        # Pad rows scattered into row 0 carry exact-zero contributions (pad activations
        # gather the zero pad column), so no spillway row is needed.
        return dw_t.t().contiguous().reshape(meta.cout, meta.q)


OPS_REGISTRY = {CompactTorchOps.name: CompactTorchOps}


def register_ops(ops) -> None:
    """Register a kernel backend's ops object (must expose name/forward/grad_input/grad_weight)."""
    OPS_REGISTRY[ops.name] = ops


# --------------------------------------------------------------------------------------
# The shared autograd Function
# --------------------------------------------------------------------------------------


class KNNConvFunction(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, x, weight, bias, meta, ops_name):
        from .knn_optimization import _autocast_dtype

        ops = OPS_REGISTRY[ops_name]
        compute_dtype = _autocast_dtype(x)
        with torch.autocast("cuda", enabled=False):
            x_c = x.contiguous().to(dtype=compute_dtype)
            w_c = weight.to(dtype=compute_dtype)
            b_c = bias.to(dtype=compute_dtype) if bias is not None else None
            y = ops.forward(meta, x_c, w_c, b_c)
        ctx.save_for_backward(x_c, w_c)
        ctx.meta = meta
        ctx.ops_name = ops_name
        ctx.grad_dtypes = (
            x.dtype,
            weight.dtype,
            bias.dtype if bias is not None else None,
        )
        return y

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_y):
        x_c, w_c = ctx.saved_tensors
        meta = ctx.meta
        ops = OPS_REGISTRY[ctx.ops_name]
        x_dtype, w_dtype, b_dtype = ctx.grad_dtypes
        grad_x = grad_w = grad_b = None
        g = grad_y.contiguous().to(dtype=x_c.dtype)
        # Optional fused entry: ops may expose backward_combined(meta, g, x, weight)
        # -> (grad_input, grad_weight), sharing one staging build across both grads.
        # Used only when both grads are needed.
        combined = getattr(ops, "backward_combined", None)
        if combined is not None and ctx.needs_input_grad[0] and ctx.needs_input_grad[1]:
            grad_x, grad_w = combined(meta, g, x_c, w_c)
            grad_x = grad_x.to(dtype=x_dtype)
            grad_w = grad_w.to(dtype=w_dtype)
        else:
            if ctx.needs_input_grad[0]:
                grad_x = ops.grad_input(meta, g, w_c).to(dtype=x_dtype)
            if ctx.needs_input_grad[1]:
                grad_w = ops.grad_weight(meta, g, x_c).to(dtype=w_dtype)
        if b_dtype is not None and ctx.needs_input_grad[2]:
            grad_b = grad_y.sum(dim=(0, 2), dtype=torch.float32).to(dtype=b_dtype)
        return grad_x, grad_w, grad_b, None, None


def compact_forward(layer, x: torch.Tensor, ops_name: str = "torch_compact") -> torch.Tensor:
    """Layer-level entry point for registry-backed backends (training and inference)."""
    meta = ensure_training_metadata(layer, x.device)
    return KNNConvFunction.apply(x, layer.weight, layer.bias, meta, ops_name)


# --------------------------------------------------------------------------------------
# torch_scatter: native-autograd replacement of the dense one-hot einsum
# --------------------------------------------------------------------------------------


def scatter_forward(layer, x: torch.Tensor) -> torch.Tensor:
    """Baseline-equivalent forward with ``scatter_add_`` instead of the one-hot einsum.

    Exact-arithmetic-identical to the baseline (padding neighbors gather the zero pad
    node and scatter zeros; rf-bin collisions sum, exactly as the einsum does), fully
    differentiable through native autograd, and valid for every dtype/device.
    """
    from .knn_optimization import _ensure_metadata

    rf_index, _, _ = _ensure_metadata(layer, x.device)
    features = layer._pad_and_gather_knns(x)  # [B, Cin, K, Nout]
    b, c, k, n = features.shape
    v = layer.local_rf.shape[2]
    features = features.permute(0, 1, 3, 2).reshape(b * c, n, k)
    index = rf_index.reshape(1, n, k).expand(b * c, n, k)
    binned = torch.zeros(b * c, n, v, device=x.device, dtype=features.dtype)
    binned = binned.scatter_add(2, index, features)
    binned = binned.reshape(b, c, n, v).permute(0, 2, 1, 3).reshape(b, n, c * v)
    return F.linear(binned, layer.weight, layer.bias).transpose(1, 2)
