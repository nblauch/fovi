"""Optimized backends for :class:`KNNConvLayer` (inference and training).

All paths exploit the fact that ``local_rf`` is one-hot: each KNN neighbor selects exactly one
reference-grid weight. Inference-only backends (``torch_cached``/``warp_cached``/``warp_memory``)
cache derived operands keyed on the weight version. Training-capable backends
(``torch_scatter``/``torch_compact``, plus kernel backends registered in
:mod:`fovi.arch.knn_autograd`) provide autograd-correct forward/backward and are selected when
gradients are enabled.
"""

from __future__ import annotations

import importlib.util
import os

import torch
import torch.nn.functional as F


VALID_BACKENDS = {
    "auto",
    "baseline",
    "torch_cached",
    "warp_cached",
    "warp_memory",
    "torch_scatter",
    "torch_compact",
    "cuda",
    "warp_train",
    "gather_gemm",
}

# Backends with an autograd-correct backward; anything else falls back to baseline
# whenever gradients are enabled. Kernel tracks extend this as they register ops.
TRAIN_CAPABLE_BACKENDS = {"torch_scatter", "torch_compact", "cuda", "warp_train", "gather_gemm"}


def _is_gather_gemm_layer(layer) -> bool:
    """K=1/V=1 layers (resnet-style downsample convs) degenerate to gather + one dense GEMM."""
    return layer._k == 1 and layer.local_rf.shape[2] == 1


def _autocast_dtype(x: torch.Tensor) -> torch.dtype:
    if x.device.type == "cuda" and torch.is_autocast_enabled():
        try:
            return torch.get_autocast_dtype("cuda")
        except AttributeError:  # PyTorch < 2.4
            return torch.get_autocast_gpu_dtype()
    return x.dtype


def _warp_available() -> bool:
    return importlib.util.find_spec("warp") is not None


def _cuda_available() -> bool:
    return importlib.util.find_spec("cupy") is not None


# Engagement heuristic from the final tuning sweep:
# W = B*Cin*K*Nout cleanly separates every cell where the compact/kernel backends win
# from every cell where they lose, across the parametric resolution/channel sweep.
WORK_VOLUME_THRESHOLD = int(float(os.environ.get("FOVI_KNN_WORK_THRESHOLD", "1e7")))


def _work_volume(layer, x: torch.Tensor) -> int:
    return x.shape[0] * layer.in_channels * layer._k * layer.knn_indices_pad_token.shape[1]


def _parameter_signature(layer) -> tuple:
    return tuple((parameter.data_ptr(), parameter._version) for parameter in layer.parameters())


def _ensure_metadata(layer, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cached = getattr(layer, "_compact_knn_metadata", None)
    if cached is not None and cached[0].device == device:
        return cached

    local_rf = layer.local_rf.to(device=device)
    rf_index = local_rf.argmax(dim=2).to(torch.int64).contiguous()  # [Nout, K]
    indices = layer.knn_indices_pad_token.to(device=device)
    nout, k = rf_index.shape
    channels = torch.arange(layer.in_channels, device=device).reshape(1, layer.in_channels, 1)
    neighbors = indices.transpose(0, 1).reshape(nout, 1, k)
    input_width = layer.in_channels * layer.in_coords.shape[0]
    input_linear = torch.where(
        neighbors < layer.in_coords.shape[0],
        channels * layer.in_coords.shape[0] + neighbors,
        torch.full_like(neighbors, input_width),
    ).reshape(nout, layer.in_channels * k)
    weight_linear = (
        channels * local_rf.shape[2] + rf_index.reshape(nout, 1, k)
    ).reshape(nout, layer.in_channels * k)
    pad_p = (-input_linear.shape[1]) % 64
    input_linear = F.pad(input_linear, (0, pad_p), value=input_width).to(torch.int32).contiguous()
    weight_linear = F.pad(weight_linear, (0, pad_p), value=0).to(torch.int32).contiguous()
    cached = (rf_index, input_linear, weight_linear)
    layer._compact_knn_metadata = cached
    return cached


def clear_cache(layer) -> None:
    """Release cached effective weights and derived index metadata (inference + training)."""
    layer._compact_effective_weight_cache = None
    layer._compact_knn_metadata = None
    layer._knn_training_metadata = None
    layer._knn_pool_cuda_meta = None


def _effective_weight(layer, weight: torch.Tensor, rf_index: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    signature = (
        weight.device,
        dtype,
        tuple(weight.shape),
        _parameter_signature(layer),
    )
    cached = getattr(layer, "_compact_effective_weight_cache", None)
    if cached is not None and cached[0] == signature:
        return cached[1]

    cout = weight.shape[0]
    reference_points = weight.shape[1] // layer.in_channels
    compact = weight.detach().to(dtype=dtype).reshape(cout, layer.in_channels, reference_points)
    compact = compact[:, :, rf_index]
    compact = compact.permute(2, 1, 3, 0).reshape(
        rf_index.shape[0], layer.in_channels * rf_index.shape[1], cout
    )
    pad_p = (-compact.shape[1]) % 64
    # Pad output columns to the Warp tile width too: tile loads must never read past
    # the source extent (tile_load overread doctrine).
    pad_o = (-compact.shape[2]) % 64
    compact = F.pad(compact, (0, pad_o, 0, pad_p)).contiguous()
    layer._compact_effective_weight_cache = (signature, compact)
    return compact


def _torch_cached_forward(layer, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    rf_index, _, _ = _ensure_metadata(layer, x.device)
    target_dtype = _autocast_dtype(x)
    effective_weight = _effective_weight(layer, weight, rf_index, target_dtype)
    features = layer._pad_and_gather_knns(x)
    nout = rf_index.shape[0]
    features = features.permute(3, 0, 1, 2).reshape(
        nout, x.shape[0], layer.in_channels * rf_index.shape[1]
    )
    # The effective cache is padded to a multiple of 64 for the Warp path. Match it with cheap
    # zero padding here so both backends share one cache.
    features = F.pad(features, (0, effective_weight.shape[1] - features.shape[2]))
    output = torch.bmm(features, effective_weight).permute(1, 2, 0)[:, : layer.out_channels]
    if layer.bias is not None:
        output = output + layer.bias.detach().to(device=output.device, dtype=output.dtype).reshape(1, -1, 1)
    return output


def _warp_forward(layer, x: torch.Tensor, weight: torch.Tensor, cached_weight: bool) -> torch.Tensor:
    from .knn_warp import run_cached, run_uncached

    rf_index, input_linear, weight_linear = _ensure_metadata(layer, x.device)
    target_dtype = _autocast_dtype(x)
    if target_dtype != torch.float16:
        raise RuntimeError("Warp KNN convolution currently supports float16 inference only")
    x_half = x if x.dtype == torch.float16 else x.to(dtype=torch.float16)
    if layer.bias is None:
        bias = torch.zeros(layer.out_channels, device=x.device, dtype=torch.float16)
    else:
        bias = layer.bias.detach().to(device=x.device, dtype=torch.float16)
    if cached_weight:
        effective_weight = _effective_weight(layer, weight, rf_index, torch.float16)
        return run_cached(
            x_half, effective_weight, bias, input_linear, cout=layer.out_channels
        )
    return run_uncached(
        x_half,
        weight.detach().to(device=x.device, dtype=torch.float16),
        bias,
        input_linear,
        weight_linear,
    )


def select_backend(layer, x: torch.Tensor) -> str:
    requested = os.environ.get("FOVI_KNN_BACKEND", getattr(layer, "kernel_backend", "auto"))
    if requested not in VALID_BACKENDS:
        raise ValueError(f"Unknown KNN backend {requested!r}; expected one of {sorted(VALID_BACKENDS)}")
    if requested == "baseline":
        return "baseline"

    if requested == "cuda":
        if not (x.is_cuda and _autocast_dtype(x) in (torch.float16, torch.bfloat16)):
            raise RuntimeError(
                "cuda backend requires CUDA tensors and float16/bfloat16 (or matching autocast)"
            )
        return "cuda"

    if requested == "warp_train":
        if not (x.is_cuda and _autocast_dtype(x) == torch.float16 and _warp_available()):
            raise RuntimeError(
                "warp_train requires CUDA, float16 (or float16 autocast), and warp-lang"
            )
        return "warp_train"

    if requested == "gather_gemm":
        if not _is_gather_gemm_layer(layer):
            raise RuntimeError("gather_gemm requires a K=1, V=1 layer")
        return "gather_gemm"

    if torch.is_grad_enabled():
        if requested in TRAIN_CAPABLE_BACKENDS:
            return requested
        if requested != "auto":
            # Explicitly requested inference-only backend under autograd: fall back to baseline.
            return "baseline"
        if not x.is_cuda:
            return "baseline"
        # K=1/V=1 downsample layers: gather_gemm wins every measured cell at every
        # batch for all dtypes.
        if _is_gather_gemm_layer(layer):
            return "gather_gemm"
        # Final training policy: above the work
        # threshold, the fused CUDA backend wins every measured cell (3.4-8.1x over
        # baseline, 17-34x lower temp memory, most-accurate AMP grads); torch_compact
        # carries fp32 (cuda kernels are fp16/bf16-only) and serves as the fallback
        # when cupy is unavailable. Below the threshold the baseline's dense
        # tensor-core einsum wins. torch_scatter was dropped from auto (superseded).
        if _work_volume(layer, x) < WORK_VOLUME_THRESHOLD:
            return "baseline"
        target_dtype = _autocast_dtype(x)
        if target_dtype in (torch.float16, torch.bfloat16) and _cuda_available():
            return "cuda"
        return "torch_compact"

    target_dtype = _autocast_dtype(x)
    warp_compatible = x.is_cuda and target_dtype == torch.float16 and _warp_available()
    if requested.startswith("warp"):
        if not warp_compatible:
            raise RuntimeError(
                f"{requested} requires CUDA, float16 (or float16 autocast), and warp-lang"
            )
        return requested
    if requested in ("torch_cached", "torch_scatter", "torch_compact"):
        return requested

    # Final inference policy, plus the K=1/V=1 special case (wins at every batch,
    # all dtypes).
    if x.is_cuda and _is_gather_gemm_layer(layer):
        return "gather_gemm"
    if x.is_cuda and target_dtype in (torch.float16, torch.bfloat16):
        # The any-batch exception is validated for large-K stems (alex0, K=121,
        # 1.9-4.5x even at B=1-10) but does NOT transfer to smaller-K stems
        # (res18_stem, K=49: 0.67-0.78x at layer B=40) — gate on K.
        dense_low_cin = (
            layer.in_channels <= 4
            and layer.knn_indices_pad_token.shape[1] >= 256
            and layer._k >= 100
        )
        # cuda wins above the work threshold everywhere, and on dense low-Cin
        # large-K layers at ANY batch — except on sm_90/Hopper, where small-batch
        # cuda loses to the cached GEMM (measured on an H100 sweep): keep the
        # any-batch exception off that arch.
        if _cuda_available():
            engage = _work_volume(layer, x) >= WORK_VOLUME_THRESHOLD
            if not engage and dense_low_cin:
                engage = torch.cuda.get_device_capability(x.device) != (9, 0)
            if engage:
                return "cuda"
        # Small-work cells without cuda: dense cuBLAS keeps high-Cin layers EXCEPT
        # when the reference frame is super-resolved (V > K): baseline's einsum
        # inflates with V at inference and the compact formulation wins every
        # measured sub-threshold cell (rf2 pathology, e.g. l4_rf2
        # baseline 2.65-5.09 ms vs torch_compact 0.16). The cached compact GEMM
        # keeps low-Cin cells. NOTE: an earlier K>=100 gate on the cached cell
        # was REVERTED by later measurement — warp_cached on small-K stems is
        # an in-model win at small batch (launch-chain context beats the isolated
        # cell measurement); the cuda any-batch exception above keeps its K gate.
        if layer.in_channels > 4:
            if layer.local_rf.shape[2] > layer._k:
                return "torch_compact"
            return "baseline"
        if layer.knn_indices_pad_token.shape[1] < 256:
            return "torch_cached"
        return "warp_cached" if warp_compatible else "torch_cached"

    # fp32 (and CPU) inference keeps the original policy verbatim: high-channel layers
    # obtain excellent utilization from the dense cuBLAS GEMM; sparse patch embedding
    # does not move end-to-end ViT latency enough to justify its persistent cache.
    if layer.in_channels > 4:
        return "baseline"
    if layer.knn_indices_pad_token.shape[1] < 256:
        return "baseline"
    if warp_compatible:
        return "warp_cached"
    return "torch_cached"


def optimized_forward(layer, x: torch.Tensor) -> torch.Tensor | None:
    """Return an optimized result, or ``None`` when the baseline should run."""
    backend = select_backend(layer, x)
    layer._last_knn_backend = backend
    if backend == "baseline":
        return None
    weight = layer.weight
    if backend == "torch_scatter":
        from .knn_autograd import scatter_forward

        return scatter_forward(layer, x)
    if backend in TRAIN_CAPABLE_BACKENDS:
        if backend == "cuda":
            from . import knn_cuda  # noqa: F401  (import registers the "cuda" ops)
        elif backend == "warp_train":
            from . import knn_warp  # noqa: F401  (import registers the "warp_train" ops)
        elif backend == "gather_gemm":
            from . import knn_gather_gemm  # noqa: F401  (import registers the ops)
        from .knn_autograd import compact_forward

        return compact_forward(layer, x, ops_name=backend)
    if backend == "torch_cached":
        return _torch_cached_forward(layer, x, weight)
    if backend == "warp_cached":
        return _warp_forward(layer, x, weight, cached_weight=True)
    if backend == "warp_memory":
        return _warp_forward(layer, x, weight, cached_weight=False)
    raise AssertionError(f"Unhandled backend {backend}")
