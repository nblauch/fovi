from fvcore.nn import FlopCountAnalysis
from torch.nn.utils import parametrize as P
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from fvcore.nn.jit_handles import get_shape
import time
import torch
from contextlib import nullcontext
from statistics import mean, median

from .. import get_trainer_from_base_fn
from . import add_to_all

__all__ = []

@torch.no_grad()
def remove_parametrizations(trainer):
    """Remove LoRA parametrizations from a model for FLOP counting.
    
    LoRA adds trainable low-rank matrices to existing weights. This function
    folds those parametrizations into the base weights so FLOP counting
    reflects the effective model without parametrization overhead.
    
    Args:
        trainer: Trainer object containing the model and config with LoRA settings.
    """
    cfg = trainer.cfg
    fovinet = trainer.model
    if cfg.pretrained_model.lora.layers is not None:
        for ii in cfg.pretrained_model.lora.layers:
            if ii == -1:
                # patch embedding -- just has a single weight, no sublayers
                layer = fovinet.network.backbone.embeddings.patch_embeddings
                P.remove_parametrizations(layer, 'weight', leave_parametrized=True)
                continue
            else:
                layer = fovinet.network.backbone.layer[ii]
            for sublayer in cfg.pretrained_model.lora.sublayers:
                parent, child = sublayer.split('.')
                P.remove_parametrizations(getattr(getattr(layer, parent), child), 'weight', leave_parametrized=True)

def _as_value(x):
    # Return the first torch._C.Value whether x is a single Value or a list/tuple of them
    return x[0] if isinstance(x, (list, tuple)) else x

def _int_list_from_value(v):
    """
    Returns a list[int] from JIT Value that’s either:
      - prim::Constant with int or tuple/list of ints
      - prim::ListConstruct of prim::Constant ints
    """
    try:
        n = v.node()
        k = n.kind()
        if k == "prim::Constant":
            iv = n.toIValue()
            if isinstance(iv, int):
                return [int(iv)]
            if isinstance(iv, (list, tuple)):
                return [int(x) for x in iv]
        if k == "prim::ListConstruct":
            vals = []
            for inp in n.inputs():
                vals.append(int(inp.node().toIValue()))
            return vals
    except Exception:
        pass
    return None

def isnan_flops(inputs, outputs):
    return elemwise_flops(inputs, outputs)

def nanmean_flops(inputs, outputs):
    # reduction over the input tensor (masking cost usually negligible vs matmuls)
    return reduction_flops(inputs, outputs)

def pad_flops(inputs, outputs):
    # padding is data movement; count 0 FLOPs
    return 0

def max_pool2d_precise_flops(inputs, outputs):
    """
    FLOPs for max pool = number of comparisons.
    Each output element is the max over a kH x kW window: (kH*kW - 1) comparisons.
    Total = (kH*kW - 1) * B * C * Hout * Wout
    """
    v_out = _as_value(outputs)
    out_shape = get_shape(v_out)  # [B, C, Hout, Wout]
    if not out_shape or len(out_shape) < 4:
        # fallback: 1 compare per output element
        return elemwise_flops(inputs, outputs)

    B, C, Hout, Wout = map(int, out_shape)

    # aten::max_pool2d signature:
    # (Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, int[2] dilation, bool ceil_mode)
    ks = _int_list_from_value(inputs[1]) if len(inputs) > 1 else None
    if not ks:
        # If stride is None it defaults to kernel_size; but if we can’t parse ks, fallback
        return elemwise_flops(inputs, outputs)

    if len(ks) == 1:
        kH = kW = ks[0]
    else:
        kH, kW = ks[:2]

    per_out_compares = max(kH * kW - 1, 0)
    return per_out_compares * B * C * Hout * Wout

def _numel_value(v):
    shape = get_shape(v)  # returns a list/tuple of ints or None
    if not shape:         # None or []
        return 0
    n = 1
    for s in shape:
        n *= int(s)
    return n

def zero_flops(inputs, outputs):
    return 0

def elemwise_flops(inputs, outputs):
    # Count per output element (cheap default for pointwise ops)
    v = _as_value(outputs)
    return _numel_value(v)

def reduction_flops(inputs, outputs):
    # Count per input element (cheap default for reductions)
    v = _as_value(inputs)
    return _numel_value(v)

def sdpa_flops(inputs, outputs):
    # aten::scaled_dot_product_attention(q, k, v, ...)
    q = _as_value(inputs)          # q
    k = inputs[1] if isinstance(inputs, (list, tuple)) else None
    v = inputs[2] if isinstance(inputs, (list, tuple)) else None

    q_shape = get_shape(q)   # [B, H, S_q, D]
    k_shape = get_shape(k)   # [B, H, S_k, D]
    v_shape = get_shape(v)   # [B, H, S_k, D_v]

    if not q_shape or not k_shape or not v_shape:
        return 0  # fall back if shape unknown

    B, H, S_q, D   = map(int, q_shape)
    _, _, S_k, Dk  = map(int, k_shape)
    _, _, _, D_v   = map(int, v_shape)

    # MACs convention (1 MAC = 1 op). If you want FLOPs=2*MACs, multiply by 2 afterwards.
    macs_qk = B * H * S_q * S_k * D        # Q @ K^T
    macs_av = B * H * S_q * S_k * D_v      # softmax(QK) @ V
    softmax = B * H * S_q * S_k            # optional, small vs matmuls

    return macs_qk + macs_av + softmax


@add_to_all(__all__)
def make_flop_counter(model, inputs, *, include_pointwise=True, include_reductions=True):
    """Create a FLOP counter with custom operation handlers.
    
    Extends fvcore's FlopCountAnalysis with handlers for common operations
    that aren't covered by default, including attention, pooling, and
    various element-wise operations.
    
    Args:
        model (nn.Module): The model to analyze.
        inputs: Input tensor(s) to trace the model with.
        include_pointwise (bool, optional): Whether to count pointwise ops
            (add, mul, div, etc.) as 1 FLOP per element. Defaults to True.
        include_reductions (bool, optional): Whether to count reduction ops
            (sum, mean, min) as 1 FLOP per input element. Defaults to True.
            
    Returns:
        FlopCountAnalysis: Configured FLOP counter. Call .total() to get count.
    """
    flops = FlopCountAnalysis(model, inputs)

    # 0-FLOP ops: allocations/meta
    for op in [
        "aten::lift_fresh", "aten::clone", "aten::new_ones",
        "aten::ones_like", "aten::tile", "aten::repeat", "aten::index_copy",
        "aten::meshgrid"
    ]:
        flops = flops.set_op_handle(op, zero_flops)

    # Pointwise (1 per output element by default)
    if include_pointwise:
        for op in [
            "aten::add", "aten::add_", "aten::sub", "aten::sub_",
            "aten::mul", "aten::mul_", "aten::div", "aten::div_",
            "aten::rsub", "aten::neg", "aten::where",
            "aten::sin", "aten::cos", "aten::gelu", "aten::silu", "aten::silu_", "aten::exp",
        ]:
            flops = flops.set_op_handle(op, elemwise_flops)

    # Reductions (1 per input element by default)
    if include_reductions:
        for op in ["aten::sum", "aten::mean", "aten::min"]:
            flops = flops.set_op_handle(op, reduction_flops)

    # Randomness / dropout
    for op in ["aten::bernoulli_", "aten::uniform_"]:
        flops = flops.set_op_handle(op, zero_flops)  # or elemwise_flops if you prefer

    # Scaled Dot-Product Attention
    flops = flops.set_op_handle("aten::scaled_dot_product_attention", sdpa_flops)

    # Indexing/select (approximate by output size)
    flops = flops.set_op_handle("aten::index_select", elemwise_flops)

    # elementwise not-equal
    flops = flops.set_op_handle("aten::ne", elemwise_flops) # 1 per element

    # simple approximation for elementwise floating‐point modulus
    flops = flops.set_op_handle("aten::fmod",  elemwise_flops) # simple: 1/elt
    flops = flops.set_op_handle("aten::fmod_",  elemwise_flops) # simple: 1/elt

    # nan processing
    flops = flops.set_op_handle("aten::isnan", isnan_flops)
    flops = flops.set_op_handle("aten::nanmean", nanmean_flops)

    # padding
    flops = flops.set_op_handle("aten::pad", pad_flops)

    # max pool
    flops = flops.set_op_handle("aten::max_pool2d", max_pool2d_precise_flops)


    return flops


@add_to_all(__all__)
class FlopWrapper(nn.Module):
    """Wrapper module for FLOP counting of a trainer's model.
    
    Prepares a model for FLOP analysis by removing LoRA parametrizations
    and freezing all parameters.
    
    Args:
        trainer: Trainer object containing the model to wrap.
        setting (str, optional): Forward pass setting (e.g., 'supervised',
            'self-supervised'). Defaults to 'supervised'.
        **kwargs: Additional keyword arguments passed to model forward.
            
    Attributes:
        trainer: The trainer object.
        kwargs (dict): Keyword arguments for the forward pass.
    """
    def __init__(self, trainer, setting='supervised', **kwargs):
        super().__init__()
        self.trainer = trainer
        # remove parametrizations: we don't want these to influence flops, since they can easily be removed after training
        remove_parametrizations(self.trainer)
        # Set requires_grad to False for all parameters in the model
        for param in self.trainer.model.parameters():
            param.requires_grad = False
        self.kwargs = kwargs
        self.kwargs['setting'] = setting

    def get_inputs(self, loader):
        """Get a batch of inputs from a data loader.
        
        Args:
            loader: DataLoader to get inputs from.
            
        Returns:
            torch.Tensor: First element (images) from the first batch.
        """
        for batch in loader:
            break
        return batch[0]

    @torch.no_grad
    def forward(self, inputs):
        """Forward pass through the wrapped model.
        
        Args:
            inputs (torch.Tensor): Input tensor.
            
        Returns:
            Model outputs.
        """
        outputs = self.trainer.model(
            inputs,
            **self.kwargs,
        )
        return outputs


@add_to_all(__all__)
def measure_latency(
    model,
    inputs,
    *,
    device='cuda',
    warmup=20,
    iters=100,
    use_autocast=True,           # set to "fp16" or True to enable autocast on CUDA
    use_inference_mode=True,      # no grad + a few micro-optimizations
    cudnn_benchmark=True,         # helps for fixed input sizes
    measure_memory=False,         # also track peak GPU memory
    add_dummy_backward=False,
):
    """Measure model inference latency with detailed statistics.
    
    Performs warmup iterations followed by timed iterations, collecting
    latency percentiles and optionally memory usage.
    
    Args:
        model (nn.Module): Model to benchmark.
        inputs: Input tensor or tuple of tensors for the model.
        device (str, optional): Device to run on. Defaults to 'cuda'.
        warmup (int, optional): Number of warmup iterations. Defaults to 20.
        iters (int, optional): Number of timed iterations. Defaults to 100.
        use_autocast (bool or str, optional): Enable autocast. True or "fp16"
            for float16, "bf16" for bfloat16. Defaults to True.
        use_inference_mode (bool, optional): Use torch.inference_mode for
            micro-optimizations. Defaults to True.
        cudnn_benchmark (bool, optional): Enable cuDNN benchmark mode.
            Defaults to True.
        measure_memory (bool, optional): Track peak GPU memory per iteration.
            Defaults to False.
        add_dummy_backward (bool, optional): Include a dummy backward pass
            to measure training latency. Defaults to False.
            
    Returns:
        dict: Dictionary containing latency statistics:
            - mean_ms, median_ms, p90_ms, p95_ms, p99_ms, min_ms, max_ms
            - iters, warmup, device, autocast, dtype
            - peak_memory_mb, mean_memory_mb (if measure_memory=True)
    """
    # device = device or next(model.parameters()).device
    if isinstance(device, str):
        device = torch.device(device)
    model = model.to(device)

    def to_device(x):
        if isinstance(x, torch.Tensor):
            return x.to(device)
        elif isinstance(x, list):
            return [to_device(item) for item in x]
        elif isinstance(x, dict):
            return {k: to_device(v) for k, v in x.items()}
        elif isinstance(x, tuple):
            return tuple(to_device(item) for item in x)
        else:
            raise ValueError(f"Unsupported type: {type(x)}")
    
    inputs = to_device(inputs)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
        if measure_memory:
            torch.cuda.reset_peak_memory_stats(device)

    # Choose contexts
    amp_dtype = torch.float16 if use_autocast in (True, "fp16") else torch.bfloat16 if use_autocast == "bf16" else None
    amp_ctx = (torch.autocast(device_type="cuda", dtype=amp_dtype) if (device.type == "cuda" and amp_dtype)
               else nullcontext())
    if add_dummy_backward:
        infer_ctx = nullcontext()
    else:
        infer_ctx = torch.inference_mode() if use_inference_mode else torch.no_grad()

    # Warmup
    with infer_ctx, amp_ctx:
        for _ in range(warmup):
            _ = model(*inputs) if isinstance(inputs, tuple) else model(inputs)
            if device.type == "cuda":
                torch.cuda.synchronize()

    # Reset memory stats after warmup
    if device.type == "cuda" and measure_memory:
        torch.cuda.reset_peak_memory_stats(device)

    # Measure
    times_ms = []
    peak_memories = []
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    with infer_ctx, amp_ctx:
        for _ in range(iters):
            if measure_memory:
                torch.cuda.reset_peak_memory_stats(device)
            starter.record()
            if add_dummy_backward:
                model.train()
            else:
                model.eval()
            out = model(*inputs) if isinstance(inputs, tuple) else model(inputs)
            if add_dummy_backward:
                model.zero_grad(set_to_none=True)
                dummy_target = torch.randn_like(out)
                loss = F.mse_loss(out, dummy_target)
                loss.backward()
                model.zero_grad(set_to_none=True)
            ender.record()
            torch.cuda.synchronize()
            times_ms.append(starter.elapsed_time(ender))  # ms
            if measure_memory:
                peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # MB
                peak_memories.append(peak_mem)

    times_ms.sort()
    p50 = median(times_ms)
    p90 = times_ms[int(0.90 * (len(times_ms)-1))]
    p95 = times_ms[int(0.95 * (len(times_ms)-1))]
    p99 = times_ms[int(0.99 * (len(times_ms)-1))]
    
    result = {
        "mean_ms": mean(times_ms),
        "median_ms": p50,
        "p90_ms": p90,
        "p95_ms": p95,
        "p99_ms": p99,
        "min_ms": times_ms[0],
        "max_ms": times_ms[-1],
        "iters": iters,
        "warmup": warmup,
        "device": str(device),
        "autocast": bool(amp_dtype),
        "dtype": str(amp_dtype) if amp_dtype else "none",
    }
    
    if measure_memory and peak_memories:
        result["peak_memory_mb"] = max(peak_memories)
        result["mean_memory_mb"] = mean(peak_memories)
    
    return result
    

@add_to_all(__all__)
def get_flops_df(runs_df, include_keys, compute_latency=False, compute_memory=False, n_fixations=None, quiet=True, **kwargs):
    """Compute FLOP counts and optionally latency/memory for multiple model runs.
    
    Iterates through a DataFrame of experimental runs, loads each model,
    and computes computational metrics.
    
    Args:
        runs_df (pd.DataFrame): DataFrame with run information, must contain
            'logging.base_fn' column with paths to model checkpoints.
        include_keys (list): List of column keys from runs_df to include
            in the output DataFrame.
        compute_latency (bool, optional): Whether to measure latency.
            Defaults to False.
        compute_memory (bool, optional): Whether to measure peak memory.
            Defaults to False.
        n_fixations (int, optional): what # of fixations to gather stats for
        **kwargs: Additional keyword arguments passed to get_trainer_from_base_fn.
            
    Returns:
        pd.DataFrame: DataFrame with GFLOPS, num_fixations, patches/fix,
            pixels/fix, GFLOPS/img, GFLOPS/img*fix, and optionally latency
            and memory columns, plus requested include_keys.
    """
    flops_df = {'GFLOPS':[], 'num_fixations':[], 'patches/fix':[], 'pixels/fix':[], 'GFLOPS/img':[], 'GFLOPS/img*fix':[], 'accuracy @ nfix':[]}
    if compute_latency:
        flops_df['latency (ms)'] = []
    if compute_memory:
        flops_df['peak_memory (MB)'] = []
    for key in include_keys:
        short_key = key.split('.')[-1]
        flops_df[short_key] = []
    for ii, row in runs_df.iterrows():
        base_fn = row['logging.base_fn']
        trainer = get_trainer_from_base_fn(base_fn, quiet=quiet, load=True, **kwargs)
        cfg = trainer.cfg
        if n_fixations is None:
            use_n_fixations = max(trainer.n_fixations_val)
        else:
            use_n_fixations = n_fixations

        assert f'top_1_val_nfix-{use_n_fixations}' in row, 'wrong number of fixations available in row'

        wrapper = FlopWrapper(trainer, **{'n_fixations': use_n_fixations})
        inputs = wrapper.get_inputs(trainer.val_loader)
        flops = make_flop_counter(wrapper, (inputs,))   # or inputs tuple matching your 
        gflops = flops.total() / 1e9
        try:
            # KNN patch embedding
            num_patches = len(trainer.model.network.backbone.embeddings.patch_embeddings.out_coords)
            num_pix = len(trainer.model.network.backbone.embeddings.patch_embeddings.in_coords)
        except:
            # standard patch embedding
            num_patches = (cfg.saccades.resize_size // cfg.model.vit.patch_size)**2
            num_pix = cfg.saccades.resize_size**2
        flops_df['GFLOPS'].append(gflops)
        flops_df['num_fixations'].append(use_n_fixations)
        flops_df['patches/fix'].append(num_patches)
        flops_df['pixels/fix'].append(num_pix)
        flops_df['GFLOPS/img'].append(gflops/inputs.shape[0])
        flops_df['GFLOPS/img*fix'].append(gflops/(inputs.shape[0] * use_n_fixations))
        flops_df[f'accuracy @ nfix'].append(row[f'top_1_val_nfix-{use_n_fixations}'])
        if compute_latency or compute_memory:
            stats = measure_latency(wrapper, inputs, measure_memory=compute_memory)
            if compute_latency:
                flops_df['latency (ms)'].append(stats['median_ms'])
            if compute_memory:
                flops_df['peak_memory (MB)'].append(stats['peak_memory_mb'])
        for key in include_keys:
            short_key = key.split('.')[-1]
            flops_df[short_key].append(row[key])
    flops_df = pd.DataFrame(flops_df)
    return flops_df