import torch
import torch.nn as nn
import torch.nn.utils.parametrize as P
import math
from . import add_to_all

__all__ = []

@add_to_all(__all__)
class LoRAParam(nn.Module):
    """Parametrization that adds low-rank updates to a weight matrix.
    
    Implements LoRA (Low-Rank Adaptation) by decomposing weight updates as:
        W_eff = W + (alpha/r) * (B @ A)
    
    For 2D weights (out, in), applies directly. For Conv weights (out, in, kH, kW),
    flattens to (out, in*kH*kW), applies BA, then reshapes back.
    
    Attributes:
        r (int): Rank of the low-rank decomposition.
        alpha (float): Scaling factor for the adaptation.
        scaling (float): Computed as alpha / r.
        is_conv (bool): Whether this is for a convolutional layer.
        B (nn.Parameter): Low-rank factor B of shape (out_dim, r).
        A (nn.Parameter): Low-rank factor A of shape (r, in_dim).
    """
    def __init__(self, weight_shape, r: int = 8, alpha: float = 8.0, init: str = "zeros", device='cuda'):
        """Initialize LoRA parametrization.
        
        Args:
            weight_shape (tuple): Shape of the weight to parametrize.
                Either (out, in) for Linear or (out, in, kH, kW) for Conv2d.
            r (int, optional): Rank of the low-rank matrices. Defaults to 8.
            alpha (float, optional): Scaling factor. Defaults to 8.0.
            init (str, optional): Initialization strategy. One of:
                - "zeros": Initialize both A and B to zeros.
                - "kaimingA_zeroB": Kaiming init for A, zeros for B.
                - "gaussian": Small Gaussian init for both.
                Defaults to "zeros".
            device (str, optional): Device for the parameters. Defaults to 'cuda'.
        """
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        if len(weight_shape) == 2:
            out_dim, in_dim = weight_shape
            self.flat_in_dim = in_dim
        elif len(weight_shape) == 3:
            # indexed conv: (out, in, n_ref)
            out_dim, in_dim, n = weight_shape
            self.flat_in_dim = in_dim * n
        elif len(weight_shape) == 4:
            # Conv2d: (out, in, kH, kW)
            out_dim, in_dim, kH, kW = weight_shape
            self.flat_in_dim = in_dim * kH * kW
        else:
            raise ValueError(f"Unsupported param shape {weight_shape}; expect (out,in) or (out,in,kH,kW).")

        # LoRA factors: B @ A has shape (out_dim, flat_in_dim)
        self.B = nn.Parameter(torch.zeros(out_dim, r, device=device))
        self.A = nn.Parameter(torch.zeros(r, self.flat_in_dim, device=device))

        # Common in LoRA: B zeros, A random small, or both zeros. Here: A kaiming, B zeros.
        if init == "zeros":
            nn.init.zeros_(self.A)
            nn.init.zeros_(self.B)
        elif init == "kaimingA_zeroB":
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            nn.init.zeros_(self.B)
        elif init == "gaussian":
            nn.init.normal_(self.A, std=1e-4)
            nn.init.normal_(self.B, std=1e-4)
        else:
            raise ValueError(f"Unknown init {init}")

    def forward(self, W_base: torch.Tensor) -> torch.Tensor:
        # W_base arrives as the original (frozen) weight tensor
        W_shape = W_base.shape
        delta = self.B @ self.A
        delta = delta.view(*W_shape)
        return W_base + self.scaling * delta

  
@add_to_all(__all__)
def apply_lora(module: nn.Module, param_name: str = "weight", r: int = 8, alpha: float = 8.0, init: str = "kaimingA_zeroB", device='cuda'):
    """
    Adds a LoRA parametrization to `module.<param_name>`. Freezes the base weight by default.
    Returns the parametrization object for convenience.
    """
    # Freeze base weight
    base = getattr(module, param_name)
    base.requires_grad_(False)

    lora = LoRAParam(base.shape, r=r, alpha=alpha, init=init, device=device)
    P.register_parametrization(module, param_name, lora)
    return lora


@add_to_all(__all__)
def remove_lora(module: nn.Module, param_name: str = "weight", merge: bool = True):
    """
    Removes LoRA parametrization. If merge=True, leaves the effective weight (folds LoRA into base).
    """
    P.remove_parametrizations(module, param_name, leave_parametrized=merge)