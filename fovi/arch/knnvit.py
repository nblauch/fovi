import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np
import math

from .knn import get_in_out_coords
from .knn import KNNConvLayer
from .vit import VisionTransformer
from ..utils import add_to_all

__all__ = []


@add_to_all(__all__)
class KNNPatchEmbedding(KNNConvLayer):
    """KNN-based patch embedding layer that replaces standard patch embedding in Vision Transformers.
    
    Instead of dividing the image into uniform non-overlapping patches, this layer divides a foveated manifold into nearly non-overlapping KNNs to create patches. 

    It then performs a standard KNNConv operation for the patch embedding.

    We typically prefer KNNPartitioningPatchEmbedding, which builds on this, as it provides an optimal tiling of patches without any visual inspection.
    """
    def __init__(self, 
                 in_channels: int,
                 embed_dim: int,
                 in_res: int,
                 fov: float,
                 cmf_a: float,
                 style: str = 'isotropic',
                 auto_match_cart_resources: bool = True,
                 in_cart_res: int = 224,
                 cart_patch_size = 16,
                 patch_overlap_factor=1,
                 device='cuda',
                 force_patches_less_than_matched=True,
                 new_parameterization=False,
                 transposed=False,
                 max_coord_val=1,
                 sample_cortex='geodesic',
                 ref_frame_side_length=None,
                 **kwargs,
                 ):
        """Initialize KNN tokenization layer.
        
        Args:
            in_channels: Number of input channels
            embed_dim: Embedding dimension for tokens
            in_res: Input resolution
            fov: Field of view parameter for foveated sampling
            cmf_a: a parameter controlling foveated sampling via the CMF
            style: Sampling style ('isotropic', etc.)
            auto_match_cart_resources: Whether to automatically match cartesian resources
            in_cart_res: Resolution of input cartesian grid
            cart_patch_size: Size of cartesian patches
            patch_overlap_factor: Factor for patch overlap
            device: Device to run on
            force_patches_less_than_matched: Whether to force patches to be less than matched
            new_parameterization: Whether to use new parameterization
            transposed: Whether to transpose output
            max_coord_val: Maximum coordinate value
            sample_cortex: Cortex sampling method
            **kwargs: Additional arguments passed to parent class
        """    

        if new_parameterization:
            # k is fixed, resolution is adapted to match overlap factor
            stride = cart_patch_size/patch_overlap_factor
            k = cart_patch_size**2
        else:
            # resolution is fixed, k is adapted to match overlap factor
            stride = cart_patch_size

        in_coords, out_coords, out_cart_res = get_in_out_coords(in_res, fov, cmf_a, stride, style=style, auto_match_cart_resources=auto_match_cart_resources, in_cart_res=in_cart_res, device=device, force_out_match_less_than=force_patches_less_than_matched, max_out_coord_val=max_coord_val)

        if not new_parameterization:
            k = int((cart_patch_size*(out_cart_res)/(np.sqrt(len(out_coords)))*patch_overlap_factor)**2)
            # int((cart_patch_size*patch_overlap_factor)**2)
            print(k)
        super().__init__(in_channels, embed_dim, k, in_coords, out_coords, device=device, sample_cortex=sample_cortex, ref_frame_side_length=ref_frame_side_length, **kwargs)

        self.transposed = transposed

    def forward(self, x):
        # reshape tokens from grid if needed
        if len(x.shape) == 4:
            x = rearrange(x, 'b n h w -> b n (h w)')
        tokens = super().forward(x)
        # flip seq and embed dims to match standard transformer style
        if not self.transposed:
            tokens = rearrange(tokens, 'b d n -> b n d')

        return tokens
    def __repr__(self):
        repr_parent = super().__repr__()
        return repr_parent.replace('KNNConvLayer', 'KNNPatchEmbedding')


@add_to_all(__all__)
class PartitioningPatchEmbedding(KNNPatchEmbedding):
    """Partitioning patch embedding layer that replaces standard patch embedding in Vision Transformers.
    
    This layer divides a foveated manifold into non-overlapping neighborhoods to create patches. 

    It turns these neighborhoods into KNNs with padding and then performs a standard KNNConv operation for the patch embedding.
    """
    def __init__(self, 
                 in_channels: int,
                 embed_dim: int,
                 in_res: int,
                 fov: float,
                 cmf_a: float,
                 style: str = 'isotropic',
                 auto_match_cart_resources: bool = True,
                 force_patches_less_than_matched: bool = True,
                 in_cart_res: int = 224,
                 cart_patch_size = 16,
                 device='cuda',
                 transposed=False,
                 max_coord_val=1,
                 ref_frame_side_length=None,
                 sample_cortex='geodesic',
                 bias=False,
                 arch_flag='',
                 in_coords=None,
                 out_coords=None,
                 ):
        """Initialize partitioning patch embedding layer.
        
        Args:
            in_channels: Number of input channels
            embed_dim: Embedding dimension for tokens
            in_res: Input resolution
            fov: Field of view parameter for foveated sampling
            cmf_a: a parameter controlling foveated sampling via the CMF
            style: Sampling style ('isotropic', etc.)
            auto_match_cart_resources: Whether to automatically match cartesian resources
            force_patches_less_than_matched: Whether to force patches to be less than matched
            in_cart_res: Resolution of input cartesian grid
            cart_patch_size: Size of cartesian patches
            device: Device to run on
            transposed: Whether to transpose output
            max_coord_val: Maximum coordinate value
            ref_frame_side_length: Reference frame side length
            sample_cortex: Cortex sampling method
            bias: Whether to use bias in linear layer
            arch_flag: Architecture flag
        """    

        nn.Module.__init__(self)

        stride = cart_patch_size # match number of tokens exactly with cartesian version

        if in_coords is not None or out_coords is not None:
            assert in_coords is not None and out_coords is not None, "in_coords and out_coords must be provided if provided"
        else:
            in_coords, out_coords, out_cart_res = get_in_out_coords(in_res, fov, cmf_a, stride, style=style, auto_match_cart_resources=auto_match_cart_resources, in_cart_res=in_cart_res, device=device, force_out_match_less_than=force_patches_less_than_matched, max_out_coord_val=max_coord_val)

        self.in_channels = in_channels
        self.out_channels = embed_dim
        self.in_coords = in_coords
        self.out_coords = out_coords
        self.device = device
        self.arch_flag = arch_flag
        self.sample_cortex = sample_cortex
        self.ref_frame_side_length = ref_frame_side_length # if we want to specify manually
    
        # compute distances in visual or cortical space
        self.k = int(len(in_coords) / len(out_coords)) # set temporary k for use in geodesic dist computation, if necessary
        self.distances = self._compute_all_distances()
        
        # compute unit partitioning
        distances_nopad = self.distances[:(len(in_coords))] # don't need padding coords in partition
        self.partitioning = torch.argmin(distances_nopad, 1)
        rfs = []
        for ii in range(len(out_coords)):
            rfs.append(torch.argwhere(self.partitioning == ii).reshape(-1))
        max_len = torch.max(torch.tensor([len(rf) for rf in rfs]))
        # pad RFs up to max len. -1 is the "index padding" token, whereas len(in_coords) is the "spatial padding" token for KNNs
        self.knn_pad_index_token_val = -1
        self.knn_pad_token_val = self.in_coords.shape[0]
        indices = self.knn_pad_index_token_val*torch.ones((max_len, len(out_coords)), dtype=int, device=device) 
        # fill in RFs 
        for ii in range(len(out_coords)):
            indices[:len(rfs[ii]),ii] = rfs[ii]

        self.knn_indices = indices
        self.k = torch.tensor(max_len)
        self._k = int(self.k.item())

        print(f'minimum k to use all inputs: {self.k}')

        # compute padding mask for use at inference - compatibility with KNNConv
        self.knn_indices_pad_mask = torch.logical_or(self.knn_indices >= self.in_coords.shape[0], self.knn_indices < 0)
        self.knn_indices_pad_token = self.knn_indices.clone()
        self.knn_indices_pad_token[self.knn_indices_pad_mask] = self.knn_pad_token_val # pad index
        # this will be updated to the correct size for a batch to be used for batches of the same size
        self.knn_indices_batch_cache = self.knn_indices_pad_token

        # compute reference coordinates
        self.compute_reference_coords(self.arch_flag)
        
        # Initialize like a conv layer (Kaiming normal with fan_out mode)
        self.weight = nn.Parameter(torch.empty(self.out_channels, self.in_channels * self.ref_coords.shape[0]))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_channels))
        else:
            self.register_parameter('bias', None)
        
        self._init_conv_like()

        self.local_rf = self.compute_local_rf()

        self.transposed = transposed

    def __repr__(self):
        repr_parent = super().__repr__()
        return repr_parent.replace('KNNPatchEmbedding', 'PartitioningPatchEmbedding')
        # return f"KNNPatchEmbedding(in_channels={self.in_channels}, embed_dim={self.out_channels}, k={self.k})"

@add_to_all(__all__)
class KNNPartitioningPatchEmbedding(KNNPatchEmbedding):
    def __init__(self, 
                in_channels: int,
                embed_dim: int,
                in_res: int,
                fov: float,
                cmf_a: float,
                style: str = 'isotropic',
                auto_match_cart_resources: bool = True,
                in_cart_res: int = 224,
                cart_patch_size = 16,
                device='cuda',
                force_patches_less_than_matched=True,
                transposed=False,
                max_coord_val='auto',
                sample_cortex='geodesic',
                 **kwargs,
                 ):
        """Initialize KNN partitioning patch embedding layer.
        
        Args:
            in_channels: Number of input channels
            embed_dim: Embedding dimension for tokens
            in_res: Input resolution
            fov: Field of view parameter for foveated sampling
            cmf_a: a parameter controlling foveated sampling via the CMF
            style: Sampling style ('isotropic', etc.)
            auto_match_cart_resources: Whether to automatically match cartesian resources
            in_cart_res: Resolution of input cartesian grid
            cart_patch_size: Size of cartesian patches
            device: Device to run on
            force_patches_less_than_matched: Whether to force patches to be less than matched
            transposed: Whether to transpose output
            max_coord_val: Maximum coordinate value
            sample_cortex: Cortex sampling method
            **kwargs: Additional arguments passed to parent class
        """    

        stride = cart_patch_size # match number of tokens exactly with cartesian version

        in_coords, out_coords, out_cart_res = get_in_out_coords(in_res, fov, cmf_a, stride, style=style, auto_match_cart_resources=auto_match_cart_resources, in_cart_res=in_cart_res, device=device, force_out_match_less_than=force_patches_less_than_matched, max_out_coord_val=max_coord_val)

        # compute a partitioning, then use the maximum RF size to set k
        k = int(len(in_coords) / len(out_coords)) # set temporary k for use in geodesic dist computation, if necessary
        KNNConvLayer.__init__(self, in_channels, embed_dim, k, in_coords, out_coords, device=device, sample_cortex=sample_cortex, **kwargs) # temporary init to set distances
        self.distances = self._compute_all_distances()

        distances_nopad = self.distances[:(len(in_coords))] # don't need padding coords in partition
        partitioning = torch.argmin(distances_nopad, 1)
        rfs = []
        for ii in range(len(out_coords)):
            rf = torch.argwhere(partitioning == ii).squeeze()
            dists = self.distances[rf,ii]
            max_dist = dists.max()
            new_rf = torch.argwhere(self.distances[:,ii] < max_dist).reshape(-1)
            rfs.append(new_rf)
        k = torch.max(torch.tensor([len(rf) for rf in rfs])).item()

        KNNConvLayer.__init__(self, in_channels, embed_dim, k, in_coords, out_coords, device=device, sample_cortex=sample_cortex, **kwargs)

        print(f'minimum k to use all inputs: {k}')
        
        self.transposed = transposed 

    def __repr__(self):
        repr_parent = super().__repr__()
        return repr_parent.replace('KNNPatchEmbedding', 'KNNPartitioningPatchEmbedding')


@add_to_all(__all__) 
class KNNViT(VisionTransformer):
    """Vision Transformer that uses KNN-based tokenization instead of patch embedding.
    
    This model inherits from VisionTransformer and only overrides the patch embedding
    to use KNN-based tokenization that creates tokens based on spatial relationships 
    in the foveated coordinate system.
    """
    
    def __init__(self,
                 fov: float,
                 cmf_a: float,
                 style: str,
                 img_size: int = 224,
                 patch_size: int = 16,
                 patch_overlap_factor: float = 1,
                 in_channels: int = 3,
                 embed_dim: int = 768,
                 num_heads: int = 12,
                 num_layers: int = 12,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.0,
                 num_outputs: int = 1000,
                 device: str = 'cuda',
                 arch_flag: str = '',
                 sample_cortex: bool = 'geodesic',
                 pos_emb_type: str = 'absolute',
                 force_patches_less_than_matched = True,
                 attn_backend: str = 'flash',
                 aggregation='cls_token',
                 ref_frame_side_length=None,
                 ):
        """Initialize KNNViT model.
        
        Args:
            fov: Field of view parameter for foveated sampling
            cmf_a: a parameter controlling foveated sampling via the CMF; smaller = stronger foveation
            style: Sampling style ('isotropic', etc.)
            img_size: Size of input image
            patch_size: Size of each patch
            patch_overlap_factor: Factor for patch overlap
            in_channels: Number of input channels
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            mlp_ratio: Ratio of MLP hidden dim to embed dim
            dropout: Dropout rate
            num_outputs: Number of output classes
            device: Device to run on
            arch_flag: Architecture flag
            sample_cortex: Whether to sample in cortical space
            pos_emb_type: Type of positional embedding ('absolute' or 'rope')
            force_patches_less_than_matched: Whether to force the number of patches to be less than a matched cartesian model, or just match as close as possible
            attn_backend: Attention backend ('flash' for Flash Attention 2, 'standard' for standard implementation)
            ref_frame_side_length: side length of reference frame for KNN-convolution in the patch embedding (None defaults to patch_size)
        """
        # Store KNN-specific parameters
        self.fov = fov
        self.cmf_a = cmf_a
        self.style = style
        self.patch_overlap_factor = patch_overlap_factor
        self.arch_flag = arch_flag
        self.sample_cortex = sample_cortex
        
        # Create KNN patch embedding
        patch_embed = KNNPatchEmbedding(
            in_channels=in_channels,
            embed_dim=embed_dim,
            in_res=img_size,
            fov=fov,
            cmf_a=cmf_a,
            style=style,
            auto_match_cart_resources=True,
            in_cart_res=img_size,
            cart_patch_size=patch_size,
            device=device,
            arch_flag=arch_flag,
            sample_cortex=sample_cortex,
            patch_overlap_factor=patch_overlap_factor,
            force_patches_less_than_matched=force_patches_less_than_matched,
            ref_frame_side_length=ref_frame_side_length,
        )
        
        # Get cartesian coordinates for positional encoding
        cartesian_coords = patch_embed.out_coords.cartesian
        
        # Initialize parent class with custom patch embedding and positional encoding
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            num_outputs=num_outputs,
            pos_emb_type=pos_emb_type,
            coords=cartesian_coords,
            patch_embed=patch_embed,
            attn_backend=attn_backend,
            aggregation=aggregation,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through KNNViT.
        
        Args:
            x: Input features [batch_size, in_channels, in_coords]
            
        Returns:
            output: Model output [batch_size, 1, embed_dim]
        """
        # Use parent class forward method
        output = super().forward(x)
        
        # Format in expected manner: [batch_size, 1, embed_dim]
        output = output.unsqueeze(1)
        
        return output
    
    def __repr__(self):
        out = super().__repr__().replace('VisionTransformer', 'KNNViT')
        return out


@add_to_all(__all__)
class FoviDinoV3RoPE(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, base: int, head_dim: int, coords: torch.Tensor, device: str = 'cuda'):
        """Initialize DinoV3RoPE positional encoding.
        
        Args:
            base: Base frequency for RoPE
            head_dim: Dimension of attention head
            coords: Coordinate tensor
            device: Device to run on
        """
        super().__init__()

        self.base = base
        self.head_dim = head_dim
        self.coords = coords
        self.device = device

        inv_freq = 1 / self.base ** torch.arange(0, 1, 4 / self.head_dim, dtype=torch.float32, device=device)  # (head_dim / 4,)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for RoPE positional encoding.
        
        Args:
            pixel_values: Input pixel values
            
        Returns:
            Tuple of cosine and sine values for RoPE
        """
        device = pixel_values.device
        device_type = device.type if isinstance(device.type, str) and device.type != "mps" else "cpu"

        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            # (height * width, 2, head_dim / 4) -> (height * width, head_dim / 2) -> (height * width, head_dim)
            angles = 2 * math.pi * self.coords[:, :, None] * self.inv_freq[None, None, :]
            angles = angles.flatten(1, 2)
            angles = angles.tile(2)

            cos = torch.cos(angles)
            sin = torch.sin(angles)

        dtype = pixel_values.dtype
        return cos.to(dtype=dtype), sin.to(dtype=dtype)

@torch.no_grad()
@add_to_all(__all__)
def resample_patch_embed_conv(
    conv: nn.Conv2d,
    target_hw=(8, 8),
    mode: str = "bicubic",
    align_corners: bool = True,
    preserve_kernel_norm: bool = False,
) -> nn.Conv2d:
    """Resample a patch-embedding Conv2d's kernels to target size.
    
    Resamples a patch-embedding Conv2d's kernels from (kH, kW) -> target_hw and
    returns a NEW Conv2d with kernel_size=stride=target_hw. Supports both upsampling
    and downsampling.
    
    Args:
        conv (nn.Conv2d): The patch embedding convolution to resample.
        target_hw (tuple, optional): Target height and width. Defaults to (8, 8).
        mode (str, optional): Interpolation mode. Defaults to "bicubic".
        align_corners (bool, optional): Whether to align corners. Defaults to True.
        preserve_kernel_norm (bool, optional): Whether to preserve kernel norm. Defaults to False.
    
    Returns:
        nn.Conv2d: A new Conv2d layer with resampled kernels.
        
    Note:
        Assumes stride == kernel_size (patch embedding), padding == 0, groups == 1, dilation == 1.
    """
    assert isinstance(conv, nn.Conv2d)
    kH, kW = conv.kernel_size if isinstance(conv.kernel_size, tuple) else (conv.kernel_size, conv.kernel_size)
    sH, sW = conv.stride if isinstance(conv.stride, tuple) else (conv.stride, conv.stride)

    # sanity checks for patch embedding
    assert (kH, kW) == (sH, sW), "Expected stride == kernel_size for patch embedding."
    assert conv.padding == (0, 0) if isinstance(conv.padding, tuple) else conv.padding == 0, "Expected no padding."
    assert conv.groups == 1, "Expected groups == 1 for patch embedding."
    assert conv.dilation == (1, 1) if isinstance(conv.dilation, tuple) else conv.dilation == 1, "Expected dilation == 1."

    device = conv.weight.device
    dtype = conv.weight.dtype
    c_out, c_in, _, _ = conv.weight.shape
    
    # flatten kernels to NCHW for interpolation: N = c_out*c_in, C=1
    w = conv.weight.detach().clone()
    n = c_out * c_in
    w_flat = w.view(n, 1, kH, kW).to(dtype=dtype, device=device)

    # resample (works for both upsampling and downsampling)
    w_resampled = F.interpolate(w_flat, size=target_hw, mode=mode, align_corners=align_corners)

    if preserve_kernel_norm:
        eps = 1e-12
        old_norm = w_flat.view(n, -1).norm(dim=1, keepdim=True).clamp_min(eps)
        new_norm = w_resampled.view(n, -1).norm(dim=1, keepdim=True).clamp_min(eps)
        w_resampled = w_resampled * (old_norm / new_norm).view(n, 1, 1, 1)

    # restore shape
    w_resampled = w_resampled.view(c_out, c_in, target_hw[0], target_hw[1]).to(dtype=dtype, device=device)

    # build the new patch-embed conv with kernel_size == stride == target_hw
    new_conv = nn.Conv2d(
        in_channels=conv.in_channels,
        out_channels=conv.out_channels,
        kernel_size=target_hw,
        stride=target_hw,
        padding=0,
        bias=(conv.bias is not None),
        device=device,
        dtype=dtype,
    )
    new_conv.weight.copy_(w_resampled)
    if conv.bias is not None:
        new_conv.bias.copy_(conv.bias.detach())
    return new_conv