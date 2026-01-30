import torch
import torch.nn as nn
from typing import Optional, Tuple
import math
from ..utils import add_to_all

# Try to import flash attention, but allow fallback if not available
try:
    from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    flash_attn_qkvpacked_func = None
    flash_attn_func = None

__all__ = []


@add_to_all(__all__)
def apply_2d_rotary_pos_emb(q, k, cos_x, sin_x, cos_y, sin_y):
    """
    Apply 2D rotary position embeddings to query and key tensors.
    
    Args:
        q: Query tensor [batch_size, num_heads, seq_len, head_dim]
        k: Key tensor [batch_size, num_heads, seq_len, head_dim]
        cos_x: Cosine embeddings for x-coordinate [seq_len, half_head_dim]
        sin_x: Sine embeddings for x-coordinate [seq_len, half_head_dim]
        cos_y: Cosine embeddings for y-coordinate [seq_len, half_head_dim]
        sin_y: Sine embeddings for y-coordinate [seq_len, half_head_dim]
        
    Returns:
        q_rot, k_rot: Query and key tensors with 2D rotary embeddings applied
    """
    # Expand cos and sin embeddings for broadcasting
    cos_x = cos_x.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, half_head_dim]
    sin_x = sin_x.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, half_head_dim]
    cos_y = cos_y.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, half_head_dim]
    sin_y = sin_y.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, half_head_dim]
    
    # Split head dimension into x and y components (first half for x, second half for y)
    head_dim = q.shape[-1]
    half_head_dim = head_dim // 2
    
    q_x, q_y = q[..., :half_head_dim], q[..., half_head_dim:]
    k_x, k_y = k[..., :half_head_dim], k[..., half_head_dim:]
    
    # Apply 2D rotary embeddings
    q_rot_x = (q_x * cos_x) + (torch.roll(q_x, shifts=1, dims=-1) * sin_x)
    q_rot_y = (q_y * cos_y) + (torch.roll(q_y, shifts=1, dims=-1) * sin_y)
    k_rot_x = (k_x * cos_x) + (torch.roll(k_x, shifts=1, dims=-1) * sin_x)
    k_rot_y = (k_y * cos_y) + (torch.roll(k_y, shifts=1, dims=-1) * sin_y)
    
    # Concatenate x and y components
    q_rot = torch.cat([q_rot_x, q_rot_y], dim=-1)
    k_rot = torch.cat([k_rot_x, k_rot_y], dim=-1)
    
    return q_rot, k_rot


@add_to_all(__all__)
class PositionalEncoding(nn.Module):
    """
    Positional encoding based on xy coordinates.
    
    This creates positional encodings using the xy coordinates of each patch,
    allowing the model to understand spatial relationships in the original image space.
    
    When coords=None, grid coordinates are computed from num_patches_h and num_patches_w.
    """
    
    def __init__(self, embed_dim: int, coords: Optional[torch.Tensor] = None, 
                 num_patches_h: Optional[int] = None, num_patches_w: Optional[int] = None,
                 device: str = 'cuda'):
        """
        Initialize xy positional encoding.
        
        Args:
            embed_dim: Embedding dimension
            coords: xy coordinates of patches [num_patches, 2]. If None, computed from grid dims.
            num_patches_h: Number of patches in height (required if coords is None)
            num_patches_w: Number of patches in width (required if coords is None)
            device: Device to run on
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Compute grid coordinates if not provided
        if coords is None:
            assert num_patches_h is not None and num_patches_w is not None, \
                "num_patches_h and num_patches_w required when coords is None"
            # Create grid coordinates
            rows = torch.arange(num_patches_h, device=device, dtype=torch.float32)
            cols = torch.arange(num_patches_w, device=device, dtype=torch.float32)
            grid_y, grid_x = torch.meshgrid(rows, cols, indexing='ij')
            coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
        
        self.num_patches = coords.shape[0]
        
        # Normalize xy coordinates to [0, 1] range
        coords_min = coords.min(dim=0, keepdim=True)[0]
        coords_max = coords.max(dim=0, keepdim=True)[0]
        normalized_coords = (coords - coords_min) / (coords_max - coords_min + 1e-8)
        
        # Create positional encoding using sine/cosine functions
        pos_encoding = torch.zeros(self.num_patches, embed_dim, device=device)
        
        # Use different frequencies for x and y coordinates
        for i in range(embed_dim // 2):
            freq_x = 1.0 / (10000 ** (2 * i / embed_dim))
            freq_y = 1.0 / (10000 ** (2 * i / embed_dim))
            
            pos_encoding[:, 2*i] = torch.sin(normalized_coords[:, 0] * freq_x)
            pos_encoding[:, 2*i + 1] = torch.cos(normalized_coords[:, 1] * freq_y)
        
        # If embed_dim is odd, add one more dimension
        if embed_dim % 2 == 1:
            freq = 1.0 / (10000 ** (2 * (embed_dim // 2) / embed_dim))
            pos_encoding[:, -1] = torch.sin(normalized_coords[:, 0] * freq)
        
        # Register as buffer so it's saved with the model
        self.register_buffer('pos_encoding', pos_encoding)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input tokens.
        
        Args:
            x: Input tokens [batch_size, num_patches, embed_dim]
            
        Returns:
            Tokens with positional encoding added
        """
        return x + self.pos_encoding.unsqueeze(0)


@add_to_all(__all__)
class RoPEPositionalEncoding(nn.Module):
    """
    2D Rotary Position Embeddings (RoPE) based on xy coordinates.
    
    This applies 2D rotary embeddings using the spatial coordinates of each patch,
    allowing the model to understand 2D spatial relationships in the original image space.
    
    Note: This is designed to work with patch tokens only. The CLS token should be handled
    separately in the attention layer.
    
    When coords=None, grid coordinates are computed from num_patches_h and num_patches_w.
    """
    
    def __init__(self, embed_dim: int, num_heads: int, coords: Optional[torch.Tensor] = None,
                 num_patches_h: Optional[int] = None, num_patches_w: Optional[int] = None,
                 device: str = 'cuda'):
        """
        Initialize 2D RoPE positional encoding.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            coords: XY coordinates of patches [num_patches, 2]. If None, computed from grid dims.
            num_patches_h: Number of patches in height (required if coords is None)
            num_patches_w: Number of patches in width (required if coords is None)
            device: Device to run on
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # Correct head dimension

        assert self.embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        assert self.embed_dim % (num_heads*2) == 0, "embed_dim must be divisible by num_heads*2 for rope"
        assert self.embed_dim % (2*num_heads*2) == 0, "embed_dim must be divisible by 2*num_heads*2 for rope"
        
        # Compute grid coordinates if not provided
        if coords is None:
            assert num_patches_h is not None and num_patches_w is not None, \
                "num_patches_h and num_patches_w required when coords is None"
            # Create grid coordinates
            rows = torch.arange(num_patches_h, device=device, dtype=torch.float32)
            cols = torch.arange(num_patches_w, device=device, dtype=torch.float32)
            grid_y, grid_x = torch.meshgrid(rows, cols, indexing='ij')
            coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
        
        self.num_patches = coords.shape[0]
        
        # Normalize coordinates to [0, 1] range
        coords_min = coords.min(dim=0, keepdim=True)[0]
        coords_max = coords.max(dim=0, keepdim=True)[0]
        normalized_coords = (coords - coords_min) / (coords_max - coords_min + 1e-8)
        
        # Create 2D position embeddings
        position_x = normalized_coords[:, 0].unsqueeze(1) #.float()  # [num_patches, 1]
        position_y = normalized_coords[:, 1].unsqueeze(1) #.float()  # [num_patches, 1]
        
        # Create frequency terms for x and y coordinates (half of head_dim for each)
        half_head_dim = self.head_dim // 2
        div_term_x = torch.exp(torch.arange(0, half_head_dim, 2, device=device) * 
                             -(math.log(10000.0) / half_head_dim))
        div_term_y = torch.exp(torch.arange(0, half_head_dim, 2, device=device) * 
                             -(math.log(10000.0) / half_head_dim))
        
        # Create cos and sin embeddings for x and y coordinates
        cos_x_emb = torch.zeros(self.num_patches, half_head_dim, device=device)
        sin_x_emb = torch.zeros(self.num_patches, half_head_dim, device=device)
        cos_y_emb = torch.zeros(self.num_patches, half_head_dim, device=device)
        sin_y_emb = torch.zeros(self.num_patches, half_head_dim, device=device)
        
        # Fill x-coordinate embeddings
        cos_x_emb[:, 0::2] = torch.cos(position_x * div_term_x)
        cos_x_emb[:, 1::2] = torch.cos(position_x * div_term_x)
        sin_x_emb[:, 0::2] = torch.sin(position_x * div_term_x)
        sin_x_emb[:, 1::2] = torch.sin(position_x * div_term_x)
        
        # Fill y-coordinate embeddings
        cos_y_emb[:, 0::2] = torch.cos(position_y * div_term_y)
        cos_y_emb[:, 1::2] = torch.cos(position_y * div_term_y)
        sin_y_emb[:, 0::2] = torch.sin(position_y * div_term_y)
        sin_y_emb[:, 1::2] = torch.sin(position_y * div_term_y)
        
        # Register as buffers so they're saved with the model
        self.register_buffer('cos_x_emb', cos_x_emb)
        self.register_buffer('sin_x_emb', sin_x_emb)
        self.register_buffer('cos_y_emb', cos_y_emb)
        self.register_buffer('sin_y_emb', sin_y_emb)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply 2D RoPE to query and key tensors.
        
        Args:
            q: Query tensor [batch_size, num_heads, seq_len, head_dim] (patch tokens only)
            k: Key tensor [batch_size, num_heads, seq_len, head_dim] (patch tokens only)
            
        Returns:
            q_rot, k_rot: Query and key tensors with 2D RoPE applied
        """
        # For fixed-length sequences, we use all embeddings in order
        cos_x = self.cos_x_emb  # [num_patches, half_head_dim]
        sin_x = self.sin_x_emb  # [num_patches, half_head_dim]
        cos_y = self.cos_y_emb  # [num_patches, half_head_dim]
        sin_y = self.sin_y_emb  # [num_patches, half_head_dim]
        
        # Apply 2D rotary embeddings
        q_rot, k_rot = apply_2d_rotary_pos_emb(q, k, cos_x.clone(), sin_x.clone(), cos_y.clone(), sin_y.clone())
        
        q_rot = q_rot.to(q.dtype)
        k_rot = k_rot.to(k.dtype)
        
        return q_rot, k_rot


@add_to_all(__all__)
class PatchEmbedding(nn.Module):
    """
    Standard patch embedding layer for Vision Transformers.
    
    Uses a strided Conv2d to divide the image into fixed-size patches and project each patch to a token.
    Positional encoding is handled separately by the transformer, not here.
    """
    
    def __init__(self, 
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_channels: int = 3,
                 embed_dim: int = 768,
                 bias: bool = False,
                 ):
        """
        Initialize patch embedding layer.
        
        Args:
            img_size: Size of input image (assumed square)
            patch_size: Size of each patch (assumed square)
            in_channels: Number of input channels
            embed_dim: Embedding dimension for tokens
            bias: Whether to use bias in convolution
        """
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        # Calculate number of patches
        self.num_patches = (img_size // patch_size) ** 2
        
        # Conv2d projection: kernel_size=stride=patch_size for non-overlapping patches
        self.weight = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: convert image to patch tokens.
        
        Args:
            x: Input image [batch_size, in_channels, height, width]
            
        Returns:
            tokens: Token embeddings [batch_size, num_patches, embed_dim]
        """
        # Conv2d projection: [batch_size, in_channels, H, W] -> [batch_size, embed_dim, H/patch_size, W/patch_size]
        x = self.weight(x)
        
        # Flatten spatial dims and transpose: [batch_size, embed_dim, num_patches_h, num_patches_w] -> [batch_size, num_patches, embed_dim]
        tokens = x.flatten(2).transpose(1, 2)
        
        return tokens
    
    def __repr__(self):
        return f"PatchEmbedding(img_size={self.img_size}, patch_size={self.patch_size}, embed_dim={self.embed_dim})"


@add_to_all(__all__)
class MultiHeadSelfAttention(nn.Module):
    """Standard multi-head self-attention layer for Vision Transformer with selectable backend."""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, attn_backend: str = 'flash'):
        """
        Initialize multi-head self-attention.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            attn_backend: Attention backend ('flash' for Flash Attention 2, 'standard' for standard implementation)
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.attn_backend = attn_backend
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Check if flash attention is requested but not available
        if attn_backend == 'flash' and not FLASH_ATTN_AVAILABLE:
            raise ImportError(
                "Flash Attention 2 is not available. Please install it with: "
                "pip install flash-attn --no-build-isolation, or use attn_backend='standard'"
            )
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout_p = dropout
        
        # Only needed for standard backend
        if attn_backend == 'standard':
            self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x)  # [batch_size, seq_len, 3 * embed_dim]
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        
        if self.attn_backend == 'flash':
            # Use Flash Attention 2 with packed QKV
            # flash_attn_qkvpacked_func expects: qkv with shape [batch, seqlen, 3, nheads, headdim]
            out = flash_attn_qkvpacked_func(qkv, dropout_p=self.dropout_p if self.training else 0.0, causal=False)
            
            # Reshape output: [batch_size, seq_len, num_heads, head_dim] -> [batch_size, seq_len, embed_dim]
            out = out.reshape(batch_size, seq_len, embed_dim)
            
        else:  # standard
            # Standard attention expects [batch_size, num_heads, seq_len, head_dim]
            qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch_size, num_heads, seq_len, head_dim]
            q, k, v = qkv[0], qkv[1], qkv[2]
            
            # Compute attention
            attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)  # [batch_size, num_heads, seq_len, seq_len]
            attn = torch.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            
            # Apply attention to values
            out = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        
        out = self.proj(out)
        
        return out


@add_to_all(__all__)
class RoPEMultiHeadSelfAttention(MultiHeadSelfAttention):
    """
    Multi-head self-attention with 2D RoPE positional embeddings based on xy coordinates.
    Applies RoPE only to patch tokens, leaving the CLS token unchanged.
    
    When coords=None, grid coordinates are computed from num_patches_h and num_patches_w.
    """
    
    def __init__(self, embed_dim: int, num_heads: int, coords: Optional[torch.Tensor] = None,
                 num_patches_h: Optional[int] = None, num_patches_w: Optional[int] = None,
                 dropout: float = 0.0, attn_backend: str = 'flash'):
        """
        Initialize RoPE multi-head self-attention.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            coords: Coordinate tensor for RoPE. If None, computed from grid dims.
            num_patches_h: Number of patches in height (required if coords is None)
            num_patches_w: Number of patches in width (required if coords is None)
            dropout: Dropout probability
            attn_backend: Attention backend ('flash' for Flash Attention 2, 'standard' for standard implementation)
        """
        super().__init__(embed_dim, num_heads, dropout, attn_backend)
        self.rope = RoPEPositionalEncoding(embed_dim, num_heads, coords=coords,
                                           num_patches_h=num_patches_h, num_patches_w=num_patches_w)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x)  # [batch_size, seq_len, 3 * embed_dim]
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        
        if self.attn_backend == 'flash':
            # For Flash Attention, we need format [batch_size, seq_len, num_heads, head_dim]
            # Unpack Q, K, V from shape [batch_size, seq_len, 3, num_heads, head_dim]
            q = qkv[:, :, 0, :, :]  # [batch_size, seq_len, num_heads, head_dim]
            k = qkv[:, :, 1, :, :]  # [batch_size, seq_len, num_heads, head_dim]
            v = qkv[:, :, 2, :, :]  # [batch_size, seq_len, num_heads, head_dim]
            
            # Apply 2D RoPE only to patch tokens (skip CLS token at position 0)
            if seq_len > 1:  # Only if we have patch tokens
                # Need to convert to [batch_size, num_heads, seq_len, head_dim] for RoPE
                q_rope = q.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
                k_rope = k.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
                
                # Apply RoPE to patch tokens (positions 1 onwards)
                q_patches, k_patches = self.rope(q_rope[:, :, 1:, :], k_rope[:, :, 1:, :])
                
                # Reconstruct full Q and K tensors
                q_rope = torch.cat([q_rope[:, :, :1, :], q_patches], dim=2)  # CLS token + rotated patch tokens
                k_rope = torch.cat([k_rope[:, :, :1, :], k_patches], dim=2)  # CLS token + rotated patch tokens
                
                # Convert back to flash attention format
                q = q_rope.transpose(1, 2)  # [batch_size, seq_len, num_heads, head_dim]
                k = k_rope.transpose(1, 2)  # [batch_size, seq_len, num_heads, head_dim]
            
            # Use Flash Attention 2
            out = flash_attn_func(q, k, v, dropout_p=self.dropout_p if self.training else 0.0, causal=False)
            
            # Reshape output: [batch_size, seq_len, num_heads, head_dim] -> [batch_size, seq_len, embed_dim]
            out = out.reshape(batch_size, seq_len, embed_dim)
            
        else:  # standard
            # For standard attention, we need format [batch_size, num_heads, seq_len, head_dim]
            qkv_rope = qkv.permute(2, 0, 3, 1, 4)  # [3, batch_size, num_heads, seq_len, head_dim]
            q_rope, k_rope, v_rope = qkv_rope[0], qkv_rope[1], qkv_rope[2]
            
            # Apply 2D RoPE only to patch tokens (skip CLS token at position 0)
            if seq_len > 1:  # Only if we have patch tokens
                # Apply RoPE to patch tokens (positions 1 onwards)
                q_patches, k_patches = self.rope(q_rope[:, :, 1:, :], k_rope[:, :, 1:, :])
                
                # Reconstruct full Q and K tensors
                q_rope = torch.cat([q_rope[:, :, :1, :], q_patches], dim=2)  # CLS token + rotated patch tokens
                k_rope = torch.cat([k_rope[:, :, :1, :], k_patches], dim=2)  # CLS token + rotated patch tokens
            
            # Compute attention
            attn = (q_rope @ k_rope.transpose(-2, -1)) * (self.head_dim ** -0.5)  # [batch_size, num_heads, seq_len, seq_len]
            attn = torch.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            
            # Apply attention to values
            out = (attn @ v_rope).transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        
        out = self.proj(out)
        
        return out


@add_to_all(__all__)
class TransformerBlock(nn.Module):
    """Standard transformer block with self-attention and MLP."""
    
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0, attn_backend: str = 'flash'):
        """
        Initialize transformer block.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            mlp_ratio: Ratio of MLP hidden dim to embed dim
            dropout: Dropout probability
            attn_backend: Attention backend ('flash' for Flash Attention 2, 'standard' for standard implementation)
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout, attn_backend)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


@add_to_all(__all__)
class RoPETransformerBlock(TransformerBlock):
    """
    Transformer block with 2D RoPE-enabled self-attention based on xy coordinates.
    
    When coords=None, grid coordinates are computed from num_patches_h and num_patches_w.
    """
    
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0,
                 coords: Optional[torch.Tensor] = None, num_patches_h: Optional[int] = None,
                 num_patches_w: Optional[int] = None, attn_backend: str = 'flash'):
        """
        Initialize RoPE transformer block.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            mlp_ratio: Ratio of MLP hidden dim to embed dim
            dropout: Dropout probability
            coords: Coordinate tensor for RoPE. If None, computed from grid dims.
            num_patches_h: Number of patches in height (required if coords is None)
            num_patches_w: Number of patches in width (required if coords is None)
            attn_backend: Attention backend ('flash' for Flash Attention 2, 'standard' for standard implementation)
        """
        super().__init__(embed_dim, num_heads, mlp_ratio, dropout, attn_backend)
        # Replace the attention layer with 2D RoPE version
        self.attn = RoPEMultiHeadSelfAttention(embed_dim, num_heads, coords=coords,
                                                num_patches_h=num_patches_h, num_patches_w=num_patches_w,
                                                dropout=dropout, attn_backend=attn_backend)


@add_to_all(__all__)
class VisionTransformer(nn.Module):
    """
    Standard Vision Transformer implementation.
    
    This is the classic ViT architecture with patch embedding and transformer blocks.
    
    When coords=None, grid coordinates are computed from img_size and patch_size.
    When coords is provided (e.g., from KNNViT), those coordinates are used directly.
    """
    
    def __init__(self,
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_channels: int = 3,
                 embed_dim: int = 768,
                 num_heads: int = 12,
                 num_layers: int = 12,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.0,
                 num_outputs: int = 1000,
                 pos_emb_type: str = 'absolute',
                 coords: Optional[torch.Tensor] = None,
                 patch_embed: Optional[nn.Module] = None,
                 attn_backend: str = 'standard',
                 aggregation: str = 'cls_token',
                 ):
        """
        Initialize Vision Transformer.
        
        Args:
            img_size: Size of input image (assumed square)
            patch_size: Size of each patch (assumed square)
            in_channels: Number of input channels
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            mlp_ratio: Ratio of MLP hidden dim to embed dim
            dropout: Dropout rate
            num_outputs: Number of output classes
            pos_emb_type: Type of positional embedding ('absolute' or 'rope')
            coords: Cartesian coordinates tensor [num_patches, 2]. If None, grid coords computed from img_size/patch_size.
            patch_embed: optional pre-specified patch embedding
            attn_backend: Attention backend ('flash' for Flash Attention 2, 'standard' for standard implementation)
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.total_embed_dim = embed_dim # always just take the cls token
        self.num_outputs = num_outputs
        self.pos_emb_type = pos_emb_type
        self.coords = coords
        self.attn_backend = attn_backend
        self.aggregation = aggregation
        
        # Compute grid dimensions for standard ViT (when coords not provided)
        num_patches_h = img_size // patch_size
        num_patches_w = img_size // patch_size
        
        # Patch embedding layer
        if patch_embed is None:
            self.patch_embed = PatchEmbedding(
                img_size=img_size,
                patch_size=patch_size,
                in_channels=in_channels,
                embed_dim=embed_dim
            )
        else:
            self.patch_embed = patch_embed
        
        # Class token (for classification)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Transformer blocks and positional encodings
        if pos_emb_type == 'rope':
            self.pos_encoding = None
            self.blocks = nn.ModuleList([
                RoPETransformerBlock(embed_dim, num_heads, mlp_ratio, dropout, coords=coords,
                                     num_patches_h=num_patches_h, num_patches_w=num_patches_w,
                                     attn_backend=attn_backend)
                for _ in range(num_layers)
            ])            
        else:
            if pos_emb_type == 'absolute':
                self.pos_encoding = PositionalEncoding(embed_dim, coords=coords,
                                                       num_patches_h=num_patches_h, num_patches_w=num_patches_w)
            else:
                raise NotImplementedError()
            self.blocks = nn.ModuleList([
                TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout, attn_backend=attn_backend)
                for _ in range(num_layers)
            ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Final projection head - allow forward pass to determine number of inputs (tokens*features if using aggregation='flatten')
        self.head = nn.LazyLinear(num_outputs) if num_outputs is not None and num_outputs > 0 else nn.Identity()  

        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights."""
        # Initialize class token
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Initialize transformer blocks
        for block in self.blocks:
            nn.init.trunc_normal_(block.attn.qkv.weight, std=0.02)
            nn.init.trunc_normal_(block.attn.proj.weight, std=0.02)
            nn.init.trunc_normal_(block.mlp[0].weight, std=0.02)
            nn.init.trunc_normal_(block.mlp[3].weight, std=0.02)
            
        # Initialize classification head
        if hasattr(self.head, 'weight'):
            if isinstance(self.head.weight, nn.UninitializedParameter):
                print('WARNING: skipping manual initialization of unitialized head')
            else:
                nn.init.trunc_normal_(self.head.weight, std=0.02)
                nn.init.zeros_(self.head.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Vision Transformer.
        
        Args:
            x: Input image [batch_size, in_channels, height, width]
            
        Returns:
            output: Model output [batch_size, num_classes] or [batch_size, embed_dim]
        """
        batch_size = x.shape[0]
        
        # Patch embedding
        tokens = self.patch_embed(x)  # [batch_size, num_patches, embed_dim]
        
        # Add custom positional encoding if provided
        if self.pos_encoding is not None:
            tokens = self.pos_encoding(tokens)
        
        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)  # [batch_size, num_patches+1, embed_dim]
        
        # Apply transformer blocks
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True if self.attn_backend == 'flash' else torch.is_autocast_enabled()):
            for block in self.blocks:
                tokens = block(tokens)
        
        # Final normalization
        tokens = self.norm(tokens)
        
        # Use class token for classification
        if self.aggregation == 'cls_token':
            out = tokens[:, 0]  # [batch_size, embed_dim]
        elif self.aggregation == 'avg_pool':
            out = tokens.mean(dim=1)  # [batch_size, embed_dim]
        elif self.aggregation is None or self.aggregation == 'none':
            out = tokens
        elif self.aggregation == 'flatten':
            out = tokens.view(tokens.shape[0], -1)
        else:
            raise ValueError(f"Invalid aggregation: {self.aggregation}")
        
        # Apply classification head (possibly identity)
        output = self.head(out)
        
        return output
    
    def __repr__(self):
        return f"VisionTransformer(embed_dim={self.embed_dim}, num_layers={len(self.blocks)}, num_outputs={self.num_outputs}, pos_emb_type={self.pos_emb_type}) \n {super().__repr__()}" 
