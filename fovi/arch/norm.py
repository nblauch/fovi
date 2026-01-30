import torch.nn as nn
import numpy as np
from ..utils import add_to_all

"""KNN normalization layers.

KNN normalization layers always take both num_coords and num_channels, even if one of these is irrelevant.
"""

__all__ = []


@add_to_all(__all__)
class KNNChannelNorm(nn.LayerNorm):
    """Simple wrapper to perform normalization over the channel dimension.
    
    Args:
        num_coords (int): Number of coordinates (unused but required for interface).
        num_channels (int): Number of channels to normalize over.
        device (str, optional): Device to place the layer on. Defaults to 'cuda'.
    """
    def __init__(self, num_coords, num_channels, device='cuda'):
        super().__init__([num_channels], device=device)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = super().forward(x)
        x = x.permute(0, 2, 1)
        return x

    def __repr__(self):
        return super().__repr__().replace('LayerNorm', 'KNNChannelNorm')


@add_to_all(__all__)
class KNNBatchNorm(nn.BatchNorm1d):
    """Simple wrapper to perform normalization over the batch dimension.
    
    Args:
        num_coords (int): Number of coordinates (unused but required for interface).
        num_channels (int): Number of channels to normalize over.
        device (str, optional): Device to place the layer on. Defaults to 'cuda'.
    """
    def __init__(self, num_coords, num_channels, device='cuda'):
        super().__init__(num_channels, device=device)

    def __repr__(self):
        return super().__repr__().replace('BatchNorm1d', 'KNNBatchNorm')

    
@add_to_all(__all__)
class KNNCoordNorm(nn.LayerNorm):
    """Simple wrapper to perform normalization over the coordinate dimension.
    
    Args:
        num_coords (int): Number of coordinates to normalize over.
        num_channels (int): Number of channels (unused but required for interface).
        device (str, optional): Device to place the layer on. Defaults to 'cuda'.
    """
    def __init__(self, num_coords, num_channels, device='cuda'):
        super().__init__([num_coords], device=device)

    def __repr__(self):
        return super().__repr__().replace('LayerNorm', 'KNNCoordNorm')


@add_to_all(__all__)
class KNNLayerNorm(nn.LayerNorm):
    """Simple wrapper to perform normalization over both coordinate and channel dimensions.
    
    Args:
        num_coords (int): Number of coordinates to normalize over.
        num_channels (int): Number of channels to normalize over.
        device (str, optional): Device to place the layer on. Defaults to 'cuda'.
    """
    def __init__(self, num_coords, num_channels, device='cuda'):
        super().__init__([num_channels, num_coords], device=device)

    def __repr__(self):
        return super().__repr__().replace('LayerNorm', 'KNNLayerNorm')

    
@add_to_all(__all__)
class KNNGroupNorm(nn.GroupNorm):
    """Simple wrapper to perform group normalization.
    
    Args:
        num_coords (int): Number of coordinates (unused but required for interface).
        num_channels (int): Number of channels to normalize over.
        base_groups (int, optional): Base number of groups. Defaults to 32.
        min_channels_per_group (int, optional): Minimum channels per group. Defaults to 4.
        device (str, optional): Device to place the layer on. Defaults to 'cuda'.
    """
    def __init__(self, num_coords, num_channels, base_groups=32, min_channels_per_group=4, device='cuda'):
        groups = find_valid_num_groups(num_channels, base_groups, min_channels_per_group=min_channels_per_group)
        super().__init__(groups, num_channels, device=device)

    def __repr__(self):
        return super().__repr__().replace('GroupNorm', 'KNNGroupNorm')


@add_to_all(__all__)
def find_valid_num_groups(num_channels, base_groups, min_channels_per_group=4):
    """Find a valid number of groups for group normalization.
    
    Args:
        num_channels (int): Total number of channels.
        base_groups (int): Desired base number of groups.
        min_channels_per_group (int, optional): Minimum channels per group. Defaults to 4.
    
    Returns:
        int: Valid number of groups that divides num_channels evenly.
    """
    # ensure there are at least min_channels_per_group per group
    max_groups = np.max([num_channels // min_channels_per_group,1])
    desired_groups = np.min([max_groups, base_groups])

    for num_groups in range(desired_groups, 0, -1):
        if num_channels % num_groups == 0:
            return num_groups
    return 1  # If no smaller divisor is found, default to 1 group


@add_to_all(__all__)
def get_norm(norm_type, num_coords, num_channels, base_groups=32, min_channels_per_group=4, device='cuda'):
    """Get a normalization layer for a KNN layer with shape (batch, num_channels, num_coords).
    
    Args:
        norm_type (str): Type of normalization ('layer_chan', 'channel', 'layer_space', 
            'coord', 'layer_all', 'layer', 'group', 'batch', or 'none').
        num_coords (int): Number of coordinates.
        num_channels (int): Number of channels.
        base_groups (int, optional): Base number of groups for group norm. Defaults to 32.
        min_channels_per_group (int, optional): Minimum channels per group. Defaults to 4.
        device (str, optional): Device to place the layer on. Defaults to 'cuda'.
    
    Returns:
        nn.Module or None: Normalization layer or None if norm_type is 'none'.
    
    Raises:
        NotImplementedError: If norm_type is not implemented.
    """
    if norm_type == 'layer_chan' or norm_type=='channel':
        norm = KNNChannelNorm(num_coords, num_channels, device=device)
    elif norm_type == 'layer_space' or norm_type=='coord':
        norm = KNNCoordNorm(num_coords, num_channels, device=device)
    elif norm_type == 'layer_all' or norm_type=='layer':
        norm = KNNLayerNorm(num_coords, num_channels, device=device)
    elif norm_type == 'group':
        norm = KNNGroupNorm(num_coords, num_channels, base_groups, min_channels_per_group, device=device)
    elif 'batch' in norm_type:
        norm = KNNBatchNorm(num_coords, num_channels, device=device)
    elif norm_type == 'none':
        norm = None
    else:
        raise NotImplementedError(f'norm_type {norm_type} not implemented')
    return norm