from pickle import NONE
import torch.nn as nn
import numpy as np

from .knn import KNNPoolingLayer, get_in_out_coords, get_knn_conv_layer
from .norm import get_norm
from ..sensing.coords import auto_match_num_coords
from ..utils import add_to_all

__all__ = []

@add_to_all(__all__)
class KNNAlexNetBlock(nn.Module):
    """A block combining KNN convolution, optional pooling, normalization and nonlinearity.
    
    This module implements a complete processing block that can include:
    - KNN convolution layer
    - Normalization (batch, layer, etc.)
    - Activation function
    - Optional pooling layer
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        k (int): Number of nearest neighbors for KNN operations.
        fov (float): Field of view parameter.
        cmf_a (float): Cortical magnification factor parameter.
        in_res (int): Input resolution.
        stride (int): Stride for the convolution operation.
        style (str, optional): Style of coordinate system. Defaults to 'isotropic'.
        conv_layer (class or str, optional): Convolution layer class or name. Defaults to 'default'.
        ref_frame_mult (float, optional): Multiplier for reference frame size. If not None,
            ref_frame_side_length = ceil(ref_frame_mult * sqrt(k)). Defaults to None.
        cart_res (int, optional): Cartesian resolution. Defaults to None.
        pool (bool, optional): Whether to include pooling layer. Defaults to True.
        pool_mode (str, optional): Pooling mode ('max', 'avg'). Defaults to 'max'.
        pool_k (int, optional): Number of neighbors for pooling. Defaults to 9.
        pool_stride (int, optional): Stride for pooling. Defaults to 2.
        norm_type (str, optional): Type of normalization. Defaults to 'batch'.
        arch_flag (str, optional): Architecture flag. Defaults to ''.
        activation (class, optional): Activation function class. Defaults to nn.ReLU.
        sample_cortex (bool, optional): Whether to sample cortex. Defaults to True.
        device (str, optional): Device to use. Defaults to 'cuda'.
        auto_match_cart_resources (int, optional): Auto-match cartesian resources. Defaults to 0.
    """
    def __init__(self, in_channels, out_channels, k, fov, cmf_a, in_res, stride,
                 style='isotropic', conv_layer='default', cart_res=None,
                 pool=True, pool_mode='max', pool_k=9, pool_stride=2, 
                 norm_type='batch', arch_flag='',
                 activation=nn.ReLU, sample_cortex=True, 
                 device='cuda', auto_match_cart_resources=0, ref_frame_mult=None,
                 ):
        super().__init__()
        
        # Resolve conv_layer from string if needed
        if isinstance(conv_layer, str):
            conv_layer = get_knn_conv_layer(conv_layer)

        self.in_coords, self.out_coords, out_cart_res = get_in_out_coords(in_res, fov, cmf_a, stride, style=style, auto_match_cart_resources=auto_match_cart_resources, in_cart_res=cart_res, device=device)
        
        # Compute ref_frame_side_length from ref_frame_mult if provided
        if ref_frame_mult is not None:
            ref_frame_side_length = int(np.ceil(ref_frame_mult * np.sqrt(k)))
        else:
            ref_frame_side_length = None  # Let the conv layer use its default
        
        # Conv layer
        if conv_layer is not None:
            self.conv = conv_layer(
                in_channels=in_channels,
                out_channels=out_channels,
                k=k,
                in_coords=self.in_coords,
                out_coords=self.out_coords,
                sample_cortex=sample_cortex,
                arch_flag=arch_flag,
                device=device,
                ref_frame_side_length=ref_frame_side_length,
            )
        else:
            self.conv = None

        # Normalization
        self.norm = get_norm(norm_type, self.out_coords.shape[0], out_channels, device=device)
            
        # Activation
        self.activation = activation() if activation is not None else None
        
        # Optional pooling layer
        if pool:
            _, pooled_coords, out_cart_res = get_in_out_coords(self.in_coords.resolution, fov, cmf_a, pool_stride, style=style, auto_match_cart_resources=auto_match_cart_resources, in_cart_res=out_cart_res, device=device)

            self.pool = KNNPoolingLayer(
                k=pool_k,
                in_coords=self.out_coords,  # Input coords are output coords from conv
                out_coords=pooled_coords,  # Keep same coords for now
                mode=pool_mode,
                device=device,
                sample_cortex=sample_cortex,
            )
            self.out_coords = pooled_coords
        else:
            self.pool = None

        self.out_res = self.out_coords.resolution
        self.out_cart_res = out_cart_res
        self.to(device)
        
    def forward(self, x):
        """Forward pass through the block.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Output tensor after processing through conv, norm, activation, and pooling.
        """
        if self.conv is not None:
            x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.pool is not None:
            x = self.pool(x)

        return x


@add_to_all(__all__)
class KNNAlexNet(nn.Module):
    """KNN-based AlexNet architecture with foveated processing.
    
    This class implements a complete KNN-based AlexNet architecture that processes
    input through multiple KNN convolution blocks with optional pooling layers.
    
    Args:
        in_res (int): Input resolution.
        in_channels (int): Number of input channels.
        features_per_layer (list[int]): List of feature dimensions for each layer.
        stride_per_layer (list[int]): List of stride values for each layer.
        pool_after (list[int]): List of layer indices after which to apply pooling.
        k_per_layer (list[int]): List of k values (nearest neighbors) for each layer.
        n_classes (int, optional): Number of output classes. If None, no classifier is added. Defaults to None.
        out_res (int, optional): Output resolution. If None, no output pooling is applied. Defaults to None.
        auto_match_cart_resources (int, optional): Auto-match cartesian resources. Defaults to 0.
        style (str, optional): Style of coordinate system. Defaults to 'isotropic'.
        layer_cls (class or str, optional): Convolution layer class or name. Defaults to 'KNNConvLayer'.
        norm_type (str, optional): Type of normalization. Defaults to 'batch'.
        arch_flag (str, optional): Architecture flag. Defaults to ''.
        fov (float, optional): Field of view parameter. Defaults to 16.
        cmf_a (float, optional): Cortical magnification factor parameter. Defaults to 0.5.
        device (str, optional): Device to use. Defaults to 'cuda'.
        sample_cortex (bool, optional): Whether to sample cortex. Defaults to True.
        ref_frame_mult (float, optional): Multiplier for reference frame size. If not None,
            ref_frame_side_length = ceil(ref_frame_mult * sqrt(k)). Defaults to None.
    """
    def __init__(self, in_res: int, in_channels: int, features_per_layer: list[int], stride_per_layer: list[int], pool_after: list[int], k_per_layer: list[int], 
                 n_classes: int = None, out_res:int = None, auto_match_cart_resources=0, 
                 style='isotropic', layer_cls='default', norm_type='batch', 
                 arch_flag='', 
                 fov=16, cmf_a=0.5, 
                 device='cuda', sample_cortex=True,
                 ref_frame_mult=None,
                 ):
        super().__init__()
        self.layers = []
        
        # Resolve layer_cls from string if needed
        if isinstance(layer_cls, str):
            layer_cls = get_knn_conv_layer(layer_cls)
        
        in_res, cart_res = auto_match_num_coords(fov, cmf_a, in_res, style, auto_match_cart_resources, device, quiet=True)
                
        for i in range(len(features_per_layer)):

            block = KNNAlexNetBlock(
                in_channels=in_channels, out_channels=features_per_layer[i], k=k_per_layer[i], fov=fov, cmf_a=cmf_a, in_res=in_res, stride=stride_per_layer[i],
                style=style, conv_layer=layer_cls, cart_res=cart_res, auto_match_cart_resources=auto_match_cart_resources,
                pool=i in pool_after, pool_mode='max', pool_k=9, pool_stride=2, 
                norm_type=norm_type, arch_flag=arch_flag,
                sample_cortex=sample_cortex,
                device=device,
                ref_frame_mult=ref_frame_mult,
            )

            in_channels = features_per_layer[i]
            in_res = block.out_res
            cart_res = block.out_cart_res

            self.layers.append(block)
        
        if out_res is not None and out_res != -1:
            cart_res = out_res
            in_coords = block.out_coords

            out_coords, out_radii, cart_res = in_coords.get_strided_coords(1, auto_match_cart_resources=auto_match_cart_resources, in_cart_res=cart_res)
            
            # set k to twice the downsampling factor squared - heuristic choice
            k = int(2*np.ceil(in_res / out_radii))**2

            layer = KNNPoolingLayer(
                k=k,
                mode='avg',
                in_coords=in_coords, out_coords=out_coords,
                sample_cortex=sample_cortex,
                device=device,
            )
            self.layers.append(layer)
        else:
            print('no output pooling layer')

        self.layers = nn.ModuleList(self.layers)
        self.n_classes = n_classes
        self.out_channels = features_per_layer[-1]
        if n_classes is not None:
            self.classifier = nn.Linear(self.out_channels*self.layers[-1].out_coords.shape[0], n_classes)

        # convenience access to final coordinates
        self.out_coords = self.layers[-1].out_coords

        # convenience access to total number of outputs units
        self.total_embed_dim = self.out_channels * self.out_coords.shape[0]

    def forward(self, x):
        """Forward pass through the entire network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, in_res, in_res).
            
        Returns:
            torch.Tensor: Output tensor. If n_classes is specified, returns logits of shape 
                (batch_size, n_classes). Otherwise, returns features of shape 
                (batch_size, out_channels, out_coords.shape[0]).
        """
        for ii, layer in enumerate(self.layers):
            x = layer(x)
        if self.n_classes is not None:
            x = x.reshape(x.shape[0], -1)
            x = self.classifier(x)
        return x
