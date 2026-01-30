import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .knn import KNNConvLayer, KNNPoolingLayer, get_in_out_coords
from .norm import get_norm
from ..utils import add_to_all

__all__ = []

@add_to_all(__all__)
class KNNResNetBasicBlock(nn.Module):
    """Basic block for KNN-based ResNet architecture.
    
    This block implements a residual connection with two KNN convolution layers,
    following the standard ResNet basic block design but using KNN convolutions
    instead of standard convolutions.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        k (int): Number of nearest neighbors for KNN convolution.
        in_res (int): Input resolution.
        stride (int): Stride for the first convolution layer.
        fov (float): Field of view parameter.
        cmf_a (float): Cortical magnification factor parameter.
        style (str, optional): Sampling style. Defaults to 'isotropic'.
        conv_layer (class, optional): Convolution layer class to use. 
            Defaults to KNNConvLayer.
        cart_res (int, optional): Cartesian resolution. Defaults to None.
        norm_type (str, optional): Normalization type. Defaults to 'batch'.
        arch_flag (str, optional): Architecture flag. Defaults to ''.
        sample_cortex (bool, optional): Whether to sample cortex. Defaults to True.
        device (str, optional): Device to use. Defaults to 'cuda'.
        auto_match_cart_resources (int, optional): Auto-match cartesian resources. 
            Defaults to 0.
    """
    def __init__(self, in_channels, out_channels, k, in_res, stride,
                 fov, cmf_a, 
                 style='isotropic', conv_layer=KNNConvLayer, cart_res=None,
                 norm_type='batch', arch_flag='',
                 sample_cortex=True, 
                 device='cuda', auto_match_cart_resources=0,
                 ):
        super().__init__()

        self.expansion = 1

        self.in_coords, self.out_coords, out_cart_res = get_in_out_coords(in_res, fov, cmf_a, stride, style=style, auto_match_cart_resources=auto_match_cart_resources, in_cart_res=cart_res, device=device)

        # first conv does the stride to out_coords
        self.conv1 = conv_layer(
            in_channels=in_channels,
            out_channels=out_channels,
            k=k,
            in_coords=self.in_coords,
            out_coords=self.out_coords,
            sample_cortex=sample_cortex,
            arch_flag=arch_flag,
            device=device,
        )

        self.norm1 = get_norm(norm_type, len(self.out_coords), out_channels)

        # second conv does no stride
        self.conv2 = conv_layer(
            in_channels=out_channels,
            out_channels=out_channels,
            k=k,
            in_coords=self.out_coords,
            out_coords=self.out_coords,
            sample_cortex=sample_cortex,
            arch_flag=arch_flag,
            device=device,
        )

        self.norm2 = get_norm(norm_type, len(self.out_coords), out_channels)

        # downsample is a 1x1 conv on the input to use as the residual
        if stride != 1:
            self.downsample = nn.Sequential(
                conv_layer(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    k=1,
                    in_coords=self.in_coords,
                    out_coords=self.out_coords,
                    sample_cortex=sample_cortex,
                    arch_flag=arch_flag,
                    device=device,
                    ),
                get_norm(norm_type, len(self.out_coords), out_channels),
            )
        else:
            self.downsample = None

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward pass through the basic block.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Output tensor after applying residual connection.
        """
        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out += identity
        out = self.relu(out)

        return out
    

@add_to_all(__all__)
class KNNResNet(nn.Module):
    """KNN-based ResNet architecture.
    
    A ResNet implementation that uses KNN convolutions instead of standard
    convolutions, designed for foveated vision processing.
    
    Args:
        block (class, optional): Block class to use for layers. 
            Defaults to KNNResNetBasicBlock.
        layers (list, optional): Number of blocks in each layer. 
            Defaults to [2, 2, 2, 2].
        in_conv_stride (int, optional): Stride for initial convolution. 
            Defaults to 2.
        in_pool_stride (int, optional): Stride for initial pooling. 
            Defaults to 2.
        fov (float, optional): Field of view parameter. Defaults to 16.
        cmf_a (float, optional): Cortical magnification factor parameter. 
            Defaults to 0.5.
        in_res (int, optional): Input resolution. Defaults to 64.
        out_res (int, optional): Output resolution. Defaults to 1.
        style (str, optional): Sampling style. Defaults to 'isotropic'.
        conv_layer (class, optional): Convolution layer class to use. 
            Defaults to KNNConvLayer.
        pool_layer (class, optional): Pooling layer class to use. 
            Defaults to KNNPoolingLayer.
        norm_type (str, optional): Normalization type. Defaults to 'batch'.
        arch_flag (str, optional): Architecture flag. Defaults to ''.
        sample_cortex (bool, optional): Whether to sample cortex. Defaults to True.
        device (str, optional): Device to use. Defaults to 'cuda'.
        auto_match_cart_resources (int, optional): Auto-match cartesian resources. 
            Defaults to 0.
        num_classes (int, optional): Number of output classes for classification. 
            If None, no classification head is added. Defaults to None.
    """
    def __init__(self, 
                 block=KNNResNetBasicBlock,
                 layers=[2, 2, 2, 2],
                 in_conv_stride=2,
                 in_pool_stride=2,
                 fov=16,
                 cmf_a=0.5,
                 in_res=64,
                 out_res=1,
                 style='isotropic',
                 conv_layer=KNNConvLayer,
                 pool_layer=KNNPoolingLayer,
                 norm_type='batch',
                 arch_flag='',
                 sample_cortex=True,
                 device='cuda',
                 auto_match_cart_resources=0,
                 num_classes=None,
                 ):
        super(KNNResNet, self).__init__()

        self.fov = fov
        self.cmf_a = cmf_a
        self.in_res = in_res
        self.style = style
        self.conv_layer = conv_layer
        self.norm_type = norm_type
        self.arch_flag = arch_flag
        self.in_channels = 64
        self.conv_layer = conv_layer
        self.num_classes = num_classes

        self.block_kwargs = dict(
            fov=fov,
            cmf_a=cmf_a,
            style=style,
            conv_layer=conv_layer,
            norm_type=norm_type,
            arch_flag=arch_flag,
            sample_cortex=sample_cortex,
            device=device,
            auto_match_cart_resources=auto_match_cart_resources,
        )

        self.in_coords, self.out_coords, out_cart_res = get_in_out_coords(in_res, fov, cmf_a, in_conv_stride, style=style, auto_match_cart_resources=auto_match_cart_resources, in_cart_res=None)
        
        # always use KNNConvLayer for the first conv
        self.conv1 = KNNConvLayer(
            in_channels=3,
            out_channels=self.in_channels,
            k=49,
            in_coords=self.in_coords,
            out_coords=self.out_coords,
            sample_cortex=sample_cortex,
        )
        
        self.bn1 = get_norm(norm_type, len(self.out_coords), self.in_channels)
        self.relu = nn.ReLU(inplace=True)

        _, self.pool_coords, self.cart_res = get_in_out_coords(self.out_coords.resolution, fov, cmf_a, in_pool_stride, style=style, auto_match_cart_resources=auto_match_cart_resources, in_cart_res=out_cart_res)
        self.in_res = self.pool_coords.resolution

        self.maxpool = pool_layer(
            k=9,
            in_coords=self.out_coords,
            out_coords=self.pool_coords,
            mode='max',
            device=device,
            sample_cortex=sample_cortex,
        )

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.out_coords = self.layer4[-1].out_coords
        self.out_channels = 512
        # reset in_channels to 3 for proper usage in SaccadeNet
        self.in_channels = 3

        if out_res is not None and out_res != -1:
            cart_res = out_res

            out_coords, out_radii, cart_res = in_coords.get_strided_coords(1, auto_match_cart_resources=auto_match_cart_resources, in_cart_res=cart_res)

            in_coords = self.out_coords
            self.out_coords = out_coords
            
            # set k to twice the downsampling factor squared - heuristic choice
            k = int(2*np.ceil(in_res / out_radii))**2

            self.avgpool = KNNPoolingLayer(
                k=k,
                mode='avg',
                in_coords=in_coords, out_coords=self.out_coords,
                sample_cortex=sample_cortex,
                device=device,
            )
        else:
            self.avgpool = None


        # linear layer for classification
        if num_classes is not None:
            self.fc = nn.Linear(self.out_channels * block.expansion, num_classes)

        # convenience access to total number of outputs units
        self.total_embed_dim = self.out_channels * self.out_coords.shape[0]

    def _make_layer(self, block, planes, blocks, stride=1):
        """Create a layer with the specified number of blocks.
        
        Args:
            block (class): Block class to use for the layer.
            planes (int): Number of output channels for the layer.
            blocks (int): Number of blocks in the layer.
            stride (int, optional): Stride for the first block. Defaults to 1.
            
        Returns:
            nn.Sequential: Sequential container with the layer blocks.
        """
        layers = [block(self.in_channels, planes, 9, self.in_res, stride, **self.block_kwargs)]
        self.in_channels = planes * layers[0].expansion

        self.in_res = layers[0].out_coords.resolution

        for _ in range(1, blocks):
            layers.append(block(self.in_channels, planes, 9, self.in_res, 1, **self.block_kwargs))
            self.in_res = layers[-1].out_coords.resolution

        return nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass through the KNN ResNet.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, height, width).
            
        Returns:
            torch.Tensor: Output tensor. If num_classes is specified, returns
                classification logits. Otherwise, returns feature embeddings.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        if self.avgpool is not None:
            x = self.avgpool(x)

        if self.num_classes is not None:
            x = x.view(x.size(0), -1)
            x = self.fc(x)

        return x