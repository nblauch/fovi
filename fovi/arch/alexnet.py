import torch
import torch.nn as nn
from functools import partial
from collections import OrderedDict

from .polar import PolarPadder
from .mlp import get_mlp
from .wrapper import BackboneProjectorWrapper
from ..utils import add_to_all

__all__ = ['baseline_alexnet_kernels']

# (out_c, ks, stride, pad, groups)
baseline_alexnet_kernels = {
    'base_': [
        (96, 11, 4, 2, 1), # alexnet baseline architecture; 'base' is preferred
        (256, 5, 1, 2, 1),
        (384, 3, 1, 1, 1),
        (384, 3, 1, 1, 1),
        (256, 3, 1, 1, 1)
        ],
    'base': [
        (96, 11, 4, 5, 1), # like 0 (baseline), but with padding p=k//2 to ensure no downsampling
        (256, 5, 1, 2, 1),
        (384, 3, 1, 1, 1),
        (384, 3, 1, 1, 1),
        (256, 3, 1, 1, 1)
        ],
    'base_4layer': [
        (96, 11, 4, 5, 1), # like 0 (baseline), but with 4 layers, designed for lower res inputs
        (256, 5, 1, 2, 1),
        (384, 3, 1, 1, 1),
        (256, 3, 1, 1, 1)
        ],
    'base_lowres': [
        (96, 11, 2, 5, 1), # like 1, but with reduced stride in layer 1 for lower-res inputs
        (256, 5, 1, 2, 1),
        (384, 3, 1, 1, 1),
        (384, 3, 1, 1, 1),
        (256, 3, 1, 1, 1)
        ],          
    '4layer_lowres': [
        (96, 5, 2, 2, 1), # 4-layer designed for low-res inputs
        (256, 5, 2, 2, 1),
        (384, 3, 1, 1, 1),
        (256, 3, 1, 1, 1)
        ],
    '4layer_nostride': [ # designed for even smaller inputs - no strides at all
        (96, 5, 1, 2, 1), 
        (256, 5, 1, 2, 1),
        (384, 3, 1, 1, 1),
        (256, 3, 1, 1, 1)
        ],
    'base_lowres_smallrf': [ # modification of 4 to have smaller RFs
        (96, 5, 2, 2, 1),
        (256, 5, 1, 2, 1),
        (384, 3, 1, 1, 1),
        (384, 3, 1, 1, 1),
        (256, 3, 1, 1, 1)
    ],
}


@add_to_all(__all__)
class ConvBlock(nn.Sequential):
    """A configurable convolutional block with optional normalization, activation, dropout, and pooling.
    
    This block provides a flexible way to construct convolutional layers with various
    preprocessing and postprocessing operations, including support for polar coordinate padding.
    
    Args:
        in_c (int): Number of input channels.
        out_c (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel.
        stride (int): Stride of the convolution.
        padding (int): Padding for the convolution.
        dilation (int, optional): Dilation rate for the convolution. Defaults to 1.
        groups (int, optional): Number of groups for grouped convolution. Defaults to 1.
        inp (callable, optional): Input preprocessing layer factory. Defaults to None.
        conv (nn.Module, optional): Convolution layer class. Defaults to nn.Conv2d.
        norm (callable, optional): Normalization layer factory. Defaults to GroupNorm(32).
        act (callable, optional): Activation function factory. Defaults to ReLU.
        dropout (callable, optional): Dropout layer factory. Defaults to None.
        pool (callable, optional): Pooling layer factory. Defaults to MaxPool2d(3,2).
        after_pool (callable, optional): Post-pooling layer factory. Defaults to None.
        polar (bool, optional): Whether to use polar coordinate padding. Defaults to False.
        out (callable, optional): Output postprocessing layer factory. Defaults to None.
    """
    def __init__(self, in_c, out_c, kernel_size, stride, padding, dilation=1, groups=1,
                 inp=None,
                 conv=nn.Conv2d, 
                 norm=partial(nn.GroupNorm, 32), 
                 act=partial(nn.ReLU, inplace=True), 
                 dropout=None,
                 pool=partial(nn.MaxPool2d(3,2)),
                 after_pool=None,
                 polar=False,
                 out=None):
        
        layers = []
        
        # input processing layer
        if inp is not None:
            layers += [inp()]

        # add polar padding for conv if necessary
        if polar:
            conv_pad = 0
            layers += [PolarPadder(((kernel_size)//2)*dilation)]
        else:
            conv_pad = padding
        
        # conv layer always included
        layers += [
            conv(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=conv_pad,
                 groups=groups, dilation=dilation, bias=False if norm is not None else True)
        ]
        
        if norm is not None:
            layers += [norm]
            #layers += [norm(out_c)]
        
        layers += [act()]
        
        if dropout is not None:
            layers += [dropout()]
            
        if pool is not None:
            # add polar padding for pool if necessary
            tmp_pool = pool()
            if polar:
                pool_pad = 0
                layers += [PolarPadder((tmp_pool.kernel_size//2) * tmp_pool.dilation)]
            else:
                pool_pad = tmp_pool.padding
            layers += [pool(padding=pool_pad)]
        
        if after_pool is not None:
            if polar:
                print(f'Warning: after_pool fun {after_pool} not configured for use with polar coordinates')
            layers += [after_pool()]
        
        # output processing layer
        if out is not None:
            layers += [out()]
            
        super(ConvBlock, self).__init__(*layers)


@add_to_all(__all__)
def get_backbone(in_channels=3,
                 kernels = baseline_alexnet_kernels['base'],
                 w=1,
                 preprocess=None,
                 inp=lambda *_: None,
                 conv=lambda *_: nn.Conv2d,
                 norm=lambda idx, args: nn.BatchNorm2d(args[0]),
                 act=lambda *_: partial(nn.ReLU, inplace=True),
                 dropout=lambda *_: None,
                 pool=lambda idx,*_: partial(nn.MaxPool2d, 3,2, padding=1) if idx in [0,1,4] else None,
                 after_pool=None,
                 out=lambda *_: None,
                 avgpool=lambda *_: nn.AdaptiveAvgPool2d((6,6)),
                 polar=False,
                ):
    """Build a configurable AlexNet-style backbone network.
    
    This function constructs a backbone network using the specified kernel configurations
    and layer factories. It provides extensive customization options for each component.
    
    Args:
        in_channels (int, optional): Number of input channels. Defaults to 3.
        kernels (list, optional): List of kernel specifications (out_c, ks, stride, pad, groups). 
            Defaults to baseline_alexnet_kernels['base'].
        w (float, optional): Width multiplier for channel counts. Defaults to 1.
        preprocess (callable, optional): Preprocessing layer factory. Defaults to None.
        inp (callable, optional): Input layer factory function(idx, args). Defaults to None.
        conv (callable, optional): Convolution layer factory function(idx, args). 
            Defaults to nn.Conv2d.
        norm (callable, optional): Normalization layer factory function(idx, args). 
            Defaults to BatchNorm2d.
        act (callable, optional): Activation layer factory function(idx, args). 
            Defaults to ReLU.
        dropout (callable, optional): Dropout layer factory function(idx, args). 
            Defaults to None.
        pool (callable, optional): Pooling layer factory function(idx, args). 
            Defaults to MaxPool2d after layers 0, 1, 4.
        after_pool (callable, optional): Post-pooling layer factory. Defaults to None.
        out (callable, optional): Output layer factory function(idx, args). Defaults to None.
        avgpool (callable, optional): Final average pooling layer factory. 
            Defaults to AdaptiveAvgPool2d((6,6)).
        polar (bool, optional): Whether to use polar coordinate padding. Defaults to False.
        
    Returns:
        nn.Sequential: The constructed backbone network.
    """
    
    layers = []

    # add a pre-processing layer (e.g., lgn transform, saccadenet fixation)
    if preprocess is not None:
         layers.append(('preprocess', preprocess()))
                     
    # scale out_channels by `w`
    kernels = [(int(out_c*w), ks, s, p, grps) for out_c, ks, s, p, grps in kernels]
    
    in_c = in_channels
    for args_ in enumerate(kernels):
        idx,(out_c, ks, stride, pad, groups) = args_
        block_name = f'conv_block_{idx+1}'
        block = ConvBlock(in_c, out_c, ks, stride, pad, groups=groups,
                          inp=inp(*args_) if inp is not None else None,
                          conv=conv(*args_),
                          norm=norm(*args_) if norm is not None else None, 
                          act=act(*args_) if act is not None else None,
                          dropout=dropout(*args_) if dropout is not None else None,
                          pool=pool(*args_) if pool is not None else None,
                          after_pool=after_pool(*args_) if after_pool is not None else None,
                          polar=polar,
                          out=out(*args_) if out is not None else None
                         )
        layers.append((block_name, block))
        in_c = out_c
    
    if avgpool is not None:
        layers.append(('avgpool', avgpool(*args_)))
    
    backbone = nn.Sequential(OrderedDict(layers))
    
    return backbone        


@torch.no_grad()
@add_to_all(__all__)
def get_repr_size(model, in_channels=3, img_size=224):
    """Compute the flattened representation size of a model's output.
    
    Runs a forward pass with a random input to determine the output size.
    
    Args:
        model (nn.Module): The model to analyze.
        in_channels (int, optional): Number of input channels. Defaults to 3.
        img_size (int, optional): Input image size (assumed square). Defaults to 224.
        
    Returns:
        int: The flattened output size of the model.
    """
    model.eval()
    x = torch.rand(1,in_channels,img_size,img_size)
    out = model(x)
    
    return out.flatten(start_dim=1).shape[1]
    

@add_to_all(__all__)
def build_model(weights=None, progress=True, 
                backbone=None, mlp=None,
                backbone_kwargs=None, mlp_kwargs=None,
                repr_size_in_channels=3, img_size=224):
    """Build a complete model with backbone and MLP projection head.
    
    Constructs a model by combining a backbone network with an MLP projection head,
    wrapped in a BackboneProjectorWrapper for unified forward pass.
    
    Args:
        weights: Optional pretrained weights to load. Defaults to None.
        progress (bool, optional): Whether to show progress bar when loading weights. 
            Defaults to True.
        backbone (nn.Module, optional): Pre-built backbone module. If None, one is created 
            using backbone_kwargs. Defaults to None.
        mlp (nn.Module, optional): Pre-built MLP module. If None, one is created using 
            mlp_kwargs. Defaults to None.
        backbone_kwargs (dict, optional): Keyword arguments for get_backbone(). 
            Defaults to None.
        mlp_kwargs (dict, optional): Keyword arguments for get_mlp(). Defaults to None.
        repr_size_in_channels (int, optional): Input channels for computing repr size. 
            Defaults to 3.
        img_size (int, optional): Image size for computing repr size. Defaults to 224.
        
    Returns:
        BackboneProjectorWrapper: The complete model with backbone and projector.
    """
    backbone_kwargs = {} if backbone_kwargs is None else backbone_kwargs
    mlp_kwargs = {} if mlp_kwargs is None else mlp_kwargs
    backbone = get_backbone(**backbone_kwargs) if backbone is None else backbone
    repr_size = get_repr_size(backbone, in_channels=repr_size_in_channels, img_size=img_size)
    mlp = get_mlp(repr_size=repr_size, **mlp_kwargs) if mlp is None else mlp(repr_size)

    model = BackboneProjectorWrapper(backbone, mlp)
    model.backbone.total_embed_dim = repr_size
    
    if weights is not None:
        msg = model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))
        print(msg)
        
    return model


@add_to_all(__all__)
def alexnet2023_baseline(mlp_kwargs=None, weights=None, progress=True, img_size=224, **backbone_kwargs):
    """Create an AlexNet-2023 baseline model with configurable backbone and MLP.
    
    This is a convenience wrapper around build_model() for creating AlexNet-style
    models with default configurations.
    
    Args:
        mlp_kwargs (dict, optional): Keyword arguments for the MLP head. Defaults to None.
        weights: Optional pretrained weights to load. Defaults to None.
        progress (bool, optional): Whether to show progress bar when loading weights. 
            Defaults to True.
        img_size (int, optional): Input image size. Defaults to 224.
        **backbone_kwargs: Additional keyword arguments passed to get_backbone().
        
    Returns:
        BackboneProjectorWrapper: The complete AlexNet-2023 model.
    """
    model = build_model(mlp_kwargs=mlp_kwargs, weights=weights, progress=progress, img_size=img_size, backbone_kwargs=backbone_kwargs)
    return model