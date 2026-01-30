import torch.nn as nn
import torch
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from .knn import KNNConvLayer, KNNDepthwiseConvLayer, get_in_out_coords
from ..sensing.coords import SamplingCoords
from ..utils import add_to_all

__all__ = []

@add_to_all(__all__)
class GRN(nn.Module):
    """GRN (Global Response Normalization) layer.
    
    Args:
        dim (int): Number of input channels.
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


@add_to_all(__all__)
class LayerNorm(nn.Module):
    """LayerNorm that supports two data formats: channels_last (default) or channels_first.
    
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, locations, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, locations).
    
    Args:
        normalized_shape (int): Expected input size from the last dimension.
        eps (float): A value added to the denominator for numerical stability. Default: 1e-6
        data_format (str): The ordering of the dimensions. Default: "channels_last"
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None] * x + self.bias[:, None]
            return x


@add_to_all(__all__)
class Block(nn.Module):
    """ConvNeXtV2 Block.
    
    Args:
        dim (int): Number of input channels.
        coords: Input/output coordinates for KNN operations.
        drop_path (float): Stochastic depth rate. Default: 0.0
        **kwargs: Additional keyword arguments.
    
    Returns:
        torch.Tensor: Output tensor with same shape as input.
    """
    def __init__(self, dim, coords, drop_path=0., **kwargs):
        super().__init__()
        self.dwconv = KNNDepthwiseConvLayer(dim, dim, k=49, in_coords=coords, out_coords=coords, bias=True, **kwargs)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x) # (B, C, N) -> (B, C, N)
        x = x.permute(0, 2, 1) # (B, C, N) -> (B, N, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 2, 1) # (B, N, C) -> (B, C, N)

        x = input + self.drop_path(x)
        return x


@add_to_all(__all__)
class ConvNeXtV2(nn.Module):
    """ConvNeXt V2.
        
    Args:
        in_res (int): Input resolution.
        fov (float): Field of view parameter.
        cmf_a (float): CMF parameter.
        style (str): Style parameter.
        auto_match_cart_resources (int): Auto match cartesian resources. Default: 1
        device (str): Device to use. Default: 'cuda'
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        first_stride (int): First stride value. Default: 4
        depths (list): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (list): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
        **kwargs: Additional keyword arguments.
    
    Returns:
        torch.Tensor: Classification logits or features.
    """
    def __init__(self, in_res, fov, cmf_a, style, 
                 auto_match_cart_resources=1, device='cuda',
                 in_chans=3, num_classes=1000, first_stride=4,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
                 drop_path_rate=0., head_init_scale=1.,
                 **kwargs,
                 ):
        super().__init__()
        self.depths = depths
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers

        self.in_coords, out_coords, out_cart_res = get_in_out_coords(in_res, fov, cmf_a, first_stride, style=style, auto_match_cart_resources=auto_match_cart_resources, in_cart_res=None, device=device)

        self.stage_coords = [out_coords]

        stem = nn.Sequential(
            KNNConvLayer(in_chans, dims[0], k=49, in_coords=self.in_coords, out_coords=out_coords, device=device, bias=True, **kwargs),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            in_coords, out_coords, out_cart_res = get_in_out_coords(out_coords.resolution, fov, cmf_a, first_stride, style=style, auto_match_cart_resources=auto_match_cart_resources, in_cart_res=out_cart_res, device=device)
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    KNNConvLayer(dims[i], dims[i+1], k=4, in_coords=in_coords, out_coords=out_coords, device=device, bias=True, **kwargs),
            )
            self.downsample_layers.append(downsample_layer)
            self.stage_coords.append(out_coords)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], coords=self.stage_coords[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        if num_classes is not None:
            self.head = nn.Linear(dims[-1], num_classes)
            self.head.weight.data.mul_(head_init_scale)
            self.head.bias.data.mul_(head_init_scale)   
        else:
            self.head = None

        self.apply(self._init_weights)

        self.out_coords = SamplingCoords(fov, cmf_a, 1, None, style=style, device=device)
        self.total_embed_dim = dims[-1]

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear, KNNDepthwiseConvLayer)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-1])) # global average pooling, (B, C, N) -> (B, C)

    def forward(self, x):
        x = self.forward_features(x)
        if self.head is not None:   
            x = self.head(x)
        else:
            x = x.unsqueeze(-1)
        return x
    

@add_to_all(__all__)
def knnconvnextv2_atto(**kwargs):
    """Create a KNN ConvNeXtV2 Atto model.
    
    Args:
        **kwargs: Additional keyword arguments passed to ConvNeXtV2.
    
    Returns:
        ConvNeXtV2: The model instance.
    """
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs)
    return model


@add_to_all(__all__)
def knnconvnextv2_femto(**kwargs):
    """Create a KNN ConvNeXtV2 Femto model.
    
    Args:
        **kwargs: Additional keyword arguments passed to ConvNeXtV2.
    
    Returns:
        ConvNeXtV2: The model instance.
    """
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], **kwargs)
    return model


@add_to_all(__all__)
def knnconvnextv2_pico(**kwargs):
    """Create a KNN ConvNeXtV2 Pico model.
    
    Args:
        **kwargs: Additional keyword arguments passed to ConvNeXtV2.
    
    Returns:
        ConvNeXtV2: The model instance.
    """
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], **kwargs)
    return model


@add_to_all(__all__)
def knnconvnextv2_nano(**kwargs):
    """Create a KNN ConvNeXtV2 Nano model.
    
    Args:
        **kwargs: Additional keyword arguments passed to ConvNeXtV2.
    
    Returns:
        ConvNeXtV2: The model instance.
    """
    model = ConvNeXtV2(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], **kwargs)
    return model


@add_to_all(__all__)
def knnconvnextv2_tiny(**kwargs):
    """Create a KNN ConvNeXtV2 Tiny model.
    
    Args:
        **kwargs: Additional keyword arguments passed to ConvNeXtV2.
    
    Returns:
        ConvNeXtV2: The model instance.
    """
    model = ConvNeXtV2(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    return model


@add_to_all(__all__)
def knnconvnextv2_base(**kwargs):
    """Create a KNN ConvNeXtV2 Base model.
    
    Args:
        **kwargs: Additional keyword arguments passed to ConvNeXtV2.
    
    Returns:
        ConvNeXtV2: The model instance.
    """
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    return model


@add_to_all(__all__)
def knnconvnextv2_large(**kwargs):
    """Create a KNN ConvNeXtV2 Large model.
    
    Args:
        **kwargs: Additional keyword arguments passed to ConvNeXtV2.
    
    Returns:
        ConvNeXtV2: The model instance.
    """
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    return model


@add_to_all(__all__)
def knnconvnextv2_huge(**kwargs):
    """Create a KNN ConvNeXtV2 Huge model.
    
    Args:
        **kwargs: Additional keyword arguments passed to ConvNeXtV2.
    
    Returns:
        ConvNeXtV2: The model instance.
    """
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816], **kwargs)
    return model