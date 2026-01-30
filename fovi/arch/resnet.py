from torch import nn
import torch
from torch import Tensor
from typing import Type, Any, Callable, Union, List, Optional
from torchvision.models.resnet import BasicBlock, Bottleneck, conv1x1
from torchvision.models import ResNet as ResNet_

from .mlp import get_mlp, MLPWrapper
from .wrapper import BackboneProjectorWrapper
from .polar import PolarPadder
from ..utils import add_to_all

__all__ = []

@add_to_all(__all__)
def conv3x3_polar(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """Create a 3x3 convolution with polar coordinate padding.
    
    Args:
        in_planes (int): Number of input channels.
        out_planes (int): Number of output channels.
        stride (int, optional): Convolution stride. Defaults to 1.
        groups (int, optional): Number of groups for grouped convolution. Defaults to 1.
        dilation (int, optional): Dilation rate. Defaults to 1.
        
    Returns:
        nn.Sequential: A sequential module with PolarPadder and Conv2d.
    """
    return nn.Sequential(
            PolarPadder(pad=dilation),
            nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=0, groups=groups, bias=False, dilation=dilation),
    )


@add_to_all(__all__)
def conv_polar(in_planes: int, out_planes: int, kernel_size, pad:int, stride: int = 1, groups: int = 1, dilation: int = 1, **kwargs) -> nn.Conv2d:
    """Create a convolution with polar coordinate padding.
    
    Args:
        in_planes (int): Number of input channels.
        out_planes (int): Number of output channels.
        kernel_size (int or tuple): Size of the convolution kernel.
        pad (int): Amount of polar padding to apply.
        stride (int, optional): Convolution stride. Defaults to 1.
        groups (int, optional): Number of groups for grouped convolution. Defaults to 1.
        dilation (int, optional): Dilation rate. Defaults to 1.
        **kwargs: Additional keyword arguments passed to Conv2d.
        
    Returns:
        nn.Sequential: A sequential module with PolarPadder and Conv2d.
    """
    return nn.Sequential(
            PolarPadder(pad=pad),
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=0, groups=groups, dilation=dilation, **kwargs),
    )


@add_to_all(__all__)
class BasicBlockPolar(nn.Module):
    """Basic residual block for polar coordinate ResNet.
    
    A basic block with two 3x3 polar convolutions and a residual connection.
    Uses polar padding to handle wraparound in the angular dimension.
    
    Attributes:
        expansion (int): Channel expansion factor (always 1 for BasicBlock).
    """
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        """Initialize the BasicBlockPolar.
        
        Args:
            inplanes (int): Number of input channels.
            planes (int): Number of output channels.
            stride (int, optional): Stride for the first convolution. Defaults to 1.
            downsample (nn.Module, optional): Downsampling module for residual path. 
                Defaults to None.
            groups (int, optional): Number of groups (must be 1). Defaults to 1.
            base_width (int, optional): Base width (must be 64). Defaults to 64.
            dilation (int, optional): Dilation rate (must be 1). Defaults to 1.
            norm_layer (callable, optional): Normalization layer factory. 
                Defaults to BatchNorm2d.
        """
        super(BasicBlockPolar, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3_polar(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3_polar(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


@add_to_all(__all__)
class BottleneckPolar(nn.Module):
    """Bottleneck residual block for polar coordinate ResNet.
    
    A bottleneck block with 1x1 -> 3x3 polar -> 1x1 convolutions and a residual connection.
    Uses polar padding for the 3x3 convolution to handle wraparound in the angular dimension.
    
    Note:
        Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
        while original implementation places the stride at the first 1x1 convolution(self.conv1)
        according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
        This variant is also known as ResNet V1.5 and improves accuracy according to
        https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    Attributes:
        expansion (int): Channel expansion factor (always 4 for Bottleneck).
    """
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        """Initialize the BottleneckPolar.
        
        Args:
            inplanes (int): Number of input channels.
            planes (int): Number of output channels (before expansion).
            stride (int, optional): Stride for the 3x3 convolution. Defaults to 1.
            downsample (nn.Module, optional): Downsampling module for residual path. 
                Defaults to None.
            groups (int, optional): Number of groups for grouped convolution. Defaults to 1.
            base_width (int, optional): Base width for computing intermediate channels. 
                Defaults to 64.
            dilation (int, optional): Dilation rate for 3x3 convolution. Defaults to 1.
            norm_layer (callable, optional): Normalization layer factory. 
                Defaults to BatchNorm2d.
        """
        super(BottleneckPolar, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3_polar(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


@add_to_all(__all__)
class ResNet(ResNet_):
    """ResNet with polar coordinate support and configurable strides.
    
    Extended ResNet implementation with support for polar coordinates and options
    to remove strides in the main blocks to preserve greater feature map resolution.
    Optionally removes pooling before first main block.
    
    Args:
        block (Type): Block class to use (BasicBlock, Bottleneck, or polar variants).
        layers (List[int]): Number of blocks in each of the 4 layers.
        num_classes (int, optional): Number of output classes. Defaults to 1000.
        zero_init_residual (bool, optional): Whether to zero-initialize the last BN 
            in each residual branch. Defaults to False.
        groups (int, optional): Number of groups for grouped convolution. Defaults to 1.
        width_per_group (int, optional): Base width per group. Defaults to 64.
        pre_block_pooling (bool, optional): Whether to apply pooling before first block. 
            Defaults to True.
        main_block_stride (int, optional): Stride for main blocks (normally 2). 
            Defaults to 1.
        polar (bool, optional): Whether to use polar coordinate padding. Defaults to False.
        no_fc (bool, optional): Whether to exclude the final FC layer. Defaults to False.
        out_map_size (int, optional): Output spatial size after adaptive pooling. 
            Defaults to 1.
        channel_mult (float, optional): Channel width multiplier. Defaults to 1.
        replace_stride_with_dilation (List[bool], optional): Whether to replace stride 
            with dilation in each of the last 3 layers. Defaults to None.
        norm_layer (callable, optional): Normalization layer factory. Defaults to BatchNorm2d.
    """
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck, BasicBlockPolar, BottleneckPolar]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        pre_block_pooling = True,
        main_block_stride = 1, # normally 2
        polar = False,
        no_fc = False,
        out_map_size=1,
        channel_mult=1,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        # super().__init__(block, layers, stride=1, zero_init_residual=zero_init_residual, **kwargs)
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.no_fc = no_fc

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        if polar:
            self.conv1 = nn.Sequential(
                PolarPadder(3),
                nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=0,
                               bias=False)
            )
        else:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                                bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        if pre_block_pooling:
            if polar:
                self.maxpool = nn.Sequential(
                    PolarPadder(1),
                    nn.MaxPool2d(kernel_size=3, stride=3, padding=0),
                )
            else:
                self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.maxpool = nn.Identity()
        self.layer1 = self._make_layer(block, int(channel_mult*64), layers[0])
        self.layer2 = self._make_layer(block, int(channel_mult*128), layers[1], stride=main_block_stride, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, int(channel_mult*256), layers[2], stride=main_block_stride, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, int(channel_mult*512), layers[3], stride=main_block_stride, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((out_map_size, out_map_size))
        if not self.no_fc:
            self.fc = nn.Linear(channel_mult*512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) or isinstance(m, BottleneckPolar):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) or isinstance(m, BasicBlockPolar):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        if not self.no_fc:
            x = self.fc(x)

        return x


def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck, BasicBlockPolar, BottleneckPolar]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        raise NotImplementedError()
    return model


@add_to_all(__all__)
def resnet18(pretrained: bool = False, progress: bool = True, polar=False, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', 
    BasicBlockPolar if polar else BasicBlock, 
    [2, 2, 2, 2], pretrained, progress, polar=polar, **kwargs)


@add_to_all(__all__)
def resnet34(pretrained: bool = False, progress: bool = True, polar=False, **kwargs: Any) -> ResNet:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlockPolar if polar else BasicBlock, 
    [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


@add_to_all(__all__)
def resnet50(pretrained: bool = False, progress: bool = True, polar=False, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', BottleneckPolar if polar else Bottleneck, 
    [3, 4, 6, 3], pretrained, progress, polar=polar,
                   **kwargs)


@add_to_all(__all__)
def resnet101(pretrained: bool = False, progress: bool = True, polar=False, **kwargs: Any) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', BottleneckPolar if polar else Bottleneck, 
    [3, 4, 23, 3], pretrained, progress, polar=polar,
                   **kwargs)


@torch.no_grad()
@add_to_all(__all__)
def get_repr_size(model, in_channels=3, img_size=224):
    model.eval()
    x = torch.rand(1,in_channels,img_size,img_size)
    out = model(x)
    
    return out.flatten(start_dim=1).shape[1]


@add_to_all(__all__)
def resnet_ssl(layers=50, mlp_kwargs='8192-8192-8192', weights=None, progress=True, polar=False, **kwargs: Any) -> ResNet:
    """
    wrapper for an SSL-style resnet model, used alongside Harvard Vision Lab model_rearing_workshop
    """
    fn = f'resnet{layers}'
    backbone = globals()[fn](pretrained=False, progress=progress, polar=polar, no_fc=True, **kwargs)
    # exec(f'backbone = resnet{layers}(pretrained=False, progress=progress, polar=polar, no_fc=True, **kwargs)')
    mlp_kwargs = {} if mlp_kwargs is None else mlp_kwargs
    repr_size = get_repr_size(backbone, in_channels=3)
    mlp = get_mlp(repr_size=repr_size, **mlp_kwargs)
    projector = MLPWrapper(mlp)
    model = BackboneProjectorWrapper(backbone, projector)

    if weights is not None:
        msg = model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))
        print(msg)
        
    return model