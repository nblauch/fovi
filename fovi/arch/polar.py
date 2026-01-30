import torch.nn as nn
from ..utils import add_to_all

__all__ = []

@add_to_all(__all__)
class PolarPadder(nn.Module):
    """Padding layer for polar coordinate representations.
    
    Applies circular padding along the angular dimension (to handle wraparound)
    and zero padding along the radial dimension.
    
    Args:
        pad (int): Amount of padding to apply on each side.
    """
    def __init__(self, pad):
        """Initialize the PolarPadder.
        
        Args:
            pad (int): Amount of padding to apply on each side.
        """
        super().__init__()
        self.pad = pad
    
    def forward(self, inputs):
        """Apply polar-aware padding to the input tensor.
        
        Args:
            inputs (torch.Tensor): Input tensor of shape (batch, channels, height, width)
                where height corresponds to radial and width to angular dimension.
                
        Returns:
            torch.Tensor: Padded tensor with circular padding on angular dimension
                and zero padding on radial dimension.
        """
        # circular padding for angular dimension
        inputs = nn.functional.pad(inputs, pad=(0, 0, self.pad, self.pad), mode='circular')
        # normal 0 padding for radial dimension
        inputs = nn.functional.pad(inputs, pad=(self.pad, self.pad, 0, 0), mode='constant')
        return inputs
    

@add_to_all(__all__)
def convert_model_to_polar(model, device, dtype):
    """
    convert a model to use polar coordinates

    Note: this only supports replacing conv2d, maxpool2d, and avgpool2d
        It notably does not support adaptiveavgpool2d, since we don't know the padding needed. however this shouldn't be a problem
        as these layers are usually used only once at the end of the network, before an FC layer which can aggregate across the angular dimension.
    """
    for name, module in dict(model.named_modules()).items():
        if isinstance(module, nn.Conv2d):
            # replace convs with polar convs
            new_conv = nn.Sequential(
                PolarPadder(module.kernel_size[0]//2),
                nn.Conv2d(module.in_channels, module.out_channels, kernel_size=module.kernel_size, stride=module.stride,
                     padding=0, groups=module.groups, bias=module.bias is not None, dilation=module.dilation, device=device, dtype=dtype),
            )
            print(f'Replacing {name} with polar conv')
            setattr(model, name, new_conv)

        elif isinstance(module, nn.MaxPool2d):
            # replace maxpool with polar maxpool
            new_maxpool = nn.Sequential(
                PolarPadder(module.kernel_size//2),
                nn.MaxPool2d(kernel_size=module.kernel_size, stride=module.stride, padding=0, device=device, dtype=dtype),
            )
            print(f'Replacing {name} with polar maxpool')
            setattr(model, name, new_maxpool)

        elif isinstance(module, nn.AvgPool2d):
            # replace avgpool with polar avgpool
            new_avgpool = nn.Sequential(
                PolarPadder(module.kernel_size//2),
                nn.AvgPool2d(kernel_size=module.kernel_size, stride=module.stride, padding=0, device=device, dtype=dtype),
            )
            print(f'Replacing {name} with polar avgpool')
            setattr(model, name, new_avgpool)
    return model
