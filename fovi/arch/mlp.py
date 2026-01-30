import torch.nn as nn
from collections import OrderedDict
from functools import partial
from ..utils import add_to_all

__all__ = []

@add_to_all(__all__)
class LayerBlock(nn.Sequential):
    """Simple wrapper that turns a list into a Sequential layer, dropping None's.
    
    This class filters out None values from the input arguments and creates
    a Sequential layer with the remaining valid layers.
    """
    def __init__(self, *args):
        assert len(args) >= 1, "LayerBlock must include at least 1 layer!"
        
        layers = [layer for layer in args if layer is not None]

        super(LayerBlock, self).__init__(*layers)

@add_to_all(__all__)
def get_mlp(repr_size=9216,
            mlp='8192-8192-8192',
            inp=None,
            linear=nn.Linear,
            act=partial(nn.ReLU, inplace=True),
            norm=nn.BatchNorm1d,
            dropout=partial(nn.Dropout, p=.5),
            dropout_all=False,
            proj_relu=False,
            output_bias=False,
            l2norm=False,
            out=None,
            offset_idx=5):
    """Create a multi-layer perceptron (MLP) with configurable architecture.
    
    Args:
        repr_size (int, optional): Input representation size. Defaults to 9216.
        mlp (str or int, optional): MLP architecture specification as a string
            of layer sizes separated by dashes (e.g., '8192-8192-8192') or
            a single integer. Defaults to '8192-8192-8192'.
        inp (callable, optional): Input layer function that takes (idx, is_output)
            and returns a layer or None. Defaults to None.
        linear (nn.Module, optional): Linear layer class to use. Defaults to nn.Linear.
        act (callable, optional): Activation function to use. Defaults to
            partial(nn.ReLU, inplace=True).
        norm (nn.Module, optional): Normalization layer class to use. Defaults to
            nn.BatchNorm1d.
        dropout (callable, optional): Dropout layer function. Defaults to
            partial(nn.Dropout, p=.5).
        dropout_all (bool, optional): Whether to apply dropout to all layers
            including the output layer. Defaults to False.
        proj_relu (bool, optional): Whether to apply ReLU to the projection
            (output) layer. Defaults to False.
        output_bias (bool, optional): Whether to use bias in the output layer.
            Defaults to False.
        l2norm (bool, optional): Whether to apply L2 normalization to the output.
            Defaults to False.
        out (callable, optional): Output layer function that takes (idx, is_output)
            and returns a layer or None. Defaults to None.
        offset_idx (int, optional): Offset for block naming. Defaults to 5.
    
    Returns:
        MLPWrapper: A wrapped MLP module that can return embeddings and
            intermediate layer outputs.
    """
    
    if isinstance(mlp, int):
        mlp = str(mlp)
    
    if mlp is None or not len(mlp):
        return MLPWrapper([]) # empty MLP with desired output format -> just passes features through
    
    mlp_spec=f'{repr_size}-{mlp}'
    mlp = [int(v) for v in mlp.split("-")]
    layers = []
    in_c = repr_size
    for idx,out_c in enumerate(mlp):
        is_output = idx==(len(mlp)-1)
        bias = output_bias if is_output else True
        inp_layer = None if inp is None else inp(idx, is_output)
        if idx == 0 and repr_size == 'lazy':
            linear_layer = nn.LazyLinear(out_c, bias=bias)
        else:
            linear_layer = linear(in_c,out_c,bias=bias)
        act_layer = None if (is_output and proj_relu==False and len(mlp)>1) else act()
        norm_layer = None if (is_output or norm is None and len(mlp)>1) else  norm(out_c)
        dropout_layer = None if ((is_output and not dropout_all and len(mlp)>1) or dropout is None) else dropout()
        l2_layer =  LPNormalize(2) if (is_output and l2norm) else None
        out_layer = None if (out is None) else out(idx, is_output)
        
        block_name = f'fc_block_{idx+1+offset_idx}'
        block = LayerBlock(inp_layer,
                           dropout_layer,
                           linear_layer, 
                           norm_layer,
                           act_layer, 
                           l2_layer,
                           out_layer)
        layers.append((block_name, block))
        in_c = out_c
    
    mlp = nn.Sequential(OrderedDict(layers))

    mlp = MLPWrapper(mlp)
    
    return mlp

@add_to_all(__all__)
class MLPWrapper(nn.Module):
    """Wrapper for MLP that provides additional functionality for layer outputs.
    
    This wrapper allows the MLP to return both the final embeddings and
    intermediate layer outputs for analysis or visualization purposes.
    """
    
    def __init__(self, mlp):
        """Initialize the MLP wrapper.
        
        Args:
            mlp (nn.Module): The MLP module to wrap.
        """
        super(MLPWrapper, self).__init__()

        self.layers = mlp
    
    def forward(self, x, return_layer_outputs=True):
        """Forward pass through the MLP.
        
        Args:
            x (torch.Tensor): Input tensor to process.
            return_layer_outputs (bool, optional): Whether to return intermediate
                layer outputs along with the final embeddings. Defaults to True.
        
        Returns:
            torch.Tensor or tuple: If return_layer_outputs is True, returns a tuple
                of (embeddings, list_outputs) where embeddings is the final output
                and list_outputs contains all intermediate layer outputs. If False,
                returns only the final embeddings.
        """
        x = x.flatten(start_dim=1)
        list_outputs = [x.detach()]
        for layer in self.layers:
            x = layer(x)
            list_outputs.append(x.detach())  # Store the output after each layer

        # The final value of x is the embedding
        embeddings = x
        
        if return_layer_outputs:
            return embeddings, list_outputs
        
        return embeddings
    
class LPNormalize(nn.Module):
    """Lp normalization layer.
    
    This module normalizes input tensors using Lp normalization along the
    feature dimension (dimension 1).
    """

    def __init__(self, power=2):
        """Initialize the Lp normalization layer.
        
        Args:
            power (int, optional): The power for Lp normalization. Defaults to 2
                (L2 normalization).
        """
        super(LPNormalize, self).__init__()
        self.power = power

    def forward(self, x):
        """Apply Lp normalization to the input tensor.
        
        Args:
            x (torch.Tensor): Input tensor to normalize.
        
        Returns:
            torch.Tensor: Lp normalized tensor.
        """
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out