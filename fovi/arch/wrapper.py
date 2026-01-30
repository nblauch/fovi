import torch.nn as nn
from ..utils import add_to_all

__all__ = []


@add_to_all(__all__)
class BackboneProjectorWrapper(nn.Module):
    """
    Wrapper for backbone models with optional projection heads.
    
    Includes support for self-supervised learning projections
    
    Args:
        backbone (nn.Module): The main backbone.
        projector (nn.Module): Projection head for downstream tasks.
        projector_ssl (nn.Module, optional): Projection head for self-supervised learning.
    """
    
    def __init__(self, backbone, projector, projector_ssl=None):
        super().__init__()
        self.backbone = backbone
        self.projector = projector
        self.projector_ssl = projector_ssl

    def forward(self, x, return_layer_outputs=False, apply_mlp=True):
        """        
        Args:
            x (torch.Tensor): Input tensor.
            return_layer_outputs (bool): Whether to return intermediate layer outputs.
            apply_mlp (bool): Whether to apply the projection head.
            
        Returns:
            torch.Tensor or tuple: Output features and optionally layer outputs.
        """
        # Forward through backbone
        features = self.backbone(x)   

        # Apply projection if requested
        if apply_mlp:
            features, layer_outputs = self.projector(features)
        else:
            layer_outputs = None
        
        if return_layer_outputs:
            return features, layer_outputs
        else:
            return features