"""Demo utilities for FoviNet.

This module provides helper functions for quickly loading and preparing
images for model inference demonstrations.
"""
import torchvision.transforms.functional as TF
from PIL import Image

__all__ = ['get_image_as_batch']

def get_image_as_batch(path='shark.png', device='cuda'):
    """Load an image and prepare it as a normalized batch tensor.
    
    Loads an image, crops it to a square (center crop), converts to tensor,
    and applies ImageNet normalization.
    
    Args:
        path (str, optional): Path to the image file. Defaults to 'shark.png'.
        device (str, optional): Device to place the tensor on. Defaults to 'cuda'.
        
    Returns:
        torch.Tensor: Normalized image tensor of shape (1, 3, H, W) where
            H and W are equal (square crop of the minimum dimension).
    """
    img = Image.open(path).convert('RGB')
    img = TF.center_crop(img, min(img.size))
    batch = TF.to_tensor(img).unsqueeze(0).to(device)
    batch = TF.normalize(batch, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return batch