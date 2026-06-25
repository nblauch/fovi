"""Demo utilities for FoviNet.

This module provides helper functions for quickly loading and preparing
images for model inference demonstrations.
"""
import torchvision.transforms.functional as TF
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

__all__ = ['get_image_as_batch', 'load_image_for_sampling']

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
    batch = TF.normalize(batch, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    return batch


def load_image_for_sampling(path, device='cuda', normalize=True, center_crop=False):
    """Load an image for foveated sampling demos.

    By default keeps the native aspect ratio. Optionally center-crops to a
    square, matching ``get_image_as_batch``.

    Args:
        path (str): Path to the image file.
        device (str, optional): Device for the batch tensor. Defaults to 'cuda'.
        normalize (bool, optional): Apply ImageNet normalization to the batch.
            Defaults to True.
        center_crop (bool, optional): Center-crop to a square before loading.
            Defaults to False.

    Returns:
        tuple: (batch, display_rgb, height, width) where batch has shape
            (1, 3, H, W), display_rgb is (H, W, 3) float in [0, 1].
    """
    img = Image.open(path).convert('RGB')
    if center_crop:
        img = TF.center_crop(img, min(img.size))
    tensor = TF.to_tensor(img).unsqueeze(0)
    display = tensor[0].permute(1, 2, 0).numpy()
    if normalize:
        tensor = TF.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    batch = tensor.to(device)
    h, w = batch.shape[2], batch.shape[3]
    return batch, display, h, w