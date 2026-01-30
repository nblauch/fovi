import numpy as np
from .fastaugs import transforms as fastT
from . import add_to_all
from . import IMAGENET_MEAN, IMAGENET_STD

__all__ = ['IMAGENET_MEAN', 'IMAGENET_STD']


@add_to_all(__all__)
def get_std_transforms(where, flip, color_jitter, gray, blur, device, dtype, pointcloud_mode=False, solarize=False, normalize=True):
    """Build standard augmentation pipelines for training.
    
    Creates three transform pipelines (loader, pre-warp, post-warp) that can
    be applied at different stages of the data processing pipeline.
    
    Args:
        where (str): Where to apply transforms. One of 'loader', 'pre_warp', 
            or 'post_warp'. Determines which pipeline contains the augmentations.
        flip (bool): Whether to include random horizontal flip.
        color_jitter (bool): Whether to include color jitter augmentation.
        gray (bool): Whether to include random grayscale.
        blur (bool): Whether to include Gaussian blur.
        device: Device to place transforms on.
        dtype: Data type for tensors.
        pointcloud_mode (bool, optional): Whether to use pointcloud-compatible
            transforms. Defaults to False.
        solarize (bool, optional): Whether to include random solarization.
            Defaults to False.
        normalize (bool, optional): Whether to include ImageNet normalization.
            Defaults to True.
            
    Returns:
        tuple: (loader_transforms, pre_transforms, post_transforms) where each
            is either a Compose object or None if empty.
    """
    loader_transforms = [fastT.ToTorchImage(device, dtype=dtype, from_numpy=True)]
    pre_transforms = []
    post_transforms = []

    assert where in ['loader', 'pre_warp', 'post_warp']

    if flip:
        if where != 'loader':
            print('Note: horizontal flip always done in the loader, to avoid differences across fixations')
        loader_transforms.append(fastT.RandomHorizontalFlip())

    if where in ['loader', 'pre_warp']:
        # same transforms, determines whether they are the same for all fixations after the initial loading
        tmp_transforms = []
        if color_jitter:
            tmp_transforms.append(fastT.RandomColorJitter(p=0.8, hue=0.10, saturation=0.20, value=0.40, contrast=0.40))
        if gray:
            tmp_transforms.append(fastT.RandomGrayscale(p=0.2))
        if solarize:
            tmp_transforms.append(fastT.RandomSolarization(p=0.2, threshold=0.5))
        if normalize:
            tmp_transforms.append(fastT.NormalizeGPU(mean=IMAGENET_MEAN, std=IMAGENET_STD, device=device, inplace=True))
        if blur:
            tmp_transforms.append(fastT.RandomGaussianBlur(kernel_size=(5, 9), sigma_range=(0.1, 2), device=device))
        if where == 'pre_warp':
            pre_transforms.extend(tmp_transforms)
        else:
            loader_transforms.extend(tmp_transforms)
    else:
        if pointcloud_mode:
            print('WARNING: some transforms (blur, solarize) not yet implemented for pointcloud mode, thus must be done pre-warp')
            if blur:
                pre_transforms.append(fastT.RandomGaussianBlur(kernel_size=(5, 9), sigma_range=(0.1, 2), device=device))
            if solarize:
                pre_transforms.append(fastT.RandomSolarization(p=0.2, threshold=0.5))
            if color_jitter:
                post_transforms.append(fastT.RandomColorJitter(p=0.8, hue=0.10, saturation=0.20, value=0.40, contrast=0.40))
            if gray:
                post_transforms.append(fastT.RandomGrayscale(p=0.2))
            if normalize:
                post_transforms.append(fastT.NormalizeGPU(mean=IMAGENET_MEAN, std=IMAGENET_STD, device=device, inplace=True))
        else:
            if color_jitter:
                post_transforms.append(fastT.RandomColorJitter(p=0.8, hue=0.10, saturation=0.20, value=0.40, contrast=0.40))
            if gray:
                post_transforms.append(fastT.RandomGrayscale(p=0.2))
            if solarize:
                post_transforms.append(fastT.RandomSolarization(p=0.2, threshold=0.5))
            if normalize:
                post_transforms.append(fastT.NormalizeGPU(mean=IMAGENET_MEAN, std=IMAGENET_STD, device=device, inplace=True))
            if blur:
                post_transforms.append(fastT.RandomGaussianBlur(kernel_size=(5, 9), sigma_range=(0.1, 2), device=device))
    
    loader_transforms = fastT.Compose(loader_transforms) if len(loader_transforms) > 0 else None
    pre_transforms = fastT.Compose(pre_transforms) if len(pre_transforms) > 0 else None
    post_transforms = fastT.Compose(post_transforms) if len(post_transforms) > 0 else None
    
    return loader_transforms, pre_transforms, post_transforms