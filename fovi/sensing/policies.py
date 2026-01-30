import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms.functional as TF
import torch.nn.functional as F

from .coords import transform_sampling_grid
from .retina import GridSampler
from ..utils import add_to_all

__all__ = []

@add_to_all(__all__)
class BaseSaccadePolicy(nn.Module):
    """Base class for SaccadeNet saccade/fixation policies.
    
    Provides functionality for sampling multiple fixation points from images.
    """    
    def __init__(self, retinal_transform, n_fixations):
        """Initialize the base saccade policy.
        
        Args:
            retinal_transform (RetinalTransform): The retinal transform object used to 
                apply retinal transformations to the images.
            n_fixations (int): The number of fixations to generate per image.
        """
        super().__init__()
        self.retinal_transform = retinal_transform
        self.n_fixations = n_fixations
        # convenience access
        self.fixation_size = self.retinal_transform.fixation_size
        self.dtype = self.retinal_transform.dtype
        self.device = self.retinal_transform.device


    def get_random_crop(self, height, width, scale, ratio):
        """
        Generate a random crop with specified scale and aspect ratio.
        
        Args:
            height (int): Image height.
            width (int): Image width.
            scale (float or tuple): Scale factor(s) for crop area.
            ratio (float or tuple): Aspect ratio(s) for the crop.
            
        Returns:
            tuple:
                - list: Normalized fixation center [y, x]
                - list: Fixation size [height, width]
        """
        area = height * width
        log_ratio = np.log(ratio)
        for _ in range(10):
            if hasattr(scale, '__len__'):
                target_area = area * np.random.uniform(scale[0], scale[1])
            else:
                target_area = area * scale
            if hasattr(ratio, '__len__'):
                aspect_ratio = np.exp(np.random.uniform(log_ratio[0], log_ratio[1]))
            else:
                aspect_ratio = ratio
            w = int(np.round(np.sqrt(target_area * aspect_ratio)))
            h = int(np.round(np.sqrt(target_area / aspect_ratio)))
            if 0 < w <= width and 0 < h <= height:
                i = int(np.random.uniform(0, height - h + 1))
                j = int(np.random.uniform(0, width - w + 1))
                # calculate normalized center of fixation
                fixation = [(i+h/2)/height,(j+w/2)/width]
                fixation_size = [h,w]
                return fixation, fixation_size
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2

        # calculate normalized center of fixation
        fixation = [(i+h/2)/height,(j+w/2)/width]
        fixation_size = [h,w]

        return fixation, fixation_size

    def get_random_nearcenter_fixation(self, height, width, scale, ratio, normalized_dist_from_center):
        """
        Generate a random fixation near the center with specified constraints.
        
        Args:
            height (int): Image height.
            width (int): Image width.
            scale (float or tuple): Scale factor(s) for crop area.
            ratio (float or tuple): Aspect ratio(s) for the crop.
            normalized_dist_from_center (float): Maximum normalized distance from center.
            
        Returns:
            tuple:
                - list: Normalized fixation center [y, x]
                - list: Fixation size [height, width]
        """
        area = height * width
        log_ratio = np.log(ratio)

        if hasattr(scale, '__len__'):
            target_area = area * np.random.uniform(scale[0], scale[1])
        else:
            target_area = area * scale
        if hasattr(ratio, '__len__'):
            aspect_ratio = np.exp(np.random.uniform(log_ratio[0], log_ratio[1]))
        else:
            aspect_ratio = ratio

        if isinstance(target_area, torch.Tensor):
            target_area = target_area.item()

        w = int(round(np.sqrt(target_area * aspect_ratio)))
        h = int(round(np.sqrt(target_area / aspect_ratio)))
        fixation_size = [h,w]

        min_frac = 0.5-normalized_dist_from_center
        max_frac = 0.5+normalized_dist_from_center
        fixation = [torch.tensor(np.random.uniform(min_frac, max_frac)), torch.tensor(np.random.uniform(min_frac, max_frac))]

        return fixation, fixation_size

    def sample_fixations(self, img_size, n=1, area_range=None, ratio=None, norm_dist_from_center=None):
        """
        Sample multiple fixations for batch processing.
        
        Args:
            img_size (tuple): Image size (height, width).
            n (int): Number of fixations to sample. Defaults to 1.
            area_range: Scale range for crop area. Defaults to None.
            ratio: Aspect ratio range. Defaults to None.
            norm_dist_from_center (float, optional): Maximum normalized distance from center. Defaults to None.
            
        Returns:
            tuple:
                - torch.Tensor: Fixation locations of shape (n, 2)
                - np.ndarray: Fixation sizes of shape (n, 2)
        """
        if area_range is None:
            area_range = self.crop_area_range if self.training else self.val_crop_size
        if ratio is None:
            ratio = self.crop_aspect_range if self.training else 1
        fixations = []
        fixation_sizes = []
        for _ in range(n):
            if norm_dist_from_center is not None:
                fixation, fixation_size = self.get_random_nearcenter_fixation(img_size[0], img_size[1], scale=area_range, ratio=ratio, 
                    normalized_dist_from_center=norm_dist_from_center,
                    )
            else:
                fixation, fixation_size = self.get_random_crop(img_size[0], img_size[1], scale=area_range, ratio=ratio)
            fixations.append(fixation)
            fixation_sizes.append(fixation_size)
        fixations = torch.tensor(fixations, dtype=self.dtype, device=self.device)
        fixation_sizes = torch.tensor(fixation_sizes)
        return fixations, fixation_sizes


@add_to_all(__all__)
class MultiRandomSaccadePolicy(BaseSaccadePolicy):
    """Multi-random saccade policy for generating fixations in images.
    
    This policy randomly selects multiple fixation points within the image,
    with configurable constraints on crop area, aspect ratio, and position.

    Attributes:
        retinal_transform (RetinalTransform): The retinal transform object used for sampling and transforming images
        n_fixations (int): The number of fixations to generate.
        fixation_size (int): The size of the fixation area.
        multi_policy (bool): Indicates if the policy is a multi-policy 
            (i.e., it can handle multiple fixations).
        nonrandom_val (bool): Whether to make validation fixations deterministic.
        norm_dist_from_center (float): If not None, changes how fixations are sampled. 
            Rather than finding any valid crop, it takes a fixation within 
            norm_dist_from_center fractional distance from the center of the image.
    """
    def __init__(self, retinal_transform, n_fixations=2, 
                    crop_area_range=[0.08, 1], 
                    add_aspect_variation=False, 
                    nonrandom_val=False, 
                    val_crop_size=1,
                    nonrandom_first=False, 
                    norm_dist_from_center=None,
                    ):
        """Initialize the multi-random saccade policy.
        
        Args:
            retinal_transform (RetinalTransform): The retinal transform object used to 
                apply retinal transformations to the images.
            n_fixations (int, optional): The number of fixations to generate. Defaults to 2.
            crop_area_range (list, optional): Range of crop area fractions [min, max]. 
                Defaults to [0.08, 1].
            add_aspect_variation (bool, optional): Whether to add aspect ratio variation 
                to crops. Defaults to False.
            nonrandom_val (bool, optional): Whether to make validation fixations 
                deterministic (center). Defaults to False.
            val_crop_size (float, optional): Crop size fraction for validation. Defaults to 1.
            nonrandom_first (bool, optional): Whether to force the first fixation to be 
                at center. Defaults to False.
            norm_dist_from_center (float, optional): Maximum normalized distance from center 
                for fixation sampling. Defaults to None.
        """
        super().__init__(retinal_transform, n_fixations)
        self.crop_area_range = crop_area_range
        self.crop_aspect_range = [3/4, 4/3] if add_aspect_variation else 1
        self.val_crop_size = val_crop_size
        self.nonrandom_first = nonrandom_first
        self.multi_policy = True
        self.nonrandom_val = nonrandom_val
        self.nonrandom_first = nonrandom_first
        self.norm_dist_from_center = norm_dist_from_center
    
    def forward(self, x, n_fixations=None, fixations=None, fixation_size=None, area_range=None):
        """
        Forward pass for the MultiRandomSaccadePolicy.

        This method generates multiple random fixations for the input images and applies the retinal transform to each fixation.

        Args:
            x (torch.Tensor): The input images of shape (n, c, h, w), where n is the batch size, c is the number of channels, h is the height, and w is the width.
            n_fixations (int, optional): The number of fixations to generate. Defaults to None, which uses the default number of fixations set in the policy.
            fixations (list of torch.Tensor, optional): A list of pre-defined fixations. Defaults to None, which generates random fixations.
            fixation_size (int, optional): The size of the fixation area. Defaults to None, which uses the default fixation size set in the policy.
            area_range (tuple, optional): The range of areas to sample from for the fixation size. Defaults to None.

        Returns:
            dict: Dictionary containing:
                - x_fixs (torch.Tensor): The transformed images.
                - fixations (torch.Tensor): The fixation coordinates.
                - fixation_sizes (torch.Tensor): The fixation sizes.
                - fix_deltas (torch.Tensor): The fixation deltas.
        """
        if n_fixations is None:
            n_fixations = self.n_fixations

        assert fixations is None or len(fixations) == n_fixations    

        fixation_sizes = []
        fixations_ = []
        for fix_i in range(n_fixations):
            fix_, fix_size = self.sample_fixations(x.shape[2:], n=x.shape[0], area_range=area_range, norm_dist_from_center=self.norm_dist_from_center)
            fixation_sizes.append(fix_size)
            if fix_i == 0 and self.nonrandom_first:
                fix_ = torch.tensor([0.5, 0.5]).to(dtype=fix_.dtype, device=fix_.device).unsqueeze(0).expand(x.shape[0],2)
            fixations_.append(fix_)
        if fixations is None:
            if self.nonrandom_val and not self.training:
                fixations = [0.5, 0.5]*n_fixations
            else:
                fixations = fixations_
        else:
            assert len(fixations) == n_fixations
            for ii, fixation in enumerate(fixations):
                if len(fixation) == 2:
                    fixation = torch.ones(x.shape[0], 2)*fixation
                    fixations[ii] = fixation
                 
        x_fixs = []
        fix_deltas = []
        for ii, (fixation, fixation_size) in enumerate(zip(fixations, fixation_sizes)):
            x_fix = self.retinal_transform(x, fixation, fixation_size=fixation_size)  
            x_fixs.append(x_fix)
            if ii > 0:
                fix_deltas.append(fixation - fixations[ii-1])
            else:
                fix_deltas.append(torch.zeros(x.shape[0], 2, device=x.device))
        
        return {
            'x_fixs': torch.stack(x_fixs, dim=1), # (B, F, C, N)
            'fixations': torch.stack(fixations, dim=1), # (B, F, 2) 
            'fixation_sizes': torch.stack(fixation_sizes, dim=1), # (B, F, 2) 
            'fix_deltas': torch.stack(fix_deltas, dim=1), # (B, F, 2) 
            }
    
    def __repr__(self):
        return (f"MultiRandomSaccadePolicy(\n"
                f"  retinal_transform={self.retinal_transform},\n"
                f"  n_fixations={self.n_fixations},\n"
                f"  nonrandom_first={getattr(self, 'nonrandom_first', None)},\n"
                f"  nonrandom_val={getattr(self, 'nonrandom_val', None)},\n"
                f"  crop_area_range={getattr(self, 'crop_area_range', None)},\n"
                f"  add_aspect_variation={getattr(self, 'add_aspect_variation', None)},\n"
                f"  val_crop_size={getattr(self, 'val_crop_size', None)},\n"
                f"  norm_dist_from_center={getattr(self, 'norm_dist_from_center', None)}\n"
                ")")

@add_to_all(__all__)
class NoSaccadePolicy(BaseSaccadePolicy):
    """
    Simple wrapper that does not apply any fixations to the input images.

    Attributes:
        retinal_transform (RetinalTransform): The retinal transform object used to apply retinal transformations to the images.
    """
    def __init__(self, retinal_transform):
        """
        Args:
            retinal_transform (RetinalTransform): The retinal transform object used to apply retinal transformations to the images.
        """
        super().__init__(retinal_transform, 1)
        self.multi_policy = False

    def forward(self, x, f1=None, area_range=None, n_fixations=None, fixation_size=None, fixations=None):
        """
        Forward pass for the NoSaccadePolicy.

        Args:
            x (torch.Tensor): The input image.
            f1 (torch.Tensor, optional): The first fixation coordinates. Must be None in order to use a center fixation. 
            area_range (tuple, optional): Unused, for compatibility with other policies.
            n_fixations (int, optional): Unused, for compatibility with other policies.

        Returns:
            dict: Dictionary containing:
                - x_fixs (torch.Tensor): The transformed image.
                - fixations (torch.Tensor): The fixation coordinates.
                - fixation_sizes (torch.Tensor): The fixation sizes.
                - fix_deltas (torch.Tensor): The fixation deltas.
        """
        assert f1 is None and area_range is None and (n_fixations is None or n_fixations == 1) and fixation_size is None and fixations is None
        x_f1 = self.retinal_transform(x, f1)
        x_f1 = x_f1.unsqueeze(1)
        return {
            'x_fixs': x_f1,
            'fixations': 0.5*torch.ones((x_f1.shape[0], 1, 2)),
            'fixation_sizes': self.fixation_size*torch.ones((x_f1.shape[0], 1, 2)),
            'fix_deltas': torch.zeros(x_f1.shape[0], 1, 2),
        }
    
    def __repr__(self):
        return (f"NoSaccadePolicy(\n"
                f"  retinal_transform={self.retinal_transform},\n"
                f"  n_fixations={self.n_fixations}\n"
                ")")


class PolicyRegistry:
    """Registry for fixation policy builder functions.
    
    This registry stores builder functions that can construct policy instances
    from a SaccadeNet object. This allows external repositories to register
    custom policies without modifying the SaccadeNet code.
    """
    def __init__(self):
        self._builders = {}
    
    def register(self, name, builder_fn):
        """Register a policy builder function.
        
        Args:
            name (str): Policy name to register
            builder_fn (callable): Function that takes a SaccadeNet instance and returns a policy instance
        """
        self._builders[name] = builder_fn
    
    def get(self, name):
        """Get a policy builder by name.
        
        Args:
            name (str): Policy name to retrieve
            
        Returns:
            callable: Builder function for the policy
            
        Raises:
            ValueError: If policy name is not found in registry
        """
        if name not in self._builders:
            raise ValueError(f"Policy '{name}' not found. Available policies: {list(self._builders.keys())}")
        return self._builders[name]
    
    def has(self, name):
        """Check if a policy is registered.
        
        Args:
            name (str): Policy name to check
            
        Returns:
            bool: True if policy is registered, False otherwise
        """
        return name in self._builders
    
    def __repr__(self):
        """Return a string representation of the registry showing all registered policies."""
        if not self._builders:
            return "PolicyRegistry(no policies registered)"
        
        policies = sorted(self._builders.keys())
        policies_str = "\n  ".join(policies)
        return f"PolicyRegistry(\n  {policies_str}\n)"


# Module-level singleton instance
FIXATION_POLICY_REGISTRY = PolicyRegistry()


# Register built-in policies
# Note: Builder functions take a SaccadeNet instance and return a policy instance

# MultiRandomSaccadePolicy - basic version
FIXATION_POLICY_REGISTRY.register(
    'multi_random',
    lambda sn: MultiRandomSaccadePolicy(
        sn.retinal_transform, 
        sn.n_fixations,
        nonrandom_first=getattr(sn.cfg.saccades, 'nonrandom_first', False),
        crop_area_range=[sn.cfg.saccades.fixation_size_min_frac, sn.cfg.saccades.fixation_size_max_frac],
        add_aspect_variation=sn.cfg.saccades.add_aspect_variation,
        val_crop_size=sn.cfg.saccades.fixation_size_frac_val,
    )
)

# MultiRandomSaccadePolicy - near center variant
FIXATION_POLICY_REGISTRY.register(
    'multi_random_nearcenter',
    lambda sn: MultiRandomSaccadePolicy(
        sn.retinal_transform, 
        sn.n_fixations,
        nonrandom_first=getattr(sn.cfg.saccades, 'nonrandom_first', False),
        norm_dist_from_center=sn.cfg.saccades.nearcenter_dist,
        crop_area_range=[sn.cfg.saccades.fixation_size_min_frac, sn.cfg.saccades.fixation_size_max_frac],
        add_aspect_variation=sn.cfg.saccades.add_aspect_variation,
        val_crop_size=sn.cfg.saccades.fixation_size_frac_val,
    )
)

# MultiRandomSaccadePolicy - near center with nonrandom validation
FIXATION_POLICY_REGISTRY.register(
    'multi_random_nearcenter_train',
    lambda sn: MultiRandomSaccadePolicy(
        sn.retinal_transform, 
        sn.n_fixations,
        norm_dist_from_center=sn.cfg.saccades.nearcenter_dist,
        nonrandom_first=getattr(sn.cfg.saccades, 'nonrandom_first', False),
        nonrandom_val=True,
        crop_area_range=[sn.cfg.saccades.fixation_size_min_frac, sn.cfg.saccades.fixation_size_max_frac],
        add_aspect_variation=sn.cfg.saccades.add_aspect_variation,
        val_crop_size=sn.cfg.saccades.fixation_size_frac_val,
    )
)

# MultiRandomSaccadePolicy - with nonrandom validation
FIXATION_POLICY_REGISTRY.register(
    'multi_random_train',
    lambda sn: MultiRandomSaccadePolicy(
        sn.retinal_transform, 
        sn.n_fixations,
        nonrandom_first=getattr(sn.cfg.saccades, 'nonrandom_first', False),
        nonrandom_val=True,
        crop_area_range=[sn.cfg.saccades.fixation_size_min_frac, sn.cfg.saccades.fixation_size_max_frac],
        add_aspect_variation=sn.cfg.saccades.add_aspect_variation,
        val_crop_size=sn.cfg.saccades.fixation_size_frac_val,
    )
)

# NoSaccadePolicy
FIXATION_POLICY_REGISTRY.register(
    'none',
    lambda sn: NoSaccadePolicy(sn.retinal_transform)
)

# Register class names as aliases for forward compatibility
FIXATION_POLICY_REGISTRY.register('MultiRandomSaccadePolicy', FIXATION_POLICY_REGISTRY.get('multi_random'))
FIXATION_POLICY_REGISTRY.register('NoSaccadePolicy', FIXATION_POLICY_REGISTRY.get('none'))

# Add PolicyRegistry and FIXATION_POLICY_REGISTRY to __all__
__all__.extend(['PolicyRegistry', 'FIXATION_POLICY_REGISTRY'])