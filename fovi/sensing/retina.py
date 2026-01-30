import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from scipy.optimize import minimize_scalar

from .coords import find_desired_res
from .samplers import GaussianKNNGridSampler, KNNGridSampler, GridSampler
from ..utils import add_to_all

__all__ = []

@add_to_all(__all__)
class RetinalTransform(nn.Module):
    """
    Implements two computational hallmarks of retinal processing:
        - spatially non-uniform (foveated) spatial sampling of the visual field
        - spatially non-uniform (foveated) color representation

    Foveated spatial sampling is based on isotropic cortical magnification of the form CMF=1/(r+a),where: 
        - r=polar radius (eccentricity)
        - a is a parameter that controls the degree of foveation. smaller = more foveation. as a->infinity, we get uniform sampling.
        - Equal sampling in cortical space is assumed, and the visual coordinates are computed by back-projection to acquire the foveated sampling grid.
    This uses a CorticalSensorManifold module to represent the V1-like manifold of retina-like samples.

    Foveated color representation is implemented by modeling hue saturation as a 1D gaussian of the visual field eccentricity, using a GaussianColorDecay module. 
    - parameterized by self.sigma
    - can be turned off during eval mode with no_color_val=True
    
    """
    def __init__(self, resolution, start_res=256,
                 fov=16, cmf_a=0.5, 
                 style='isotropic',
                 sampler='grid_nn',
                 fixation_size=None, 
                 device='cuda', 
                 dtype=torch.float, 
                 auto_match_cart_resources=True,
                 pre_transforms=None, post_transforms=None,
                 sigma=None,
                 no_color_val=False,
                 **kwargs, # passed to the sampler
                 ):
        """
        Initialize the RetinalTransform module.
        
        Args:
            resolution (int): Target resolution for the retinal transform.
            start_res (int, optional): Starting resolution. Defaults to 256.
            fov (float, optional): Field of view diameter in degrees. Defaults to 16.
            cmf_a (float, optional): Cortical magnification factor parameter. Defaults to 0.5.
            style (str, optional): Sampling style. Defaults to 'isotropic'.
            sampler (str, optional): Sampler type. Defaults to 'grid_nn'.
            fixation_size (int, optional): Fixation size in pixels. Defaults to None.
            device (str, optional): Device to use. Defaults to 'cuda'.
            dtype (torch.dtype, optional): Data type. Defaults to torch.float.
            auto_match_cart_resources (bool, optional): Whether to auto-match cartesian resources. Defaults to True.
            pre_transforms (callable, optional): Pre-processing transforms. Defaults to None.
            post_transforms (callable, optional): Post-processing transforms. Defaults to None.
            sigma (float, optional): Standard deviation for Gaussian color decay. Defaults to None.
            no_color_val (bool, optional): Whether to disable color in eval mode. Defaults to False.
            **kwargs: Additional arguments passed to warping function.
        """
        super().__init__()
        self.sigma = sigma
        self.cmf_a = cmf_a
        self.fov = fov
        full_fov = self.fov
        self.fixation_size = start_res if fixation_size is None else fixation_size # this is the maximum fixation size
        self.start_res = start_res
        self.device = device
        self.dtype = dtype

        if auto_match_cart_resources != 0:
            in_resolution = resolution
            num_coords = resolution**2
            if 'fixn' in style:
                # if fixn, we are going to force the resolution later
                resolution = resolution
            else:
                # otherwise we determine it now
                resolution, num_coords = find_desired_res(fov, cmf_a, num_coords, style=style, device=self.device, force_less_than=True, quiet=True)

        self.resolution = resolution

        # for compatibility
        self.mode = f'pointcloud_{style}'
        self.out_size = resolution

        # for use with standard CNNs
        if '_as_grid' in self.mode:
            self.reshape_as_grid = True
        else:
            self.reshape_as_grid = False

        self.no_color_val = no_color_val
        self.foveal_color = GaussianColorDecay(sigma)

        if sampler == 'gaussian_pooling':
            self.sampler = GaussianKNNGridSampler(self.fov, self.cmf_a, resolution, fixation_size=self.fixation_size, device=device, style=style, **kwargs)
        elif sampler == 'pooling':
            self.sampler = KNNGridSampler(self.fov, self.cmf_a, resolution, fixation_size=self.fixation_size, device=device, style=style, **kwargs)
        elif sampler == 'grid_nn':
            self.sampler = GridSampler(self.fov, self.cmf_a, resolution, device=device, mode='nearest', style=style)
        elif sampler == 'grid_bilinear':
            self.sampler = GridSampler(self.fov, self.cmf_a, resolution, device=device, mode='bilinear', style=style)

        else:
            raise ValueError(f'Invalid sampler: {sampler}')
        self.device = device
        self.dtype = dtype

        self.pre_transforms = pre_transforms
        self.post_transforms = post_transforms    

        self.scatter_sizes = self.sampler.coords.get_scatter_sizes().cpu().numpy()

    @torch.autocast(device_type='cuda', enabled=False)
    def forward(self, x, fix_loc, fixation_size=None, **kwargs):
        """
        Forward pass of the retinal transform.
        
        Args:
            x (torch.Tensor): Input tensor.
            fix_loc (torch.Tensor or tuple): Fixation location.
            fixation_size (int, optional): Fixation size. Defaults to None.
            **kwargs: Additional arguments.
            
        Returns:
            torch.Tensor: Transformed tensor.
        """

        # check fixation_size
        fixation_size = self._check_fixation_size(fixation_size, x.shape[0])
            
        # check fix_loc
        fix_loc = self._check_fix_loc(fix_loc, x.shape[0])
        
        if self.pre_transforms is not None and self.training:
            x = self.pre_transforms(x.clone())

        x = self.sampler(x, fix_loc=fix_loc, fixation_size=fixation_size, **kwargs)

        if self.post_transforms is not None and self.training:
            x = self.post_transforms(x.unsqueeze(3)).clone().squeeze(3)

        if self.foveal_color is not None:
            x = self.foveal_color(x, self.sampler.polar_radius)

        if not self.training and self.no_color_val:
            x = TF.rgb_to_grayscale(x.unsqueeze(3), num_output_channels=3).clone().squeeze(3)

        if self.reshape_as_grid:
            x = x.reshape(x.shape[0], x.shape[1], self.resolution, self.resolution)

        return x.to(self.dtype)

    def change_sigma(self, sigma):
        """
        Change the sigma parameter for the Gaussian color decay.
        
        Args:
            sigma (float): New sigma value for the Gaussian color decay.
        """
        self.foveal_color = GaussianColorDecay(sigma)

    def get_warp_params(self):
        """
        Get the warping parameters.
        
        Returns:
            tuple: (fov, cmf_a, resolution, fixation_size)
        """
        return self.fov, self.cmf_a, self.resolution, self.fixation_size

    def _check_fixation_size(self, fixation_size, batch_size):
        """
        Validate and format fixation size for batch processing.
        
        Args:
            fixation_size (int, tuple, or None): Fixation size specification.
            batch_size (int): Number of samples in the batch.
            
        Returns:
            np.ndarray: Formatted fixation size array of shape (batch_size, 2).
        """
        if fixation_size is None or (isinstance(fixation_size, str) and fixation_size == 'none'):
            if hasattr(self.fixation_size, '__len__'):
                fixation_size = np.array(self.fixation_size)
            else:
                fixation_size = np.array([self.fixation_size, self.fixation_size])
        elif isinstance(fixation_size, int) or isinstance(fixation_size, float):
            fixation_size = np.array([fixation_size, fixation_size])
        elif hasattr(fixation_size, '__len__') and len(fixation_size) == 2:
            fixation_size = np.array(fixation_size)
        elif isinstance(fixation_size, torch.Tensor):
            fixation_size = fixation_size.cpu().numpy()
        fixation_size = fixation_size.squeeze()
        if len(fixation_size.shape) == 1:
            if fixation_size.shape[0] == 2:
                fixation_size = np.expand_dims(np.array(fixation_size), axis=0).repeat(batch_size, axis=0)
            elif fixation_size.shape[0] == batch_size:
                fixation_size = np.expand_dims(np.array(fixation_size), axis=1).repeat(2, axis=1)
            else:
                raise ValueError(f'fixation_size shape: {fixation_size.shape}')
        else:
            assert fixation_size.shape[0] == batch_size
        
        return fixation_size

    def _check_fix_loc(self, fix_loc, batch_size):
        """
        Validate and format fixation location for batch processing.
        
        Args:
            fix_loc (tuple, list, torch.Tensor, or None): Fixation location specification.
            batch_size (int): Number of samples in the batch.
            
        Returns:
            torch.Tensor: Formatted fixation location tensor of shape (batch_size, 2).
        """
        if fix_loc is None:
            fix_loc = np.array([0.5, 0.5])
        elif isinstance(fix_loc, float) or isinstance(fix_loc, int):
            fix_loc = np.array([fix_loc, fix_loc])
        elif isinstance(fix_loc, torch.Tensor):
            fix_loc = fix_loc.cpu().numpy()
        elif isinstance(fix_loc, list) or isinstance(fix_loc, tuple):
            fix_loc = np.array(fix_loc)
        if len(fix_loc.squeeze().shape) == 1:
            fix_loc = np.expand_dims(np.array(fix_loc.squeeze()), axis=0).repeat(batch_size, axis=0)
        else:
            assert fix_loc.shape[0] == batch_size

        fix_loc = torch.tensor(fix_loc, dtype=self.dtype, device=self.device)

        return fix_loc

    def _check_aspect_ratio(self, aspect_ratio, batch_size):
        """
        Validate and format aspect ratio for batch processing.
        
        Args:
            aspect_ratio (float, list, or None): Aspect ratio specification.
            batch_size (int): Number of samples in the batch.
            
        Returns:
            np.ndarray: Formatted aspect ratio array of shape (batch_size,).
        """
        if aspect_ratio is None:
            aspect_ratio = np.ones(batch_size)
        elif isinstance(aspect_ratio, int) or isinstance(aspect_ratio, float):
            aspect_ratio = np.ones(batch_size)*aspect_ratio
        elif isinstance(aspect_ratio, list) or isinstance(aspect_ratio, tuple) or isinstance(aspect_ratio, np.ndarray):
            if len(aspect_ratio) == 2:
                # it is a range for sampling. use log space to handle ratios properly
                log_ratio = np.log(aspect_ratio)
                aspect_ratio = np.exp(np.random.uniform(log_ratio[0], log_ratio[1], batch_size))
            else:
                assert len(aspect_ratio) == batch_size
                aspect_ratio = np.array(aspect_ratio)
        elif isinstance(aspect_ratio, torch.Tensor):
            aspect_ratio = aspect_ratio.cpu().numpy()

        assert aspect_ratio.shape[0] == batch_size

        return aspect_ratio

    def _apply_aspect_ratio(self, fixation_size, aspect_ratio, batch_size):
        """
        Apply aspect ratio to fixation size while preserving area.
        
        Args:
            fixation_size (torch.Tensor): Current fixation size.
            aspect_ratio (float): Aspect ratio to apply.
            batch_size (int): Number of samples in the batch.
            
        Returns:
            np.ndarray: Modified fixation size with applied aspect ratio.
        """
        if len(fixation_size.shape) == 1:
            fixation_size = fixation_size.unsqueeze(0).repeat(batch_size, 1)
        assert fixation_size.shape[1] == 2
        if aspect_ratio is not None:
            fix_h = fixation_size[:,0]
            fix_w = fixation_size[:,1]
            # compute fixation widths and heights while preserving fixation areas
            area = fix_w * fix_h
            fix_w = (area * aspect_ratio) ** 0.5
            fix_h = (area / aspect_ratio) ** 0.5
            fixation_size = np.stack((fix_h, fix_w), axis=1)
        return fixation_size

@add_to_all(__all__)
def min_diff_for_cmf_a(cmf_a, fov, output_res, fixation_size, force_n_points=None, disallow_undersampling=True, force_less_than=False, device='cuda'):
    """
    Helper function that computes minimum difference between radii for a given cmf_a value.
    
    Args:
        cmf_a (float): Cortical magnification factor parameter.
        fov (float): Field of view diameter in degrees.
        output_res (int): Output resolution.
        fixation_size (int): Fixation size in pixels.
        force_n_points (int, optional): Force number of points. Defaults to None.
        disallow_undersampling (bool, optional): Whether to disallow undersampling. Defaults to True.
        force_less_than (bool, optional): Whether to force less than. Defaults to False.
        device (str, optional): Device to use. Defaults to 'cuda'.
        
    Returns:
        float: Minimum difference between radii.
    """
    if force_n_points is not None:
        output_res = find_desired_res(fov, cmf_a, force_n_points, 'isotropic', device=device, force_less_than=force_less_than, quiet=True)

    w_min = np.log(cmf_a)
    w_max = np.log(cmf_a + fov/2)

    log_radius = np.linspace(w_min, w_max, output_res//2)
    lin_radius = np.exp(log_radius) - cmf_a

    lin_radius_pix = (lin_radius*fixation_size/2)/(fov/2)
    diff = lin_radius_pix[1:] - lin_radius_pix[:-1]
    min_diff = np.min(diff)
    if min_diff > 1 and disallow_undersampling:
        # we undersampled, so by returning cmf_a, it will suggest to the optimizer to reduce the cmf_a next time, which is good
        return np.maximum(cmf_a, 1)
    return np.abs(1 - min_diff)  # Return difference from target min_diff of 1

@add_to_all(__all__)
def get_min_cmf_a(fixation_size, output_res, start_res=5496, fov=65, start_cmf_a=0.15, style='isotropic', maxiters=200, disallow_undersampling=True, use_scaled_fov=True, device='cuda'):
    """
    Find the minimum cmf_a value that satisfies the constraints.
    
    Args:
        fixation_size (int): Fixation size in pixels.
        output_res (int): Output resolution.
        start_res (int, optional): Starting resolution. Defaults to 5496.
        fov (float, optional): Field of view diameter in degrees. Defaults to 65.
        start_cmf_a (float, optional): Starting cmf_a value. Defaults to 0.15.
        style (str, optional): Sampling style. Defaults to 'isotropic'.
        maxiters (int, optional): Maximum iterations for optimization. Defaults to 200.
        disallow_undersampling (bool, optional): Whether to disallow undersampling. Defaults to True.
        use_scaled_fov (bool, optional): Whether to use scaled FOV. Defaults to True.
        
    Returns:
        float or None: Minimum cmf_a value, or None if not found.
    """
    if use_scaled_fov:
        fov = fov * (fixation_size / start_res)
    
    def loss_fn(cmf_a):
        return min_diff_for_cmf_a(cmf_a, fov, output_res, fixation_size, disallow_undersampling=disallow_undersampling, device=device)
    
    try:
        result = minimize_scalar(loss_fn, bounds=(0.01, 1000), method='bounded', options=dict(maxiter=maxiters))
        if result.success:
            return result.x
        else:
            return None
    except Exception as e:
        print(e)
        return None

@add_to_all(__all__)
class GaussianColorDecay(nn.Module):
    """
    Implements foveated color representation using Gaussian decay based on eccentricity.
    
    This module models hue saturation as a 1D Gaussian function of visual field eccentricity,
    simulating the reduced color sensitivity in the periphery.
    """
    
    def __init__(self, sigma):
        """
        Initialize the Gaussian color decay module.
        
        Args:
            sigma (float): Standard deviation of the Gaussian decay function.
        """
        super().__init__()
        self.sigma = sigma

    def forward(self, x, radius):
        """
        Apply Gaussian color decay based on eccentricity.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
            radius (torch.Tensor): Eccentricity radius tensor.
            
        Returns:
            torch.Tensor: Color-decayed tensor with same shape as input.
        """
        if self.sigma is None:
            return x
        
        # Apply Gaussian decay to color channels
        decay = torch.exp(-radius / self.sigma)
        decay = decay.unsqueeze(1)  # Add channel dimension
        
        # Apply decay to all channels except luminance
        x_decayed = x * decay
        
        return x_decayed

    def __repr__(self):
        """String representation of the GaussianColorDecay module."""
        return f'GaussianColorDecay(sigma={self.sigma})'