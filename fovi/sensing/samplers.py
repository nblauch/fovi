import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import torchvision.transforms.functional as TF

from .coords import SamplingCoords, transform_sampling_grid, xy_to_colrow
from ..arch.knn import KNNPoolingLayer
from ..utils import add_to_all

__all__ = []

@add_to_all(__all__)
class BaseGridSampler(nn.Module):
    """
    Base class for grid samplers.

    Note: objects of the BaseGridSampler family should not be used directly; it is much more convenient to use RetinalTransform, which stores a BaseGridSampler child, due to its handling of fixation parameters. 
    """

    def _transform_fix_grid(self, img_shape, fix_loc, fixation_size):
        """
        Transform fixation grid to image coordinates.
        
        Args:
            img_shape (tuple): Image shape (height, width).
            fix_loc (torch.Tensor): Fixation location.
            fixation_size (torch.Tensor): Fixation size.
            
        Returns:
            torch.Tensor: Transformed grid coordinates.
        """
        return transform_sampling_grid(self.sampling_grid, fix_loc, fixation_size, img_shape)

    def _prep_grid_for_grid_sample(self, cartesian_grid):
        """
        Prepare grid for torch.nn.functional.grid_sample.

        Args:
            cartesian_grid (torch.Tensor): (n,2) coordinates, where each 2-vector coordinate is specified in (x,y) in the typical math sense (normalized to [-1,1])
            
        Returns:
            torch.Tensor: (1,1,n,2) coordinates, where each 2-vector coordinate is specified in (-y, x) or (row, col) format for grid_sample (normalized to [-1,1])
        """
        out_grid = xy_to_colrow(cartesian_grid.clone(), do_norm=False, format='-11')
        out_grid = out_grid.unsqueeze(0).unsqueeze(0)
        return out_grid


@add_to_all(__all__)
class GridSampler(BaseGridSampler):
    """
    Grid sampler for foveated vision using regular grid sampling.
    
    This sampler uses standard grid sampling with nearest neighbor or bilinear interpolation
    to sample from the foveated sampling grid.
    """
    
    def __init__(self, fov, cmf_a, resolution, device='cuda', dtype=torch.float, mode='nearest', style='isotropic', coords=None):
        """
        Initialize the GridSampler.
        
        Args:
            fov (float): Field of view diameter in degrees.
            cmf_a (float): A parameter from the CMF: M(r)=1/(r+a). Smaller = stronger foveation.
            resolution (int): Resolution parameter.
            device (str, optional): Device to run on. Defaults to 'cuda'.
            dtype (torch.dtype, optional): Data type. Defaults to torch.float.
            mode (str, optional): Sampling mode ('nearest' or 'bilinear'). Defaults to 'nearest'.
            style (str, optional): Sampling style. Defaults to 'isotropic'.
            coords (SamplingCoords, optional): Pre-computed sampling coordinates. Defaults to None.
        """
        super().__init__()
        self.fov = fov
        self.cmf_a = cmf_a
        self.resolution = resolution
        self.device = device
        self.dtype = dtype
        self.mode = mode
        self.style = style
        
        if coords is None:
            self.coords = SamplingCoords(fov, cmf_a, resolution, device=device, style=style, dtype=dtype)
        else:
            self.coords = coords
            
        self.sampling_grid = self._prep_grid_for_grid_sample(self.coords.cartesian)
        self.out_sampling_grid = self.sampling_grid
        self.polar_radius = self.coords.polar[:, 0]

    def forward(self, img, fix_loc=None, fixation_size=None, return_coords=False):
        """
        Forward pass for grid sampling.
        
        Args:
            img (torch.Tensor): Input image tensor.
            fix_loc (torch.Tensor, optional): Fixation location. Defaults to None.
            fixation_size (torch.Tensor, optional): Fixation size. Defaults to None.
            return_coords (bool, optional): Whether to return sampling coordinates. Defaults to False.
            
        Returns:
            torch.Tensor: Sampled image tensor.
        """
        # Transform relative fixation grid to absolute image coordinates (col, row)
        grid = self._transform_fix_grid(img.shape[-2:], fix_loc, fixation_size)
        
        # Apply grid sampling
        sampled = torch.nn.functional.grid_sample(img, grid, mode=self.mode, align_corners=False).squeeze(2)
        
        if return_coords:
            return sampled, grid
        return sampled

    def all_coords(self, device=None):
        """
        Get all sampling coordinates.
        
        Args:
            device (str, optional): Device to place coordinates on. Defaults to None.
            
        Returns:
            torch.Tensor: All sampling coordinates.
        """
        if device is None:
            device = self.device
        return self.coords.cartesian.to(device)

    def __repr__(self):
        """String representation of the GridSampler."""
        return f'GridSampler(fov={self.fov}, cmf_a={self.cmf_a}, style={self.style}, resolution={self.resolution}, mode={self.mode}, n={len(self.coords)})'
    

@add_to_all(__all__)
class KNNGridSampler(BaseGridSampler):
    """
    K-Nearest Neighbors grid sampler for foveated vision.
    
    This sampler uses KNN-based sampling to perform local average pooling over a high-res sensor array into a lower-res sensor array, with the same CorticalSensorManifold.
    - highres_coords: akin to photoreceptors: there are more of them
    - coords: akin to retinal ganglion cells: there are less of them, and they integrate over a local pool of photoreceptors (highres_coords)
    """
    
    def __init__(self, fov, cmf_a, resolution, res_mult=3, cmf_a_mult=1, fixation_size=3000, k=None, style='isotropic', sample_cortex=True, dtype=torch.float, device='cuda'):
        """
        Initialize the KNNGridSampler.
        
        Args:
            fov (float): Field of view diameter in degrees.
            cmf_a (float): A parameter from the CMF: M(r)=1/(r+a). Smaller = stronger foveation.
            resolution (int): Resolution parameter.
            res_mult (int, optional): Resolution multiplier for photoreceptor layer vs. rgc layer. Defaults to 3.
            cmf_a_mult (int, optional): CMF_a multiplier for photoreceptor layer vs. rgc layer. Defaults to 1.
            fixation_size (int, optional): Fixation size in pixels. Defaults to 3000.
            k (int, optional): Number of nearest neighbors. Defaults to None.
            style (str, optional): Sampling style. Defaults to 'isotropic'.
            sample_cortex (bool, optional): Whether to sample from cortex. Defaults to True.
            dtype (torch.dtype, optional): Data type. Defaults to torch.float.
            device (str, optional): Device to run on. Defaults to 'cuda'.
        """
        super().__init__()
        self.highres_coords = SamplingCoords(fov, cmf_a_mult*cmf_a, res_mult*resolution, device=device, style=style, dtype=dtype)
        self.coords = SamplingCoords(fov, cmf_a, resolution, device=device, style=style, dtype=dtype)

        if k is None:
            # default to the ratio of the number of pixels in the retinal and cortical grids
            k = int(np.round(len(self.highres_coords) / len(self.coords)))
            print(f'auto-set knngridsampler k={k}')

        self.pooler = KNNPoolingLayer(k, self.highres_coords, self.coords, mode='avg', device=device, sample_cortex=sample_cortex)

        self.sampling_grid = self._prep_grid_for_grid_sample(self.highres_coords.cartesian)
        self.out_sampling_grid = self._prep_grid_for_grid_sample(self.coords.cartesian)

        self.polar_radius = self.coords.polar[:,0]
        self.fov = fov
        self.cmf_a = cmf_a
        self.resolution = resolution
        self.fixation_size = fixation_size
        self.k = k # number of neighbors to consider, the later ones will be weighted less or not at all
        self.dtype = dtype
        self.device = device
        self.style = style
        self.num_coords = len(self.coords)
        self.sample_cortex = sample_cortex

        self.rf_sizes = self.coords.get_scatter_sizes()

    def forward(self, img, fix_loc=None, fixation_size=None):
        """
        Forward pass for KNN grid sampling.
        
        Args:
            img (torch.Tensor): Input image tensor.
            fix_loc (torch.Tensor, optional): Fixation location. Defaults to None.
            fixation_size (torch.Tensor, optional): Fixation size. Defaults to None.
            
        Returns:
            torch.Tensor: Pooled samples from KNN grid sampling.
        """
        if not isinstance(img, torch.Tensor):
            img = TF.to_tensor(img).unsqueeze(0)
        img = img.to(self.device).to(self.dtype)
            
        # account for fixation location, size, and possible bounding box
        fix_grid = self._transform_fix_grid(img.shape[-2:], fix_loc, fixation_size)
        # do the first layer of sampling
        ret_samples = torch.nn.functional.grid_sample(img, fix_grid, mode='nearest').squeeze(2)
        # pool to get the final retinal samples
        pooled_samples = self.pooler(ret_samples)

        return pooled_samples

    def all_coords(self, device=None):
        """
        Get all sampling coordinates.
        
        Args:
            device (str, optional): Device to place coordinates on. Defaults to None.
            
        Returns:
            tuple: Cartesian, polar, and plotting coordinates.
        """
        if device is None:
            return self.coords.cartesian, self.coords.polar, self.coords.plotting
        else:
            return self.coords.cartesian.to(device), self.coords.polar.to(device), self.coords.plotting.to(device)

    def __repr__(self):
        """String representation of the KNNGridSampler."""
        return f'KNNGridSampler(fov={self.fov}, cmf_a={self.cmf_a}, resolution={self.resolution}, num_coords={self.num_coords}, fixation_size={self.fixation_size}, k={self.k}, dtype={self.dtype}, device={self.device}, style={self.style})'
    

@add_to_all(__all__)
class GaussianKNNGridSampler(KNNGridSampler):
    """K-Nearest Neighbors grid sampler with Gaussian-weighted pooling.
    
    Similar to KNNGridSampler, but uses Gaussian-weighted pooling rather than 
    simple averaging. The Gaussian weighting gives higher weight to photoreceptors 
    that are closer to the center of each retinal ganglion cell's receptive field,
    providing a more biologically plausible pooling mechanism.
    
    Inherits all attributes and methods from KNNGridSampler, with the pooler
    replaced by a Gaussian-weighted version.
    """
    def __init__(self, *args, **kwargs):
        """Initialize the GaussianKNNGridSampler.
        
        Args:
            *args: Variable length argument list passed to KNNGridSampler.
                See KNNGridSampler.__init__ for details on positional arguments:
                fov, cmf_a, resolution, etc.
            **kwargs: Arbitrary keyword arguments passed to KNNGridSampler.
                See KNNGridSampler.__init__ for details on keyword arguments:
                res_mult, cmf_a_mult, fixation_size, k, style, sample_cortex, 
                dtype, device.
        """
        super().__init__(*args, **kwargs)

        # just adjust the pooler
        self.pooler = KNNPoolingLayer(self.k, self.highres_coords, self.coords, mode='gaussian', device=self.device, sample_cortex=self.sample_cortex)

def compute_knn_indices_chunked(in_coords, out_coords, chunk_size=200, max_k=1000, use_tqdm=True):
    """
    Compute K-nearest neighbor indices in chunks to handle large coordinate sets.
    
    Args:
        in_coords (torch.Tensor): Input coordinates.
        out_coords (torch.Tensor): Output coordinates.
        chunk_size (int, optional): Size of chunks for processing. Defaults to 200.
        max_k (int, optional): Maximum number of neighbors. Defaults to 1000.
        use_tqdm (bool, optional): Whether to show progress bar. Defaults to True.
        
    Returns:
        tuple:
            - torch.Tensor: KNN indices
            - torch.Tensor: KNN distances
    """
    knn_indices = []
    knn_distances = []
    for i in tqdm(range(0, out_coords.size(0), chunk_size)) if use_tqdm else range(0, out_coords.size(0), chunk_size):
        chunk = out_coords[i:i+chunk_size]
        distances_chunk = torch.cdist(in_coords, chunk)  # Pairwise Euclidean distances for the chunk
        _, knn_indices_chunk = torch.topk(distances_chunk, max_k, dim=0, largest=False)
        knn_indices.append(knn_indices_chunk)
        knn_distances.append(torch.gather(distances_chunk, 0, knn_indices_chunk))
        # delete chunk of distances
        del distances_chunk
        torch.cuda.empty_cache()

    knn_indices = torch.cat(knn_indices, dim=1)
    knn_distances = torch.cat(knn_distances, dim=1)

    return knn_indices, knn_distances  
