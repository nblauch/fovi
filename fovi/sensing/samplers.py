import os
import warnings

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

    def _prepare_sampling_args(
            self, img, fix_loc, fixation_size, dtype=torch.float32):
        """Return batched fixation tensors on ``img.device``.

        Keeping conversion in one place makes standalone samplers accept the numpy/list
        fixation forms that ``RetinalTransform`` historically normalizes for them.
        """
        batch = img.shape[0]
        fix_loc = torch.as_tensor(fix_loc, device=img.device, dtype=dtype)
        fixation_size = torch.as_tensor(
            fixation_size, device=img.device, dtype=dtype)
        while fix_loc.ndim > 2 and fix_loc.shape[1] == 1:
            fix_loc = fix_loc.squeeze(1)
        while fixation_size.ndim > 2 and fixation_size.shape[1] == 1:
            fixation_size = fixation_size.squeeze(1)
        if fix_loc.ndim == 1:
            fix_loc = fix_loc.unsqueeze(0).expand(batch, -1)
        if fixation_size.ndim == 0:
            fixation_size = fixation_size.repeat(2)
        if fixation_size.ndim == 1:
            if fixation_size.numel() == 2:
                fixation_size = fixation_size.unsqueeze(0).expand(batch, -1)
            elif fixation_size.numel() == batch:
                fixation_size = fixation_size.unsqueeze(1).expand(-1, 2)
        if fix_loc.shape != (batch, 2):
            raise ValueError(f"fix_loc must have shape ({batch}, 2), got {tuple(fix_loc.shape)}")
        if fixation_size.shape != (batch, 2):
            raise ValueError(
                f"fixation_size must have shape ({batch}, 2), got {tuple(fixation_size.shape)}")
        return fix_loc.contiguous(), fixation_size.contiguous()

    def _direct_pixel_coords(self, img_shape, fix_loc, fixation_size):
        """Compute canonical pixel-boundary coordinates for direct sampling."""
        height, width = img_shape
        base = self.sampling_grid[0, 0].to(
            device=fix_loc.device, dtype=fix_loc.dtype)
        pixel_x = (
            base[None, :, 0] * (fixation_size[:, None, 1] * 0.5)
            + fix_loc[:, None, 1] * width
        )
        pixel_y = (
            base[None, :, 1] * (fixation_size[:, None, 0] * 0.5)
            + fix_loc[:, None, 0] * height
        )
        return pixel_x, pixel_y

    def _direct_grid(self, img_shape, fix_loc, fixation_size):
        """Return a grid_sample-format grid describing the direct pixel coordinates."""
        height, width = img_shape
        pixel_x, pixel_y = self._direct_pixel_coords(img_shape, fix_loc, fixation_size)
        return torch.stack(
            (2.0 * pixel_x / width - 1.0, 2.0 * pixel_y / height - 1.0), dim=-1
        ).unsqueeze(1)

    @staticmethod
    def _mask_invalid_samples(samples, valid_mask):
        """Apply deterministic FoV padding without changing forward signatures."""
        if bool(valid_mask.all()):
            return samples
        return samples * valid_mask.to(
            device=samples.device, dtype=samples.dtype)[None, None, :]


@add_to_all(__all__)
class GridSampler(BaseGridSampler):
    """
    Grid sampler for foveated vision using regular grid sampling.
    
    This sampler uses standard grid sampling with nearest neighbor or bilinear interpolation
    to sample from the foveated sampling grid.
    """
    
    def __init__(self, fov, cmf_a, resolution, device='cuda', dtype=torch.float,
                 mode='nearest', style='isotropic', coords=None,
                 isotropic_plotting_type='v1like', backend='auto',
                 output_dtype=None, fov_type='circular'):
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
            backend (str, optional): Sampling backend: ``auto``, ``torch``, or ``cuda``.
                For floating inputs, ``torch`` selects ``torch.grid_sample``; for uint8 it
                selects direct indexing. ``auto`` prefers eligible native CUDA kernels.
            output_dtype (torch.dtype, optional): Optional dtype for the compact output.
                This never rescales values. Nearest sampling otherwise preserves dtype;
                bilinear integer sampling naturally produces float32.
        """
        super().__init__()
        self.fov = fov
        self.cmf_a = cmf_a
        self.resolution = resolution
        self.device = device
        self.dtype = dtype
        self.mode = mode
        self.style = style
        if mode not in ('nearest', 'bilinear'):
            raise ValueError(f"Unsupported sampling mode {mode!r}")
        if backend not in ('auto', 'torch', 'cuda'):
            raise ValueError("backend must be one of 'auto', 'torch', or 'cuda'")
        if mode == 'bilinear' and output_dtype is not None and not output_dtype.is_floating_point:
            raise ValueError("bilinear sampling requires a floating output dtype")
        self.backend = backend
        self.output_dtype = output_dtype
        self._last_backend = None
        self._native_errors = {}
        self._native_uint8_sample_fn = None
        self._native_float_sample_fn = None
        self.fov_type = fov_type
        
        if coords is None:
            self.coords = SamplingCoords(
                fov, cmf_a, resolution, device=device, style=style,
                dtype=dtype, isotropic_plotting_type=isotropic_plotting_type,
                fov_type=fov_type)
        else:
            self.coords = coords
            
        self.sampling_grid = self._prep_grid_for_grid_sample(self.coords.cartesian)
        self.out_sampling_grid = self.sampling_grid
        self.polar_radius = self.coords.polar[:, 0]
        self.valid_mask = self.coords.valid_mask

    def _requested_backend(self):
        requested = os.environ.get('FOVI_GRID_SAMPLER_BACKEND', self.backend)
        if requested not in ('auto', 'torch', 'cuda'):
            raise ValueError(
                "FOVI_GRID_SAMPLER_BACKEND/backend must be one of 'auto', 'torch', or 'cuda'"
            )
        return requested

    def _torch_direct_sample(self, img, fix_loc, fixation_size):
        """Direct moving-grid gather used as the canonical-coordinate oracle."""
        height, width = img.shape[-2:]
        pixel_x, pixel_y = self._direct_pixel_coords(
            (height, width), fix_loc, fixation_size)
        batch_index = torch.arange(img.shape[0], device=img.device)[:, None]
        image_hwc = img.permute(0, 2, 3, 1)

        def gather(y, x):
            valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
            x = x.clamp(0, width - 1)
            y = y.clamp(0, height - 1)
            values = image_hwc[batch_index, y, x].permute(0, 2, 1)
            return values, valid

        if self.mode == 'nearest':
            # align_corners=False maps a boundary coordinate p to source-center p - 0.5;
            # round-to-even matches grid_sample's nearest tie convention.
            values, valid = gather(
                torch.round(pixel_y - 0.5).long(), torch.round(pixel_x - 0.5).long())
            return values * valid[:, None, :].to(values.dtype)

        source_x = pixel_x - 0.5
        source_y = pixel_y - 0.5
        x0 = torch.floor(source_x).long()
        y0 = torch.floor(source_y).long()
        wx = source_x - x0
        wy = source_y - y0
        arithmetic_dtype = (
            torch.float32 if img.dtype == torch.uint8
            else self._coordinate_dtype(img.dtype))
        out = torch.zeros(
            (img.shape[0], img.shape[1], pixel_x.shape[1]),
            device=img.device, dtype=arithmetic_dtype)
        for yy, xx, weight in (
            (y0, x0, (1.0 - wy) * (1.0 - wx)),
            (y0, x0 + 1, (1.0 - wy) * wx),
            (y0 + 1, x0, wy * (1.0 - wx)),
            (y0 + 1, x0 + 1, wy * wx),
        ):
            values, valid = gather(yy, xx)
            out.add_(
                values.to(arithmetic_dtype)
                * (weight * valid).to(arithmetic_dtype)[:, None, :])
        if img.dtype != torch.uint8 and out.dtype != img.dtype:
            out = out.to(img.dtype)
        return out

    @staticmethod
    def _coordinate_dtype(image_dtype):
        if image_dtype in (torch.float16, torch.bfloat16, torch.float32):
            return torch.float32
        if image_dtype == torch.float64:
            return torch.float64
        raise TypeError(f"unsupported floating input dtype {image_dtype}")

    def _native_uint8_sample(self, img, fix_loc, fixation_size):
        if self._native_uint8_sample_fn is None:
            from .grid_sample_cuda import sample_uint8
            self._native_uint8_sample_fn = sample_uint8
        return self._native_uint8_sample_fn(
            img, self.sampling_grid, fix_loc, fixation_size, mode=self.mode)

    def _native_float_sample(self, img, fix_loc, fixation_size):
        if self._native_float_sample_fn is None:
            from .grid_sample_cuda import sample_float
            self._native_float_sample_fn = sample_float
        return self._native_float_sample_fn(
            img, self.sampling_grid, fix_loc, fixation_size, mode=self.mode)

    def _native_eligible(self, img, fix_loc, fixation_size):
        supported_dtype = img.dtype == torch.uint8 or img.dtype in (
            torch.float16, torch.float32, torch.float64)
        return img.is_cuda and supported_dtype and not (
            img.requires_grad or fix_loc.requires_grad
            or fixation_size.requires_grad or self.sampling_grid.requires_grad)

    def _sample_uint8(self, img, fix_loc, fixation_size):
        requested = self._requested_backend()
        can_native = self._native_eligible(img, fix_loc, fixation_size)
        if requested == 'cuda' and not can_native:
            raise RuntimeError(
                "cuda uint8 sampling requires CUDA input and non-differentiable coordinates"
            )
        error_key = (img.dtype, self.mode)
        if requested != 'torch' and can_native and (
                requested == 'cuda' or error_key not in self._native_errors):
            try:
                sampled = self._native_uint8_sample(img, fix_loc, fixation_size)
                self._last_backend = 'cuda'
                return sampled
            except Exception as exc:
                if requested == 'cuda':
                    raise
                self._native_errors[error_key] = (
                    f"{type(exc).__name__}: {exc}")
                warnings.warn(
                    f"native uint8 sampler unavailable ({exc}); using Torch gather",
                    RuntimeWarning, stacklevel=2)
        self._last_backend = 'torch_gather'
        return self._torch_direct_sample(img, fix_loc, fixation_size)

    def _torch_grid_sample(self, img, fix_loc, fixation_size):
        """Sample through grid_sample using the floating opmath coordinate dtype."""
        coordinate_dtype = self._coordinate_dtype(img.dtype)
        base_grid = self.sampling_grid.to(
            device=img.device, dtype=coordinate_dtype)
        grid = transform_sampling_grid(
            base_grid, fix_loc, fixation_size, img.shape[-2:])
        sample_input = img.to(coordinate_dtype)
        sampled = torch.nn.functional.grid_sample(
            sample_input, grid, mode=self.mode, align_corners=False).squeeze(2)
        if sampled.dtype != img.dtype:
            sampled = sampled.to(img.dtype)
        self._last_backend = 'torch_grid_sample'
        return sampled, grid

    def _sample_float(self, img, fix_loc, fixation_size):
        requested = self._requested_backend()
        can_native = self._native_eligible(img, fix_loc, fixation_size)
        if requested == 'cuda' and not can_native:
            raise RuntimeError(
                "cuda floating sampling requires a CUDA float16/float32/float64 "
                "input and no required gradients")
        error_key = (img.dtype, self.mode)
        if requested != 'torch' and can_native and (
                requested == 'cuda' or error_key not in self._native_errors):
            try:
                sampled = self._native_float_sample(
                    img, fix_loc, fixation_size)
                self._last_backend = 'cuda'
                return sampled, None
            except Exception as exc:
                if requested == 'cuda':
                    raise
                self._native_errors[error_key] = (
                    f"{type(exc).__name__}: {exc}")
                warnings.warn(
                    f"native floating sampler unavailable ({exc}); using grid_sample",
                    RuntimeWarning, stacklevel=2)
        return self._torch_grid_sample(img, fix_loc, fixation_size)

    def _convert_output(self, sampled):
        if self.output_dtype is not None and sampled.dtype != self.output_dtype:
            sampled = sampled.to(self.output_dtype)
        return sampled

    def forward(self, img, fix_loc=None, fixation_size=None, return_coords=False,
                direct=False):
        """
        Forward pass for grid sampling.
        
        Args:
            img (torch.Tensor): Input image tensor.
            fix_loc (torch.Tensor, optional): Fixation location. Defaults to None.
            fixation_size (torch.Tensor, optional): Fixation size. Defaults to None.
            return_coords (bool, optional): Whether to return sampling coordinates. Defaults to False.
            direct (bool, optional): Force the explicit Torch direct-index oracle.
                Defaults to False.
            
        Returns:
            torch.Tensor: Sampled image tensor.
        """
        if not isinstance(img, torch.Tensor):
            raise TypeError("GridSampler input must be a torch.Tensor")
        if img.ndim != 4:
            raise ValueError(f"GridSampler expects NCHW input, got shape {tuple(img.shape)}")

        if img.dtype == torch.uint8:
            fix_loc_t, fixation_size_t = self._prepare_sampling_args(
                img, fix_loc, fixation_size)
            if direct:
                sampled = self._torch_direct_sample(
                    img, fix_loc_t, fixation_size_t)
                self._last_backend = 'torch_direct'
            else:
                sampled = self._sample_uint8(
                    img, fix_loc_t, fixation_size_t)
            grid = None
            if return_coords:
                grid = self._direct_grid(img.shape[-2:], fix_loc_t, fixation_size_t)
        else:
            if not img.is_floating_point():
                raise TypeError(
                    f"integer sampling currently supports torch.uint8, got {img.dtype}")
            coordinate_dtype = self._coordinate_dtype(img.dtype)
            fix_loc_t, fixation_size_t = self._prepare_sampling_args(
                img, fix_loc, fixation_size, dtype=coordinate_dtype)
            if direct:
                sampled = self._torch_direct_sample(
                    img, fix_loc_t, fixation_size_t)
                self._last_backend = 'torch_direct'
                grid = None
            else:
                sampled, grid = self._sample_float(
                    img, fix_loc_t, fixation_size_t)
            if return_coords and grid is None:
                grid = self._direct_grid(
                    img.shape[-2:], fix_loc_t, fixation_size_t)

        sampled = self._convert_output(sampled)
        sampled = self._mask_invalid_samples(sampled, self.valid_mask)
        
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
        return (f'GridSampler(fov={self.fov}, cmf_a={self.cmf_a}, '
                f'fov_type={self.fov_type!r}, style={self.style}, '
                f'resolution={self.resolution}, mode={self.mode}, backend={self.backend}, '
                f'output_dtype={self.output_dtype}, n={len(self.coords)})')
    

@add_to_all(__all__)
class KNNGridSampler(BaseGridSampler):
    """
    K-Nearest Neighbors grid sampler for foveated vision.
    
    This sampler uses KNN-based sampling to perform local average pooling over a high-res sensor array into a lower-res sensor array, with the same CorticalSensorManifold.
    - highres_coords: akin to photoreceptors: there are more of them
    - coords: akin to retinal ganglion cells: there are less of them, and they integrate over a local pool of photoreceptors (highres_coords)
    """
    
    def __init__(self, fov, cmf_a, resolution, res_mult=3, cmf_a_mult=1,
                 fixation_size=3000, k=None, style='isotropic', sample_cortex=True,
                 dtype=torch.float, device='cuda', isotropic_plotting_type='v1like',
                 backend='auto', output_dtype=None, fov_type='circular'):
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
        self.highres_coords = SamplingCoords(
            fov, cmf_a_mult*cmf_a, res_mult*resolution, device=device,
            style=style, dtype=dtype,
            isotropic_plotting_type=isotropic_plotting_type,
            fov_type=fov_type)
        self.coords = SamplingCoords(
            fov, cmf_a, resolution, device=device, style=style, dtype=dtype,
            isotropic_plotting_type=isotropic_plotting_type,
            fov_type=fov_type)

        if k is None:
            # default to the ratio of the number of pixels in the retinal and cortical grids
            k = int(np.round(len(self.highres_coords) / len(self.coords)))
            print(f'auto-set knngridsampler k={k}')

        self.pooler = KNNPoolingLayer(k, self.highres_coords, self.coords, mode='avg', device=device, sample_cortex=sample_cortex)

        if output_dtype is not None and not output_dtype.is_floating_point:
            raise ValueError("KNN pooling requires a floating output dtype")
        self.input_sampler = GridSampler(
            fov, cmf_a_mult * cmf_a, res_mult * resolution, device=device, dtype=dtype,
            mode='nearest', style=style, coords=self.highres_coords,
            isotropic_plotting_type=isotropic_plotting_type, backend=backend,
            fov_type=fov_type)
        self.backend = backend
        self.output_dtype = output_dtype
        self._last_backend = None

        self.sampling_grid = self._prep_grid_for_grid_sample(self.highres_coords.cartesian)
        self.out_sampling_grid = self._prep_grid_for_grid_sample(self.coords.cartesian)

        self.polar_radius = self.coords.polar[:,0]
        self.highres_valid_mask = self.highres_coords.valid_mask
        self.valid_mask = self.coords.valid_mask
        self.fov = fov
        self.cmf_a = cmf_a
        self.resolution = resolution
        self.fixation_size = fixation_size
        self.k = k # number of neighbors to consider, the later ones will be weighted less or not at all
        self.dtype = dtype
        self.device = device
        self.style = style
        self.fov_type = fov_type
        self.num_coords = len(self.coords)
        self.sample_cortex = sample_cortex

        self.rf_sizes = self.coords.get_scatter_sizes()

    def forward(self, img, fix_loc=None, fixation_size=None, direct=False):
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
        img = img.to(self.device)

        # Sample integer images before casting, so only the compact photoreceptor array is
        # promoted for the arithmetic required by KNN pooling.
        ret_samples = self.input_sampler(
            img, fix_loc=fix_loc, fixation_size=fixation_size,
            direct=direct).to(self.dtype)
        self._last_backend = self.input_sampler._last_backend
        ret_samples = self._mask_invalid_samples(
            ret_samples, self.highres_valid_mask)
        # pool to get the final retinal samples
        pooled_samples = self.pooler(ret_samples)
        pooled_samples = self._mask_invalid_samples(
            pooled_samples, self.valid_mask)

        if self.output_dtype is not None:
            pooled_samples = pooled_samples.to(self.output_dtype)

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
    def __init__(self, *args, gauss_sigma, **kwargs):
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
        self.pooler = KNNPoolingLayer(self.k, self.highres_coords, self.coords, mode='gaussian', device=self.device, sample_cortex=self.sample_cortex, gauss_sigma=gauss_sigma)

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
