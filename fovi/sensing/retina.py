import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from scipy.optimize import minimize_scalar

from .coords import find_desired_res
from .samplers import GaussianKNNGridSampler, KNNGridSampler, GridSampler
from ..utils import add_to_all
from ..utils.fastaugs import transforms as fastT
from ..utils.fastaugs import functional as fastF
from ..utils.fastaugs import functional_tensor as fastFT

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
                 isotropic_plotting_type='v1like',
                 sampler_backend='auto',
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
            sampler_backend (str, optional): Sampler backend (``auto``, ``torch``, or
                ``cuda``) for both uint8 and floating inputs. Defaults to ``auto``.
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
            self.sampler = GaussianKNNGridSampler(self.fov, self.cmf_a, resolution, fixation_size=self.fixation_size, device=device, style=style, isotropic_plotting_type=isotropic_plotting_type, backend=sampler_backend, **kwargs)
        elif sampler == 'pooling':
            self.sampler = KNNGridSampler(self.fov, self.cmf_a, resolution, fixation_size=self.fixation_size, device=device, style=style, isotropic_plotting_type=isotropic_plotting_type, backend=sampler_backend, **kwargs)
        elif sampler == 'grid_nn':
            self.sampler = GridSampler(self.fov, self.cmf_a, resolution, device=device, mode='nearest', style=style, isotropic_plotting_type=isotropic_plotting_type, backend=sampler_backend, **kwargs)
        elif sampler == 'grid_bilinear':
            self.sampler = GridSampler(self.fov, self.cmf_a, resolution, device=device, mode='bilinear', style=style, isotropic_plotting_type=isotropic_plotting_type, backend=sampler_backend, **kwargs)

        else:
            raise ValueError(f'Invalid sampler: {sampler}')
        self.device = device
        self.dtype = dtype

        self.pre_transforms = pre_transforms
        self.post_transforms = post_transforms

        # Whether the sample-then-augment fast path may be used when eligible
        # (see _fast_pre_transforms_supported). Set to False to force the reference path.
        self.fast_pre_transforms = True
        self._padding_mask_src_cache = None

        self.scatter_sizes = self.sampler.coords.get_scatter_sizes().cpu().numpy()

    @torch.autocast(device_type='cuda', enabled=False)
    def forward(self, x, fix_loc, fixation_size=None, **kwargs):
        """
        Forward pass of the retinal transform.

        Args:
            x (torch.Tensor): NCHW input tensor. Floating inputs retain their existing
                value-range semantics; uint8 inputs are interpreted as the equivalent
                unit-range image and converted after sampling whenever semantics permit.
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

        input_is_uint8 = x.dtype == torch.uint8
        apply_pre = self.pre_transforms is not None and self.training
        if apply_pre and not kwargs and self._fast_pre_transforms_supported():
            # Fast path: sample first (nearest-neighbor), then apply the pointwise pre-warp
            # transforms to the ~N sampled points instead of the full H x W image, once per
            # fixation. Bit-exact with the reference path (see _sample_then_pre_transform)
            # and consumes the RNG streams identically.
            self._announce_pre_transform_execution(fast=True)
            x = self._sample_then_pre_transform(x, fix_loc, fixation_size)
        else:
            if apply_pre:
                self._announce_pre_transform_execution(fast=False)
                if input_is_uint8:
                    # Reference semantics: camera uint8 represents the same unit-range
                    # image that ToTorchImage currently supplies before RetinalTransform.
                    x = self._uint8_to_unit(x)
                x = self.pre_transforms(x.clone())
            sample_kwargs = kwargs
            if input_is_uint8 and apply_pre:
                # The image has been converted because pre-warp transforms needed it, but
                # its sampling coordinates must retain the canonical raw-input definition.
                sample_kwargs = {**kwargs, 'direct': True}
            x = self.sampler(
                x, fix_loc=fix_loc, fixation_size=fixation_size, **sample_kwargs)
            if input_is_uint8 and not apply_pre:
                # No pre-warp operation needs the full image, so convert only the compact
                # sampled tensor.  Bilinear and pooling outputs remain native-scale float.
                x = self._uint8_to_unit(x)

        if self.post_transforms is not None and self.training:
            x = self.post_transforms(x.unsqueeze(3)).clone().squeeze(3)

        if self.foveal_color is not None:
            x = self.foveal_color(x, self.sampler.polar_radius)

        if not self.training and self.no_color_val:
            x = TF.rgb_to_grayscale(x.unsqueeze(3), num_output_channels=3).clone().squeeze(3)

        if self.reshape_as_grid:
            x = x.reshape(x.shape[0], x.shape[1], self.resolution, self.resolution)

        return x.to(self.dtype)

    def _uint8_to_unit(self, x):
        """Convert native uint8-scale values to the current unit-float contract.

        This deliberately mirrors the tensor path in ``ToTorchImage``: cast to the target
        dtype first, then divide by 255.  It is representation conversion, not statistical
        normalization; normalization remains in the supplied transform pipelines.
        """
        return x.to(dtype=self.dtype).div_(255.0)

    # ------------------------------------------------------------------
    # Sample-then-augment fast path for pre-warp transforms
    # ------------------------------------------------------------------

    _FAST_PRE_TRANSFORM_TYPES = (fastT.RandomColorJitter, fastT.RandomGrayscale,
                                 fastT.NormalizeGPU)

    def _announce_pre_transform_execution(self, fast):
        """One-time notice per decision, making the semantics/execution distinction
        explicit: the ``transforms.where`` config declares SEMANTICS (pre_warp =
        augmentations are defined on the full pre-warp image); the implementation is
        free to choose any execution that preserves those semantics exactly, and
        semantics-preserving post-warp execution (sample-then-augment) is preferred
        because it touches ~11x fewer elements. Re-announces only if the decision
        changes (e.g. transforms were swapped at runtime)."""
        if getattr(self, '_announced_pre_transform_fast', None) == fast:
            return
        self._announced_pre_transform_fast = fast
        names = [type(t).__name__ for t in getattr(self.pre_transforms, 'transforms', [])]
        if fast:
            print(
                f"Note: pre_warp transforms {names} execute POST-warp (sample-then-augment) "
                "with exact pre_warp semantics (bit-identical outputs and RNG streams). "
                "pre_warp/post_warp declares semantics, not implementation; semantics-"
                "preserving post-warp execution is preferred."
            )
        else:
            reason = (
                "fast_pre_transforms is disabled"
                if not self.fast_pre_transforms
                else "at least one transform/sampler is not on the verified "
                "semantics-preserving allowlist for post-warp execution"
            )
            print(
                f"Note: pre_warp transforms {names} execute on the full pre-warp image "
                f"(reference path) - {reason} "
                "(see RetinalTransform._fast_pre_transforms_supported)."
            )

    def _fast_pre_transforms_supported(self):
        """The fast path requires (a) a plain nearest-neighbor GridSampler, so that sampling
        commutes exactly with pointwise (per-pixel, per-image-parameter) transforms, and
        (b) pre_transforms composed solely of such pointwise transforms.

        Strict allowlist, checked by exact type: the container must be a plain
        ``fastT.Compose`` (other containers such as ``RandomApply``/``MultiSample`` also
        expose ``.transforms`` but have different call semantics), and every member must be
        exactly one of the verified pointwise classes — no subclasses, whose overridden
        behavior the fast path's replicated arithmetic would not reproduce. Anything else
        (e.g., a future spatial transform in pre_transforms) falls back to the bit-exact
        reference path rather than silently producing point-cloud-augmented results."""
        if not self.fast_pre_transforms:
            return False
        if type(self.sampler) is not GridSampler or self.sampler.mode != 'nearest':
            return False
        if type(self.pre_transforms) is not fastT.Compose:
            return False
        return all(type(t) in self._FAST_PRE_TRANSFORM_TYPES for t in self.pre_transforms.transforms)

    def _random_grayscale_points(self, s4, idx, num_output_channels):
        """Same arithmetic as ``fastF.random_grayscale`` (ITU-R 601-2 luma, index_copy on the
        selected images), but with the 3-element luma weights cached on-device instead of an
        H2D copy per call."""
        weights = getattr(self, '_grayscale_weights', None)
        if weights is None or weights.device != s4.device:
            weights = fastF.grayscale[None, :, :, :].to(s4.device)
            self._grayscale_weights = weights
        sel = s4.index_select(0, idx)
        L = (sel * weights).sum(dim=1, keepdim=True).to(dtype=sel.dtype)
        if num_output_channels == 3:
            L = torch.cat(3 * [L], dim=1)
        return s4.index_copy(0, idx, L)

    def _padding_mask_src(self, x):
        """(B,1,H,W) ones (batch-expanded view of a cached (1,1,H,W) tensor) used to mark
        which sampled points fall inside the image (grid_sample zero-pads outside)."""
        key = (x.shape[-2], x.shape[-1], x.device, x.dtype)
        cached = self._padding_mask_src_cache
        if cached is None or cached[0] != key:
            ones = torch.ones(1, 1, x.shape[-2], x.shape[-1], device=x.device, dtype=x.dtype)
            self._padding_mask_src_cache = (key, ones)
            cached = self._padding_mask_src_cache
        return cached[1].expand(x.shape[0], 1, x.shape[-2], x.shape[-1])

    def _sample_then_pre_transform(self, x, fix_loc, fixation_size):
        """Equivalent to ``self.sampler(self.pre_transforms(x.clone()), ...)`` for a
        nearest-neighbor GridSampler and pointwise pre_transforms, but ~10x cheaper:
        the image is sampled once, and the transforms run on the (B, C, N) point cloud.

        Exactness notes (all verified bit-exact against the reference path):
        - nearest-neighbor sampling selects pixel values, so pointwise ops commute with it;
        - the contrast-jitter reference mean is a full-image statistic and is still computed
          on the (brightness-adjusted) full image, exactly as FT.adjust_contrast does;
        - grid_sample zero-pads out-of-bounds points AFTER the transforms in the reference
          path, so the transformed samples are re-masked with a grid-sampled ones image;
        - RNG draws (bernoulli masks + per-image jitter parameters) have identical shapes
          and order, leaving the CUDA generator stream identical to the reference path.
        """
        input_is_uint8 = x.dtype == torch.uint8
        if input_is_uint8:
            sampled = self.sampler(
                x, fix_loc=fix_loc, fixation_size=fixation_size)
            sampled = self._uint8_to_unit(sampled)
            fix_loc_t, fixation_size_t = self.sampler._prepare_sampling_args(
                x, fix_loc, fixation_size)
            pixel_x, pixel_y = self.sampler._direct_pixel_coords(
                x.shape[-2:], fix_loc_t, fixation_size_t)
            sample_x = torch.round(pixel_x - 0.5)
            sample_y = torch.round(pixel_y - 0.5)
            mask = (
                (sample_x >= 0) & (sample_x < x.shape[-1])
                & (sample_y >= 0) & (sample_y < x.shape[-2])
            )[:, None, :].to(sampled.dtype)
        else:
            grid = self.sampler._transform_fix_grid(x.shape[-2:], fix_loc, fixation_size)
            sampled = torch.nn.functional.grid_sample(
                x, grid, mode=self.sampler.mode, align_corners=False).squeeze(2)
            mask = torch.nn.functional.grid_sample(
                self._padding_mask_src(x), grid, mode=self.sampler.mode,
                align_corners=False).squeeze(2)

        s4 = sampled.unsqueeze(3)  # (B, C, N, 1): a 4D "image" for the batch transforms
        for t in self.pre_transforms.transforms:
            if isinstance(t, fastT.RandomColorJitter):
                # same RNG consumption as t(full_image): bernoulli(B) + 4 x uniform(len(idx))
                t.before_call(s4)
                idx = t.idx
                sel = s4.index_select(0, idx)
                full_sel = x.index_select(0, idx)
                if input_is_uint8:
                    full_sel = self._uint8_to_unit(full_sel)
                sel = _color_jitter_points(sel, full_sel, t.h, t.s, t.v, t.c)
                s4 = s4.index_copy(0, idx, sel.to(dtype=s4.dtype))
            elif isinstance(t, fastT.RandomGrayscale):
                t.before_call(s4)
                s4 = self._random_grayscale_points(s4, t.idx, t.num_output_channels)
            elif isinstance(t, fastT.NormalizeGPU):
                s4 = t(s4)
            else:  # pragma: no cover - guarded by _fast_pre_transforms_supported
                raise RuntimeError(f'unsupported pre_transform for fast path: {t}')

        return s4.squeeze(3) * mask

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
            fixation_size (int, tuple, torch.Tensor, or None): Fixation size specification.
            batch_size (int): Number of samples in the batch.

        Returns:
            np.ndarray or torch.Tensor: Formatted fixation size of shape (batch_size, 2).
            Tensor inputs stay tensors on their device (no host round-trip / device sync).
        """
        if isinstance(fixation_size, torch.Tensor):
            # pure-tensor handling: never .cpu()/.numpy() a (possibly CUDA) tensor here
            fixation_size = fixation_size.squeeze()
            if fixation_size.dim() <= 1:
                fixation_size = fixation_size.reshape(-1)
                if fixation_size.shape[0] == 2:
                    fixation_size = fixation_size.unsqueeze(0).expand(batch_size, 2)
                elif fixation_size.shape[0] == batch_size:
                    fixation_size = fixation_size.unsqueeze(1).expand(batch_size, 2)
                else:
                    raise ValueError(f'fixation_size shape: {tuple(fixation_size.shape)}')
            else:
                assert fixation_size.shape[0] == batch_size
            # move small CPU fixation sizes to the sampling device here (once) rather than
            # inside the per-fixation grid transform
            return fixation_size.to(self.device)

        if fixation_size is None or (isinstance(fixation_size, str) and fixation_size == 'none'):
            if hasattr(self.fixation_size, '__len__'):
                fixation_size = np.array(self.fixation_size)
            else:
                fixation_size = np.array([self.fixation_size, self.fixation_size])
        elif isinstance(fixation_size, int) or isinstance(fixation_size, float):
            fixation_size = np.array([fixation_size, fixation_size])
        elif hasattr(fixation_size, '__len__'):
            fixation_size = np.array(fixation_size)
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
        if isinstance(fix_loc, torch.Tensor):
            # pure-tensor handling: the previous .cpu().numpy() round-trip forced a device
            # sync per fixation for CUDA fixations on the training hot path
            fix_loc = fix_loc.squeeze()
            if fix_loc.dim() <= 1:
                fix_loc = fix_loc.reshape(-1).unsqueeze(0).expand(batch_size, -1)
            else:
                assert fix_loc.shape[0] == batch_size
            return fix_loc.to(device=self.device, dtype=self.dtype)

        if fix_loc is None:
            fix_loc = np.array([0.5, 0.5])
        elif isinstance(fix_loc, float) or isinstance(fix_loc, int):
            fix_loc = np.array([fix_loc, fix_loc])
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

def _color_jitter_points(pts, full_sel, hue, sat, val, con):
    """Replicate ``fastFT.color_jitter`` (brightness -> contrast -> saturation -> hue) on
    nearest-sampled points instead of full images.

    Args:
        pts (torch.Tensor): (n, 3, N, 1) sampled points of the selected images.
        full_sel (torch.Tensor): (n, 3, H, W) the corresponding full images; needed only for
            the contrast-jitter reference mean, which is a full-image statistic.
        hue/sat/val/con (torch.Tensor or None): per-image factors of shape (n,), exactly as
            drawn by RandomColorJitter.before_call.

    Every operation is the same arithmetic as the corresponding ``fastFT.adjust_*`` call on
    the full image (bit-exact), evaluated only at the sampled points. The wrapper asserts of
    the ``adjust_*`` functions (``.any()``/``.all()`` device syncs) are intentionally skipped.
    """
    if val is not None:
        v4 = val[:, None, None, None]
        pts = fastFT._blend(pts, torch.zeros_like(pts), v4)
    if con is not None:
        c4 = con[:, None, None, None]
        if val is not None:
            # brightness-adjusted full image; _blend(img, zeros, v) == clamp(v * img, 0, 1)
            full_b = full_sel.mul(val[:, None, None, None]).clamp_(0.0, 1.0)
        else:
            full_b = full_sel
        mean = torch.mean(fastFT.rgb_to_grayscale(full_b).to(pts.dtype),
                          dim=(-3, -2, -1), keepdim=True)
        pts = fastFT._blend(pts, mean, c4)
    if sat is not None:
        s4 = sat[:, None, None, None]
        pts = fastFT._blend(pts, fastFT.rgb_to_grayscale(pts), s4)
    if hue is not None:
        h3 = hue[:, None, None]
        if not pts.is_contiguous():
            pts = pts.contiguous()
        pts = fastFT._rgb2hsv_fast(pts)  # in-place on pts (owned copy)
        h_chan = pts.select(dim=-3, index=0)
        h_chan.add_(h3).fmod_(1.0)
        pts = fastFT._hsv2rgb_fast(pts)
    return pts


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
