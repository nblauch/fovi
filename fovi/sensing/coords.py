import math

import numpy as np
import torch

from ..utils import normalize, add_to_all
from .manifold import spherical_radius_limit, validate_field_geometry
from .manifold import vis_cartesian_to_cortical_cartesian_coords as vis_to_sensor_manifold

__all__ = []

SAMPLING_STYLES = (
    'isotropic', 'isotropic_fixn',
    'logpolar', 'logpolar_as_grid',
    'warped_cartesian', 'warped_cartesian_as_grid',
    'uniform', 'uniform_as_grid',
)
FOV_TYPES = ('circular', 'square', 'wang')
WARPED_CARTESIAN_STYLES = (
    'warped_cartesian', 'warped_cartesian_as_grid')
#: Norms available for measuring native-plane radius in the Cartesian warp.
#: ``2.0`` gives circular iso-eccentricity shells, ``inf`` gives square ones.
RADIUS_NORMS = (2.0, math.inf)


def _resolution_shape(res):
    """Return grid height and width while retaining scalar API compatibility."""
    if isinstance(res, (int, np.integer)) and not isinstance(res, bool):
        if res <= 0:
            raise ValueError("Resolution must be positive")
        return int(res), int(res)
    if isinstance(res, (tuple, list, np.ndarray)) and len(res) == 2:
        height, width = res
        if all(isinstance(value, (int, np.integer)) and not isinstance(value, bool)
               and value > 0 for value in (height, width)):
            return int(height), int(width)
    raise ValueError("Resolution must be a positive integer or (height, width) pair")


def _field_aspect(res):
    height, width = _resolution_shape(res)
    return width / height


def _footprint_radius(angle, fov_type, aspect, max_val=1.0):
    """Visual-chart distance from the origin to the footprint edge."""
    cosine = torch.cos(angle).abs()
    sine = torch.sin(angle).abs()
    if fov_type == 'circular':
        return max_val / torch.sqrt((cosine / aspect) ** 2 + sine ** 2)
    if fov_type in ('square', 'wang'):
        return max_val * torch.minimum(
            aspect / cosine.clamp_min(1e-30),
            1 / sine.clamp_min(1e-30))
    raise ValueError(f"Unsupported footprint {fov_type!r}")


def _rectangular_native_to_visual(coordinates, fov, cmf_a, fov_type,
                                  aspect, max_val, radius_norm):
    radius = torch.linalg.vector_norm(coordinates, dim=-1, keepdim=True)
    angle = torch.atan2(coordinates[..., 1], coordinates[..., 0])
    boundary = _footprint_radius(angle, fov_type, aspect, max_val)[..., None]
    if radius_norm == math.inf:
        native_fraction = torch.amax(
            coordinates.abs() / coordinates.new_tensor((aspect, 1.0)),
            dim=-1, keepdim=True) / max_val
    else:
        native_fraction = radius / boundary
    log_boundary = torch.log1p(boundary * (fov / 2) / cmf_a)
    visual_fraction = cmf_a * torch.expm1(native_fraction * log_boundary) / (boundary * fov / 2)
    scale = torch.where(
        native_fraction > 0, visual_fraction / native_fraction.clamp_min(1e-30),
        log_boundary * cmf_a / (boundary * fov / 2))
    return coordinates * scale


def _rectangular_visual_to_native(coordinates, fov, cmf_a, fov_type,
                                  aspect, max_val, radius_norm):
    radius = torch.linalg.vector_norm(coordinates, dim=-1, keepdim=True)
    angle = torch.atan2(coordinates[..., 1], coordinates[..., 0])
    boundary = _footprint_radius(angle, fov_type, aspect, max_val)[..., None]
    if radius_norm == math.inf:
        visual_fraction = torch.amax(
            coordinates.abs() / coordinates.new_tensor((aspect, 1.0)),
            dim=-1, keepdim=True) / max_val
    else:
        visual_fraction = radius / boundary
    log_boundary = torch.log1p(boundary * (fov / 2) / cmf_a)
    native_fraction = torch.log1p(
        visual_fraction * boundary * (fov / 2) / cmf_a) / log_boundary
    scale = torch.where(
        visual_fraction > 0, native_fraction / visual_fraction.clamp_min(1e-30),
        boundary * (fov / 2) / (cmf_a * log_boundary))
    return coordinates * scale


def _validate_fov_type(fov_type, style=None):
    """Validate an FoV type and its optional sensor-style constraint."""
    if fov_type not in FOV_TYPES:
        raise ValueError(
            f"fov_type must be one of {FOV_TYPES}, got {fov_type!r}")
    if (fov_type == 'wang'
            and style not in (None, *WARPED_CARTESIAN_STYLES)):
        raise ValueError(
            "fov_type='wang' is only supported by "
            f"{WARPED_CARTESIAN_STYLES}, got style={style!r}")


def _validate_radius_norm(radius_norm, fov_type=None, style=None):
    """Validate the native-plane radius norm against its FoV and style."""
    if radius_norm not in RADIUS_NORMS:
        raise ValueError(
            f"radius_norm must be one of {RADIUS_NORMS}, got {radius_norm!r}")
    if radius_norm == math.inf and fov_type == 'wang':
        raise ValueError(
            "fov_type='wang' normalizes the Euclidean radius at the native "
            "square's side centers and is undefined for radius_norm=inf")
    # Only the Cartesian warp measures a native-plane radius, so a non-default
    # norm would be silently discarded by every other style.
    if (radius_norm != 2.0 and style is not None
            and style not in WARPED_CARTESIAN_STYLES):
        raise ValueError(
            f"radius_norm={radius_norm!r} only applies to "
            f"{WARPED_CARTESIAN_STYLES}, got style={style!r}")


def _native_radius(coordinates, radius_norm, dim=-1, keepdim=True):
    """Measure native-plane radius in the requested norm."""
    if radius_norm == math.inf:
        return coordinates.abs().amax(dim=dim, keepdim=keepdim)
    return torch.linalg.vector_norm(coordinates, dim=dim, keepdim=keepdim)


def _fov_valid_mask(cartesian, fov_type, max_val=1.0, aspect=1.0):
    """Return which normalized visual coordinates lie in the requested FoV."""
    _validate_fov_type(fov_type)
    eps = 16 * torch.finfo(cartesian.dtype).eps
    if fov_type == 'wang':
        return torch.ones(
            cartesian.shape[0], device=cartesian.device, dtype=torch.bool)
    if fov_type == 'circular':
        return torch.linalg.vector_norm(
            cartesian / cartesian.new_tensor((aspect, 1.0)), dim=1) <= max_val + eps
    square_half_extent = max_val
    return ((cartesian[:, 0].abs() <= aspect * square_half_extent + eps)
            & (cartesian[:, 1].abs() <= square_half_extent + eps))


def _validate_square_shell_geometry(fov: float, cmf_a: float, res: int,
                                    max_val: float,
                                    field_geometry: str = 'planar') -> None:
    """Validate an infinity-norm sensor, whose corners reach sqrt(2) * max_val."""
    if not all(np.isfinite(value) and value > 0 for value in (fov, cmf_a, max_val)):
        raise ValueError("Square-shell fov, cmf_a, and max_val must be finite and positive")
    _resolution_shape(res)
    aspect = _field_aspect(res)
    if field_geometry == 'spherical' and (fov > 180 or np.hypot(aspect, 1) * max_val * fov / 2 >= 180):
        raise ValueError("Spherical square footprint must stay below 180 degrees eccentricity and FoV must not exceed 180 degrees")


def _warp_law(fov, cmf_a, radius_norm, fov_type='circular', max_val=1.0):
    """Return the ``(radius_normalizer, rho_axis)`` pair defining the CMF warp.

    Both norms integrate the same inverse-linear CMF ``M(e) ~ 1/(e + cmf_a)``;
    they differ only in which norm measures the native radius, and in how
    ``max_val`` reaches the normalizer. Under the Euclidean norm the native
    unit circle maps to the nominal visual radius and ``max_val`` bounds the
    footprint by masking. Under the infinity norm the native square of
    half-side ``max_val`` maps exactly onto the visual square of half-side
    ``max_val``, so nothing needs masking.
    """
    half_fov = fov / 2.0
    if radius_norm == math.inf:
        return max_val, np.log1p(max_val * half_fov / cmf_a)
    normalizer = _warped_cartesian_radius_normalizer(
        fov_type, fov, cmf_a, max_val)
    return normalizer, np.log1p(half_fov / cmf_a)


def _inverse_warped_cartesian(
        plotting_coords, fov, cmf_a, radius_normalizer=1.0,
        rho_axis=None, radius_norm=2.0):
    """Inverse the radial CMF warp from its Cartesian plane to visual space.

    ``radius_normalizer=1`` maps the native unit circle to the nominal visual
    radius. The Wang-style footprint uses a CMF-dependent normalizer so the
    side centers of the native square map to the outer-square half-side.
    ``radius_norm`` selects the norm measuring native radius: ``2.0`` gives
    circular iso-eccentricity shells, ``inf`` gives square ones.
    """
    if rho_axis is None:
        rho_axis = np.log1p((fov / 2.0) / cmf_a)
    plotting_radius = _native_radius(
        plotting_coords, radius_norm, dim=1, keepdim=False)
    warped_radius = plotting_radius / radius_normalizer
    visual_radius = (
        cmf_a * torch.expm1(warped_radius * rho_axis) / (fov / 2.0))
    scale = torch.where(
        plotting_radius > 0,
        visual_radius / plotting_radius.clamp_min(torch.finfo(plotting_radius.dtype).tiny),
        torch.full_like(plotting_radius, cmf_a * rho_axis / (radius_normalizer * fov / 2.0)))
    cartesian = plotting_coords * scale[:, None]
    # ``visual_radius`` is measured in ``radius_norm``; polar radius is always
    # Euclidean, so recompute it when the two differ.
    polar_radius = (
        torch.linalg.vector_norm(cartesian, dim=1)
        if radius_norm == math.inf else visual_radius)
    polar = torch.stack((
        polar_radius,
        torch.atan2(cartesian[:, 1], cartesian[:, 0])), dim=1)
    return cartesian, polar


def _warped_cartesian_radius_normalizer(
        fov_type, fov=None, cmf_a=None, max_val=1.0):
    """Return the radial normalizer for a Wang FoV.

    Its minimum perimeter radius occurs at the native square's side centers.
    Map that radius to the half-side of the square circumscribing the nominal
    visual circle, so the warped footprint contains and touches that square.
    """
    _validate_fov_type(fov_type, style='warped_cartesian')
    if fov_type == 'wang':
        if fov is None or cmf_a is None:
            raise ValueError(
                "fov and cmf_a are required for a Wang FoV")
        nominal_radius = fov / 2.0
        square_half_extent = max_val
        rho_at_square_side = (
            np.log1p(square_half_extent * nominal_radius / cmf_a)
            / np.log1p(nominal_radius / cmf_a))
        return max_val / rho_at_square_side
    return 1.0


def _append_missing_square_corners(coords, polar_coords, max_val):
    """Ensure the enclosing radial manifold contributes all square corners."""
    square_half_extent = max_val
    corners = torch.tensor(
        [[-square_half_extent, -square_half_extent],
         [-square_half_extent, square_half_extent],
         [square_half_extent, -square_half_extent],
         [square_half_extent, square_half_extent]],
        device=coords.device, dtype=coords.dtype)
    missing = []
    for corner in corners:
        if not torch.any(torch.all(torch.isclose(
                coords, corner[None], rtol=1e-5, atol=1e-6), dim=1)):
            missing.append(corner)
    if not missing:
        return coords, polar_coords
    missing = torch.stack(missing)
    missing_polar = torch.stack((
        torch.linalg.vector_norm(missing, dim=1),
        torch.atan2(missing[:, 1], missing[:, 0])), dim=1)
    return (torch.cat((coords, missing), dim=0),
            torch.cat((polar_coords, missing_polar), dim=0))

@add_to_all(__all__)
class SamplingCoords():
    """Object for storing multiple coordinate systems relevant to a set of visual field samples.
    
    Args:
        fov (float): Field-of-view in degrees.
        cmf_a (float): A parameter from the CMF: M(r)=1/(r+a). Smaller a = stronger foveation.
        res (int or tuple[int, int]): Scalar resolution or (height, width).
            A pair sets the visual-field aspect and the grid sample budget.
        device (str): What device to operate on.
        style (str): What type of sampling.
        dtype (torch.dtype): What data type to use.
        fov_type (str): ``'circular'`` becomes an oval and ``'square'`` a
            rectangle when the requested resolution is non-square; ``'wang'``
            remains available for warped Cartesian styles.
            The last is only valid for warped Cartesian sensor styles.
        radius_norm (float): Norm measuring native-plane radius for warped
            Cartesian styles. ``2.0`` gives circular iso-eccentricity shells;
            ``inf`` gives square shells, which pair with ``fov_type='square'``
            to retain every cell.
    """
    def __init__(self, fov, cmf_a, res, device='cpu', style='isotropic',
                 dtype=torch.float,
                 max_val=1,
                 isotropic_plotting_type='v1like',
                 fov_type='circular', field_geometry='planar',
                 radius_norm=2.0,
                 ):

        validate_field_geometry(field_geometry)
        self.field_geometry = field_geometry
        _validate_radius_norm(radius_norm, fov_type=fov_type, style=style)
        if radius_norm == math.inf and style in WARPED_CARTESIAN_STYLES:
            _validate_square_shell_geometry(
                fov, cmf_a, res, max_val, field_geometry)
        if field_geometry == 'spherical' and 'logpolar' in style and isinstance(res, (int, np.integer)):
            raise ValueError("Spherical geometry requires an angular sample set, not a flattened grid topology")
        if 'warped_cartesian' in style:
            if not all(np.isfinite(value) and value > 0 for value in (fov, cmf_a)):
                raise ValueError("Warped Cartesian fov and cmf_a must be finite and positive")
            if field_geometry == 'spherical' and fov > 180:
                raise ValueError("Spherical field diameter must not exceed 180 degrees")
        self.fov = fov
        self.cmf_a = cmf_a
        self.resolution = res
        self.grid_shape = _resolution_shape(res)
        self.aspect = _field_aspect(res)
        self.device = device
        self.style = style
        self.dtype = dtype
        self.max_val = max_val
        self.isotropic_plotting_type = isotropic_plotting_type
        _validate_fov_type(fov_type, style=style)
        self.fov_type = fov_type
        self.radius_norm = radius_norm

        if self.grid_shape == (1, 1):
            self.cartesian = torch.zeros(1, 2, device=device, dtype=dtype)
            self.polar = torch.zeros(1, 2, device=device, dtype=dtype)
            self.plotting = torch.zeros(1, 2, device=device, dtype=dtype)
            self.valid_mask = torch.ones(1, device=device, dtype=torch.bool)
            self.fov_padding_coords = torch.empty(
                0, 2, device=device, dtype=dtype)
            self.cartesian_pad_coords = torch.empty(
                0, 2, device=device, dtype=dtype)
            if style in WARPED_CARTESIAN_STYLES or 'logpolar' in style:
                self.cortical = torch.zeros(
                    1, 2, device=device, dtype=dtype)
                self.cortical_pad_coords = torch.empty(
                    0, 2, device=device, dtype=dtype)
            elif 'uniform' in style:
                self.cortical = None
            else:
                self.cortical = torch.zeros(
                    1, 3, device=device, dtype=dtype)
                self.cortical_pad_coords = torch.empty(
                    0, 3, device=device, dtype=dtype)
        else:
            (self.cartesian, self.polar, self.plotting,
             self.valid_mask, self.fov_padding_coords) = get_sampling_coords(
                fov, cmf_a, res, device=device, style=style, max_val=max_val,
                isotropic_plotting_type=isotropic_plotting_type,
                fov_type=self.fov_type, field_geometry=self.field_geometry, return_valid_mask=True,
                return_masked_coords=True, radius_norm=self.radius_norm)
            if field_geometry == 'spherical' and 'warped_cartesian' in style:
                active_angles = self.polar[self.valid_mask, 0] * (fov / 2.0)
                if torch.any(active_angles >= 180):
                    raise ValueError("Spherical warped Cartesian samples must stay below 180 degrees eccentricity")
            # image format (row, col)
            self.cartesian_rowcol = xy_to_rowcol(self.cartesian, do_norm=False, format='-11')

            outer_padding_coords = self.pad_cartesian(
                device=device, dtype=dtype)
            self.fov_padding_coords = self.fov_padding_coords.to(
                device=device, dtype=dtype)
            self.cartesian_pad_coords = torch.cat((
                self.fov_padding_coords, outer_padding_coords), dim=0)

            if field_geometry == 'spherical' and 'isotropic' in style:
                extent = float(self.cartesian_pad_coords.norm(dim=1).max())
                pad_radius = extent * fov / 2
                limit = spherical_radius_limit(cmf_a)
                if pad_radius > limit:
                    raise ValueError(
                        f"Spherical FoV {fov:g} degrees with cmf_a={cmf_a:g} "
                        f"requires padding radius {pad_radius:g} degrees, exceeding "
                        f"the manifold radius limit {limit:g}. This normalized padded "
                        f"grid requires FoV <= {2 * limit / extent:g} degrees; "
                        "padding changes when the grid is rebuilt at a different FoV.")

            if style in WARPED_CARTESIAN_STYLES:
                # The native topology of this sensor is its regular Cartesian
                # lattice in the warped plane.
                self.cortical = self.plotting.clone()
                self.cortical_pad_coords = self._warped_cartesian_pad_plotting.to(
                    device=device, dtype=dtype)
            elif 'logpolar' in style:
                # log polar image, meaning it does not use the proper sensor manifold, but a simplified one
                self.cortical = self.polar.clone()
                if not isinstance(self.resolution, (int, np.integer)):
                    self.cortical[:, 0] = self.plotting[:, 0]
                else:
                    radius = ((self.cartesian[:,0]**2 + self.cartesian[:,1]**2)**.5)*self.fov/2
                    cmf_a_tensor = torch.tensor(
                        self.cmf_a, device=radius.device, dtype=radius.dtype)
                    max_radius = radius.max()
                    rho_max = torch.log(
                        (max_radius + cmf_a_tensor) / cmf_a_tensor)
                    self.cortical[:,0] = torch.log(
                        (radius + cmf_a_tensor) / cmf_a_tensor) / rho_max
                # Preserve the missing endpoint of the periodic angular axis:
                # the samples are 0, 1/res, ..., (res-1)/res, not two copies
                # of the same 0/2pi position.
                self.cortical[:,1] = self.polar[:,1] / (2 * torch.pi)
                self.plotting = self.cortical
                self.cortical_pad_coords = self._logpolar_pad_cortical.to(
                    device=device, dtype=dtype)
            elif 'uniform' in style:
                self.cortical = None
            else:
                # cortical coordinates to be used for sampling RFs
                self.cortical = vis_to_sensor_manifold(self.cartesian.cpu().numpy(), cmf_a, fov, as_tensor=True, device=device, field_geometry=field_geometry)
                self.cortical_pad_coords = vis_to_sensor_manifold(self.cartesian_pad_coords.cpu().numpy(), cmf_a, fov, as_tensor=True, device=device, field_geometry=field_geometry).to(dtype=dtype)

        self.cartesian = self.cartesian.to(dtype)
        self.cartesian_rowcol = xy_to_rowcol(self.cartesian, do_norm=False, format='-11')
        if self.cortical is not None:
            self.cortical = self.cortical.to(dtype)
        self.polar = self.polar.to(dtype)
        self.plotting = self.plotting.to(dtype)
        self.shape = self.cartesian.shape

    def as_grid(self, values: torch.Tensor, sample_dim: int = -1) -> torch.Tensor:
        """Replace a canonical sample axis with the sensor's image axes.

        Cartesian grids return upright rows down and columns right. Log-polar
        grids retain their radius/angle ordering. Other axes are unchanged.

        Args:
            values: Tensor with one axis containing all canonical samples.
            sample_dim: Sample axis; use zero for coordinates shaped (N, 2).

        Returns:
            Tensor with the sample axis replaced by height and width.
        """
        if not self.style.endswith('_as_grid'):
            raise ValueError(f"Grid layout requires an _as_grid style, got {self.style!r}")
        sample_dim %= values.ndim
        height, width = self.grid_shape
        if values.shape[sample_dim] != height * width:
            raise ValueError("Sample axis does not match the sensor grid resolution")
        # Canonical Cartesian order has x as the first axis; log-polar order
        # already has angle rows and radius columns.
        axes = (width, height) if self.style in ('uniform_as_grid', 'warped_cartesian_as_grid') else (height, width)
        shape = (*values.shape[:sample_dim], *axes,
                 *values.shape[sample_dim + 1:])
        grid = values.reshape(shape)
        if self.style in ('uniform_as_grid', 'warped_cartesian_as_grid'):
            grid = grid.transpose(sample_dim, sample_dim + 1).flip(sample_dim)
        return grid.contiguous()

    def native_to_visual(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Map (..., 2) native Cartesian warp coordinates to the visual chart.

        Visual chart radius one is half the configured FoV. For spherical
        geometry this is angular eccentricity, not a pinhole projection.
        The unbounded chart includes masked cells outside the model footprint;
        renderers must handle their own angular padding domain.
        """
        if self.style not in WARPED_CARTESIAN_STYLES:
            raise ValueError("Analytic native mapping requires a Cartesian warp sensor")
        if coordinates.shape[-1] != 2:
            raise ValueError("Native coordinates must end in a two-coordinate axis")
        if not isinstance(self.resolution, (int, np.integer)):
            return _rectangular_native_to_visual(
                coordinates, self.fov, self.cmf_a, self.fov_type,
                self.aspect, self.max_val, self.radius_norm)
        normalizer, rho_axis = _warp_law(
            self.fov, self.cmf_a, self.radius_norm, self.fov_type, self.max_val)
        visual, _ = _inverse_warped_cartesian(
            coordinates.reshape(-1, 2), self.fov, self.cmf_a,
            radius_normalizer=normalizer, rho_axis=rho_axis,
            radius_norm=self.radius_norm)
        return visual.reshape(coordinates.shape)

    def visual_to_native(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Map (..., 2) normalized visual-chart positions to the native grid."""
        if self.style not in WARPED_CARTESIAN_STYLES:
            raise ValueError("Analytic native mapping requires a Cartesian warp sensor")
        if coordinates.shape[-1] != 2:
            raise ValueError("Visual coordinates must end in a two-coordinate axis")
        if not isinstance(self.resolution, (int, np.integer)):
            return _rectangular_visual_to_native(
                coordinates, self.fov, self.cmf_a, self.fov_type,
                self.aspect, self.max_val, self.radius_norm)
        radius = _native_radius(coordinates, self.radius_norm)
        normalizer, axis_scale = _warp_law(
            self.fov, self.cmf_a, self.radius_norm, self.fov_type, self.max_val)
        native_radius = normalizer * torch.log1p(radius * (self.fov / 2.0) / self.cmf_a) / axis_scale
        scale = torch.where(radius > 0, native_radius / radius.clamp_min(torch.finfo(radius.dtype).tiny),
                            torch.full_like(radius, normalizer * (self.fov / 2.0) / self.cmf_a / axis_scale))
        return coordinates * scale
    
    def pad_cartesian(self, padding_distance=0.5, device=None, dtype=None):
        """Generate additional cartesian coordinates for padding around the sampling grid.
        
        Args:
            padding_distance (float): Padding extent in normalized coordinates.
                For spherical radial grids, measured from the real retinal rim,
                retaining at least one complete neighboring ring. Planar/legacy
                radial grids retain the historical interval starting at the first
                outer ring, which can overshoot the requested rim margin by one
                ring. Preserve that behavior for checkpoint KNN compatibility.
                Warped Cartesian and log-polar layouts pad in their own grid
                coordinates using a whole number of grid intervals.
            device (str, optional): Device to place the coordinates on. Defaults to None.
            dtype (torch.dtype, optional): Data type for the coordinates. Defaults to None.
            
        Returns:
            torch.Tensor: Additional cartesian coordinates for padding.
        """
        if 'uniform' in self.style and not isinstance(self.resolution, (int, np.integer)):
            height, width = self.grid_shape
            step_x = 2 * self.max_val * self.aspect / width
            step_y = 2 * self.max_val / height
            count_x = max(1, int(np.ceil(float(padding_distance) / step_x)))
            count_y = max(1, int(np.ceil(float(padding_distance) / step_y)))
            x_index = torch.arange(-count_x, width + count_x, device=device)
            y_index = torch.arange(-count_y, height + count_y, device=device)
            x = (x_index.to(dtype) + 0.5) * step_x - self.max_val * self.aspect
            y = (y_index.to(dtype) + 0.5) * step_y - self.max_val
            xx, yy = torch.meshgrid(x, y, indexing='ij')
            inner = ((x_index[:, None] >= 0) & (x_index[:, None] < width)
                     & (y_index[None, :] >= 0) & (y_index[None, :] < height))
            return torch.stack((xx, yy), -1)[~inner]
        if self.style in WARPED_CARTESIAN_STYLES:
            if not isinstance(self.resolution, (int, np.integer)):
                height, width = self.grid_shape
                step_x = 2 * self.max_val * self.aspect / width
                step_y = 2 * self.max_val / height
                count_x = max(1, int(np.ceil(float(padding_distance) / step_x)))
                count_y = max(1, int(np.ceil(float(padding_distance) / step_y)))
                x_index = torch.arange(-count_x, width + count_x, device=device)
                y_index = torch.arange(-count_y, height + count_y, device=device)
                x = (x_index.to(dtype) + 0.5) * step_x - self.max_val * self.aspect
                y = (y_index.to(dtype) + 0.5) * step_y - self.max_val
                xx, yy = torch.meshgrid(x, y, indexing='ij')
                plotting = torch.stack((xx, yy), -1)
                inner = ((x_index[:, None] >= 0) & (x_index[:, None] < width)
                         & (y_index[None, :] >= 0) & (y_index[None, :] < height))
                self._warped_cartesian_pad_plotting = plotting[~inner]
                return self.native_to_visual(self._warped_cartesian_pad_plotting)
            step = 2 * self.max_val / self.resolution
            centers = torch.linspace(
                -self.max_val + step / 2, self.max_val - step / 2,
                self.resolution, device=device, dtype=dtype)
            num_layers = max(
                1, int(np.ceil(float(padding_distance) / step)))
            offsets = torch.arange(
                -num_layers, self.resolution + num_layers,
                device=device, dtype=dtype)
            padded_axis = centers[0] + offsets * step
            xx, yy = torch.meshgrid(
                padded_axis, padded_axis, indexing='ij')
            plotting_grid = torch.stack((xx, yy), dim=-1)
            inner = torch.logical_and(
                torch.logical_and(
                    offsets[:, None] >= 0,
                    offsets[:, None] < self.resolution),
                torch.logical_and(
                    offsets[None, :] >= 0,
                    offsets[None, :] < self.resolution))
            plotting_pad = plotting_grid[~inner]
            self._warped_cartesian_pad_plotting = plotting_pad
            pad_coords = self.native_to_visual(plotting_pad)
            return pad_coords.to(device=device, dtype=dtype)

        if 'logpolar' in self.style:
            if not isinstance(self.resolution, (int, np.integer)):
                height, width = self.grid_shape
                radial_step = 1.0 / (width - 1)
                num_layers = max(1, int(np.ceil(float(padding_distance) / radial_step)))
                padded_rho = torch.cat((
                    -radial_step * torch.arange(num_layers, 0, -1, device=device, dtype=dtype),
                    1 + radial_step * torch.arange(1, num_layers + 1, device=device, dtype=dtype)))
                angles = torch.arange(height, device=device, dtype=dtype) * (2 * torch.pi / height)
                boundary = _footprint_radius(angles, self.fov_type, self.aspect, self.max_val)
                radius = self.cmf_a * torch.expm1(
                    padded_rho[:, None] * torch.log1p(boundary[None, :] * (self.fov / 2) / self.cmf_a)
                ) / (self.fov / 2)
                theta = angles[None, :].expand_as(radius)
                self._logpolar_pad_cortical = torch.stack((
                    padded_rho[:, None].expand_as(radius), theta / (2 * torch.pi)), -1).reshape(-1, 2)
                return torch.stack((radius * theta.cos(), radius * theta.sin()), -1).reshape(-1, 2)
            radial_step = 1.0 / (self.resolution - 1)
            num_layers = max(
                1, int(np.ceil(float(padding_distance) / radial_step)))
            inner_radii = -radial_step * torch.arange(
                num_layers, 0, -1, device=device, dtype=dtype)
            outer_radii = 1.0 + radial_step * torch.arange(
                1, num_layers + 1, device=device, dtype=dtype)
            padded_radii = torch.cat((inner_radii, outer_radii))
            angles = torch.arange(
                self.resolution, device=device, dtype=dtype)
            angles = angles / self.resolution
            rho, theta = torch.meshgrid(
                padded_radii, angles, indexing='ij')
            cortical_pad = torch.stack((rho, theta), dim=-1).reshape(-1, 2)
            self._logpolar_pad_cortical = cortical_pad

            max_radius = self.polar[:, 0].max() * (self.fov / 2.0)
            rho_max = torch.log(
                (max_radius + self.cmf_a) / self.cmf_a)
            visual_radius = (
                self.cmf_a * torch.expm1(cortical_pad[:, 0] * rho_max)
                / (self.fov / 2.0))
            visual_angle = cortical_pad[:, 1] * (2 * torch.pi)
            pad_coords = torch.stack((
                visual_radius * torch.cos(visual_angle),
                visual_radius * torch.sin(visual_angle)), dim=1)
            return pad_coords.to(device=device, dtype=dtype)

        pad_coords = []
        sorted_radii = torch.sort(torch.unique(self.polar[:,0])).values
        if sorted_radii.numel() > 1:
            radius_diff = sorted_radii[-1] - sorted_radii[-2]
        else:
            # A very small square fixed-count grid can retain only its four
            # equal-radius corners. Use the nominal radial interval so it can
            # still receive a surrounding padding ring.
            radius_diff = sorted_radii[-1] / max(self.grid_shape[0] - 1, 1)
        start_radius = sorted_radii[-1] + radius_diff
        padding_radii = torch.arange(
                start_radius, start_radius + padding_distance, radius_diff,
                device=self.polar.device, dtype=self.polar.dtype)
        if self.field_geometry == 'spherical':
            # Measure the margin from the real rim, not the first padding ring.
            # The extra ring can exceed the spherical embedding domain at
            # coarse resolutions. Keep one full neighbor ring when its spacing
            # exceeds the requested margin; domain validation still applies.
            padding_radii = padding_radii[
                (padding_radii <= sorted_radii[-1] + padding_distance)
                | (padding_radii == start_radius)]
        for radius in padding_radii:
            for angle in torch.arange(
                    0, 2*np.pi, radius_diff, device=self.polar.device,
                    dtype=self.polar.dtype):
                pad_coords.append(torch.stack((
                    radius * torch.cos(angle), radius * torch.sin(angle))))
        pad_coords = torch.stack(pad_coords, 0).to(device=device, dtype=dtype)
        return pad_coords

    def get_strided_coords(self, stride, auto_match_cart_resources=0, in_cart_res=None, force_less_than=True, max_val=1):
        """Return a strided version of the coordinates.
        
        Args:
            stride (int): Stride factor for downsampling coordinates.
            auto_match_cart_resources (int): Automatic matching parameter for cartesian resources.
            in_cart_res (int, optional): Input cartesian resolution. Required for 'fixn' style.
            force_less_than (bool, optional): Whether to force less than target resolution. Defaults to True.
            max_val (float, optional): Maximum value for coordinates. Defaults to 1.
            
        Returns:
            tuple: A tuple containing:
                - SamplingCoords: New SamplingCoords object with strided coordinates.
                - int: Number of output radii.
                - int: Corresponding output cartesian resolution.
        """
        out_cart_res = None
        if 'fixn' in self.style: 
            """
            fixn means to fix the number (n) of sampling coordinates exactly to that of the target cartesian grid
            this requires adding in some additional samples across radii, deviating a bit from local isotropy
            """
            assert in_cart_res is not None, 'in_cart_res must be provided if using fixn style'
            out_radii = (tuple(max(1, side // stride) for side in _resolution_shape(in_cart_res))
                         if not isinstance(in_cart_res, (int, np.integer)) else in_cart_res // stride)
            out_cart_res = out_radii
        elif auto_match_cart_resources > 0 and 'fixn' not in self.style and in_cart_res is not None:
            """
            this means to get as close as possible to resolution of the cartesian grid (same or less), while maintaining isotropic sampling
            """
            assert in_cart_res is not None, 'in_cart_res must be provided if auto_match_cart_resources is True'
            out_cart_res = (tuple(max(1, side // stride) for side in in_cart_res)
                            if not isinstance(in_cart_res, (int, np.integer)) else in_cart_res // stride)
            if isinstance(out_cart_res, tuple):
                out_radii = out_cart_res
            else:
                out_radii, num_coords = find_desired_res(
                    self.fov, self.cmf_a, out_cart_res**2, style=self.style,
                    device=self.device, force_less_than=force_less_than, quiet=True,
                    fov_type=self.fov_type, field_geometry=self.field_geometry,
                    radius_norm=self.radius_norm)
        else:
            """
            no matching to cartesian sampling resolution -- not recommended since it makes comparisons very difficult
            """
            out_radii = (tuple(max(1, side // stride) for side in self.grid_shape)
                         if not isinstance(self.resolution, (int, np.integer)) else self.resolution // stride)
        if min(_resolution_shape(out_radii)) < 1:
            raise ValueError(f'Out radii decreased to less than 1: {out_radii}')
        
        out_coords = SamplingCoords(
            self.fov, self.cmf_a, out_radii, self.device, self.style, self.dtype,
            max_val=max_val, isotropic_plotting_type=self.isotropic_plotting_type,
            fov_type=self.fov_type, field_geometry=self.field_geometry,
            radius_norm=self.radius_norm)

        return out_coords, out_radii, out_cart_res
    
    def to(self, device=None, dtype=None):
        """Move the SamplingCoords object to a different device and/or change data type.
        
        Args:
            device (str, optional): Target device. Defaults to None.
            dtype (torch.dtype, optional): Target data type. Defaults to None.
            
        Returns:
            SamplingCoords: SamplingCoords object on the specified device/dtype.
        """
        dtype = self.dtype if dtype is None else dtype
        device = self.device if device is None else device
        self.device = device
        self.dtype = dtype
        self.cartesian = self.cartesian.to(device=device, dtype=dtype)
        if self.cortical is not None:
            self.cortical = self.cortical.to(device=device, dtype=dtype)
        self.polar = self.polar.to(device=device, dtype=dtype)
        self.plotting = self.plotting.to(device=device, dtype=dtype)
        self.valid_mask = self.valid_mask.to(device=device)
        if hasattr(self, "cartesian_rowcol"):
            self.cartesian_rowcol = self.cartesian_rowcol.to(device=device, dtype=dtype)
        if hasattr(self, "cartesian_pad_coords"):
            self.cartesian_pad_coords = self.cartesian_pad_coords.to(device=device, dtype=dtype)
        if hasattr(self, "fov_padding_coords"):
            self.fov_padding_coords = self.fov_padding_coords.to(device=device, dtype=dtype)
        if hasattr(self, "cortical_pad_coords"):
            self.cortical_pad_coords = self.cortical_pad_coords.to(device=device, dtype=dtype)
        return self
    
    def get_scatter_sizes(self):
        """Calculate approximate size to be used for plotting in scatter plots.

        Scales linearly with eccentricity, as occurs in KNN-sampling of the sensor manifold. 
        However, this is not precisely tuned to the particular warping parameters.
        
        Returns:
            torch.Tensor: Receptive field sizes for each sampling point.
        """
        return self.polar[:, 0] * self.fov / self.grid_shape[0]
    
    def clone(self, fov=None, cmf_a=None, resolution=None, device=None, style=None,
              dtype=None, max_val=None, isotropic_plotting_type=None,
              fov_type=None, field_geometry=None, radius_norm=None):
        """Return a deep copy of the SamplingCoords object with optional parameter overrides.
        
        Args:
            fov (float, optional): Field-of-view in degrees. Defaults to current value.
            cmf_a (float, optional): A parameter from the CMF. Defaults to current value.
            resolution (int, optional): Resolution parameter. Defaults to current value.
            device (str, optional): Device to place coordinates on. Defaults to current value.
            style (str, optional): Sampling style. Defaults to current value.
            dtype (torch.dtype, optional): Data type. Defaults to current value.
            max_val (float, optional): Maximum coordinate value. Defaults to current value.
            
        Returns:
            SamplingCoords: A new SamplingCoords object with the specified parameters.
        """
        new_coords = SamplingCoords(
            self.fov if fov is None else fov, 
            self.cmf_a if cmf_a is None else cmf_a, 
            self.resolution if resolution is None else resolution, 
            self.device if device is None else device, 
            self.style if style is None else style,  
            self.dtype if dtype is None else dtype,
            self.max_val if max_val is None else max_val,
            self.isotropic_plotting_type if isotropic_plotting_type is None else isotropic_plotting_type,
            self.fov_type if fov_type is None else fov_type,
            self.field_geometry if field_geometry is None else field_geometry,
            self.radius_norm if radius_norm is None else radius_norm,
            )
        return new_coords

    def __len__(self):
        """Return the number of sampling coordinates."""
        return self.cartesian.shape[0]

    def __repr__(self):
        """String representation of the SamplingCoords object."""
        return (f'SamplingCoords(length={len(self)}, fov={self.fov}, cmf_a={self.cmf_a}, '
                f'resolution={self.resolution}, style={self.style}, '
                f'fov_type={self.fov_type!r}, radius_norm={self.radius_norm!r})')


def _get_sampling_plotting_coords(
        coords, polar_coords, fov, cmf_a, plotting_type):
    """Map visual-field coordinates into the requested plotting layout."""
    assert plotting_type in ('v1like', 'schwartz', 'warp'), \
        f"plotting_type must be one of 'v1like', 'schwartz', 'warp'; got {plotting_type!r}"

    if plotting_type == 'warp':
        r_norm = polar_coords[:, 0]
        theta = polar_coords[:, 1]
        r_deg = r_norm * (fov / 2.0)
        rho_max = np.log((fov / 2.0 + cmf_a) / cmf_a)
        rho = torch.log((r_deg + cmf_a) / cmf_a) / rho_max
        return torch.stack(
            [rho * torch.cos(theta), rho * torch.sin(theta)], dim=1)

    hemi_inds = coords[:, 0] < 0
    fov_coords = coords * (fov / 2)
    plotting_coords = torch.log(
        torch.abs(fov_coords[:, 0]) + 1j * fov_coords[:, 1] + cmf_a)
    if plotting_type == 'v1like':
        plotting_coords = torch.stack(
            [plotting_coords.real, -plotting_coords.imag], 1)
    else:
        plotting_coords = torch.stack(
            [plotting_coords.real, plotting_coords.imag], 1)

    max_fov_rad = np.log(fov / 2 + cmf_a)
    min_fov_rad = np.log(cmf_a)
    std = torch.std(plotting_coords[:, 0]) * .5

    if plotting_type == 'v1like':
        plotting_coords[hemi_inds, 0] = (
            std + max_fov_rad - plotting_coords[hemi_inds, 0])
        plotting_coords[~hemi_inds, 0] = (
            plotting_coords[~hemi_inds, 0] - (std + max_fov_rad))
    else:
        gap = std * 0.1
        plotting_coords[~hemi_inds, 0] = (
            plotting_coords[~hemi_inds, 0] - min_fov_rad + gap)
        plotting_coords[hemi_inds, 0] = -(
            plotting_coords[hemi_inds, 0] - min_fov_rad) - gap

    return plotting_coords


@add_to_all(__all__)
def get_isotropic_sampling_coords(
        fov, cmf_a, res, fov_type='circular', device='cpu',
        constant_num_angles=False, force_n_points=None, max_norm_rad=1,
        plotting_type='v1like', filter_to_fov=True,
        return_masked_coords=False, field_geometry='planar'):
    """Sample coordinates isotropically with the cortical magnification function of the complex log mapping w=log(z+a), where z=x+iy.

    Args:
        fov (float): Field of view diameter in degrees.
        cmf_a (float): A parameter in the CMF controlling foveation; smaller = stronger foveation.
        res (int): Number of sampling points.
        fov_type (str, optional): ``'circular'`` or ``'square'``. Defaults to
            ``'circular'``.
        device (str, optional): Device to run the computation on. Defaults to 'cpu'.
        constant_num_angles (bool, optional): If True, the number of angles is constant for all radii, implementing log polar image sampling. Defaults to False.
        force_n_points (int, optional): If not None, forces the number of points to exactly this value. Useful for controlled comparisons with other sensors. Defaults to None.
        max_norm_rad (float, optional): Maximum normalized radius. Defaults to 1.
        plotting_type (str, optional): Layout style for plotting coords. One of:
            - 'v1like': V1-like flat complex-log with inverted hemifields facing outwards.
            - 'schwartz': upright complex-log (Schwartz's log(z+a) model) with hemifields facing inwards.
            - 'warp': cortical polar disc (log-polar radial warp in a disc, fovea at centre);
              matches fovi-rtx samples_uv_cortical_disc layout.
        return_masked_coords (bool, optional): Also return radial-manifold
            coordinates removed by the square FoV. These can be used
            as spatial padding candidates by KNN layers. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - torch.Tensor: Sampling cartesian coordinates in visual space, normalized to (-1,1).
            - torch.Tensor: Sampling polar coordinates in visual space.
            - torch.Tensor: Plotting coordinates in complex log space, useful for visualizing the sampling.
            - torch.Tensor: Masked spatial-padding coordinates, when
              ``return_masked_coords`` is True.
    """

    _validate_fov_type(fov_type, style='isotropic')
    if not isinstance(res, (int, np.integer)):
        height, width = _resolution_shape(res)
        if force_n_points is not None and force_n_points != height * width:
            raise ValueError("Rectangular fixed-count sampling requires height * width samples")
        if constant_num_angles:
            coords, polar, plotting = _rectangular_logpolar_sampling_coords(
                fov, cmf_a, res, device, max_norm_rad, fov_type)
            masked = coords.new_empty((0, 2))
        else:
            coords, polar, plotting, masked = _rectangular_isotropic_sampling_coords(
                fov, cmf_a, res, device, max_norm_rad, fov_type,
                plotting_type, field_geometry, force_n_points is not None)
        return (coords, polar, plotting, masked) if return_masked_coords else (coords, polar, plotting)

    if force_n_points is not None:
        res, _ = find_desired_res(
            fov, cmf_a, force_n_points, style='isotropic', device=device,
            quiet=True, fov_type=fov_type, field_geometry=field_geometry)

    # compute log-sampled radii, and angles for each radius
    radius, n_angles = _compute_isotropic_r_and_num_theta(
        fov, cmf_a, res, fov_type=fov_type, field_geometry=field_geometry, device=device)
    if constant_num_angles:
        # overwrite isotropic angle sampling to use standard log polar image sampling
        n_angles = torch.tensor([res]*res, device=device)

    radius = radius * max_norm_rad

    # if force_n_points is not None, we need to adjust the number of angles for the last radius
    if force_n_points is not None and fov_type == 'circular':
        # fix # of points by removing angles evenly across radii
        diff = n_angles.sum() - force_n_points
        # insert/remove new radii
        if diff > 0:
            add = -1
            action = 'removing'
            verb = 'from'
        else:
            add = 1
            action = 'adding'
            verb = 'to'
        if diff != 0:
            rad_idx = []
            print(np.abs(diff))
            while len(rad_idx) < np.abs(diff):
                this_diff = np.minimum(np.abs(diff) - len(rad_idx), len(radius)-1).item()
                print(f'{action} {this_diff} angles {verb} {len(radius)} radii')
                # set some heuristic choices for which radii to presever
                rad_min = int(0.2*len(radius))
                rad_idx = np.concatenate([rad_idx, np.random.choice(np.arange(rad_min, len(radius)), size=np.minimum(this_diff, len(radius)-rad_min), replace=False)])
            rad_idx = rad_idx.astype(int)
            for ii in range(np.abs(diff)):
                # make sure we don't remove the only angle from a radius
                this_idx = rad_idx[ii]
                while n_angles[this_idx] <= 1 and action == 'removing':
                    this_idx = this_idx + 1
                n_angles[this_idx] = n_angles[this_idx] + add
            assert n_angles.min() > 0, 'some radii have no angles'
                
    # compute angles and store coordinates
    coords = []
    polar_coords = []
    for ii, radius_i in enumerate(radius):
        n_angles_i = int(n_angles[ii].item())
        angles = torch.arange(
            n_angles_i, device=device, dtype=radius.dtype)
        angles = angles * (2 * torch.pi / n_angles_i)
        for angle in angles:
            polar_coords.append(torch.stack([radius_i, angle]))
            coords.append(torch.stack([
                radius_i * torch.cos(angle),
                radius_i * torch.sin(angle)]))

    coords = torch.stack(coords)
    polar_coords = torch.stack(polar_coords)

    masked_coords = coords.new_empty((0, 2))
    if fov_type == 'square' and filter_to_fov and int(res) > 1:
        coords, polar_coords = _append_missing_square_corners(
            coords, polar_coords, max_norm_rad)
        valid_mask = _fov_valid_mask(
            coords, fov_type='square', max_val=max_norm_rad)
        masked_coords = coords[~valid_mask]
        coords = coords[valid_mask]
        polar_coords = polar_coords[valid_mask]

    # For square fixed-count sampling, matching and selection must happen after
    # removing radial-manifold points outside the square FoV.
    if force_n_points is not None and fov_type == 'square':
        if coords.shape[0] < force_n_points:
            raise RuntimeError(
                f"square FoV produced {coords.shape[0]} samples, fewer than the "
                f"requested {force_n_points}")
        if coords.shape[0] > force_n_points:
            square_half_extent = max_norm_rad
            corner_mask = torch.all(torch.isclose(
                torch.abs(coords),
                torch.full_like(coords, square_half_extent),
                rtol=1e-5, atol=1e-6), dim=1)
            corner_inds = torch.where(corner_mask)[0]
            other_inds = torch.where(~corner_mask)[0]
            n_other = force_n_points - corner_inds.numel()
            selected = torch.linspace(
                0, other_inds.numel() - 1, n_other,
                device=coords.device).round().long()
            keep = torch.cat((other_inds[selected], corner_inds)).sort().values
            coords = coords[keep]
            polar_coords = polar_coords[keep]

    plotting_coords = _get_sampling_plotting_coords(
        coords, polar_coords, fov, cmf_a, plotting_type)

    result = (coords, polar_coords, plotting_coords)
    if return_masked_coords:
        return (*result, masked_coords)
    return result

@add_to_all(__all__)
def get_logpolar_image_sampling_coords(
        fov, cmf_a, res, device='cpu', force_n_points=None,
        max_norm_rad=1, fov_type='circular', field_geometry='planar'):
    """Convenience wrapper for log polar image sampling.
    
    Sample coordinates with the cortical magnification function of the complex log mapping w=log(z+a), where z=x+iy.
    This is not isotropic, rather, it produces a square log polar image using
    an equal number of angular samples for all radii. Flattened coordinates are
    emitted in row-major image order with angle as height and eccentricity as
    width.
    
    Args:
        fov (float): Field of view diameter in degrees.
        cmf_a (float): A parameter from the CMF: M(r)=1/(r+a). Smaller a = stronger foveation.
        res (int): Number of sampling points.
        device (str, optional): Device to run the computation on. Defaults to 'cpu'.
        force_n_points (int, optional): If not None, the number of points is forced to be exactly this, useful for controlled comparisons with other sensors. Defaults to None.
        max_norm_rad (float, optional): Maximum normalized radius. Defaults to 1.

    Returns:
        tuple: A tuple containing:
            - torch.Tensor: Sampling cartesian coordinates in visual space, normalized to (-1,1).
            - torch.Tensor: Sampling polar coordinates in visual space.
            - torch.Tensor: Plotting coordinates in complex log space, useful for visualizing the sampling.
    """
    _validate_fov_type(fov_type, style='logpolar')
    if not isinstance(res, (int, np.integer)):
        if force_n_points is not None:
            raise ValueError("Rectangular log-polar grids use their full height * width sample count")
        return _rectangular_logpolar_sampling_coords(
            fov, cmf_a, res, device, max_norm_rad, fov_type)
    if force_n_points is not None:
        return get_isotropic_sampling_coords(
            fov, cmf_a, res, fov_type=fov_type, field_geometry=field_geometry, device=device,
            constant_num_angles=True, force_n_points=force_n_points,
            max_norm_rad=max_norm_rad, filter_to_fov=False)

    radius, _ = _compute_isotropic_r_and_num_theta(
        fov, cmf_a, res, fov_type=fov_type, field_geometry=field_geometry, device=device)
    radius = radius * max_norm_rad
    angles = torch.arange(res, device=device, dtype=radius.dtype)
    angles = angles * (2 * torch.pi / res)
    angle_grid, radius_grid = torch.meshgrid(
        angles, radius, indexing='ij')
    polar_coords = torch.stack(
        (radius_grid, angle_grid), dim=-1).reshape(-1, 2)
    radius_flat, angle_flat = polar_coords.unbind(dim=1)
    coords = torch.stack((
        radius_flat * torch.cos(angle_flat),
        radius_flat * torch.sin(angle_flat)), dim=1)
    plotting_coords = _get_sampling_plotting_coords(
        coords, polar_coords, fov, cmf_a, 'v1like')
    return coords, polar_coords, plotting_coords


@add_to_all(__all__)
def get_warped_cartesian_sampling_coords(
        fov, cmf_a, res, device='cpu', max_val=1,
        fov_type='circular', return_valid_mask=False, radius_norm=2.0):
    """Build a regular Cartesian lattice in the radial-CMF warp plane.

    Each output location is inverse-mapped analytically into normalized visual
    coordinates. Circular FoVs retain the mapped circle, square FoVs retain
    the square circumscribing that circle, and Wang FoVs map the full native
    square without masking.

    ``radius_norm=inf`` measures native radius in the infinity norm, so
    iso-eccentricity shells are squares rather than circles and the native
    square maps exactly onto the visual square, leaving every cell valid.
    """
    _validate_fov_type(fov_type, style='warped_cartesian')
    _validate_radius_norm(radius_norm, fov_type=fov_type)
    if not isinstance(res, (int, np.integer)):
        height, width = _resolution_shape(res)
        aspect = width / height
        x = (torch.arange(width, device=device) + 0.5) * (2 * max_val * aspect / width) - max_val * aspect
        y = (torch.arange(height, device=device) + 0.5) * (2 * max_val / height) - max_val
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        plotting_coords = torch.stack((xx, yy), -1).reshape(-1, 2)
        coords = _rectangular_native_to_visual(
            plotting_coords, fov, cmf_a, fov_type, aspect, max_val, radius_norm)
        polar_coords = torch.stack((
            torch.linalg.vector_norm(coords, dim=1),
            torch.atan2(coords[:, 1], coords[:, 0])), dim=1)
        valid_mask = _fov_valid_mask(coords, fov_type, max_val, aspect)
        result = (coords, polar_coords, plotting_coords)
        return (*result, valid_mask) if return_valid_mask else result
    step = max_val / res
    axis = torch.linspace(
        -max_val + step, max_val - step, res, device=device)
    xx, yy = torch.meshgrid(axis, axis, indexing='ij')
    plotting_coords = torch.stack((xx, yy), dim=-1).reshape(-1, 2)
    normalizer, rho_axis = _warp_law(
        fov, cmf_a, radius_norm, fov_type, max_val)
    coords, polar_coords = _inverse_warped_cartesian(
        plotting_coords, fov, cmf_a, radius_normalizer=normalizer,
        rho_axis=rho_axis, radius_norm=radius_norm)
    valid_mask = _fov_valid_mask(coords, fov_type, max_val=max_val)
    result = (coords, polar_coords, plotting_coords)
    if return_valid_mask:
        return (*result, valid_mask)
    return result


def _compute_isotropic_r_and_num_theta(
        fov, cmf_a, res, fov_type='circular', device='cpu', field_geometry='planar'):
    """Compute the radii and angles for isotropic logarithmic sampling.
    
    Args:
        fov (float): Field of view diameter in degrees.
        cmf_a (float): A parameter from the CMF: M(r)=1/(r+a). Smaller a = stronger foveation.
        res (int): Number of sampling radii.
        fov_type (str, optional): ``'circular'`` or ``'square'``.
        device (str, optional): Device to run computation on. Defaults to 'cpu'.
        
    Returns:
        tuple: A tuple containing:
            - torch.Tensor: Radii for each sampling ring.
            - torch.Tensor: Number of angles for each radius.
    """
    _validate_fov_type(fov_type, style='isotropic')
    res = int(res)
    if res == 1:
        return (torch.zeros(1, device=device),
                torch.ones(1, device=device, dtype=torch.long))
    if fov is not None:
        # The circular FoV ends at the nominal radius. The outer square is
        # constructed from a radial manifold reaching its corners.
        r_max = fov / 2
        if fov_type == 'square':
            r_max *= np.sqrt(2.0)
    else:
        r_max = None

    # sample evenly in cortical radius (w=log(r + cmf_a)) and solve for visual radius (r = exp(w) - cmf_a)
    w_min = np.log(cmf_a)
    w_max = np.log(r_max + cmf_a)
    # add one extra point to cortical radius so that we can accurately compute the angle delta for the last visual radius
    w_delta = (w_max - w_min)/(res-1)

    w = torch.linspace(w_min, w_max+w_delta, steps=res+1, device=device) # even sampling in the cortical radius
    radius = torch.exp(w) - cmf_a # back-projection into visual radius

    # fulfill approximate isotropy: make the difference between neighboring angles equal to the difference in neighboring radii
    n_angles_init = 1
    n_angles = [n_angles_init]
    for ii in range(1,res):
        # average curr to prev and curr to next radius dists
        radius_diff = ((radius[ii] - radius[ii-1]) + (radius[ii+1] - radius[ii])) / 2 
        tangential_radius = radius[ii]
        if field_geometry == 'spherical':
            tangential_radius = torch.rad2deg(torch.sin(torch.deg2rad(radius[ii])))
        n_angles.append(len(torch.arange(0, 2 * torch.pi * tangential_radius, radius_diff, device=device)))
    n_angles = torch.tensor(n_angles)  

    # remove extra radius
    radius = radius[:-1]

    # normalize to [0,1]
    radius = radius / (fov/2)

    return radius, n_angles
    

@add_to_all(__all__)
def isotropic_foveal_ring(fov: float, cmf_a: float, res: int,
                         fov_type: str = 'circular', field_geometry: str = 'planar') -> torch.Tensor:
    """Return the first noncentral ring as normalized (N, 2) visual coordinates."""
    if res < 2:
        raise ValueError("Foveal spacing requires at least two sampling rings")
    radii, counts = _compute_isotropic_r_and_num_theta(
        fov, cmf_a, res, fov_type=fov_type, field_geometry=field_geometry)
    angles = torch.arange(int(counts[1])) * (2 * torch.pi / counts[1])
    return radii[1] * torch.stack((angles.cos(), angles.sin()), dim=-1)


@add_to_all(__all__)
def num_sampling_coords_isotropic(
        fov, cmf_a, res, fov_type='circular', device='cpu', field_geometry='planar'):
    """Quickly compute the number of sampling coordinates for isotropic sampling.

    Useful for optimizing the res (# of radii) to match a certain output n (# of points).
    
    Args:
        fov (float): Field of view diameter in degrees.
        cmf_a (float): A parameter from the CMF: M(r)=1/(r+a). Smaller a = stronger foveation.
        res (int): Number of sampling radii.
        fov_type (str, optional): ``'circular'`` or ``'square'``.
        device (str, optional): Device to run computation on. Defaults to 'cpu'.
        
    Returns:
        int: Total number of sampling coordinates.
    """
    _validate_fov_type(fov_type, style='isotropic')
    if not isinstance(res, (int, np.integer)):
        return len(get_isotropic_sampling_coords(
            fov, cmf_a, res, fov_type=fov_type, device=device,
            field_geometry=field_geometry)[0])
    if int(res) == 1:
        return 1
    radius, n_angles = _compute_isotropic_r_and_num_theta(
        fov, cmf_a, res, fov_type=fov_type, field_geometry=field_geometry, device=device)
    if fov_type == 'circular':
        return n_angles.sum().item()

    count = 0
    for radius_i, n_angles_i in zip(radius, n_angles):
        angles = torch.arange(n_angles_i, device=device) * (2 * torch.pi / n_angles_i)
        ring = torch.stack((
            radius_i * torch.cos(angles),
            radius_i * torch.sin(angles)), dim=1)
        count += int(_fov_valid_mask(ring, fov_type='square').sum().item())
    outer_n = n_angles[-1]
    outer_angles = torch.arange(
        outer_n, device=device) * (2 * torch.pi / outer_n)
    outer_ring = torch.stack((
        radius[-1] * torch.cos(outer_angles),
        radius[-1] * torch.sin(outer_angles)), dim=1)
    square_half_extent = 1.0
    corners = torch.tensor(
        [[-square_half_extent, -square_half_extent],
         [-square_half_extent, square_half_extent],
         [square_half_extent, -square_half_extent],
         [square_half_extent, square_half_extent]],
        device=device, dtype=outer_ring.dtype)
    for corner in corners:
        if not torch.any(torch.all(torch.isclose(
                outer_ring, corner[None], rtol=1e-5, atol=1e-6), dim=1)):
            count += 1
    return count


@add_to_all(__all__)
def find_desired_res(
        fov, cmf_a, n_points_desired, style, device='cpu',
        bounds=(1,1000), force_less_than=False, quiet=False,
        fov_type='circular', field_geometry='planar', radius_norm=2.0):
    """Find the resolution that gives the desired number of sampling points using binary search.
    
    Args:
        fov (float): Field of view diameter in degrees.
        cmf_a (float): A parameter from the CMF: M(r)=1/(r+a). Smaller a = stronger foveation.
        n_points_desired (int): Desired number of sampling points.
        style (str): Which sampling style, e.g. 'isotropic'.
        device (str, optional): Device to run computation on. Defaults to 'cpu'.
        bounds (tuple, optional): Bounds for resolution search. Defaults to (1,1000).
        force_less_than (bool, optional): Whether to force less than target resolution. Defaults to False.
        quiet (bool, optional): Whether to suppress output. Defaults to False.
        
    Returns:
        tuple: A tuple containing:
            - int: Resolution that gives the desired number of points.
            - int: Actual number of points achieved.
    """
    validate_field_geometry(field_geometry)
    _validate_fov_type(fov_type, style=style)
    # Try a range of integer values directly instead of using minimize_scalar
    best_res = None
    best_diff = float('inf')
    
    # Binary search through the range since the function is monotonic
    left, right = bounds
    while left <= right:
        mid = (left + right) // 2
        n = num_sampling_coords(
            fov, cmf_a, mid, style=style, device=device,
            fov_type=fov_type, field_geometry=field_geometry,
            radius_norm=radius_norm)
        diff = abs(n - n_points_desired)
        
        if diff < best_diff:
            best_diff = diff
            best_res = mid
            
        if n < n_points_desired:
            left = mid + 1
        else:
            right = mid - 1
    
    n = num_sampling_coords(
        fov, cmf_a, best_res, style=style, device=device,
        fov_type=fov_type, field_geometry=field_geometry,
        radius_norm=radius_norm)

    if force_less_than:
        while n > n_points_desired:
            best_res = best_res - 1
            n = num_sampling_coords(
                fov, cmf_a, best_res, style=style, device=device,
                fov_type=fov_type, field_geometry=field_geometry,
                radius_norm=radius_norm)
    else:
        while n < n_points_desired:
            # make sure we overshoot slightly so that we can remove angles rather than adding them
            best_res = best_res + 1
            n = num_sampling_coords(
                fov, cmf_a, best_res, style=style, device=device,
                fov_type=fov_type, field_geometry=field_geometry,
                radius_norm=radius_norm)

    if not quiet:
        print(f'found resolution {best_res} giving {n} points (desired: {n_points_desired})')
        
    return best_res, n


def _rectangular_isotropic_sampling_coords(
        fov, cmf_a, res, device, max_val, fov_type,
        plotting_type, field_geometry, fixed_count):
    """Generate angularly isotropic rings, then retain the requested footprint."""
    height, width = _resolution_shape(res)
    aspect = width / height
    target = height * width
    outer_radius = max_val * (np.hypot(aspect, 1) if fov_type == 'square' else max(aspect, 1))

    def generate(rings):
        cartesian, _, _ = get_isotropic_sampling_coords(
            fov * outer_radius, cmf_a, rings, device=device,
            fov_type='circular', field_geometry=field_geometry)
        cartesian = cartesian * outer_radius
        valid = _fov_valid_mask(cartesian, fov_type, max_val, aspect)
        return cartesian[valid], cartesian[~valid]

    low, high = 1, max(2, int(np.sqrt(target)))
    while len(generate(high)[0]) < target:
        high *= 2
    while low < high:
        mid = (low + high) // 2
        if len(generate(mid)[0]) < target:
            low = mid + 1
        else:
            high = mid
    rings = low if fixed_count else max(1, low - 1)
    cartesian, masked = generate(rings)
    if fov_type == 'square':
        if target < 4:
            raise ValueError("A rectangular square footprint requires at least four samples")
        corners = cartesian.new_tensor([
            [-aspect * max_val, -max_val], [-aspect * max_val, max_val],
            [aspect * max_val, -max_val], [aspect * max_val, max_val]])
        keep_count = target - len(corners)
        if fixed_count and len(cartesian) < keep_count:
            raise RuntimeError("Rectangular isotropic manifold has too few samples")
        if len(cartesian) > keep_count and (fixed_count or len(cartesian) + len(corners) > target):
            keep = torch.linspace(0, len(cartesian) - 1, keep_count, device=device).round().long()
            cartesian = cartesian[keep]
        cartesian = torch.cat((cartesian, corners), dim=0)
    elif fixed_count:
        if len(cartesian) < target:
            raise RuntimeError("Rectangular isotropic manifold has too few samples")
        if len(cartesian) > target:
            keep = torch.linspace(0, len(cartesian) - 1, target, device=device).round().long()
            cartesian = cartesian[keep]
    polar = torch.stack((torch.linalg.vector_norm(cartesian, dim=1),
                         torch.atan2(cartesian[:, 1], cartesian[:, 0])), dim=1)
    plotting = _get_sampling_plotting_coords(cartesian, polar, fov, cmf_a, plotting_type)
    return cartesian, polar, plotting, masked


def _rectangular_logpolar_sampling_coords(fov, cmf_a, res, device, max_val, fov_type):
    """Use angle rows and radius columns with a boundary for each angle."""
    height, width = _resolution_shape(res)
    angles = torch.arange(height, device=device) * (2 * torch.pi / height)
    boundary = _footprint_radius(angles, fov_type, width / height, max_val)
    rho = torch.linspace(0, 1, width, device=device)
    radius = cmf_a * torch.expm1(
        rho[None, :] * torch.log1p(boundary[:, None] * (fov / 2) / cmf_a)) / (fov / 2)
    angle = angles[:, None].expand(-1, width)
    polar = torch.stack((radius, angle), -1).reshape(-1, 2)
    cartesian = torch.stack((radius * angle.cos(), radius * angle.sin()), -1).reshape(-1, 2)
    plotting = torch.stack((rho[None, :].expand(height, -1),
                            angle / (2 * torch.pi)), -1).reshape(-1, 2)
    return cartesian, polar, plotting


@add_to_all(__all__)
def get_sampling_coords(
        fov, cmf_a, res, device='cpu', style='isotropic', max_val=1,
        isotropic_plotting_type='v1like', fov_type='circular',
        return_valid_mask=False, return_masked_coords=False, field_geometry='planar',
        radius_norm=2.0):
    """Generate sampling coordinates based on the specified style.
    
    Args:
        fov (float): Field of view diameter in degrees.
        cmf_a (float): A parameter from the CMF: M(r)=1/(r+a). Smaller a = stronger foveation.
        res (int or tuple[int, int]): Scalar resolution or (height, width).
        device (str, optional): Device to run computation on. Defaults to 'cpu'.
        style (str): Sampling style, including ``isotropic``, ``logpolar``,
            ``warped_cartesian``, and their supported ``_as_grid`` variants.
        max_val (float, optional): Maximum x/y value. Defaults to 1.
        fov_type (str, optional): FoV geometry. ``'wang'`` is only valid with
            a warped-Cartesian sensor style.
        return_valid_mask (bool, optional): Append a validity mask to the
            returned tuple. Defaults to False.
        return_masked_coords (bool, optional): Append isotropic samples from
            the radial manifold that are excluded by the square FoV
            for use as KNN padding candidates. Grid styles retain such samples
            in-place and return an empty tensor. Defaults to False.
        
    Returns:
        tuple: A tuple containing:
            - torch.Tensor: Cartesian coordinates.
            - torch.Tensor: Polar coordinates.
            - torch.Tensor: Plotting coordinates.
            - torch.Tensor: Validity mask, when requested.
            - torch.Tensor: Masked spatial-padding coordinates, when requested.
    """
    if style not in SAMPLING_STYLES:
        raise ValueError(
            f"style must be one of {SAMPLING_STYLES}, got {style!r}")
    validate_field_geometry(field_geometry)
    _validate_fov_type(fov_type, style=style)
    _validate_radius_norm(radius_norm, fov_type=fov_type, style=style)
    if radius_norm == math.inf and style in WARPED_CARTESIAN_STYLES:
        _validate_square_shell_geometry(
            fov, cmf_a, res, max_val, field_geometry)
    if not isinstance(res, (int, np.integer)):
        height, width = _resolution_shape(res)
        aspect = width / height
        if style in ('uniform', 'uniform_as_grid'):
            x = (torch.arange(width, device=device) + 0.5) * (2 * max_val * aspect / width) - max_val * aspect
            y = (torch.arange(height, device=device) + 0.5) * (2 * max_val / height) - max_val
            xx, yy = torch.meshgrid(x, y, indexing='ij')
            coords = torch.stack((xx, yy), -1).reshape(-1, 2)
            polar_coords = torch.stack((torch.linalg.vector_norm(coords, dim=1),
                                        torch.atan2(coords[:, 1], coords[:, 0])), dim=1)
            plotting_coords = coords.clone()
            valid_mask = _fov_valid_mask(coords, fov_type, max_val, aspect)
            masked_coords = coords.new_empty((0, 2))
        elif style in WARPED_CARTESIAN_STYLES:
            coords, polar_coords, plotting_coords, valid_mask = get_warped_cartesian_sampling_coords(
                fov, cmf_a, res, device=device, max_val=max_val,
                fov_type=fov_type, return_valid_mask=True, radius_norm=radius_norm)
            masked_coords = coords.new_empty((0, 2))
        elif style in ('logpolar', 'logpolar_as_grid'):
            coords, polar_coords, plotting_coords = _rectangular_logpolar_sampling_coords(
                fov, cmf_a, res, device, max_val, fov_type)
            valid_mask = torch.ones(len(coords), dtype=torch.bool, device=device)
            masked_coords = coords.new_empty((0, 2))
        elif style in ('isotropic', 'isotropic_fixn'):
            coords, polar_coords, plotting_coords, masked_coords = _rectangular_isotropic_sampling_coords(
                fov, cmf_a, res, device, max_val, fov_type,
                isotropic_plotting_type, field_geometry, style == 'isotropic_fixn')
            valid_mask = torch.ones(len(coords), dtype=torch.bool, device=device)
        else:
            raise ValueError(f"Unsupported sampling style {style!r}")
        result = (coords, polar_coords, plotting_coords)
        if return_valid_mask:
            result = (*result, valid_mask)
        if return_masked_coords:
            result = (*result, masked_coords)
        return result
    if style == 'uniform' or style == 'uniform_as_grid':
        step = max_val / res
        coords = torch.linspace(-max_val + step, max_val - step, res)
        coords = torch.stack(torch.meshgrid(coords, coords), dim=2).reshape(-1, 2).to(device)
        polar_coords = torch.stack([torch.sqrt(coords[:,0]**2 + coords[:,1]**2), torch.arctan2(coords[:,1], coords[:,0])], dim=1)
        plotting_coords = coords.clone()
        valid_mask = torch.ones(coords.shape[0], device=device, dtype=torch.bool)
        masked_coords = coords.new_empty((0, 2))
    elif 'warped_cartesian' in style:
        coords, polar_coords, plotting_coords, valid_mask = (
            get_warped_cartesian_sampling_coords(
                fov, cmf_a, res, device=device, max_val=max_val,
                fov_type=fov_type, return_valid_mask=True,
                radius_norm=radius_norm))
        # These styles retain masked cells in their native rectangular layout.
        # KNN layers use valid_mask to treat them as padding in-place.
        masked_coords = coords.new_empty((0, 2))
    elif 'isotropic' in style or 'logpolar' in style:
        if 'fixn' in style:
            force_n_points = res**2
        else:
            force_n_points = None
        if style == 'logpolar' or style == 'logpolar_as_grid':
            coords, polar_coords, plotting_coords = get_logpolar_image_sampling_coords(
                fov, cmf_a, res, device=device, force_n_points=None,
                max_norm_rad=max_val, fov_type=fov_type, field_geometry=field_geometry)
        else:
            (coords, polar_coords, plotting_coords,
             masked_coords) = get_isotropic_sampling_coords(
                fov, cmf_a, res, device=device,
                force_n_points=force_n_points, max_norm_rad=max_val,
                plotting_type=isotropic_plotting_type,
                fov_type=fov_type, field_geometry=field_geometry, return_masked_coords=True)
        if style == 'logpolar' or style == 'logpolar_as_grid':
            masked_coords = coords.new_empty((0, 2))
        valid_mask = _fov_valid_mask(
            coords, fov_type, max_val=max_val)
        if 'fixn' in style:
            assert coords.shape[0] == res**2
    else:
        raise NotImplementedError('')

    result = (coords, polar_coords, plotting_coords)
    if return_valid_mask:
        result = (*result, valid_mask)
    if return_masked_coords:
        result = (*result, masked_coords)
    return result


@add_to_all(__all__)
def rowcol_to_xy(coords, do_norm=True, format='01'):
    """Convert row-column coordinates to xy coordinates.
    
    Args:
        coords (torch.Tensor): Input coordinates in row-column format.
        do_norm (bool, optional): Whether to normalize coordinates. Defaults to True.
        format (str): Coordinate format ('01' for [0,1] or '-11' for [-1,1]). Defaults to '01'.
        
    Returns:
        torch.Tensor: Coordinates in xy format.
    """
    assert format in ['01', '-11']
    if format == '01':
        min = 0
        max = 1
    else:
        min = -1
        max = 1
    if do_norm:   
        coords = normalize(coords, min=min, max=max)
    row, col = coords[:,0], coords[:,1]
    x = col
    if format == '01':
        y = 1-row
    else:
        y = -row
    return torch.stack((x, y), dim=1)


@add_to_all(__all__)
def xy_to_rowcol(coords, do_norm=True, format='01'):
    """Convert xy coordinates to row-column coordinates.
    
    Args:
        coords (torch.Tensor): Input coordinates in xy format.
        do_norm (bool, optional): Whether to normalize coordinates. Defaults to True.
        format (str): Coordinate format ('01' for [0,1] or '-11' for [-1,1]). Defaults to '01'.
        
    Returns:
        torch.Tensor: Coordinates in row-column format.
    """
    assert format in ['01', '-11']
    if format == '01':
        min = 0
        max = 1
    else:
        min = -1
        max = 1
    if do_norm:
        coords = normalize(coords, min=min, max=max)
    x, y = coords[:,0], coords[:,1]
    col = x
    if format == '01':
        row = 1-y
    else:
        row = -y
    return torch.stack((row, col), dim=1)


@add_to_all(__all__)
def xy_to_colrow(coords, do_norm=True, format='01'):
    """Convert xy coordinates to column-row coordinates.
    
    Args:
        coords (torch.Tensor): Input coordinates in xy format.
        do_norm (bool, optional): Whether to normalize coordinates. Defaults to True.
        format (str): Coordinate format ('01' for [0,1] or '-11' for [-1,1]). Defaults to '01'.
        
    Returns:
        torch.Tensor: Coordinates in row-column format.
    """
    rowcol = xy_to_rowcol(coords, do_norm, format)
    return torch.stack((rowcol[:,1], rowcol[:,0]), dim=1)


@add_to_all(__all__)
def num_sampling_coords(
        fov, cmf_a, res, style='isotropic', device='cpu',
        fov_type='circular', field_geometry='planar', radius_norm=2.0):
    """Calculate the number of sampling coordinates for a given style.
    
    Args:
        fov (float): Field of view diameter in degrees.
        cmf_a (float): A parameter from the CMF: M(r)=1/(r+a). Smaller a = stronger foveation.
        res (int): Resolution parameter.
        style (str): Sampling style. Defaults to 'isotropic'.
        device (str, optional): Device to run computation on. Defaults to 'cpu'.
        
    Returns:
        int: Number of sampling coordinates.
    """
    validate_field_geometry(field_geometry)
    _validate_fov_type(fov_type, style=style)
    _validate_radius_norm(radius_norm, fov_type=fov_type, style=style)
    if not isinstance(res, (int, np.integer)):
        height, width = _resolution_shape(res)
        if style == 'isotropic':
            return len(get_sampling_coords(
                fov, cmf_a, res, style=style, device=device,
                fov_type=fov_type, field_geometry=field_geometry)[0])
        return height * width
    if style == 'isotropic':
        return num_sampling_coords_isotropic(
            fov, cmf_a, res, device=device, fov_type=fov_type, field_geometry=field_geometry)
    elif style in [
            'logpolar', 'logpolar_as_grid', 'warped_cartesian',
            'warped_cartesian_as_grid', 'isotropic_fixn', 'uniform',
            'uniform_as_grid']:
        return res**2
    else:
        raise ValueError(f'Style {style} not recognized')


@add_to_all(__all__)
def transform_sampling_grid(sampling_grid, fix_loc, fixation_size, image_size):
    """Transform sampling grid coordinates from fixation space to full image space.

    Args:
        sampling_grid (torch.Tensor): Sampling grid of shape (1, n_coords, 2) in [-1, 1] range. It is in (x,y) format.
        fix_loc (tuple or torch.Tensor): Fixation center in normalized image coordinates (h, w), e.g., (0.5, 0.5).
        fixation_size (tuple or torch.Tensor): Size of the fixation region in pixels (fix_h, fix_w).
        image_size (tuple): Full image size (H, W).

    Returns:
        torch.Tensor: Transformed sampling grid in the full image space.
    """
    if isinstance(fix_loc, tuple) or isinstance(fix_loc, np.ndarray):
        fix_loc = torch.tensor(fix_loc)
        if fix_loc.ndim == 1:
            fix_loc = fix_loc.unsqueeze(0)
    fix_loc = fix_loc.clone()
    if isinstance(fixation_size, tuple) or isinstance(fixation_size, np.ndarray):
        fixation_size = torch.tensor(fixation_size)
        if fixation_size.ndim == 1:
            fixation_size = fixation_size.unsqueeze(0)
    fixation_size = fixation_size.clone()
    if isinstance(sampling_grid, np.ndarray):
        sampling_grid = torch.tensor(sampling_grid)

    # Unpack inputs
    fix_center_h = fix_loc[:,0].reshape(-1,1,1).to(sampling_grid.device)
    fix_center_w = fix_loc[:,1].reshape(-1,1,1).to(sampling_grid.device)
    fix_h = fixation_size[:,0].reshape(-1,1,1).to(sampling_grid.device)
    fix_w = fixation_size[:,1].reshape(-1,1,1).to(sampling_grid.device)
    H, W = image_size
    batch_size = fix_center_h.shape[0]

    # Convert fixation center from normalized to pixel space
    fix_center_w *= W
    fix_center_h *= H

    # Scale grid from [-1, 1] to fixation size in pixels
    scaled_grid = sampling_grid.clone()
    if scaled_grid.shape[0] == 1:
        scaled_grid = scaled_grid.repeat(batch_size, 1, 1, 1)

    scaled_grid[:, :, :, 0] *= (fix_w / 2)  # Scale x-coordinates
    scaled_grid[:, :, :, 1] *= (fix_h / 2)  # Scale y-coordinates

    # Offset grid by fixation center
    scaled_grid[:, :, :, 0] += fix_center_w  # Shift x-coordinates
    scaled_grid[:, :, :, 1] += fix_center_h  # Shift y-coordinates

    scaled_grid[:, :, :, 0] = (2 * scaled_grid[:, :, :, 0] / W) - 1  # Normalize x-coordinates
    scaled_grid[:, :, :, 1] = (2 * scaled_grid[:, :, :, 1] / H) - 1  # Normalize y-coordinates

    return scaled_grid


@add_to_all(__all__)
def auto_match_num_coords(
        fov, cmf_a, cart_res, style, auto_match_cart_resources, device,
        force_less_than=True, quiet=False, fov_type='circular', field_geometry='planar',
        radius_norm=2.0):
    """Automatically match the number of coordinates to cartesian resolution.
    
    Args:
        fov (float): Field of view diameter in degrees.
        cmf_a (float): A parameter from the CMF: M(r)=1/(r+a). Smaller a = stronger foveation.
        cart_res (int): Cartesian resolution.
        style (str): Sampling style.
        auto_match_cart_resources (int): Auto-matching parameter (-1: auto-match in_res, 0: no auto-matching, >0: auto-match everything).
        device (str): Device to run computation on.
        force_less_than (bool, optional): Whether to force less than target resolution. Defaults to True.
        
    Returns:
        tuple: A tuple containing:
            - int: Input resolution.
            - int: Cartesian resolution.
    """
    if not isinstance(cart_res, (int, np.integer)):
        _resolution_shape(cart_res)
        return tuple(cart_res), tuple(cart_res)
    if 'fixn' in style:
        in_res = cart_res
    elif auto_match_cart_resources != 0 and 'fixn' not in style:
        # -1: auto-match in_res, 0: no auto-matching, >0: auto-match everything
        in_res, num_coords = find_desired_res(
            fov, cmf_a, cart_res**2, style, device=device,
            force_less_than=force_less_than, quiet=quiet,
            fov_type=fov_type, field_geometry=field_geometry,
            radius_norm=radius_norm)
    else:
        in_res = cart_res
    return in_res, cart_res


@add_to_all(__all__)
def logpolar_radius(cartesian, fov, cmf_a):
    """Utility for computing logpolar radius from normalized cartesian coordinates.
    
    Args:
        cartesian (torch.Tensor): (nx2) cartesian coordinates normalized to (-1,1).
        fov (float): Field-of-view diameter in degrees.
        cmf_a (float): A parameter from the CMF: M(r)=1/(r+a). Smaller a = stronger foveation.
        
    Returns:
        torch.Tensor: (nx1) log radius as in logpolar mapping.
    """
    radius = ((cartesian[:,0]**2 + cartesian[:,1]**2)**.5)*fov/2
    cmf_a_tensor = torch.tensor(cmf_a)
    log_radius = (torch.log(radius + cmf_a) - torch.log(cmf_a_tensor))/(torch.log(fov/2 + cmf_a_tensor) - torch.log(cmf_a_tensor))
    return log_radius


@add_to_all(__all__)
def cart_to_polar(cartesian):
    """Convert cartesian coordinates to polar coordinates.
    
    Args:
        cartesian (torch.Tensor or array-like): Input cartesian coordinates.
        
    Returns:
        torch.Tensor: Polar coordinates (radius, angle).
    """
    if not isinstance(cartesian, torch.Tensor):
        cartesian = torch.tensor(cartesian)
    polar_coords = torch.stack([
        torch.sqrt(cartesian[:,0]**2 + cartesian[:,1]**2), 
        torch.arctan2(cartesian[:,1], cartesian[:,0]),
        ], dim=1)
    return polar_coords


@add_to_all(__all__)
def polar_to_cart(polar):
    """Convert polar coordinates to cartesian coordinates.
    
    Args:
        polar (torch.Tensor or array-like): Input polar coordinates (radius, angle).
        
    Returns:
        torch.Tensor: Cartesian coordinates (x, y).
    """
    if not isinstance(polar, torch.Tensor):
        polar = torch.tensor(polar)
    cartesian_coords = torch.stack([
        polar[:,0] * torch.cos(polar[:,1]),
        polar[:,0] * torch.sin(polar[:,1])
    ], dim=1)
    return cartesian_coords


@add_to_all(__all__)
def cart_to_complex_log(cartesian, fov, cmf_a, postproc=True):
    """Convert cartesian coordinates to complex log space coordinates.
    
    Args:
        cartesian (torch.Tensor or array-like): Input cartesian coordinates.
        fov (float): Field of view diameter in degrees.
        cmf_a (float): A parameter from the CMF: M(r)=1/(r+a). Smaller a = stronger foveation.
        postproc (bool, optional): Whether to apply post-processing for hemisphere separation. Defaults to True.
        
    Returns:
        torch.Tensor: Complex log space coordinates.
    """
    if not isinstance(cartesian, torch.Tensor):
        cartesian = torch.tensor(cartesian)

    # compute hemisphere indices based on x coordinate sign
    hemi_inds = (cartesian[:,0] <= 0)

    # use log(z+a) model to compute plotting coordinates (i.e. cortical visualization) 
    fov_coords = cartesian*(fov/2)
    plotting_coords = torch.log(torch.abs(fov_coords[:,0]) + 1j*fov_coords[:,1] + cmf_a)
    plotting_coords = torch.stack([plotting_coords.real, -plotting_coords.imag],1)
    
    # make plotting coords separated nicely across hemifields/hemispheres
    if postproc:
        std = torch.std(plotting_coords[:,0])*.5
        add = std + torch.max(plotting_coords[:,0])
        sub = (std + torch.max(plotting_coords[hemi_inds == 0,0]))
    else:
        add = 0
        sub = 0

    if any(hemi_inds == 1):
        plotting_coords[hemi_inds == 1,0] =  add - plotting_coords[hemi_inds == 1,0]
    if any(hemi_inds == 0):
        plotting_coords[hemi_inds == 0,0] = plotting_coords[hemi_inds == 0,0] - sub

    return plotting_coords
