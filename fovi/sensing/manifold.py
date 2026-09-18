from scipy.integrate import cumulative_trapezoid
from scipy.optimize import root_scalar
import trimesh
import numpy as np
import torch
from scipy.interpolate import interp1d

from ..utils import normalize
from ..utils import add_to_all

__all__ = []


def validate_field_geometry(field_geometry: str) -> None:
    """Validate the visual-field metric independently of image projection."""
    if field_geometry not in ("planar", "spherical", "legacy"):
        raise ValueError(f"Unknown field_geometry {field_geometry!r}")


def spherical_radius_limit(cmf_a: float) -> float:
    """Maximum embeddable eccentricity in degrees, allowing for integration spacing."""
    if not np.isfinite(cmf_a) or cmf_a <= 0:
        raise ValueError("cmf_a must be finite and positive")
    return root_scalar(
        # Factor out the known zero at 180 degrees to avoid cancellation there.
        lambda r: (cmf_a + r) * np.cos(np.deg2rad(r / 2)) - (180 / np.pi) * np.sin(np.deg2rad(r / 2)),
        bracket=(90.0, 180.0), method="brentq",
    ).root - 0.0001

@add_to_all(__all__)
class CorticalSensorManifold():
    r"""
    3-D cortical sensor manifold based on Rovamu and Virsu (1984) (see also Motter (2009)).
    Relevant coordinate systems:

    * :math:`(x, y)` -> visual cartesian coordinates
    * :math:`(r, \theta)` -> visual polar coordinates
    * :math:`(\rho, z, \phi)` -> cortical cylindrical coordinates
    * :math:`(x_c, y_c, z)` -> cortical cartesian coordinates

    We use the magnification function: :math:`M(r)=\frac{k}{r+a}`, where:

    * :math:`k`: scaling factor that gives a good match to cortical mm. irrelevant for foveated sampling.
    * :math:`a`: critical parameter controlling magnification. smaller == stronger magnification / foveation

    Due to our choice of magnification function, this is essentially a 3d extension of the complex logarithmic map (Schwartz, 1980), where both preserve local isotropy unlike the Schwartz (1980) model, our 3D version also preserves global/meridional isotropy, since there is no warping due to flattening

    """
    def __init__(self, cmf_a, fov, k=10, field_geometry="planar"):
        r"""
        Args:
            cmf_a (float): a parameter in cortical magnification function (CMF), in degrees
            fov (float): visual field size in degrees
            k (float): scaling factor that gives a good match to cortical mm. irrelevant for foveated sampling.
        """
        self.cmf_a = cmf_a
        self.fov = fov
        self.k = k
        validate_field_geometry(field_geometry)
        self.field_geometry = field_geometry
        if not all(np.isfinite(v) and v > 0 for v in (cmf_a, fov, k)):
            raise ValueError("cmf_a, fov, and k must be finite and positive")
        if field_geometry == "spherical" and fov > 180:
            raise ValueError("Spherical field diameter must not exceed 180 degrees")

        # compute cortical radius (z) over a fine mesh of the visual field by integration of the CMF
        spacing = 0.0001
        self.max_radius = 2 * fov
        if field_geometry == "planar":
            self.max_radius = np.inf
            return
        if field_geometry == "spherical":
            self.max_radius = min(self.max_radius, spherical_radius_limit(cmf_a))
        mesh = np.arange(0, self.max_radius, spacing)
        if field_geometry == "legacy":
            # Preserve the original mesh endpoints and integration arithmetic:
            # tiny coordinate changes can change discrete checkpoint neighborhoods.
            self.max_radius = mesh[-1]
        else:
            mesh = np.append(mesh, self.max_radius)
        z_integrand_vec = self.z_integrand(mesh) 
        integral_vals = cumulative_trapezoid(z_integrand_vec, x=mesh, initial=0)  # Integral from 0 to each x_grid point
        self.z_integral_interp = interp1d(mesh, integral_vals, kind='linear')

    def m(self, r):
        r"""
        cortical magnification as a function of eccentricity r

        Args:
            r (float): visual radius
        Returns:
            float: cortical magnification in mm/deg, evaluated at r
        """
        return self.k/(self.cmf_a + r) # mm/deg

    def rho_3d(self, r):
        r"""
        compute cortical radius in 3d cylindrical coordinates: :math:`m(r)*\sin(r)`

        Counterintuitively from the equation, the units are mm rather than mm/rad:

        *   this is because there is a left over radian term from the derivation
        *   :math:`r=m(r)*\sin(r)*d\theta/d\phi`. While :math:`d\theta` and :math:`d\phi` cancel out, :math:`d\theta` is in radians while :math:`d\phi` is unitless. 
        *   :math:`d\phi` is unitless because it is part of an infinitesimal distance :math:`r*d\phi` along the cortical surface that is in units mm. r keeps the mm units and :math:`d\phi` is unitless. 
        *   We use :math:`m(r)` in mm/deg: thus :math:`m(r)*\sin(r)` -> mm*rad/deg, so we convert to mm by multiplying by 180 deg/pi rad.

        Args:
            r (float): visual radius
        Returns:
            float: cortical radius in mm

        """
        deg_per_rad = 180/np.pi
        if self.field_geometry == "planar":
            return self.m(r) * r
        return self.m(r)*np.sin(np.deg2rad(r))*deg_per_rad # mm

    def phi_3d(self, theta):
        r"""
        cortical phi in 3d cylindrical coordinates. :math:`\phi=\theta`
        
        Args:
            theta (float): Visual polar angle (in radians).

        Returns:
            float: Cortical cylindrical coordinate :math:`\phi` (in radians).
        """
        return theta

    def dm_dr(self, r):
        r"""
        derivative of cortical magnification function with respect to radius (:math:`dm/dr`)

        Args:
            r (float): Visual radius in degrees

        Returns:
            float: :math:`dm/dr` in mm/deg^2
        """
        return -self.k*((self.cmf_a + r)**(-2))

    def drho_dr(self, r):
        r"""
        derivative of cortical cylindrical radius with respect to visual field radius (:math:`d\rho/dr`)

        Args:
            r (float): Visual radius in degrees
        Returns:
            float: :math:`d\rho/dr` in mm/deg
        """
        deg_per_rad = 180/np.pi
        if self.field_geometry == "planar":
            return self.dm_dr(r) * r + self.m(r)
        return self.dm_dr(r)*np.sin(np.deg2rad(r))*deg_per_rad + self.m(r)*np.cos(np.deg2rad(r))

    def z_integrand(self, r):
        r"""
        integrand for cortical z in 3d cylindrical coordinates: :math:`(m(r)^2 - (d\rho/dr)^2)^{0.5}`
        
        Args:
            r (float): Visual radius in degrees
        Returns:
            float: :math:`(m(r)^2 - (d\rho/dr)^2)^{0.5}` in mm/deg
        """
        return ((self.m(r))**2 - (self.drho_dr(r))**2)**0.5

    def z_3d(self, r):
        r"""
        cortical z in 3d cylindrical coordinates computed with numerical integration via interpolation of fine mesh of precomputed values

        Args:
            r (float): Visual radius in degrees
        Returns:
            float: cortical z in mm
        """
        if np.any(np.asarray(r) < 0) or np.any(np.asarray(r) > self.max_radius):
            raise ValueError(f"Visual radius outside manifold domain [0, {self.max_radius}] degrees")
        if self.field_geometry == "planar":
            # Analytic integral avoids absolute mesh spacing breaking scale invariance.
            t = np.sqrt(np.maximum((np.asarray(r) / self.cmf_a + 1) ** 2 - 1, 0))
            return self.k * (np.arcsinh(t) - t / np.sqrt(1 + t * t))
        return self.z_integral_interp(r)

    def map_3d(self, x, y):
        r"""
        map cartesian :math:`(x,y)` coordinates to 3D cortical cylindrical coordinates :math:`(\rho,z,\phi)`

        Args:
            x (float): visual cartesian x coordinate
            y (float): visual cartesian y coordinate
        Returns:
            rho_z_phi (tuple[float, float, float]): 3D cortical cylindrical coordinates :math:`(\rho, z, \phi)`
        """
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        rho = self.rho_3d(r)
        phi = self.phi_3d(theta)
        z = self.z_3d(r)
        return rho, z, phi

    def map_to_xyz(self, rho_z_phi):
        r"""
        map 3d cortical cylindrical coordinates to cortical cartesian coordinates for plotting convenience

        Args:
            rho_z_phi (tuple): 3D cortical cylindrical coordinates
        Returns:
            result (tuple): A tuple containing:
                - x_c (float): cortical cartesian x coordinate
                - y_c (float): cortical cartesian y coordinate
                - z (float): cortical cartesian z coordinate
        """
        rho, z, phi = rho_z_phi
        x_c = rho*np.cos(phi)
        y_c = rho*np.sin(phi)
        return x_c, y_c, z

    def normalize_coords(self, coords):
        """
        normalize coordinates to lie between -1 and 1

        Args:
            coords (np.ndarray): un-normalized coordinates (n, d)
        Returns:
            np.ndarray: normalized coordinates (n, d) 
        """
        # normalize between -1 and 1
        return 2*normalize(coords) - 1

    def cort_cartesian_to_cort_cylindrical(self, x_y_z):
        r"""
        Map cortical cartesian coordinates to cortical cylindrical coordinates for reverse mapping

        Args:
            x_y_z (tuple): cortical cartesian coordinates :math:`(x_c, y_c, z)`
        Returns:
            result (tuple): A tuple containing:
                - rho (float): r in cortical cylindrical coordinates
                - z (float): z in cortical cylindrical coordinates
                - phi (float): :math:`\phi` in cortical cylindrical coordinates
        """
        x, y, z = x_y_z
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        return rho, z, phi

    def vis_cartesian_to_cort_cartesian(self, x_y):
        r"""
        Map visual cartesian coordinates to cortical cartesian coordinates

        Args:
            x_y (np.ndarray): visual cartesian coordinates :math:`(x,y)`
        Returns:
            np.ndarray: cortical cartesian coordinates :math:`(x_c, y_c, z)`
        """
        return np.array([(self.map_to_xyz(self.map_3d(x,y))) for x, y in x_y])

    def r_from_z(self, z):
        r"""
        Use a numerical root-finding method to determine eccentricity from the cortical z coordinate

        Args:
            z (float): cortical z coordinate in mm
        Returns:
            float: visual eccentricity (radius) in degrees
        """
        def func(r):
            return self.z_3d(r) - z
        result = root_scalar(func, bracket=[0, self.fov/2 + 1], method='brentq')
        return result.root

    def cylindrical_to_visual_polar(self, rho_z_phi):
        r"""
        reverse map from cortical cylindrical coordinates to visual polar coordinates

        Args:
            rho_z_phi (tuple): tuple :math:`(\rho, z, \phi)` of cortical cylindrical coordinates
        Returns:
            result (tuple): A tuple containing:
                - r (float): visual eccentricity (radius) in degrees
                - phi (float): visual polar angle in radians
        """
        rho, z, phi = rho_z_phi
        r = self.r_from_z(z)
        return r, phi

    def init_visual_mesh(self, rs=None):
        r"""

        initialize a grid of visual points to specify the bounds of the 3d mesh before sampling points on it

        Args:
            rs (np.ndarray, optional): array of r values to test.
        Returns:
            result (tuple): A tuple containing:
                - grid_pts_3d_xyz (np.ndarray): (n, 3) array of cortical cartesian coordinates
                - grid_pts_polar (np.ndarray): (n, 2) array of visual polar coordinates
        """
        grid_pts = []
        grid_pts_polar = []
        theta_range = [0, 2*np.pi]
        rs = np.geomspace(0.001,self.fov/2+1,20) if rs is None else rs
        # we go to self.fov/2+1 so that we can remove 1 deg worth of visual angle from closed surface mesh to open it back up later
        for r in np.geomspace(0.0001,self.fov/2+1,500):
            for theta in np.linspace(theta_range[0],theta_range[1],10):
                grid_pts.append( (r*np.cos(theta),r*np.sin(theta)))
                grid_pts_polar.append((r, theta))
        for r in rs:
            for theta in np.linspace(theta_range[0],theta_range[1],100):
                grid_pts.append( (r*np.cos(theta),r*np.sin(theta)))
                grid_pts_polar.append((r, theta))

        grid_pts = np.array(grid_pts)
        grid_pts_polar = np.array(grid_pts_polar)

        # map the grid to 3D
        grid_pts_3d = np.array([(self.map_3d(x, y)) for x, y in grid_pts])
        grid_pts_3d_xyz = np.array([self.map_to_xyz(rho_z_phi) for rho_z_phi in grid_pts_3d])

        return grid_pts_3d_xyz, grid_pts_polar

    def init_cortical_mesh(self, grid_pts_3d_xyz, num_coords=10000):
        r"""
        initialize a 3d mesh of approximately evenly spaced cortical points

        NOTE:

        * this is not evenly-spaced enough to be useful, but we keep it around for reference
        * instead, we define the mesh by sampling visual radii according to the CMF, and performing locally isotropic sampling of angles. 
        * this is equivalent to uniform sampling on the cortical sensor manifold
        * the sensor manifold is thus used primarily for receptive field sampling once the mesh has been defined

        Args:
            grid_pts_3d_xyz (np.ndarray): (n,3) array of cortical cartesian coordinates used as a starting point to determine bounds
            num_coords (int): number of coordinates to generate on the surface
        Returns:
            np.ndarray: (n,3) array of cortical cartesian coordinates that are approximately evenly spaced
        
        """
        # Convert grid_pts_3d_uniform to a mesh 
        points = grid_pts_3d_xyz[~np.isnan(grid_pts_3d_xyz).any(axis=1)]  
        mesh = trimesh.Trimesh(vertices=points, process=False)
        
        # Extract the outer surface of the mesh
        mesh = mesh.convex_hull  

        max_z = self.z_3d(self.fov / 2 - 1)  # Z threshold for 1 degree of visual angle

        # sample the surface evenly (even sampling in cortical space)
        num_coords = 0
        mult = 1.2
        while num_coords < num_coords:
            uniform_points_3d = trimesh.sample.sample_surface_even(mesh, count=int(mult*num_coords), seed=0)[0]

            # Remove points corresponding to 1 degree worth of visual angle
            uniform_points_3d = uniform_points_3d[uniform_points_3d[:,2] < max_z]

            num_coords = uniform_points_3d.shape[0]
            mult *= 1.1

        # sort by z
        uniform_points_3d = uniform_points_3d[np.argsort(uniform_points_3d[:,2])]

        # # sample num_coords point randomly to get correct number
        np.random.seed(0) # ensure reproducibility
        uniform_points_3d = uniform_points_3d[np.random.choice(len(uniform_points_3d), num_coords, replace=False)]

        return uniform_points_3d    
    
    def reverse_map(self, mesh_points_xyz):
        r"""
        reverse map from cortical cartesian coordinates to visual cartesian and polar coordinates

        Args:
            mesh_points_xyz (np.ndarray): (n,3) cortical cartesian mesh pts
        Returns:
            result (tuple): A tuple containing:
                - cartesian_visual (np.ndarray): (n, 2) visual cartesian mesh pts
                - polar_visual (np.ndarray): (n, 2) visual polar mesh pts
        """
        # get points in cylindrical coordinates to do reverse mapping
        mesh_points_cylindrical = np.array([self.cort_cartesian_to_cort_cylindrical(xyz) for xyz in mesh_points_xyz])
        # reverse map into visual space using polar/spherical coordinat3es
        polar_visual = np.array([self.cylindrical_to_visual_polar(rho_z_phi) for rho_z_phi in mesh_points_cylindrical])
        # convert to cartesian coordinates for visual space
        cartesian_visual = np.stack([np.cos(polar_visual[:,1])*polar_visual[:,0], np.sin(polar_visual[:,1])*polar_visual[:,0]], 1)

        return cartesian_visual, polar_visual


@add_to_all(__all__)
def vis_cartesian_to_cortical_cartesian_coords(cartesian_coords, cmf_a, fov, as_tensor=False, device='cpu', k=10, field_geometry='planar'):
    r"""

    * Map visual cartesian coordinates to 3d cortical cartesian coordinates using the 3D cortical model. 
    * We use this when coordinates are sampled elsewhere, and we want to map them into the 3d model for receptive field sampling.
    * This is the main function we use to implement the cortical sensor manifold, in combination with isotropically magnified visual field sampling 
    
    Note, the CMF is :math:`M(r)=\frac{k}{r+a}`

    Args:
        cartesian_coords (np.ndarray): (n,2) visual cartesian coordinates in (x,y) format
        cmf_a (float): :math:`a` value in the CMF, in degrees
        fov (float): field-of-view in degrees
        as_tensor (bool, optional): whether to return as a tensor. Defaults to False.
        device (str or torch.Device, optional): if as_tensor=True, which device to use
        k (float, optional): scaling value for CMF
    Returns:
        np.ndarray or torch.Tensor: (n, 3) array of cortical cartesian points :math:`(x_c, y_c, z)`

    """
    cartesian_fov_coords = cartesian_coords*(fov/2)
    model = CorticalSensorManifold(cmf_a, fov, k=k, field_geometry=field_geometry)
    grid_pts_3d = model.vis_cartesian_to_cort_cartesian(cartesian_fov_coords)

    if as_tensor:
        return torch.tensor(grid_pts_3d, device=device)
    else:
        return grid_pts_3d


@add_to_all(__all__)
def vis_cartesian_to_cortical_cylindrical(cartesian_coords, cmf_a, fov, as_tensor=False, device='cpu', k=10, field_geometry='planar'):
    r"""
    Map visual cartesian coordinates to cortical cylindrical coordinates using the 3D cortical model. 

    * This is not usually used, since the cortical cartesian coordinates are more useful typically
    * We use this when coordinates are sampled elsewhere, and we want to map them into the 3D model for receptive field sampling.

    Note, the CMF is :math:`M(r)=\frac{k}{r+a}`

    Args:
        cartesian_coords (np.ndarray): (n,2) visual cartesian coordinates in (x,y) format
        cmf_a (float): :math:`a` value in the CMF, in degrees
        fov (float): field-of-view in degrees
        as_tensor (bool, optional): whether to return as a tensor. Defaults to False.
        device (str or torch.Device, optional): if as_tensor=True, which device to use
        k (float, optional): scaling value for CMF
    Returns:
        np.ndarray or torch.Tensor: (n, 3) array of cortical cylindrical points :math:`(\rho, z, \phi)`

    """
    cartesian_fov_coords = cartesian_coords*(fov/2)
    model = CorticalSensorManifold(cmf_a, fov, k=k, field_geometry=field_geometry)
    grid_pts_3d = np.array([(model.map_3d(x, y)) for x, y in cartesian_fov_coords])

    if as_tensor:
        return torch.tensor(grid_pts_3d, device=device)
    else:
        return grid_pts_3d


@add_to_all(__all__)
def cortical_cylindrical_to_cortical_cartesian(rho_z_phi, cmf_a, fov, k=10, field_geometry='planar'):
    r"""
    map cortical cylindrical coordinates to cortical cartesian coordinates
    
    Args:
        rho_z_phi (np.ndarray): (n, 3) array of cortical cylindrical points :math:`(\rho, z, \phi)`
        cmf_a (float): :math:`a` value in CMF
        fov (float): field-of-view in degrees
        k (float, optional): scaling value for CMF
    Returns:
        np.ndarray: (n,3) array of cortical cartesian coordinates :math:`(x_c, y_c, z)`
    """
    model = CorticalSensorManifold(cmf_a, fov, k=k, field_geometry=field_geometry)
    grid_pts_3d = np.array([(model.map_to_xyz(rho_z_phi_i)) for rho_z_phi_i in rho_z_phi])
    return grid_pts_3d


def sample_coords_from_manifold(cmf_a, fov, num_coords=10000, device='cpu', field_geometry='planar'):
    r"""
    Get sensor manifold coordinates using a 3D cortical model. 
    
    * This samples the manifold approximately uniformly, but in practice it is not uniform enough. 
    * Instead of using this, we typically:

        * sample isotropically in visual space (see .coords.SamplingCoords), then map into the 3d model for receptive field sampling. (vis_cartesian_to_cortical_cartesian_coords)
        * this leads to uniformly spaced points on the cortical manifold
        * thus, it is an indirect approach for even sampling on the manifold
        * it works much better than the direct method thus far implemented

    Args:
        cmf_a (float): a value in CMF
        fov (float): field-of-view in degrees
        num_coords (int, optional): number of points to sample
        device (str or torch.device, optional): which device to use
    Returns:
        result (tuple): A tuple containing:
            - coords (torch.Tensor): (n, 2) visual cartesian coordinates
            - polar_coords (torch.Tensor): (n, 2) visual polar coordinates
            - cortical_coords (torch.Tensor): (n, 3) cortical cartesian coordinates

    """
    model = CorticalSensorManifold(cmf_a, fov, field_geometry=field_geometry)
    # define a grid of visual points to specify the bounds of the 3d mesh
    grid_pts_3d_xyz, _ = model.init_visual_mesh()
    # sample the mesh evenly
    coords_3d = model.init_cortical_mesh(grid_pts_3d_xyz, num_coords)
    # reverse map into visual space using polar/spherical coordinat3es
    coords, polar_coords = model.reverse_map(coords_3d)
    # get logz coordinates for plotting in 2D
    coords = torch.tensor(model.normalize_coords(coords), device=device)
    coords = coords.flip(1) # flip x and y
    polar_coords = torch.tensor(polar_coords, device=device)
    # divide radius by fov/2 to get it into the range [0,1]
    polar_coords[:,0] = polar_coords[:,0]/(fov/2)
    cortical_coords = torch.tensor(coords_3d, device=device)
    return coords, polar_coords, cortical_coords
