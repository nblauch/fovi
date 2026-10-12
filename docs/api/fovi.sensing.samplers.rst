fovi.sensing.samplers
=====================

Retinal pooling
---------------

Pooling samplers (``RetinalTransform(sampler="pooling" | "gaussian_pooling")``,
``PoolingGridSampler``) select their pooler from the sensor style: ``_as_grid``
sensors use masked 2D pooling and sample-set sensors use KNN pooling.

For 2D pooling, ``k`` is the neighborhood size, i.e. the squared kernel side
(``k=9`` is a 3x3 window), and must be a perfect square. ``None`` uses the
squared integer high/low resolution ratio. That ratio also sets the stride.
Input and output must share their warp parameters (``cmf_a_mult=1``).
Invalid retinal units and spatial padding are excluded and weights
renormalized. Missing camera coverage retains zero padding. Kernel/stride
parity mismatches use asymmetric SAME-style padding, placing the window center
half a high-resolution sample toward larger image indices. All output
coordinates and dimensions are retained.

Log-polar grids (angle x eccentricity) pad the angular axis circularly, so
windows wrap around and are centered on each output angle. The eccentricity
axis uses the spatial convention above; since log-polar radii span the same
endpoints at every resolution, those windows are index-aligned rather than
centered on the output radii.

.. automodule:: fovi.sensing.samplers
   :members:
   :undoc-members:
   :show-inheritance:
   :imported-members:
   :special-members: __init__
   :exclude-members: __weakref__
