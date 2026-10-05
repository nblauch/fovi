fovi.sensing.samplers
=====================

Retinal pooling
---------------

``RetinalTransform(sampler="pooling", pool_method="auto")`` uses masked 2D
pooling for Cartesian ``_as_grid`` sensors and KNN for sample sets.
``sampler="gaussian_pooling"`` applies Gaussian weights with the same selection.
Select ``pool_method="knn"`` explicitly to retain the previous grid behavior.
``KNNGridSampler`` and ``GaussianKNNGridSampler`` still default to KNN.

For ``grid2d``, ``k`` is the square kernel side and ``None`` uses the integer
high/low resolution ratio. That ratio also sets the stride. Input and output
must share their warp parameters (``cmf_a_mult=1``); polar grids are unsupported.
Invalid retinal units are excluded and weights renormalized. Missing camera
coverage retains zero padding. Kernel/stride parity mismatches use asymmetric
SAME-style padding, placing the window center half a high-resolution sample
toward larger image indices. All output coordinates and dimensions are retained.

.. automodule:: fovi.sensing.samplers
   :members:
   :undoc-members:
   :show-inheritance:
   :imported-members:
   :special-members: __init__
   :exclude-members: __weakref__
