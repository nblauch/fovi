"""Surface geodesics from actual patch centers for fixed-count geometry banks."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import numpy as np
import scipy
import torch
from scipy.spatial import Delaunay

from .coords import SamplingCoords


def patch_geodesic_distances(
    inputs: SamplingCoords, patches: SamplingCoords, cache_dir: Path | None = None
) -> torch.Tensor:
    """Return [real + padding samples, patches] distances on the cortical mesh.

    Triangulate in the visual plane, but measure on the embedded cortical
    surface. Include the actual patch centers as mesh vertices, rather than
    moving each source to its nearest retinal sample. Distances are exact on
    this piecewise-linear mesh, an approximation to the smooth manifold.
    Construction is CPU-only and stays outside the training hot path.
    """
    import pygeodesic

    if inputs.field_geometry != "planar" or patches.field_geometry != "planar":
        raise ValueError("Fixed-count geodesic banks currently require planar geometry")
    if inputs.fov != patches.fov or inputs.cmf_a != patches.cmf_a:
        raise ValueError("Geodesic samples and patches must share FoV and CMF")
    visual = (
        torch.cat((inputs.cartesian, inputs.cartesian_pad_coords, patches.cartesian))
        .double()
        .cpu()
        .numpy()
    )
    cortical = (
        torch.cat((inputs.cortical, inputs.cortical_pad_coords, patches.cortical))
        .double()
        .cpu()
        .numpy()
    )
    sample_count = len(inputs) + len(inputs.cartesian_pad_coords)
    expected_shape = (sample_count, len(patches))
    signature = hashlib.sha256(
        f"surface-geodesic-v1:{np.__version__}:{scipy.__version__}:{pygeodesic.__version__}:{expected_shape}".encode()
        + visual.tobytes()
        + cortical.tobytes()
    ).hexdigest()
    cache_path = None if cache_dir is None else cache_dir / f"{signature}.npz"
    if cache_path is not None and cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cached:
            distances = cached["distances"]
            checksum = hashlib.sha256(distances.tobytes()).hexdigest()
            if (
                str(cached["geometry"]) != signature
                or str(cached["sha256"]) != checksum
                or distances.shape != expected_shape
                or distances.dtype != np.float32
                or not np.isfinite(distances).all()
                or (distances < 0).any()
            ):
                raise ValueError(f"Invalid geodesic cache: {cache_path}")
        return torch.from_numpy(distances)
    print(
        f"Building cortical surface geodesics: a={inputs.cmf_a:g}, samples={sample_count}, patches={len(patches)}",
        flush=True,
    )
    distances = _solve_surface_geodesics(visual, cortical, sample_count)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=cache_path.parent, suffix=".npz", delete=False
        ) as stream:
            temporary = Path(stream.name)
            try:
                np.savez_compressed(
                    stream,
                    distances=distances,
                    geometry=signature,
                    sha256=hashlib.sha256(distances.tobytes()).hexdigest(),
                )
                stream.close()
                temporary.replace(cache_path)
            finally:
                temporary.unlink(missing_ok=True)
    return torch.from_numpy(distances)


def _solve_surface_geodesics(
    visual: np.ndarray, cortical: np.ndarray, sample_count: int
) -> np.ndarray:
    """Solve all sample/patch pairs on one joint triangulated surface."""
    from pygeodesic import geodesic

    unique, first, inverse = np.unique(
        visual, axis=0, return_index=True, return_inverse=True
    )
    faces = Delaunay(unique).simplices.astype(np.int32)
    if len(np.unique(faces)) != len(unique):
        raise RuntimeError("Geodesic triangulation omitted a sample or patch center")
    solver = geodesic.PyGeodesicAlgorithmExact(cortical[first], faces)
    targets = inverse[:sample_count].astype(np.int32)
    sources = inverse[sample_count:].astype(np.int32)
    distances = np.empty((sample_count, len(sources)), dtype=np.float32)
    for column, source in enumerate(sources):
        values, _ = solver.geodesicDistances(
            np.array([source], dtype=np.int32), targets
        )
        if not np.isfinite(values).all() or (values < 0).any():
            raise RuntimeError("Geodesic solver returned invalid cortical distances")
        distances[:, column] = values
    return distances
