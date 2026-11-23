"""Parity of fovi's exact surface geodesics with pygeodesic, the reference they replace.

pygeodesic is a test-only dependency (requirements-test.txt). Every test here
solves the same meshes with both and requires the same distances, up to float64
rounding, and the same neighbor tables in the pooling and patch-embedding layers
built from them.
"""

import numpy as np
import pytest
import torch
from scipy.spatial import Delaunay

import fovi.arch.knn as knn
import fovi.sensing.mesh_geodesic as mesh_geodesic
from fovi.arch.knn import KNNPoolingLayer, get_in_out_coords
from fovi.sensing import geodesic as surface_geodesic
from fovi.sensing.coords import SamplingCoords
from fovi.sensing.mesh_geodesic import ExactGeodesicSolver, distances_from_first_vertex

pygeodesic = pytest.importorskip("pygeodesic.geodesic")

DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
    ),
]
# Relative to the mesh extent; both solvers are exact, so only rounding differs.
TOLERANCE = 1e-11


def reference(vertices, faces, source):
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.shape[1] == 2:
        vertices = np.pad(vertices, ((0, 0), (0, 1)))
    algorithm = pygeodesic.PyGeodesicAlgorithmExact(vertices, np.asarray(faces, np.int32))
    values, _ = algorithm.geodesicDistances(np.array([source], np.int32))
    return values


def assert_parity(vertices, faces, sources, actual):
    scale = np.ptp(np.asarray(vertices), axis=0).max()
    for source, row in zip(sources, actual):
        expected = reference(vertices, faces, source)
        np.testing.assert_allclose(row, expected, rtol=0, atol=TOLERANCE * scale)


def synthetic_meshes():
    rng = np.random.default_rng(0)
    points = rng.uniform(-1, 1, (160, 2))
    faces = Delaunay(points).simplices
    radius = rng.uniform(0.2, 1, 200)
    angle = rng.uniform(0, 1.6 * np.pi, 200)
    c_shape = np.c_[radius * np.cos(angle), radius * np.sin(angle)]
    return {
        "flat": (points, faces),
        "bumpy": (np.c_[points, 0.3 * np.sin(3 * points[:, 0]) * np.cos(2 * points[:, 1])], faces),
        "saddle": (np.c_[points, 0.5 * (points[:, 0] ** 2 - points[:, 1] ** 2)], faces),
        # Non-convex: shortest paths bend around the boundary's reflex vertices.
        "c-shape": (c_shape, Delaunay(np.c_[np.log(radius), angle]).simplices),
    }


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", ["flat", "bumpy", "saddle", "c-shape"])
def test_synthetic_meshes_match_pygeodesic(name: str, device: str) -> None:
    vertices, faces = synthetic_meshes()[name]
    sources = list(range(0, len(vertices), 17))
    actual = ExactGeodesicSolver(vertices, faces, device=device).distances_from_each(sources)
    assert_parity(vertices, faces, sources, actual)


def test_heap_overflow_reruns_with_more_capacity(monkeypatch: pytest.MonkeyPatch) -> None:
    vertices, faces = synthetic_meshes()["saddle"]
    monkeypatch.setattr(mesh_geodesic, "_HEAP_PER_VERTEX", 0)
    actual = ExactGeodesicSolver(vertices, faces).distances_from_each([0, 50])
    assert_parity(vertices, faces, [0, 50], actual)


def test_unreferenced_vertices_are_unreachable() -> None:
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [5.0, 5.0]])
    distances = ExactGeodesicSolver(vertices, [[0, 1, 2]]).distances(0)
    np.testing.assert_allclose(distances[:3], [0.0, 1.0, 1.0])
    assert distances[3] == np.inf


def omni_coords(geometry: str, cmf_a: float, resolution: int = 32, patch: int = 8):
    coords = SamplingCoords(16.0, cmf_a, resolution, style="isotropic_fixn_calibrated",
                            field_geometry=geometry)
    _, patches, _ = get_in_out_coords(
        resolution, 16.0, cmf_a, patch, style="isotropic_fixn_calibrated", in_coords=coords,
        in_cart_res=resolution, device="cpu", max_out_coord_val="auto", field_geometry=geometry)
    return coords, patches


@pytest.mark.parametrize(("geometry", "cmf_a"), [("planar", 0.1), ("planar", 10.0), ("spherical", 1.0)])
def test_omni_patch_surface_matches_pygeodesic(
    geometry: str, cmf_a: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    solved = []

    class Recording(ExactGeodesicSolver):
        def __init__(self, vertices, faces, device=None):
            super().__init__(vertices, faces, device)
            self.mesh = (vertices, faces)

        def distances_from_each(self, sources):
            result = super().distances_from_each(sources)
            solved.append((*self.mesh, list(sources), result))
            return result

    monkeypatch.setattr(surface_geodesic, "ExactGeodesicSolver", Recording)
    coords, patches = omni_coords(geometry, cmf_a)
    surface_geodesic.patch_geodesic_distances(coords, patches, None)
    ((vertices, faces, sources, actual),) = solved
    assert_parity(vertices, faces, sources, actual)


def with_pygeodesic(meshes, device):
    return [reference(vertices, faces, 0) for vertices, faces in meshes]


def recorded_local_parity(monkeypatch: pytest.MonkeyPatch) -> list:
    """Make KNN layers check every local mesh they solve against pygeodesic."""
    solved = []

    def checked(meshes, device):
        result = distances_from_first_vertex(meshes, device)
        solved.append((meshes, result))
        return result

    monkeypatch.setattr(knn, "_local_geodesic_distances", checked)
    return solved


# (fov, cmf_a, res, highres multiple, k): the test_knn_neighbors geometry, the
# camera's default cmf_a with its default k, and a larger k.
POOLING_CASES = [(90.0, 9.0, 8, 2, 3), (90.0, 0.591, 16, 2, 4), (90.0, 0.591, 12, 2, 9)]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(("fov", "cmf_a", "res", "mult", "k"), POOLING_CASES)
def test_geodesic_pooling_tables_match_pygeodesic(
    fov, cmf_a, res, mult, k, device, monkeypatch: pytest.MonkeyPatch
) -> None:
    def build():
        out_coords = SamplingCoords(fov=fov, cmf_a=cmf_a, res=res, style="isotropic",
                                    field_geometry="spherical", device=device)
        in_coords = out_coords.clone(resolution=res * mult)
        return KNNPoolingLayer(k, in_coords, out_coords, mode="avg", device=device,
                               sample_cortex="geodesic")

    solved = recorded_local_parity(monkeypatch)
    actual = build()
    ((meshes, distances),) = solved
    for (vertices, faces), row in zip(meshes, distances):
        assert_parity(vertices, faces, [0], [row])
    monkeypatch.setattr(knn, "_local_geodesic_distances", with_pygeodesic)
    expected = build()
    assert torch.equal(actual.knn_indices, expected.knn_indices)
    assert torch.equal(actual.knn_indices_pad_token, expected.knn_indices_pad_token)
    torch.testing.assert_close(actual.knn_distances, expected.knn_distances, rtol=1e-6, atol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_zero_area_faces_on_warped_grids_match_pygeodesic(
    device: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Warped-Cartesian samples lie on a lattice on the cortex, so hull slivers of
    # the visual-field triangulation become collinear, zero-area cortical faces.
    def build():
        in_coords = SamplingCoords(16.0, 0.5, 16, style="warped_cartesian", device=device)
        out_coords = SamplingCoords(16.0, 0.5, 8, style="warped_cartesian", device=device)
        return KNNPoolingLayer(4, in_coords, out_coords, mode="avg", device=device,
                               sample_cortex="geodesic")

    solved = recorded_local_parity(monkeypatch)
    actual = build()
    ((meshes, distances),) = solved
    flat = 0
    for (vertices, faces), row in zip(meshes, distances):
        corners = np.pad(vertices, ((0, 0), (0, 3 - vertices.shape[1])))[faces]
        area = np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1)
        flat += int((area < 1e-12).sum())
        assert_parity(vertices, faces, [0], [row])
    assert flat > 0, "Need zero-area faces to exercise them"
    monkeypatch.setattr(knn, "_local_geodesic_distances", with_pygeodesic)
    assert torch.equal(actual.knn_indices_pad_token, build().knn_indices_pad_token)


@pytest.mark.parametrize("cmf_a", [2.785765, 60.94])
def test_geodesic_patch_partition_matches_pygeodesic(
    cmf_a: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fovi.models.knnvit import KNNPartitioningPatchEmbedding

    def build():
        return KNNPartitioningPatchEmbedding(
            in_channels=3, embed_dim=8, in_res=32, fov=16.0, cmf_a=cmf_a, style="isotropic",
            in_cart_res=32, max_coord_val="auto", device="cpu", sample_cortex="geodesic",
            force_patches_less_than_matched=False, ref_frame_side_length=16, transposed=True,
            bias=True, cart_patch_size=8)

    actual = build()
    monkeypatch.setattr(knn, "_local_geodesic_distances", with_pygeodesic)
    expected = build()
    assert torch.equal(actual.knn_indices, expected.knn_indices)
    assert torch.equal(actual.knn_indices_pad_token, expected.knn_indices_pad_token)
