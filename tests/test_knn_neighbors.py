"""KNN neighbor tables are chosen deterministically, independent of float32 rounding and device.

Run with: python -m pytest tests/test_knn_neighbors.py
"""

import copy
import unittest

import torch

from fovi.arch.knn import KNNPoolingLayer, _nearest_neighbors, _topk_candidates
from fovi.sensing.coords import SamplingCoords

# (style, res, highres multiple, k, sample_cortex): visual-field, cortical and geodesic
# distances on isotropic rings, whose symmetry creates many ties, and the native
# warped-grid tree.
CASES = [
    ("isotropic", 8, 2, 3, False),
    ("isotropic", 8, 2, 3, True),
    ("isotropic", 8, 2, 3, "geodesic"),
    ("isotropic", 12, 2, 6, True),
    ("warped_cartesian", 12, 2, 4, False),
    ("warped_cartesian", 12, 2, 4, True),
]
COORD_FIELDS = ("cartesian", "cortical", "cartesian_pad_coords", "cortical_pad_coords")


def _coords(style, res, device):
    return SamplingCoords(
        fov=90.0, cmf_a=9.0, res=res, style=style, field_geometry="spherical", device=device)


def _pooler(in_coords, out_coords, k, sample_cortex, device):
    return KNNPoolingLayer(
        k, in_coords, out_coords, mode="avg", device=device, sample_cortex=sample_cortex)


def _jitter(coords, scale, seed):
    """A copy of ``coords`` with every coordinate changed by float32-rounding-sized noise."""
    jittered = copy.deepcopy(coords)
    generator = torch.Generator().manual_seed(seed)
    for name in COORD_FIELDS:
        values = getattr(jittered, name, None)
        if values is None or values.numel() == 0:
            continue
        noise = torch.rand(values.shape, generator=generator).to(values) * 2 - 1
        values.add_(noise * scale * values.abs())
    return jittered


class TestNearestNeighbors(unittest.TestCase):

    def test_ties_at_the_cutoff_go_to_the_lowest_index(self):
        # Candidates 2, 3 and 4 tie at 0.3 (4 by less than the tolerance); 1 is nearest.
        distances = torch.tensor([[0.5], [0.2], [0.3], [0.3], [0.3 + 1e-9], [0.9]])
        query = torch.ones(1, 2)
        for k, expected in ((2, [1, 2]), (3, [1, 2, 3]), (4, [1, 2, 3, 4])):
            indices, values = _nearest_neighbors(_topk_candidates(distances), k, query, 6)
            self.assertEqual(indices[:, 0].tolist(), expected)
            torch.testing.assert_close(values[:, 0], distances[expected, 0])

    def test_ties_reaching_past_the_first_candidates_are_resolved(self):
        # Ten candidates tie with the cutoff, more than the first search holds; the lowest
        # indices win, and a query without ties keeps its nearest neighbors.
        distances = torch.cat([torch.full((10, 1), 0.3), torch.arange(10.0)[:, None] / 10 + 1], dim=1)
        distances = torch.cat([torch.tensor([[0.1, 0.0]]), distances.flip(0)])
        indices, _ = _nearest_neighbors(_topk_candidates(distances), 3, torch.ones(2, 2), 11)
        self.assertEqual(indices[:, 0].tolist(), [0, 1, 2])
        self.assertEqual(indices[:, 1].tolist(), [0, 10, 9])


class TestNeighborTables(unittest.TestCase):

    def test_rounding_sized_coordinate_changes_keep_the_tables(self):
        # Coordinates computed on another device differ by about 1e-7 to 1e-5 of their
        # magnitude; neighbor tables built from them must not change.
        for style, res, mult, k, sample_cortex in CASES:
            with self.subTest(style=style, res=res, k=k, sample_cortex=sample_cortex):
                out_coords = _coords(style, res, "cpu")
                in_coords = out_coords.clone(resolution=res * mult)
                reference = _pooler(in_coords, out_coords, k, sample_cortex, "cpu").knn_indices_pad_token
                for seed in range(3):
                    jittered = _pooler(
                        _jitter(in_coords, 1e-6, seed), _jitter(out_coords, 1e-6, seed + 10),
                        k, sample_cortex, "cpu").knn_indices_pad_token
                    self.assertTrue(torch.equal(reference, jittered))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cpu_and_cuda_tables_match(self):
        for style, res, mult, k, sample_cortex in CASES:
            with self.subTest(style=style, res=res, k=k, sample_cortex=sample_cortex):
                tables = []
                for device in ("cpu", "cuda"):
                    out_coords = _coords(style, res, device)
                    in_coords = out_coords.clone(resolution=res * mult)
                    tables.append(_pooler(in_coords, out_coords, k, sample_cortex, device).knn_indices_pad_token.cpu())
                self.assertTrue(torch.equal(*tables))


if __name__ == "__main__":
    unittest.main()
