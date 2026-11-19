"""Modal pooling for label maps: exact values, tie-breaking, excluded and unseen neighbors.

Run with: python -m pytest tests/test_modal_pooling.py
"""

import unittest
from unittest import mock

import torch

from fovi.arch import modal
from fovi.arch.knn import KNNPoolingLayer
from fovi.arch.modal import modal_pool
from fovi.arch.pooling import GridPoolingLayer, build_retinal_pooler
from fovi.sensing.coords import SamplingCoords

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

# Three outputs over five inputs; 5 marks an excluded neighbor. Columns are outputs,
# rows are neighbor slots in tie-break order.
HAND_TABLE = torch.tensor([[0, 3, 5], [1, 4, 5], [2, 5, 5]])


def reference_modal_pool(values, table, visible=None):
    """Brute-force modal pooling of ``(B, C, N_in)`` values over a ``(k, N_out)`` table."""
    batch, channels, num_inputs = values.shape
    out = torch.zeros(batch, channels, table.shape[1], dtype=values.dtype)
    for b in range(batch):
        for n in range(table.shape[1]):
            voters = [i for i in table[:, n].tolist()
                      if i != num_inputs and (visible is None or bool(visible[b, i]))]
            labels = [tuple(values[b, :, i].tolist()) for i in voters]
            if labels:
                # max() keeps the first of equally common labels, i.e. the earliest slot.
                out[b, :, n] = torch.tensor(max(labels, key=labels.count), dtype=values.dtype)
    return out


def _isotropic(res, device="cpu"):
    lowres = SamplingCoords(fov=90.0, cmf_a=9.0, res=res, style="isotropic", field_geometry="spherical", device=device)
    return lowres.clone(resolution=2 * res), lowres


class TestModalPool(unittest.TestCase):

    def test_majority_color_vector_wins_and_ties_go_to_the_earliest_neighbor(self):
        # Output 0 sees [0,0,0,255] twice and [0,0,1,255] once; a per-channel max would give [0,0,1,255].
        # Output 1 sees one vote each for two colors (its third slot is excluded): the first wins.
        # Output 2 has no voting neighbor and is zero.
        colors = torch.tensor(
            [[0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 1, 255], [10, 20, 30, 255], [200, 100, 0, 255]],
            dtype=torch.uint8)
        for device in DEVICES:
            with self.subTest(device=device):
                values = colors.T[None].to(device)
                pooled = modal_pool(values, HAND_TABLE.to(device))
                expected = torch.tensor([[0, 0, 0, 255], [10, 20, 30, 255], [0, 0, 0, 0]], dtype=torch.uint8)
                self.assertEqual(pooled.dtype, torch.uint8)
                self.assertTrue(torch.equal(pooled[0].T.cpu(), expected))
                swapped = HAND_TABLE[[1, 0, 2]].to(device)
                self.assertTrue(torch.equal(modal_pool(values, swapped)[0, :, 1].cpu(), colors[4]))

    def test_ids_beyond_float_precision_come_back_exactly(self):
        # 2**24 + 1 and 2**24 + 3 round to other values in float32.
        for dtype, big in ((torch.int32, 2**30), (torch.int64, 2**40)):
            ids = torch.tensor([2**24 + 1, 2**24 + 1, 2**24 + 3, big + 7, big + 7], dtype=dtype)
            values = torch.stack([ids, ids.flip(0)])[:, None]
            pooled = modal_pool(values, HAND_TABLE)
            expected = torch.tensor([[2**24 + 1, big + 7, 0], [big + 7, 2**24 + 1, 0]], dtype=dtype)
            self.assertEqual(pooled.dtype, dtype)
            self.assertTrue(torch.equal(pooled[:, 0], expected))

    def test_wide_vectors_are_compared_exactly(self):
        # Two int64 channels exceed the 8-byte packed key; vectors differing only in the second channel stay distinct.
        vectors = torch.tensor([[1, 2**40], [1, 2**40 + 1], [1, 2**40 + 1], [5, 6], [7, 8]], dtype=torch.int64)
        pooled = modal_pool(vectors.T[None], HAND_TABLE)
        self.assertTrue(torch.equal(pooled[0].T, torch.tensor([[1, 2**40 + 1], [5, 6], [0, 0]])))

    def test_unseen_inputs_do_not_vote(self):
        # Output 0 sees labels [1, 1, 2]; hiding the 1s leaves 2. Output 1 sees [4, 5]; hiding both leaves nothing.
        labels = torch.tensor([1, 1, 2, 4, 5], dtype=torch.int32)[None, None].expand(2, -1, -1)
        visible = torch.tensor([[False, False, True, False, False], [True, True, True, True, True]])
        pooled = modal_pool(labels, HAND_TABLE, visible)
        self.assertTrue(torch.equal(pooled[:, 0], torch.tensor([[2, 0, 0], [1, 4, 0]], dtype=torch.int32)))
        with self.assertRaises(ValueError):
            modal_pool(labels, HAND_TABLE, visible[:, :4])

    def test_chunked_vote_matches_the_reference(self):
        # Few labels so neighborhoods hold repeats and ties; tiny chunks exercise the batched loop.
        generator = torch.Generator().manual_seed(1)
        table = torch.randint(0, 41, (6, 30), generator=generator)
        labels = torch.randint(0, 3, (3, 2, 40), generator=generator).to(torch.uint8)
        visible = torch.rand(3, 40, generator=generator) > 0.3
        with mock.patch.object(modal, "_CHUNK_ELEMENTS", 1):
            pooled = modal_pool(labels, table, visible)
        self.assertTrue(torch.equal(pooled, reference_modal_pool(labels, table, visible)))


class TestModalPoolers(unittest.TestCase):

    def test_knn_modal_votes_over_the_neighbor_table(self):
        for sample_cortex in (False, True):
            with self.subTest(sample_cortex=sample_cortex):
                highres, lowres = _isotropic(8)
                pooler = KNNPoolingLayer(4, highres, lowres, mode="modal", device="cpu", sample_cortex=sample_cortex)
                labels = torch.randint(0, 3, (2, 1, len(highres)), generator=torch.Generator().manual_seed(0))
                pooled = pooler(labels)
                self.assertEqual(pooled.dtype, labels.dtype)
                self.assertTrue(torch.equal(pooled, reference_modal_pool(labels, pooler.knn_indices_pad_token)))

    def test_grid_modal_windows_match_the_grid_pooler(self):
        cases = [
            ("warped_cartesian_as_grid", "spherical", 6, 2, 4),
            ("warped_cartesian_as_grid", "spherical", 6, 2, 9),
            ("warped_cartesian_as_grid", "spherical", 6, 3, 4),
            ("logpolar_as_grid", "planar", 6, 2, 9),
        ]
        for style, geometry, res, mult, k in cases:
            with self.subTest(style=style, res=res, mult=mult, k=k):
                lowres = SamplingCoords(fov=90.0, cmf_a=9.0, res=res, style=style, field_geometry=geometry, device="cpu")
                highres = lowres.clone(resolution=mult * res)
                modal_pooler = GridPoolingLayer(k, highres, lowres, mode="modal")
                max_pooler = GridPoolingLayer(k, highres, lowres, mode="max")
                table = modal_pooler.neighbor_table
                # The table holds exactly the samples the max pooler reduces over.
                values = torch.rand(1, 1, len(highres), generator=torch.Generator().manual_seed(0)) + 1
                padded = torch.cat([values[0, 0], torch.zeros(1)])
                torch.testing.assert_close(padded[table].max(dim=0).values, max_pooler(values)[0, 0])
                labels = torch.randint(0, 3, (2, 2, len(highres)), generator=torch.Generator().manual_seed(1))
                self.assertTrue(torch.equal(modal_pooler(labels), reference_modal_pool(labels, table)))

    def test_modal_pooling_through_the_retinal_pooler_builder(self):
        highres, lowres = _isotropic(8)
        pooler, _ = build_retinal_pooler(highres, lowres, k=None, mode="modal", sample_cortex=True,
                                         gauss_sigma=10.0, device="cpu")
        labels = torch.arange(len(highres), dtype=torch.int64)[None, None] + 2**40
        pooled = pooler(labels)
        # Unique labels each get one vote, so every output takes its nearest valid neighbor's label exactly.
        table = pooler.knn_indices_pad_token
        for n in range(table.shape[1]):
            neighbors = [i for i in table[:, n].tolist() if i != len(highres)]
            expected = int(labels[0, 0, neighbors[0]]) if neighbors else 0
            self.assertEqual(int(pooled[0, 0, n]), expected)

    def test_visible_is_rejected_by_arithmetic_modes(self):
        highres, lowres = _isotropic(8)
        pooler = KNNPoolingLayer(4, highres, lowres, mode="avg", device="cpu", sample_cortex=True)
        with self.assertRaises(ValueError):
            pooler(torch.rand(1, 1, len(highres)), visible=torch.ones(1, len(highres), dtype=torch.bool))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_modal_vote_on_cuda_matches_cpu(self):
        lowres = SamplingCoords(fov=90.0, cmf_a=9.0, res=6, style="warped_cartesian_as_grid",
                                field_geometry="spherical", device="cpu")
        highres = lowres.clone(resolution=12)
        table = GridPoolingLayer(9, highres, lowres, mode="modal").neighbor_table
        labels = torch.randint(0, 3, (2, 2, len(highres)), generator=torch.Generator().manual_seed(2))
        visible = torch.rand(2, len(highres), generator=torch.Generator().manual_seed(3)) > 0.2
        expected = modal_pool(labels, table, visible)
        actual = modal_pool(labels.cuda(), table.cuda(), visible.cuda())
        self.assertTrue(torch.equal(actual.cpu(), expected))
        self.assertTrue(torch.equal(expected, reference_modal_pool(labels, table, visible)))


if __name__ == "__main__":
    unittest.main()
