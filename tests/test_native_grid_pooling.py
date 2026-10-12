"""Native-grid pooling builds local neighborhoods without all-pairs distances."""

from unittest import mock

import pytest
import torch
from fovi.arch.knn import KNNPoolingLayer
from fovi.sensing.coords import SamplingCoords


@pytest.mark.parametrize("k", [1, 4, 9])
@pytest.mark.parametrize("resolution", [7, 16])
def test_native_grid_neighbors_match_euclidean_distances(
    k: int, resolution: int
) -> None:
    incoming = SamplingCoords(50, 5, resolution, style="warped_cartesian_as_grid")
    outgoing = incoming.clone(resolution=8)
    layer = KNNPoolingLayer(k, incoming, outgoing, device="cpu", sample_cortex=True)
    candidates = torch.cat((incoming.cortical, incoming.cortical_pad_coords))
    distances = torch.cdist(
        candidates.double(),
        outgoing.cortical.double(),
        compute_mode="donot_use_mm_for_euclid_dist",
    )
    expected = distances.topk(k, dim=0, largest=False).values
    selected = distances.gather(0, layer.knn_indices)
    torch.testing.assert_close(selected.sort(dim=0).values, expected)
    torch.testing.assert_close(
        layer.knn_distances.double(), selected, atol=1e-7, rtol=1e-6
    )


def test_native_grid_pooling_avoids_dense_search() -> None:
    incoming = SamplingCoords(50, 5, 32, style="warped_cartesian_as_grid")
    outgoing = incoming.clone(resolution=16)
    # A dense search scales quadratically; reject it even when this small fixture fits.
    with mock.patch("torch.cdist", side_effect=AssertionError("all-pairs search")):
        layer = KNNPoolingLayer(4, incoming, outgoing, device="cpu", sample_cortex=True)
    values = torch.full((1, 1, len(incoming)), 100.0)
    values[..., ~incoming.valid_mask] = 1e6
    expected = outgoing.valid_mask.float().reshape(1, 1, -1) * 100
    torch.testing.assert_close(layer(values), expected)


def test_twofold_grid_pooling_selects_the_corresponding_cell() -> None:
    incoming = SamplingCoords(
        50, 5, 16, style="warped_cartesian_as_grid", fov_type="wang"
    )
    outgoing = incoming.clone(resolution=8)
    layer = KNNPoolingLayer(
        4, incoming, outgoing, mode="avg", device="cpu", sample_cortex=True
    )
    values = torch.arange(len(incoming), dtype=torch.float32).reshape(1, 1, -1)
    expected = torch.nn.functional.avg_pool2d(incoming.as_grid(values), 2)
    actual = outgoing.as_grid(layer(values))
    torch.testing.assert_close(actual, expected)
