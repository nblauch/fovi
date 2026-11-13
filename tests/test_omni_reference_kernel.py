"""Reference resolution changes weight placement, not the sampling budget."""

import pytest
import torch

from fovi.models.omni_isotropic import OmniIsotropicPatchEmbedding


def make_patch(side: int) -> OmniIsotropicPatchEmbedding:
    return OmniIsotropicPatchEmbedding(
        fov=16,
        cmf_a=[0.001, 1000],
        resolution=16,
        patch_size=4,
        reference_side=side,
        in_channels=1,
        embed_dim=1,
        sample_cortex="geodesic",
        device="cpu",
    )


def test_upsampling_preserves_orientation_and_pointwise_amplitude() -> None:
    patch = make_patch(9)
    conv = torch.nn.Conv2d(1, 1, 3, bias=True)
    with torch.no_grad():
        # An affine field has an exact interpolation oracle, including orientation.
        conv.weight.copy_(
            torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
        )
        conv.bias.fill_(2)
    patch.load_conv2d_weights(conv)
    x = torch.linspace(0, 1, 9)
    expected = 6 + 2 * x[:, None] - 6 * x[None, :]
    torch.testing.assert_close(patch.weight.reshape(9, 9), expected)
    torch.testing.assert_close(patch.bias, conv.bias)
    with pytest.raises(ValueError, match="channel shape"):
        patch.load_conv2d_weights(torch.nn.Conv2d(2, 1, 3))


def test_larger_kernel_preserves_neighbors_and_constant_field_response() -> None:
    small, large = make_patch(4), make_patch(8)
    torch.testing.assert_close(small.positions, large.positions, atol=0, rtol=0)
    torch.testing.assert_close(small.geometry.reverse_ptr, large.geometry.reverse_ptr)
    # Reverse incidence identifies each sample's patch independently of binning.
    for level in range(2):
        count = int(small.geometry.ptr[level, -1])
        small_patches = small.geometry.reverse_indices[level, :count] // 16
        large_patches = large.geometry.reverse_indices[level, :count] // 64
        torch.testing.assert_close(small_patches, large_patches, atol=0, rtol=0)
    conv = torch.nn.Conv2d(1, 1, 4, bias=True)
    with torch.no_grad():
        conv.weight.fill_(1.25)
        conv.bias.fill_(-0.5)
    small.load_conv2d_weights(conv)
    large.load_conv2d_weights(conv)
    images = torch.randn(2, 1, 256)
    levels = torch.arange(2)
    torch.testing.assert_close(
        small(images, levels), large(images, levels), atol=1e-5, rtol=1e-5
    )
