"""Dense Cartesian sensors must preserve ordinary image row/column directions."""

import pytest
import torch
from fovi.sensing.retina import RetinalTransform


def make_retina(style: str, sampler: str, device: str = "cpu") -> RetinalTransform:
    return RetinalTransform(
        resolution=16,
        start_res=32,
        fixation_size=32,
        fov=16.0,
        cmf_a=0.5,
        style=style,
        sampler=sampler,
        auto_match_cart_resources=False,
        sampler_backend="torch",
        device=device,
    ).eval()


@pytest.mark.parametrize("style", ["uniform_as_grid", "warped_cartesian_as_grid"])
@pytest.mark.parametrize("sampler", ["grid_nn", "grid_bilinear"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.uint8])
def test_coordinate_ramps_are_upright(
    style: str, sampler: str, dtype: torch.dtype
) -> None:
    retina = make_retina(style, sampler)
    yy, xx = torch.meshgrid(torch.arange(32), torch.arange(32), indexing="ij")
    image = torch.stack((xx, yy, torch.ones_like(xx))).to(dtype)[None]
    samples = retina(image, (0.5, 0.5))

    assert samples.is_contiguous()
    assert samples[0, 0, 8, 12] > samples[0, 0, 8, 3]
    assert samples[0, 1, 12, 8] > samples[0, 1, 3, 8]
    torch.testing.assert_close(samples[0, 0, 12, 8], samples[0, 0, 3, 8])
    torch.testing.assert_close(samples[0, 1, 8, 12], samples[0, 1, 8, 3])


@pytest.mark.parametrize("style", ["uniform", "warped_cartesian"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_grid_and_vector_samples_and_gradients_match(style: str, device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    flat_retina = make_retina(style, "grid_bilinear", device)
    grid_retina = make_retina(f"{style}_as_grid", "grid_bilinear", device)
    images = torch.rand(2, 3, 32, 32, device=device, requires_grad=True)
    fixations = torch.tensor([[0.5, 0.5], [0.35, 0.7]], device=device)
    flat = flat_retina(images, fixations)
    grid = grid_retina(images, fixations)
    expected = flat.reshape(2, 3, 16, 16).transpose(-2, -1).flip(-2)
    torch.testing.assert_close(grid, expected, atol=0, rtol=0)
    torch.testing.assert_close(
        grid_retina.sampler.coords.cartesian,
        flat_retina.sampler.coords.cartesian,
        atol=0,
        rtol=0,
    )
    weights = torch.randn_like(grid)
    actual_grad = torch.autograd.grad(
        (grid * weights).sum(), images, retain_graph=True
    )[0]
    expected_grad = torch.autograd.grad((expected * weights).sum(), images)[0]
    torch.testing.assert_close(actual_grad, expected_grad)


def test_uniform_grid_preserves_an_asymmetric_image_exactly() -> None:
    retina = RetinalTransform(
        resolution=16,
        start_res=16,
        fixation_size=16,
        fov=16,
        cmf_a=None,
        style="uniform_as_grid",
        sampler="grid_nn",
        device="cpu",
        auto_match_cart_resources=False,
        sampler_backend="torch",
    ).eval()
    image = torch.arange(3 * 16 * 16, dtype=torch.float32).reshape(1, 3, 16, 16)
    torch.testing.assert_close(retina(image, (0.5, 0.5)), image, atol=0, rtol=0)
