"""Calibrated retinal pooling preserves retinal support and camera padding."""

import pytest
import torch
from fovi.sensing.projection import CameraModel, gaze_rotation
from fovi.sensing.retina import RetinalTransform
from fovi.sensing.samplers import GridSampler, PoolingGridSampler


def camera_model(model: str = "fisheye") -> CameraModel:
    return CameraModel(model, (48, 64), (35, 35, 31.5, 23.5))


def make_sampler(
    mode: str,
    device: str = "cpu",
    backend: str = "torch",
    model: str = "fisheye",
    fov_type: str = "circular",
    style: str = "warped_cartesian",
) -> PoolingGridSampler:
    return PoolingGridSampler(
        50,
        5,
        8,
        res_mult=2,
        style=style,
        pool_mode=mode,
        gauss_sigma=0.5,
        sample_cortex=True,
        device=device,
        backend=backend,
        field_geometry="spherical",
        camera_model=camera_model(model),
        fov_type=fov_type,
        gaze_convention="pan_tilt",
    )


@pytest.mark.parametrize("style", ["warped_cartesian", "warped_cartesian_as_grid"])
@pytest.mark.parametrize("mode", ["avg", "gaussian"])
@pytest.mark.parametrize("explicit_rotation", [False, True])
@pytest.mark.parametrize("model", ["fisheye", "pinhole"])
def test_spherical_pooling_matches_calibrated_sampling(
    mode: str,
    explicit_rotation: bool,
    model: str,
    style: str,
) -> None:
    sampler = make_sampler(mode, model=model, style=style)
    reference = GridSampler(
        50,
        5,
        16,
        style=style,
        device="cpu",
        backend="torch",
        field_geometry="spherical",
        camera_model=camera_model(model),
        gaze_convention="pan_tilt",
    )
    image = torch.arange(48 * 64).remainder(251).to(torch.uint8).reshape(1, 1, 48, 64)
    image = image.expand(2, 3, -1, -1)
    fixation = torch.tensor([[0.4, 0.65], [0.6, 0.3]])
    rotation = None
    if explicit_rotation:
        rotation = gaze_rotation(
            torch.tensor([[0.2, -0.3, 1.0], [-0.3, 0.1, 1.0]]), "pan_tilt"
        )
    expected = sampler.pooler(reference(image, fixation, rotation=rotation).float())
    actual = sampler(image, fixation, rotation=rotation)
    torch.testing.assert_close(actual, expected)
    assert sampler.camera_model is sampler.input_sampler.camera_model


@pytest.mark.parametrize("mode", ["avg", "gaussian"])
@pytest.mark.parametrize("fov_type", ["circular", "square"])
def test_retinal_mask_excludes_values_and_normalizes_boundary(
    mode: str, fov_type: str
) -> None:
    sampler = make_sampler(mode, fov_type=fov_type)
    incoming, outgoing = sampler.highres_coords, sampler.coords
    values = torch.full((1, 3, len(incoming)), 100.0)
    values[..., ~incoming.valid_mask] = 1e6
    result = sampler.pooler(values)
    counts = (~sampler.pooler.knn_indices_pad_mask).sum(0)
    assert ((counts > 0) & (counts < sampler.k) & outgoing.valid_mask).any()
    torch.testing.assert_close(
        result[..., outgoing.valid_mask],
        torch.full_like(result[..., outgoing.valid_mask], 100.0),
    )
    assert torch.count_nonzero(result[..., ~outgoing.valid_mask]) == 0
    sampled = sampler(torch.full((1, 3, 48, 64), 100.0), torch.tensor([[0.5, 0.5]]))
    torch.testing.assert_close(sampled, result)


def test_camera_padding_remains_in_average_denominator() -> None:
    sampler = make_sampler("avg")
    image = torch.full((1, 1, 48, 64), 100.0)
    fixation = torch.tensor([[0.5, 0.98]])
    highres = sampler.input_sampler(image, fixation)
    indices = sampler.pooler.knn_indices_pad_token
    gathered = torch.nn.functional.pad(highres, (0, 1))[..., indices]
    counts = (~sampler.pooler.knn_indices_pad_mask).sum(0)
    expected = gathered.sum(-2) / counts.clamp_min(1)
    actual = sampler(image, fixation)
    torch.testing.assert_close(actual, expected)
    assert ((actual > 0) & (actual < 100)).any()


@pytest.mark.parametrize("sampler", ["pooling", "gaussian_pooling"])
def test_spherical_transform_retains_output_grid(sampler: str) -> None:
    kwargs = {
        "resolution": 8,
        "fov": 50,
        "cmf_a": 5,
        "style": "warped_cartesian_as_grid",
        "device": "cpu",
        "auto_match_cart_resources": 0,
        "field_geometry": "spherical",
        "camera_model": camera_model(),
        "sampler_backend": "torch",
        "sigma": None,
    }
    original = RetinalTransform(**kwargs)
    extra = {"gauss_sigma": 0.5} if sampler == "gaussian_pooling" else {}
    pooled = RetinalTransform(**kwargs, sampler=sampler, res_mult=2, **extra)
    torch.testing.assert_close(
        pooled.sampler.coords.cartesian, original.sampler.coords.cartesian
    )
    assert torch.equal(pooled.valid_mask, original.valid_mask)
    assert pooled.camera_model == original.camera_model
    image = torch.full((1, 3, 48, 64), 100, dtype=torch.uint8)
    fixation = torch.tensor([[0.5, 0.5]])
    torch.testing.assert_close(pooled(image, fixation), original(image, fixation))


def test_multilevel_pooling_remains_unsupported() -> None:
    with pytest.raises(ValueError, match="Multi-level cmf_a"):
        RetinalTransform(
            8,
            fov=50,
            cmf_a=[5, 10],
            style="warped_cartesian_as_grid",
            sampler="pooling",
            field_geometry="spherical",
            camera_model=camera_model(),
            device="cpu",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("mode", ["avg", "gaussian"])
def test_cuda_spherical_pooling_matches_eager(mode: str) -> None:
    sampler = make_sampler(mode, "cuda", "cuda")
    image = (
        torch.arange(48 * 64, device="cuda")
        .remainder(251)
        .to(torch.uint8)
        .reshape(1, 1, 48, 64)
    )
    fixation = torch.tensor([[0.4, 0.9]], device="cuda")
    expected = sampler(image, fixation, direct=True)
    actual = sampler(image, fixation)
    torch.testing.assert_close(actual, expected)
