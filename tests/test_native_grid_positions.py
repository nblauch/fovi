"""Native image ordering and configurable dense DINO patch coordinates."""

import copy

import pytest
import torch
from fovi.models.dinov3 import configure_dinov3_positions
from fovi.sensing.coords import SamplingCoords
from fovi.sensing.projection import CameraModel
from fovi.sensing.retina import RetinalTransform
from transformers import DINOv3ViTConfig, DINOv3ViTModel


def make_model() -> DINOv3ViTModel:
    return DINOv3ViTModel(
        DINOv3ViTConfig(
            image_size=32,
            patch_size=8,
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=64,
            num_register_tokens=0,
        )
    ).eval()


@pytest.mark.parametrize("space", ["cortical", "cartesian"])
@pytest.mark.parametrize("training", [False, True])
def test_uniform_positions_match_native_dino(space: str, training: bool) -> None:
    model = make_model()
    model.train(training)
    original = copy.deepcopy(model)
    coords = SamplingCoords(16, None, 32, style="uniform_as_grid")
    configure_dinov3_positions(
        model, sensor_coords=coords, patch_size=8, position_coordinate_space=space
    )
    inputs = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        torch.manual_seed(321)
        actual = model(inputs).last_hidden_state
        torch.manual_seed(321)
        expected = original(inputs).last_hidden_state
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert model.config.position_coordinate_space == space
    assert model.config.fovi_sensor["style"] == "uniform_as_grid"


def test_cartesian_positions_follow_upright_warp() -> None:
    model = make_model()
    coords = SamplingCoords(16, 0.576336, 32, style="warped_cartesian_as_grid")
    configure_dinov3_positions(
        model, sensor_coords=coords, patch_size=8, position_coordinate_space="cartesian"
    )
    patches = coords.clone(resolution=4)
    expected = patches.cartesian_rowcol.reshape(4, 4, 2).transpose(0, 1).flip(0)
    torch.testing.assert_close(model.rope_embeddings.coords, expected.reshape(-1, 2))
    assert "coords" in dict(model.rope_embeddings.named_buffers())
    assert "rope_embeddings.coords" not in model.state_dict()


@pytest.mark.parametrize("space", ["cortical", "cartesian"])
def test_rectangular_dense_positions_follow_image_shape(space: str) -> None:
    model = make_model()
    coords = SamplingCoords((24, 40), None, 31, style="uniform_as_grid")
    assert coords.grid_shape == (24, 40)
    configure_dinov3_positions(
        model, sensor_coords=coords, patch_size=8, position_coordinate_space=space
    )
    with torch.no_grad():
        output = model(torch.randn(1, 3, 24, 40)).last_hidden_state
    assert output.shape[1] == 1 + 3 * 5
    assert model.config.fovi_sensor["resolution"] == 31


@pytest.mark.parametrize(
    "patch_size,space,match",
    [(7, "cortical", "divisible"), (8, "unknown", "position_coordinate_space")],
)
def test_invalid_dense_position_config_fails(
    patch_size: int, space: str, match: str
) -> None:
    coords = SamplingCoords(16, 0.5, 32, style="warped_cartesian_as_grid")
    with pytest.raises(ValueError, match=match):
        configure_dinov3_positions(
            make_model(),
            sensor_coords=coords,
            patch_size=patch_size,
            position_coordinate_space=space,
        )


@pytest.mark.parametrize("fraction", [1e6, 1e12, 1e16])
@pytest.mark.parametrize("coverage", ["circular", "square", "wang"])
def test_large_a_converges_to_uniform(fraction: float, coverage: str) -> None:
    coords = SamplingCoords(
        16, 16 * fraction, 16, style="warped_cartesian_as_grid", fov_type=coverage
    )
    torch.testing.assert_close(coords.cartesian, coords.plotting, atol=3e-7, rtol=3e-7)


def test_grid_ordering_handles_masks_coordinates_and_images() -> None:
    coords = SamplingCoords(16, 0.5, 8, style="warped_cartesian_as_grid")
    xy = coords.as_grid(coords.cartesian, sample_dim=0)
    assert xy.shape == (8, 8, 2)
    assert xy[0, 0, 0] < 0 and xy[0, 0, 1] > 0
    image = coords.as_grid(coords.cartesian.T[None])
    torch.testing.assert_close(image[0].permute(1, 2, 0), xy)
    mask = coords.as_grid(coords.valid_mask)
    assert mask.shape == (8, 8)


@pytest.mark.parametrize("geometry", ["planar", "spherical"])
@pytest.mark.parametrize("coverage", ["circular", "square", "wang"])
def test_analytic_native_map_roundtrip(geometry: str, coverage: str) -> None:
    coords = SamplingCoords(
        120,
        120 * 0.036021,
        32,
        style="warped_cartesian_as_grid",
        fov_type=coverage,
        field_geometry=geometry,
    )
    native = torch.tensor([[[0.0, 0.0], [0.5, -0.75], [1.1, 1.2]]], requires_grad=True)
    visual = coords.native_to_visual(native)
    reconstructed = coords.visual_to_native(visual)
    torch.testing.assert_close(reconstructed, native)
    gradient = torch.autograd.grad(reconstructed.sum(), native)[0]
    torch.testing.assert_close(gradient, torch.ones_like(gradient))
    torch.testing.assert_close(coords.cortical, coords.plotting)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("backend", ["torch", "cuda"])
def test_spherical_grid_sampling_masks_unbounded_corners(
    device: str, backend: str
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    retina = RetinalTransform(
        32,
        fov=150,
        cmf_a=150 * 0.01,
        style="warped_cartesian_as_grid",
        field_geometry="spherical",
        sampler_backend=backend,
        device=device,
        auto_match_cart_resources=False,
        camera_model=CameraModel("fisheye", (128, 128), (40, 40, 63.5, 63.5)),
    )
    coords = retina.sampler.coords
    assert (coords.polar[:, 0] * 75).max() > 180
    images = torch.rand(2, 3, 128, 128, device=device)
    fixations = torch.tensor([[0.5, 0.5], [0.45, 0.55]], device=device)
    if device == "cpu" and backend == "cuda":
        with pytest.raises(
            RuntimeError,
            match="CUDA calibrated sampling requires supported CUDA inputs",
        ):
            retina(images, fixations)
        return
    output = retina(images, fixations)
    assert retina.sampler.last_backend == f"{backend}_calibrated"
    assert output.shape == (2, 3, 32, 32)
    assert torch.isfinite(output).all()
    mask = coords.as_grid(coords.valid_mask)
    assert torch.count_nonzero(output[:, :, ~mask]) == 0
    _, projected_validity = retina.sampler.calibrated_pixels(fixations)
    assert not projected_validity[:, ~coords.valid_mask].any()


def test_cartesian_rope_move_preserves_coordinate_precision() -> None:
    model = make_model()
    coords = SamplingCoords(16, 0.5, 32, style="warped_cartesian_as_grid")
    configure_dinov3_positions(
        model, sensor_coords=coords, patch_size=8, position_coordinate_space="cartesian"
    )
    expected = model.rope_embeddings.coords.clone()
    model.bfloat16()
    torch.testing.assert_close(model.rope_embeddings.coords, expected, atol=0, rtol=0)


def test_saved_position_space_is_preserved_unless_overridden() -> None:
    model = make_model()
    coords = SamplingCoords(16, 0.5, 32, style="warped_cartesian_as_grid")
    configure_dinov3_positions(
        model, sensor_coords=coords, patch_size=8, position_coordinate_space="cartesian"
    )
    config = DINOv3ViTConfig.from_dict(model.config.to_dict())
    restored = DINOv3ViTModel(config)
    configure_dinov3_positions(restored, sensor_coords=coords, patch_size=8)
    assert restored.config.position_coordinate_space == "cartesian"
    torch.testing.assert_close(
        restored.rope_embeddings.coords, model.rope_embeddings.coords
    )
    configure_dinov3_positions(
        restored,
        sensor_coords=coords,
        patch_size=8,
        position_coordinate_space="cortical",
    )
    assert restored.config.position_coordinate_space == "cortical"


def test_wrong_patch_convolution_fails_before_config_mutation() -> None:
    model = make_model()
    original_config = model.config.to_dict()
    coords = SamplingCoords(16, 0.5, 32, style="warped_cartesian_as_grid")
    with pytest.raises(ValueError, match="kernel and stride"):
        configure_dinov3_positions(model, sensor_coords=coords, patch_size=16)
    assert model.config.to_dict() == original_config


def test_spherical_wang_rejects_active_samples_beyond_angular_chart() -> None:
    with pytest.raises(ValueError, match="180 degrees eccentricity"):
        SamplingCoords(
            150,
            1.5,
            32,
            style="warped_cartesian_as_grid",
            field_geometry="spherical",
            fov_type="wang",
        )
