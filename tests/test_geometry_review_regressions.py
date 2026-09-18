"""Configuration boundaries must preserve the selected geometry and calibration."""

from __future__ import annotations

import copy
import io
from dataclasses import asdict

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from fovi.arch.knn import get_in_out_coords
from fovi.models.architectures import rescale_fov
from fovi.sensing.coords import SamplingCoords
from fovi.sensing.projection import CameraModel, gaze_rotation
from fovi.sensing.retina import RetinalTransform
from fovi.sensing.samplers import (
    BaseGridSampler,
    GaussianKNNGridSampler,
    GridSampler,
    KNNGridSampler,
)


def test_yaml_camera_compiles_with_fullgraph() -> None:
    reference = CameraModel(
        "fisheye",
        (80, 120),
        (75, 75, 59.5, 39.5),
        (0.01, 0.001, 0.0, 0.0),
        (59.5, 39.5, 65.0),
    )
    camera = CameraModel.from_config(OmegaConf.create(asdict(reference)))
    rays = torch.tensor([[0.1, 0.2, 1.0], [0.0, 0.0, 1.0]])
    actual = torch.compile(camera.project, backend="eager", fullgraph=True)(rays)
    expected = reference.project(rays)
    for value, target in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, target)
    assert hash(camera) == hash(reference)


def test_yaml_sampler_fullgraph_preserves_fixation_gradients() -> None:
    camera = CameraModel("pinhole", (40, 60), (40, 40, 29.5, 19.5))
    sampler = GridSampler(
        20,
        0.5,
        6,
        device="cpu",
        mode="bilinear",
        field_geometry="spherical",
        camera_model=OmegaConf.create(asdict(camera)),
    )
    image = torch.arange(60, dtype=torch.float32)[None, None, None].expand(1, 1, 40, 60)
    fixation = torch.tensor([[0.4, 0.6]], requires_grad=True)
    compiled = torch.compile(
        sampler.calibrated_forward, backend="eager", fullgraph=True
    )
    actual, pixels = compiled(image, fixation, None)
    expected, expected_pixels = sampler.calibrated_forward(image, fixation, None)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(pixels, expected_pixels)
    gradient = torch.autograd.grad(actual.sum(), fixation, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected.sum(), fixation)[0]
    torch.testing.assert_close(gradient, expected_gradient)
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


def test_direct_config_builder_requires_resolved_geometry() -> None:
    cfg = OmegaConf.create({"saccades": {}})
    with pytest.raises(AttributeError, match="field_geometry"):
        rescale_fov(cfg)


@pytest.mark.parametrize("geometry", ["legacy", "spherical"])
def test_supplied_coordinates_must_match_geometry(geometry: str) -> None:
    coords = SamplingCoords(16, 0.5, 8, device="cpu", field_geometry=geometry)
    with pytest.raises(ValueError, match="field_geometry"):
        get_in_out_coords(
            8,
            16,
            0.5,
            2,
            in_coords=coords,
            device="cpu",
            auto_match_cart_resources=False,
            field_geometry="planar",
        )
    actual, output, _ = get_in_out_coords(
        8,
        16,
        0.5,
        2,
        in_coords=coords,
        device="cpu",
        auto_match_cart_resources=False,
        field_geometry=geometry,
    )
    assert actual is coords
    assert output.field_geometry == geometry


def test_planar_calibrated_pixels_reports_geometry_precondition() -> None:
    sampler = GridSampler(16, 0.5, 8, device="cpu")
    with pytest.raises(ValueError, match="requires spherical field_geometry"):
        sampler.calibrated_pixels(torch.tensor([[0.5, 0.5]]))


@pytest.mark.parametrize("geometry", ["planar", "legacy"])
@pytest.mark.parametrize("pooling", ["pooling", "gaussian_pooling"])
def test_retinal_pooling_accepts_model_camera_defaults(
    geometry: str, pooling: str
) -> None:
    kwargs = {"gauss_sigma": 1.0} if pooling == "gaussian_pooling" else {}
    transform = RetinalTransform(
        8,
        start_res=32,
        fixation_size=32,
        device="cpu",
        auto_match_cart_resources=False,
        sampler=pooling,
        sampler_backend="torch",
        field_geometry=geometry,
        camera_model=None,
        gaze_convention="camera_xyz",
        **kwargs,
    )
    image = torch.ones(1, 3, 32, 32)
    actual = transform(image, torch.tensor([[0.5, 0.5]]))
    assert actual.shape == (1, 3, len(transform.sampler.coords))
    assert torch.isfinite(actual).all()
    assert actual.max() > 0
    assert transform.sampler.coords.field_geometry == geometry


def test_mode_change_rejects_integer_bilinear_output() -> None:
    sampler = GridSampler(16, 0.5, 8, device="cpu", output_dtype=torch.uint8)
    with pytest.raises(
        ValueError, match="bilinear sampling requires a floating output dtype"
    ):
        sampler.mode = "bilinear"
    assert sampler.mode == "nearest"


@pytest.mark.parametrize(
    "field,value",
    [
        ("intrinsics", ("75", 75, 59.5, 39.5)),
        ("distortion", (b"0", 0, 0, 0)),
        ("image_circle", (59.5, 39.5, "65")),
        ("image_size", ("80", 120)),
        ("max_angle_deg", "80"),
    ],
)
def test_camera_rejects_string_calibration(
    field: str, value: str | tuple[str | bytes | float, ...]
) -> None:
    config = asdict(CameraModel("pinhole", (80, 120), (75, 75, 59.5, 39.5)))
    config[field] = value
    with pytest.raises(TypeError, match=field):
        CameraModel.from_config(config)


def test_camera_accepts_numpy_and_tensor_real_scalars() -> None:
    camera = CameraModel(
        "pinhole",
        (np.int64(80), torch.tensor(120)),
        (np.float32(75), torch.tensor(75.0), 59.5, 39.5),
        max_angle_deg=torch.tensor(80.0),
    )
    assert camera == CameraModel(
        "pinhole", (80, 120), (75, 75, 59.5, 39.5), max_angle_deg=80
    )


@pytest.mark.parametrize("geometry", ["planar", "legacy", "spherical"])
def test_output_dtype_and_mode_changes_are_atomic(geometry: str) -> None:
    camera = CameraModel("fisheye", (32, 32), (30, 30, 15.5, 15.5))
    sampler = GridSampler(
        16,
        0.5,
        8,
        device="cpu",
        mode="bilinear",
        backend="torch",
        field_geometry=geometry,
        camera_model=camera if geometry == "spherical" else None,
    )
    image = torch.arange(32, dtype=torch.uint8).view(1, 1, 1, 32).expand(1, 1, 32, 32)
    args = (
        (image, [0.43, 0.57]) if geometry == "spherical" else (image, [0.43, 0.57], 24)
    )
    before = sampler(*args)
    assert (before != before.round()).any()
    with pytest.raises(ValueError, match="bilinear.*floating output dtype"):
        sampler.output_dtype = torch.uint8
    assert sampler.output_dtype is None
    torch.testing.assert_close(sampler(*args), before, rtol=0, atol=0)
    sampler.output_dtype = torch.float64
    torch.testing.assert_close(sampler(*args), before.double(), rtol=0, atol=0)
    sampler.output_dtype = None
    sampler.mode = "nearest"
    sampler.output_dtype = torch.uint8
    nearest = sampler(*args)
    with pytest.raises(ValueError, match="bilinear.*floating output dtype"):
        sampler.mode = "bilinear"
    assert sampler.mode == "nearest"
    torch.testing.assert_close(sampler(*args), nearest, rtol=0, atol=0)


@pytest.mark.parametrize(
    "sampler", ["grid_nn", "grid_bilinear", "pooling", "gaussian_pooling"]
)
@pytest.mark.parametrize("geometry", ["planar", "legacy"])
def test_all_retinal_samplers_validate_gaze_convention(
    sampler: str, geometry: str
) -> None:
    with pytest.raises(ValueError, match="Unknown gaze convention.*bogus"):
        RetinalTransform(
            8,
            device="cpu",
            sampler=sampler,
            field_geometry=geometry,
            gaze_convention="bogus",
            auto_match_cart_resources=False,
            **({"gauss_sigma": 1.0} if sampler == "gaussian_pooling" else {}),
        )


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
@pytest.mark.parametrize("axis", [0, 1])
def test_image_dimensions_reject_nonfinite_values(value: float, axis: int) -> None:
    size = [80.0, 120.0]
    size[axis] = value
    with pytest.raises(ValueError, match="image_size.*finite"):
        CameraModel("pinhole", size, (75, 75, 59.5, 39.5))


@pytest.mark.parametrize(
    "field", ["image_size", "intrinsics", "distortion", "image_circle"]
)
@pytest.mark.parametrize("kind", ["float", "tensor", "numpy", "none"])
def test_calibration_sequence_errors_name_field(field: str, kind: str) -> None:
    config = asdict(CameraModel("pinhole", (80, 120), (75, 75, 59.5, 39.5)))
    config[field] = {
        "float": 0.0,
        "tensor": torch.tensor(0.0),
        "numpy": np.array(0.0),
        "none": None,
    }[kind]
    if field == "image_circle" and kind == "none":
        assert CameraModel.from_config(config).image_circle is None
        return
    with pytest.raises(TypeError, match=field):
        CameraModel.from_config(config)


def test_sampler_output_dtype_serialization_preserves_mutation_validation() -> None:
    sampler = GridSampler(
        16, 0.5, 8, device="cpu", mode="bilinear", output_dtype=torch.float64
    )
    stream = io.BytesIO()
    torch.save(sampler, stream)
    stream.seek(0)
    for restored in (copy.deepcopy(sampler), torch.load(stream, weights_only=False)):
        assert restored.output_dtype == torch.float64
        with pytest.raises(ValueError, match="bilinear.*floating output dtype"):
            restored.output_dtype = torch.uint8


@pytest.mark.parametrize("mode", ["nearest", "bilinear"])
def test_invalid_output_dtype_type_is_rejected_at_both_boundaries(mode: str) -> None:
    with pytest.raises(TypeError, match="output_dtype"):
        GridSampler(16, 0.5, 8, device="cpu", mode=mode, output_dtype="float32")
    sampler = GridSampler(16, 0.5, 8, device="cpu", mode=mode)
    with pytest.raises(TypeError, match="output_dtype"):
        sampler.output_dtype = "float32"
    assert sampler.output_dtype is None


@pytest.mark.parametrize("kind", ["numpy", "tensor", "tuple", "list", "yaml"])
def test_camera_sequence_normalization_preserves_projection(kind: str) -> None:
    camera = CameraModel(
        "fisheye",
        (80, 120),
        (75, 75, 59.5, 39.5),
        (0.01, 0.001, 0, 0),
        (59.5, 39.5, 65),
    )
    config = asdict(camera)
    if kind == "yaml":
        config = OmegaConf.create(config)
    else:
        convert = {
            "numpy": np.array,
            "tensor": lambda x: torch.tensor(x, dtype=torch.float64),
            "tuple": tuple,
            "list": list,
        }[kind]
        for field in ("image_size", "intrinsics", "distortion", "image_circle"):
            config[field] = convert(config[field])
    actual = CameraModel.from_config(config)
    assert actual == camera
    assert hash(actual) == hash(camera)
    rays = torch.tensor([[0.1, -0.2, 1.0], [0.0, 0.0, 1.0]])
    for value, expected in zip(actual.project(rays), camera.project(rays), strict=True):
        torch.testing.assert_close(value, expected, rtol=0, atol=0)


@pytest.mark.parametrize("gaussian", [False, True])
def test_pooling_dtype_mutations_preserve_values_and_validation(gaussian: bool) -> None:
    cls = GaussianKNNGridSampler if gaussian else KNNGridSampler
    kwargs = {"gauss_sigma": 1.0} if gaussian else {}
    sampler = cls(16, 0.5, 8, device="cpu", backend="torch", **kwargs)
    image = torch.rand(1, 3, 32, 32)
    args = (image, [0.4, 0.6], 24)
    reference = sampler(*args)
    for invalid, error in ((torch.uint8, ValueError), ("float32", TypeError)):
        with pytest.raises(error, match="output dtype|output_dtype"):
            sampler.output_dtype = invalid
        assert sampler.output_dtype is None
        torch.testing.assert_close(sampler(*args), reference, rtol=0, atol=0)
        with pytest.raises(error, match="output dtype|output_dtype"):
            cls(16, 0.5, 8, device="cpu", output_dtype=invalid, **kwargs)
    sampler.output_dtype = torch.float64
    torch.testing.assert_close(sampler(*args), reference.double(), rtol=0, atol=0)
    stream = io.BytesIO()
    torch.save(sampler, stream)
    stream.seek(0)
    for restored in (copy.deepcopy(sampler), torch.load(stream, weights_only=False)):
        assert restored.output_dtype == torch.float64
        with pytest.raises(ValueError, match="floating output dtype"):
            restored.output_dtype = torch.uint8
    sampler.output_dtype = None
    torch.testing.assert_close(sampler(*args), reference, rtol=0, atol=0)


@pytest.mark.parametrize("pooling", [False, True])
def test_missing_serialized_output_dtype_uses_attribute_error(pooling: bool) -> None:
    cls = KNNGridSampler if pooling else GridSampler
    sampler = cls(16, 0.5, 8, device="cpu")
    del sampler.__dict__["output_dtype"]
    assert not hasattr(sampler, "output_dtype")
    assert getattr(sampler, "output_dtype", None) is None
    with pytest.raises(AttributeError, match="output_dtype"):
        _ = sampler.output_dtype


def test_rotation_rejects_unknown_gaze_convention() -> None:
    with pytest.raises(ValueError, match="Unknown gaze convention.*bogus"):
        gaze_rotation(torch.tensor([[0.0, 0.0, 1.0]]), "bogus")


def test_base_sampler_output_dtype_remains_assignable() -> None:
    sampler = BaseGridSampler()
    for dtype in (torch.uint8, torch.float32, None):
        sampler.output_dtype = dtype
        assert sampler.output_dtype == dtype
