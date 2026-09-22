"""Counting and resolution matching enforce the coordinate radius-norm contract."""

from __future__ import annotations

import math

import pytest

from fovi.sensing.coords import SamplingCoords, find_desired_res, num_sampling_coords


@pytest.mark.parametrize("find_resolution", [False, True], ids=["count", "match"])
@pytest.mark.parametrize(
    "kwargs",
    [
        {"style": "warped_cartesian_as_grid", "radius_norm": 1.0},
        {"style": "warped_cartesian_as_grid", "radius_norm": math.nan},
        {"style": "warped_cartesian_as_grid", "radius_norm": "inf"},
        {"style": "warped_cartesian", "radius_norm": math.inf, "fov_type": "wang"},
        {
            "style": "warped_cartesian_as_grid",
            "radius_norm": math.inf,
            "fov_type": "wang",
        },
        {"style": "isotropic", "radius_norm": math.inf},
        {"style": "isotropic_fixn", "radius_norm": math.inf},
        {"style": "logpolar", "radius_norm": math.inf},
        {"style": "logpolar_as_grid", "radius_norm": math.inf},
        {"style": "uniform", "radius_norm": math.inf},
        {"style": "uniform_as_grid", "radius_norm": math.inf},
    ],
)
def test_invalid_radius_norm_rejected(
    kwargs: dict[str, float | str], find_resolution: bool
) -> None:
    with pytest.raises(ValueError, match="radius_norm"):
        SamplingCoords(16.0, 0.5, 16, **kwargs)
    with pytest.raises(ValueError, match="radius_norm"):
        if find_resolution:
            find_desired_res(16.0, 0.5, 256, quiet=True, **kwargs)
        else:
            num_sampling_coords(16.0, 0.5, 16, **kwargs)


@pytest.mark.parametrize("style", ["warped_cartesian", "warped_cartesian_as_grid"])
@pytest.mark.parametrize(
    ("radius_norm", "fov_type"),
    [
        (2.0, "circular"),
        (2.0, "square"),
        (2.0, "wang"),
        (math.inf, "circular"),
        (math.inf, "square"),
    ],
)
def test_supported_radius_norm_preserves_count_and_resolution(
    style: str, radius_norm: float, fov_type: str
) -> None:
    kwargs = {"style": style, "radius_norm": radius_norm, "fov_type": fov_type}
    coords = SamplingCoords(16.0, 0.5, 16, **kwargs)
    assert num_sampling_coords(16.0, 0.5, 16, **kwargs) == len(coords) == 256
    assert find_desired_res(16.0, 0.5, 256, quiet=True, **kwargs) == (16, 256)
