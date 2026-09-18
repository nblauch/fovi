"""Shared validation for sampling configuration boundaries."""

from __future__ import annotations

import torch


def validate_gaze_convention(convention: str) -> None:
    """Reject unsupported gaze conventions."""
    if convention not in ("camera_xyz", "pan_tilt"):
        raise ValueError(f"Unknown gaze convention {convention!r}")


def validate_sampling_mode(mode: str) -> None:
    """Reject unsupported interpolation modes."""
    if mode not in ("nearest", "bilinear"):
        raise ValueError(f"Unsupported sampling mode {mode!r}")


def validate_output_dtype(
    output_dtype: torch.dtype | None, floating_operation: str | None = None
) -> None:
    """Validate an optional output cast and an operation's floating requirement."""
    if output_dtype is not None:
        if not isinstance(output_dtype, torch.dtype):
            raise TypeError("output_dtype must be a torch.dtype or None")
        if floating_operation is not None and not output_dtype.is_floating_point:
            raise ValueError(f"{floating_operation} requires a floating output dtype")
