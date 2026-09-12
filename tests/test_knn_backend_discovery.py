import importlib.util
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from fovi.arch.knn_optimization import select_backend


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_inference_backend_selection_does_not_repeat_package_discovery(
    dtype: torch.dtype,
) -> None:
    layer = SimpleNamespace(
        kernel_backend="auto",
        in_channels=3,
        _k=121,
        local_rf=torch.empty(1, 1, 121),
        knn_indices_pad_token=torch.empty(121, 256),
    )
    value = SimpleNamespace(
        is_cuda=True,
        device=torch.device("cuda"),
        dtype=dtype,
        shape=(32, 3, 1000),
    )
    with (
        torch.inference_mode(),
        mock.patch.object(torch.cuda, "get_device_capability", return_value=(12, 0)),
    ):
        expected = select_backend(layer, value)
        # Import hooks may retain caller frames; no lookup may capture a live batch.
        with mock.patch.object(
            importlib.util,
            "find_spec",
            side_effect=AssertionError("Package discovery reached the inference path"),
        ):
            for _ in range(3):
                assert select_backend(layer, value) == expected
