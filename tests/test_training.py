from __future__ import annotations

from types import SimpleNamespace

import pandas
import torch

from admet.models.rf import RandomForestModel
from admet.training import _library_versions


def test_library_versions_record_pandas_and_the_random_forest_device(
    monkeypatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    versions = _library_versions(RandomForestModel(n_tasks=1))

    assert versions["pandas"] == pandas.__version__
    assert versions["torch_device"] == "cpu"


def test_library_versions_record_the_selected_cuda_device(monkeypatch) -> None:
    selected_device = torch.device("cuda:1")
    seen_devices = []

    def get_device_name(device=None):
        seen_devices.append(device)
        return "Test GPU"

    monkeypatch.setattr(torch.cuda, "get_device_name", get_device_name)
    versions = _library_versions(SimpleNamespace(device=selected_device))

    assert seen_devices == [selected_device]
    assert versions["torch_device"] == "cuda:Test GPU"
