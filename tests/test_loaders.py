"""Offline tests for dataset CSV cleaning and label handling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from admet.data.loaders import DATASETS, Dataset, DatasetSpec, load_dataset


def _load_csv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    contents: str,
    task_columns: tuple[str, ...] = ("active",),
) -> Dataset:
    """Load a tiny registered dataset without invoking the downloader."""
    spec = DatasetSpec(
        name="toy",
        url="https://example.invalid/toy.csv",
        filename="toy.csv",
        smiles_column="smiles",
        task_columns=task_columns,
    )
    (tmp_path / spec.filename).write_text(contents, encoding="utf-8")
    monkeypatch.setitem(DATASETS, spec.name, spec)
    return load_dataset(spec.name, cache_dir=tmp_path)


def test_unparseable_smiles_are_dropped_and_counted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _load_csv(
        tmp_path,
        monkeypatch,
        "smiles,active\nCCO,1\nnot-a-smiles,0\nCCN,0\n",
    )

    assert len(dataset) == 2
    assert dataset.n_unparseable == 1
    assert dataset.summary()["n_unparseable_dropped"] == 1


def test_equivalent_smiles_are_merged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _load_csv(
        tmp_path,
        monkeypatch,
        "smiles,active\nCCO,1\nOCC,1\nCCN,0\n",
    )

    assert len(dataset) == 2
    assert dataset.smiles.count("CCO") == 1
    assert dataset.n_duplicates == 1


def test_missing_label_remains_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _load_csv(
        tmp_path,
        monkeypatch,
        "smiles,active\nCCO,1\nCCN,\n",
    )

    assert dataset.y[0, 0] == 1.0
    assert np.isnan(dataset.y[1, 0])


def test_label_density_matches_observed_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _load_csv(
        tmp_path,
        monkeypatch,
        "smiles,active,other\nCCO,1,0\nCCN,1,\nCCC,,0\n",
        task_columns=("active", "other"),
    )

    assert dataset.label_density == 4 / 6
