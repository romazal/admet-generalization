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


def test_conflicting_duplicate_labels_are_warned_about(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Merging keeps the first label and discards the other, so say so.

    `CCO` and `OCC` are the same molecule. When their labels disagree the
    loader silently keeps one, which is a judgement call being made on the
    caller's behalf; it has to be audible. On ClinTox this fires for every
    one of the 19 merged duplicates.
    """
    with pytest.warns(RuntimeWarning, match="disagreed between duplicate"):
        dataset = _load_csv(
            tmp_path, monkeypatch, "smiles,active\nCCO,1\nOCC,0\nCCN,1\n"
        )

    assert len(dataset) == 2
    assert dataset.n_duplicates == 1
    assert dataset.n_label_conflicts == 1


def test_agreeing_duplicates_are_merged_without_a_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Merging is routine; only a discarded disagreement is worth interrupting for."""
    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("error", RuntimeWarning)
        dataset = _load_csv(
            tmp_path, monkeypatch, "smiles,active\nCCO,1\nOCC,1\nCCN,0\n"
        )

    assert dataset.n_duplicates == 1
    assert dataset.n_label_conflicts == 0


def test_the_warning_separates_conflicts_from_duplicate_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two counts that are easy to confuse, and do differ.

    A duplicate row can disagree on more than one assay, so the conflict
    count is per assay and exceeds the row count. ClinTox is 19 rows and 38
    conflicts; quoting one as the other understates the problem by half.
    """
    with pytest.warns(RuntimeWarning) as record:
        dataset = _load_csv(
            tmp_path,
            monkeypatch,
            "smiles,a,b\nCCO,1,1\nOCC,0,0\nCCN,1,0\n",
            task_columns=("a", "b"),
        )

    assert dataset.n_duplicates == 1
    assert dataset.n_label_conflicts == 2
    message = str(record[0].message)
    assert "2 assay label(s)" in message
    assert "1 duplicate row(s)" in message


# --------------------------------------------------------------------------- #
# The registry itself
#
# Adding a dataset is a hand edit to `DATASETS` (TODO §3), and the mistake
# that edit invites is listing the SMILES column as a label: the model then
# trains on its own input and scores suspiciously well. These run over every
# registered dataset so a fourth one is covered the day it is added.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", sorted(DATASETS))
def test_no_dataset_lists_its_smiles_column_as_a_task(name: str) -> None:
    spec = DATASETS[name]
    if spec.task_columns is None:
        assert spec.smiles_column in spec.exclude_columns, (
            f"{name} takes every non-excluded column as a task but does not "
            f"exclude {spec.smiles_column!r}, so the model would be trained to "
            f"predict its own input"
        )
    else:
        assert spec.smiles_column not in spec.task_columns


@pytest.mark.parametrize("name", sorted(DATASETS))
def test_every_dataset_names_a_source_and_describes_itself(name: str) -> None:
    spec = DATASETS[name]
    assert spec.name == name, "registry key and spec name must agree"
    assert spec.url and spec.filename
    assert len(spec.description) > 40, "the description is what `admet datasets` prints"


def test_bbbp_is_registered_with_its_single_endpoint() -> None:
    """BBBP is the third dataset (TODO §3); its label column is `p_np`."""
    spec = DATASETS["bbbp"]
    assert spec.task_columns == ("p_np",)
    assert spec.smiles_column == "smiles"
    # `name` holds free text such as "Propanolol"; it must never be a task.
    assert "name" not in (spec.task_columns or ())
