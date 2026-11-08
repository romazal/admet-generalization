"""Tests for the aggregation tables.

The one covered hardest is the path column of ``runs.csv``. That file is
committed, so an absolute path in it would bake one machine's home directory
into the repository and guarantee the file differs for everyone who
regenerates it.
"""

from __future__ import annotations

import json
from pathlib import Path

from admet.analysis.aggregate import aggregate, load_results


def _write_run(
    directory: Path,
    *,
    dataset: str = "tox21",
    model: str = "rf",
    split: str = "random",
    seed: int = 0,
) -> Path:
    name = f"{dataset}_{model}_{split}_frac100_seed{seed}"
    payload = {
        "run": name,
        "dataset": dataset,
        "model": model,
        "split": split,
        "seed": seed,
        "train_fraction": 1.0,
        "metrics": {"test": {"roc_auc": 0.8, "pr_auc": 0.4}},
        "split_sizes": {"train": 100, "valid": 20, "test": 20},
    }
    into = directory / dataset
    into.mkdir(parents=True, exist_ok=True)
    path = into / f"{name}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_run_path_is_relative_and_posix(tmp_path: Path) -> None:
    """The committed runs.csv must not contain a machine-specific path."""
    _write_run(tmp_path)
    (record,) = load_results(tmp_path)

    assert record.path == "tox21/tox21_rf_random_frac100_seed0.json"
    assert not Path(record.path).is_absolute()
    assert "\\" not in record.path


def test_paths_do_not_leak_the_results_directory(tmp_path: Path) -> None:
    """Positive control: the directory name must appear nowhere in the output."""
    marker = tmp_path / "a-very-distinctive-directory-name"
    marker.mkdir()
    _write_run(marker)

    records = load_results(marker)
    assert records
    for record in records:
        assert "a-very-distinctive-directory-name" not in record.path
        assert str(marker) not in record.path


def test_runs_csv_contains_no_absolute_paths(tmp_path: Path) -> None:
    _write_run(tmp_path, seed=0)
    _write_run(tmp_path, seed=1)
    aggregate(tmp_path, write=True)

    text = (tmp_path / "runs.csv").read_text(encoding="utf-8")
    assert str(tmp_path) not in text
    assert "tox21/tox21_rf_random_frac100_seed0.json" in text


def test_results_are_found_in_dataset_subdirectories(tmp_path: Path) -> None:
    """The per-dataset layout must not hide runs from the aggregation."""
    _write_run(tmp_path, dataset="tox21")
    _write_run(tmp_path, dataset="clintox")

    records = load_results(tmp_path)
    assert {r.dataset for r in records} == {"tox21", "clintox"}


def test_a_flat_file_is_still_picked_up(tmp_path: Path) -> None:
    """Files written by an older version sat at the top level."""
    _write_run(tmp_path)
    flat = tmp_path / "tox21_rf_scaffold_frac100_seed0.json"
    flat.write_text(
        json.dumps(
            {
                "run": "flat",
                "dataset": "tox21",
                "model": "rf",
                "split": "scaffold",
                "seed": 0,
                "train_fraction": 1.0,
                "metrics": {"test": {"roc_auc": 0.7, "pr_auc": 0.3}},
                "split_sizes": {"train": 100, "valid": 20, "test": 20},
            }
        ),
        encoding="utf-8",
    )
    assert len(load_results(tmp_path)) == 2
