"""Tests for the aggregation tables.

The one covered hardest is the path column of ``runs.csv``. That file is
committed, so an absolute path in it would bake one machine's home directory
into the repository and guarantee the file differs for everyone who
regenerates it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

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


# --------------------------------------------------------------------------- #
# CSV byte-for-byte stability
# --------------------------------------------------------------------------- #


def test_tables_are_written_with_crlf_and_no_index_column(tmp_path: Path) -> None:
    """The two ways the pandas writer could silently rewrite every table.

    `csv.writer` through a handle opened with newline="" emits CRLF on every
    platform; `DataFrame.to_csv` defaults to os.linesep and, left alone, also
    prepends an unnamed index column. Either would rewrite all four committed
    tables with no visible change in a text diff, so this reads bytes.
    """
    _write_run(tmp_path, seed=0)
    _write_run(tmp_path, seed=1)
    aggregate(tmp_path, write=True)

    for name in ("runs.csv", "summary.csv"):
        raw = (tmp_path / name).read_bytes()
        assert raw.count(b"\r\n") > 0, f"{name}: no CRLF line endings"
        assert raw.count(b"\n") == raw.count(b"\r\n"), f"{name}: a bare LF slipped in"
        first = raw.split(b"\r\n", 1)[0].decode("utf-8")
        assert not first.startswith(","), f"{name}: an index column was written"
        assert first.split(",")[0] in {"dataset", "run"}, f"{name}: unexpected header"


def test_the_line_terminator_is_pinned_rather_than_left_to_the_platform(
    tmp_path: Path, monkeypatch
) -> None:
    """Checks the argument, because on Windows the bytes cannot tell you.

    `to_csv` defaults to os.linesep, which *is* CRLF on Windows — so dropping
    `lineterminator="\\r\\n"` changes nothing here and every byte-level test
    above still passes. It would break on the first POSIX machine to run
    `make aggregate`, rewriting all four committed tables with no visible
    difference in a text diff. This is the only check that fails on the
    machine the project is developed on.
    """
    seen: dict[str, object] = {}
    original = pd.DataFrame.to_csv

    def spy(self, *args, **kwargs):
        seen.update(kwargs)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_csv", spy)
    _write_run(tmp_path, seed=0)
    aggregate(tmp_path, write=True)

    assert seen.get("lineterminator") == "\r\n", (
        "write_csv must pin the line terminator; the platform default would "
        "emit LF on POSIX and rewrite every committed table"
    )
    assert seen.get("index") is False


def test_regenerating_a_table_is_byte_identical(tmp_path: Path) -> None:
    """Regeneration must not change a single byte.

    The committed tables are the checkable record of the runs: if `aggregate`
    is not a pure function of the result files down to the byte, every
    teammate who regenerates gets a dirty working tree and the tables stop
    being verifiable by reproduction. Compared as bytes rather than text,
    because reading as text normalises line endings and would pass while the
    files differ.
    """
    _write_run(tmp_path, seed=0)
    _write_run(tmp_path, seed=1)

    aggregate(tmp_path, write=True)
    first = {n: (tmp_path / n).read_bytes() for n in ("runs.csv", "summary.csv")}
    aggregate(tmp_path, write=True)
    second = {n: (tmp_path / n).read_bytes() for n in ("runs.csv", "summary.csv")}

    assert first == second


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #


def test_seeding_pins_torch_to_deterministic_kernels() -> None:
    """Seeding alone does not make a GPU run reproducible.

    Measured before this was added: three runs of the same config, seed,
    machine and torch build gave test ROC-AUCs of 0.7369, 0.7413 and 0.7690.
    That 0.032 spread is wider than the 0.011 across-seed SD the report
    quotes as its uncertainty, and it moved which hyperparameters won. The
    cause is float addition order in the scatter-add and in cuBLAS, which
    seeding cannot touch.
    """
    import os

    import torch

    from admet.training import deterministic_mode, set_global_seeds

    set_global_seeds(0)
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.benchmark is False
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
    assert deterministic_mode() is True


def test_the_escape_hatch_is_recorded_rather_than_silent(monkeypatch) -> None:
    """Opting out must show up in the result file, not just in someone's shell.

    `use_deterministic_algorithms` raises on any op without a deterministic
    kernel, so a future torch could stop the pipeline dead and someone will
    need to finish a run. A result produced that way is not comparable with
    the committed ones, so it has to say so.
    """
    from admet.training import deterministic_mode

    monkeypatch.setenv("ADMET_NONDETERMINISTIC", "1")
    assert deterministic_mode() is False
