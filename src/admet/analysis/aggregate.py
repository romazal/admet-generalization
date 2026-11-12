"""Turn per-run result files into the summary tables the report quotes.

``results/`` holds one JSON file per run — one model, one split, one training
fraction, one seed. Nothing in the report is read off a single run: every
number is a mean and standard deviation over seeds, and the quantity both
hypotheses are stated in terms of is the **GNN-minus-baseline gap** per
condition (proposal §3.3).

This module is the only thing that reads those files, and everything
downstream (``plots.py``, the README tables) reads what it produces, so no
number in the report can come from anywhere but ``results/``.

Two conventions worth knowing:

*Standard deviations are sample standard deviations* (``ddof=1``), because the
seeds are a sample of the possible runs rather than the population of them.
With a single seed the deviation is undefined and reported as ``None``, not as
zero — zero would read as "perfectly reproducible" when it means "not measured".

*The gap's uncertainty is pooled, not differenced.* The GNN and baseline runs
at a given seed share a split, but they are separate fits, so the gap's
standard deviation is combined in quadrature from the two arms.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import pandas as pd

from admet.evaluation import METRICS

__all__ = [
    "RunRecord",
    "ConditionSummary",
    "GapRow",
    "PerTaskRow",
    "load_results",
    "summarize",
    "gap_table",
    "per_task_table",
    "write_csv",
    "aggregate",
]


# --------------------------------------------------------------------------- #
# Containers
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RunRecord:
    """The handful of fields the summary needs out of one result file."""

    run: str
    dataset: str
    model: str
    split: str
    seed: int
    train_fraction: float
    test_roc_auc: float | None
    test_pr_auc: float | None
    n_train: int
    #: Carried for the same reason as the other two: a reader checking that a
    #: learning-curve point varied only the *training* set needs to see that
    #: validation and test did not move with it.
    n_valid: int
    n_test: int
    #: Location of the result file *relative to the results directory*, with
    #: forward slashes — e.g. ``tox21/tox21_rf_random_frac100_seed0.json``.
    #: It is never absolute: runs.csv is committed, so an absolute path would
    #: bake one machine's home directory into the repository and guarantee
    #: the file differs for everyone who regenerates it.
    path: str

    @property
    def condition(self) -> tuple[str, str, str, float]:
        """Everything but the seed — runs sharing this are averaged together."""
        return (self.dataset, self.model, self.split, self.train_fraction)


@dataclass(frozen=True)
class ConditionSummary:
    """Mean and standard deviation over seeds for one condition."""

    dataset: str
    model: str
    split: str
    train_fraction: float
    n_seeds: int
    roc_auc_mean: float | None
    roc_auc_std: float | None
    pr_auc_mean: float | None
    pr_auc_std: float | None
    n_train: int

    @property
    def key(self) -> tuple[str, str, str, float]:
        """``(dataset, model, split, train_fraction)``."""
        return (self.dataset, self.model, self.split, self.train_fraction)


@dataclass(frozen=True)
class GapRow:
    """GNN-minus-baseline, the quantity H1 and H2 are stated in."""

    dataset: str
    split: str
    train_fraction: float
    metric: str
    baseline_mean: float | None
    gnn_mean: float | None
    gap: float | None
    gap_std: float | None
    n_seeds: int


@dataclass(frozen=True)
class PerTaskRow:
    """Mean and standard deviation for one assay under one condition."""

    dataset: str
    model: str
    split: str
    train_fraction: float
    task: str
    n_seeds: int
    roc_auc_mean: float | None
    roc_auc_std: float | None
    pr_auc_mean: float | None
    pr_auc_std: float | None


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _std(values: Sequence[float]) -> float | None:
    """Sample standard deviation, or ``None`` for fewer than two values."""
    if len(values) < 2:
        return None
    average = sum(values) / len(values)
    variance = sum((v - average) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def _relative_path(path: Path, directory: Path) -> str:
    """``path`` relative to ``directory``, POSIX-style.

    Falls back to the bare filename if the two are unrelated, which can only
    happen if a caller passes a file from outside the results tree. Forward
    slashes regardless of platform, so the committed runs.csv is identical on
    Windows and Linux.
    """
    try:
        return path.resolve().relative_to(directory.resolve()).as_posix()
    except ValueError:
        return path.name


def load_results(results_dir: str | Path) -> list[RunRecord]:
    """Read every ``*.json`` result file under ``results_dir``, recursively.

    Runs are written to a per-dataset subdirectory (``results/tox21/…``), so
    the search recurses. It stays recursive rather than listing the known
    datasets, so a new dataset needs no change here, and files left in the
    top level by an older version are still picked up.

    Files that are not run results — anything missing the expected keys — are
    skipped rather than raising, so a stray JSON file in the directory cannot
    break the aggregation. A malformed file that *is* a result raises.

    Returns:
        One record per run, sorted by condition and then seed.
    """
    directory = Path(results_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"results directory not found: {directory}")

    records: list[RunRecord] = []
    for path in sorted(directory.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} is not valid JSON: {exc}") from exc
        if not isinstance(payload, dict) or "metrics" not in payload:
            continue

        test = (payload.get("metrics") or {}).get("test") or {}
        sizes = payload.get("split_sizes") or {}
        records.append(
            RunRecord(
                run=payload.get("run", path.stem),
                dataset=payload["dataset"],
                model=payload["model"],
                split=payload["split"],
                seed=int(payload["seed"]),
                train_fraction=float(payload["train_fraction"]),
                test_roc_auc=test.get("roc_auc"),
                test_pr_auc=test.get("pr_auc"),
                n_train=int(sizes.get("train", 0)),
                n_valid=int(sizes.get("valid", 0)),
                n_test=int(sizes.get("test", 0)),
                path=_relative_path(path, directory),
            )
        )
    records.sort(key=lambda r: (*r.condition, r.seed))
    return records


# --------------------------------------------------------------------------- #
# Summaries
# --------------------------------------------------------------------------- #


def summarize(records: Iterable[RunRecord]) -> list[ConditionSummary]:
    """Collapse runs to one row per condition, averaging over seeds."""
    grouped: dict[tuple[str, str, str, float], list[RunRecord]] = {}
    for record in records:
        grouped.setdefault(record.condition, []).append(record)

    summaries: list[ConditionSummary] = []
    for (dataset, model, split, fraction), runs in sorted(grouped.items()):
        roc = [r.test_roc_auc for r in runs if r.test_roc_auc is not None]
        pr = [r.test_pr_auc for r in runs if r.test_pr_auc is not None]
        summaries.append(
            ConditionSummary(
                dataset=dataset,
                model=model,
                split=split,
                train_fraction=fraction,
                n_seeds=len(runs),
                roc_auc_mean=_mean(roc),
                roc_auc_std=_std(roc),
                pr_auc_mean=_mean(pr),
                pr_auc_std=_std(pr),
                n_train=max((r.n_train for r in runs), default=0),
            )
        )
    return summaries


def gap_table(
    summaries: Iterable[ConditionSummary],
    *,
    baseline: str = "rf",
    advanced: str = "gnn",
    metrics: Sequence[str] = METRICS,
) -> list[GapRow]:
    """Build the GNN-minus-baseline table, one row per condition and metric.

    A condition with only one of the two models present is skipped: there is
    no gap to report, and emitting a half-filled row invites it being read as
    a zero gap.
    """
    by_key = {(s.dataset, s.split, s.train_fraction, s.model): s for s in summaries}
    conditions = sorted({(s.dataset, s.split, s.train_fraction) for s in summaries})

    rows: list[GapRow] = []
    for dataset, split, fraction in conditions:
        base = by_key.get((dataset, split, fraction, baseline))
        adv = by_key.get((dataset, split, fraction, advanced))
        if base is None or adv is None:
            continue

        for metric in metrics:
            base_mean = getattr(base, f"{metric}_mean")
            adv_mean = getattr(adv, f"{metric}_mean")
            base_std = getattr(base, f"{metric}_std")
            adv_std = getattr(adv, f"{metric}_std")

            gap = (
                None if base_mean is None or adv_mean is None else adv_mean - base_mean
            )
            # Separate fits, so combine the two arms' spread in quadrature.
            if base_std is None or adv_std is None:
                gap_std = None
            else:
                gap_std = math.sqrt(base_std**2 + adv_std**2)

            rows.append(
                GapRow(
                    dataset=dataset,
                    split=split,
                    train_fraction=fraction,
                    metric=metric,
                    baseline_mean=base_mean,
                    gnn_mean=adv_mean,
                    gap=gap,
                    gap_std=gap_std,
                    n_seeds=min(base.n_seeds, adv.n_seeds),
                )
            )
    return rows


def per_task_table(results_dir: str | Path) -> list[PerTaskRow]:
    """Collapse assay-level metrics across seeds to a per-condition table.

    Each run already contains a ``metrics.test.per_task`` list, so this is a
    thin aggregation over those values rather than a recomputation from raw
    predictions.
    """
    directory = Path(results_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"results directory not found: {directory}")

    grouped: dict[
        tuple[str, str, str, float, str], list[tuple[float | None, float | None]]
    ] = {}
    for path in sorted(directory.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} is not valid JSON: {exc}") from exc
        if not isinstance(payload, dict) or "metrics" not in payload:
            continue

        test = (payload.get("metrics") or {}).get("test") or {}
        per_task = test.get("per_task") or []
        if not isinstance(per_task, list):
            continue

        dataset = payload.get("dataset")
        model = payload.get("model")
        split = payload.get("split")
        train_fraction = payload.get("train_fraction")
        if not all(isinstance(value, str) for value in (dataset, model, split)):
            continue
        try:
            fraction = float(train_fraction)
        except (TypeError, ValueError):
            continue

        for entry in per_task:
            if not isinstance(entry, dict):
                continue
            task = entry.get("task")
            if not isinstance(task, str):
                continue
            key = (dataset, model, split, fraction, task)
            grouped.setdefault(key, []).append(
                (entry.get("roc_auc"), entry.get("pr_auc"))
            )

    rows: list[PerTaskRow] = []
    for (dataset, model, split, fraction, task), values in sorted(grouped.items()):
        rocs = [v for v, _ in values if isinstance(v, (int, float))]
        prs = [v for _, v in values if isinstance(v, (int, float))]
        rows.append(
            PerTaskRow(
                dataset=dataset,
                model=model,
                split=split,
                train_fraction=fraction,
                task=task,
                n_seeds=len(values),
                roc_auc_mean=_mean(rocs),
                roc_auc_std=_std(rocs),
                pr_auc_mean=_mean(prs),
                pr_auc_std=_std(prs),
            )
        )
    return rows


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #


def write_csv(rows: Sequence[Any], path: str | Path) -> Path:
    """Write a sequence of dataclass rows to ``path`` as CSV.

    An empty sequence writes nothing and returns the path, so a sweep that has
    not produced a given table yet does not leave a misleading empty file
    behind from an earlier run.
    """
    target = Path(path)
    if not rows:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    header = [f.name for f in fields(rows[0])]
    frame = pd.DataFrame([asdict(row) for row in rows], columns=header)
    # `lineterminator` is the one thing that has to be said out loud. The
    # committed tables were written by `csv.writer` through a handle opened
    # with newline="", which emits CRLF on every platform; pandas defaults to
    # os.linesep instead. Leaving it to the default would rewrite all four
    # files on a POSIX machine with no visible change, breaking the
    # regenerates-with-zero-diff property that makes the tables checkable.
    frame.to_csv(target, index=False, lineterminator="\r\n")
    return target


def aggregate(
    results_dir: str | Path, *, write: bool = True
) -> tuple[list[RunRecord], list[ConditionSummary], list[GapRow]]:
    """Load, summarize and (by default) write ``summary.csv`` and ``gap.csv``.

    Returns:
        ``(records, summaries, gaps)``.
    """
    directory = Path(results_dir)
    records = load_results(directory)
    summaries = summarize(records)
    gaps = gap_table(summaries)

    if write:
        write_csv(records, directory / "runs.csv")
        write_csv(summaries, directory / "summary.csv")
        write_csv(gaps, directory / "gap.csv")
        write_csv(per_task_table(directory), directory / "per_task.csv")

    return records, summaries, gaps
