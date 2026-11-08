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

import csv
import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from admet.evaluation import METRICS

__all__ = [
    "RunRecord",
    "ConditionSummary",
    "GapRow",
    "load_results",
    "summarize",
    "gap_table",
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
    n_test: int
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


def load_results(results_dir: str | Path) -> list[RunRecord]:
    """Read every ``*.json`` result file in ``results_dir``.

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
    for path in sorted(directory.glob("*.json")):
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
                n_test=int(sizes.get("test", 0)),
                path=str(path),
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
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
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

    return records, summaries, gaps
