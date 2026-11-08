"""Metrics for masked multitask binary classification.

Tox21 is a multitask dataset with **many missing labels**: a molecule was not
screened in every assay, so its label matrix is sparse. Proposal §3.4 and §6
single out the handling of those missing labels as one of the things that can
silently invalidate the comparison, because there are two wrong ways to do it
that both produce plausible-looking numbers:

* treating a missing label as a negative, which inflates the number of
  negatives and makes every model look better than it is; and
* dropping a molecule entirely when *any* of its labels is missing, which on
  Tox21 would discard most of the dataset.

The rule used here is the one the proposal states: a molecule with a missing
label for an assay is excluded **from that assay only**, and contributes
normally to every assay it does have a label for.

Scores are reported per assay and then **macro-averaged** — each assay counts
once, regardless of how many molecules it was measured on. An assay that cannot
be scored (no labelled molecules, or only one class present among them) is
skipped and named in the result rather than being silently counted as zero;
scoring it as zero would drag the mean down for a reason that has nothing to do
with the model. This matters under the scaffold split, where a small test part
can easily end up with no positives for a rare assay.

ROC-AUC is the primary metric and PR-AUC the secondary one; Tox21 assays have
roughly 4-12% positives, and PR-AUC is the more honest of the two under that
imbalance (proposal §3.3).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

__all__ = [
    "METRICS",
    "PRIMARY_METRIC",
    "TaskScore",
    "EvaluationResult",
    "evaluate",
]

#: Metrics computed for every assay, in report order.
METRICS: tuple[str, ...] = ("roc_auc", "pr_auc")

#: The metric H1 and H2 are judged on (proposal §3.3).
PRIMARY_METRIC = "roc_auc"


# --------------------------------------------------------------------------- #
# Containers
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TaskScore:
    """Scores for a single assay, or the reason it could not be scored."""

    task: str
    roc_auc: float | None
    pr_auc: float | None
    n_labelled: int
    n_positive: int
    skipped_reason: str | None = None

    @property
    def scored(self) -> bool:
        """Whether this assay contributed to the macro-average."""
        return self.skipped_reason is None

    @property
    def positive_rate(self) -> float | None:
        """Share of positives among the labelled molecules, or ``None``."""
        if self.n_labelled == 0:
            return None
        return self.n_positive / self.n_labelled

    def as_dict(self) -> dict[str, object]:
        """JSON-serialisable form."""
        return {
            "task": self.task,
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "n_labelled": self.n_labelled,
            "n_positive": self.n_positive,
            "positive_rate": self.positive_rate,
            "skipped_reason": self.skipped_reason,
        }


@dataclass(frozen=True)
class EvaluationResult:
    """Macro-averaged scores over assays, plus the per-assay breakdown."""

    roc_auc: float | None
    pr_auc: float | None
    per_task: tuple[TaskScore, ...] = field(default=())

    @property
    def scored_tasks(self) -> tuple[str, ...]:
        """Assays that contributed to the macro-average."""
        return tuple(t.task for t in self.per_task if t.scored)

    @property
    def skipped_tasks(self) -> dict[str, str]:
        """Assays left out of the macro-average, mapped to the reason."""
        return {
            t.task: t.skipped_reason
            for t in self.per_task
            if t.skipped_reason is not None
        }

    def __getitem__(self, metric: str) -> float | None:
        """Look a macro-average up by metric name, e.g. ``result["roc_auc"]``."""
        if metric not in METRICS:
            raise KeyError(
                f"unknown metric {metric!r}; expected one of {', '.join(METRICS)}"
            )
        return getattr(self, metric)

    def as_dict(self) -> dict[str, object]:
        """JSON-serialisable form, as stored in a result file."""
        return {
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "n_tasks_scored": len(self.scored_tasks),
            "n_tasks_skipped": len(self.skipped_tasks),
            "skipped_tasks": self.skipped_tasks,
            "per_task": [t.as_dict() for t in self.per_task],
        }


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #


def _as_2d(array: object, name: str) -> np.ndarray:
    """Coerce to a float 2-D array, promoting a 1-D single-task column."""
    values = np.asarray(array, dtype=float)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2:
        raise ValueError(f"{name} must be 1- or 2-dimensional, got {values.ndim}-D")
    return values


def evaluate(
    y_true: object,
    y_score: object,
    *,
    tasks: Sequence[str] | None = None,
) -> EvaluationResult:
    """Score predictions against masked multitask labels.

    Args:
        y_true: ``(n_molecules, n_tasks)`` labels in ``{0, 1}``, with ``NaN``
            marking a molecule that was not screened in that assay. A 1-D array
            is treated as a single task.
        y_score: ``(n_molecules, n_tasks)`` predicted scores — probabilities or
            any monotone transform of them, since both metrics are
            rank-based. ``NaN`` is not allowed here: a model owes a prediction
            for every molecule it is asked about.
        tasks: assay names, used for reporting. Defaults to ``task_0 …``.

    Returns:
        An :class:`EvaluationResult` whose ``roc_auc`` and ``pr_auc`` are
        macro-averages over the assays that could be scored, and which is
        ``None`` for a metric when **no** assay could be scored.

    Raises:
        ValueError: on shape mismatches, labels outside ``{0, 1, NaN}``, or
            non-finite predictions.
    """
    true = _as_2d(y_true, "y_true")
    score = _as_2d(y_score, "y_score")

    if true.shape != score.shape:
        raise ValueError(
            f"y_true and y_score must have the same shape, got "
            f"{true.shape} and {score.shape}"
        )
    if true.shape[0] == 0:
        raise ValueError("cannot evaluate an empty set of molecules")

    n_tasks = true.shape[1]
    if tasks is None:
        names = [f"task_{i}" for i in range(n_tasks)]
    else:
        names = list(tasks)
        if len(names) != n_tasks:
            raise ValueError(
                f"got {len(names)} task name(s) for {n_tasks} task column(s)"
            )

    if not np.all(np.isfinite(score)):
        raise ValueError("y_score contains NaN or infinity")

    observed = true[~np.isnan(true)]
    if observed.size and not np.all(np.isin(observed, (0.0, 1.0))):
        bad = np.unique(observed[~np.isin(observed, (0.0, 1.0))])
        raise ValueError(
            f"y_true must contain only 0, 1 or NaN; found {bad[:5].tolist()}"
        )

    per_task: list[TaskScore] = []
    for index, name in enumerate(names):
        column = true[:, index]
        mask = ~np.isnan(column)  # the masking rule: this assay's labels only.
        labels = column[mask]
        n_labelled = int(labels.size)
        n_positive = int(labels.sum())

        if n_labelled == 0:
            reason = "no labelled molecules"
        elif n_positive == 0:
            reason = "no positive molecules"
        elif n_positive == n_labelled:
            reason = "no negative molecules"
        else:
            reason = None

        if reason is not None:
            per_task.append(
                TaskScore(name, None, None, n_labelled, n_positive, reason)
            )
            continue

        predictions = score[mask, index]
        per_task.append(
            TaskScore(
                task=name,
                roc_auc=float(roc_auc_score(labels, predictions)),
                pr_auc=float(average_precision_score(labels, predictions)),
                n_labelled=n_labelled,
                n_positive=n_positive,
            )
        )

    def macro(metric: str) -> float | None:
        values = [getattr(t, metric) for t in per_task if t.scored]
        return float(np.mean(values)) if values else None

    return EvaluationResult(
        roc_auc=macro("roc_auc"),
        pr_auc=macro("pr_auc"),
        per_task=tuple(per_task),
    )
