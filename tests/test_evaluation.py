"""Tests for :mod:`admet.evaluation`, and for the masked loss that mirrors it.

Proposal §3.4 asks for metrics checked on small hand-made examples including
missing labels, and §6 names "correct masking of missing labels in both the
loss and the metrics" as one of the few errors that would invalidate the
comparison outright. Both halves of that are tested here: the metrics in
:mod:`admet.evaluation`, and :func:`admet.models.gnn.masked_bce_loss`, which
has to follow the same rule on the training side.

The important tests are the ones with a **positive control** — they show that
the masking is doing something, by checking that the obvious wrong
implementation (missing label read as a negative) would give a *different*
answer. Without that, a test that "masking works" passes just as happily on
code that ignores the mask on data where it makes no difference.
"""

from __future__ import annotations

import csv
import json
import math

import numpy as np
import pytest

from admet.analysis.aggregate import aggregate, per_task_table
from admet.evaluation import METRICS, PRIMARY_METRIC, EvaluationResult, evaluate


# --------------------------------------------------------------------------- #
# Known answers on tiny examples
# --------------------------------------------------------------------------- #


def test_perfect_ranking_scores_one() -> None:
    y = np.array([[0.0], [0.0], [1.0], [1.0]])
    scores = np.array([[0.1], [0.2], [0.8], [0.9]])
    result = evaluate(y, scores)
    assert result.roc_auc == pytest.approx(1.0)
    assert result.pr_auc == pytest.approx(1.0)


def test_inverted_ranking_scores_zero() -> None:
    y = np.array([[0.0], [0.0], [1.0], [1.0]])
    scores = np.array([[0.9], [0.8], [0.2], [0.1]])
    assert evaluate(y, scores).roc_auc == pytest.approx(0.0)


def test_constant_prediction_scores_chance() -> None:
    """A model with nothing to say gets ROC-AUC 0.5, not an error.

    The baseline falls back to a constant score for an assay with no positives
    in training, so this path is reached in real runs.
    """
    y = np.array([[0.0], [1.0], [0.0], [1.0]])
    scores = np.full((4, 1), 0.3)
    assert evaluate(y, scores).roc_auc == pytest.approx(0.5)


def test_roc_auc_matches_a_hand_computed_value() -> None:
    """Two positives, three negatives, ranks known by hand.

    ROC-AUC is the share of (positive, negative) pairs the model orders
    correctly. Positives score 0.9 and 0.4; negatives score 0.8, 0.3, 0.2.
    The 0.9 positive beats all three negatives; the 0.4 positive beats two of
    them and loses to 0.8. So 5 of 6 pairs are correct.
    """
    y = np.array([[1.0], [1.0], [0.0], [0.0], [0.0]])
    scores = np.array([[0.9], [0.4], [0.8], [0.3], [0.2]])
    assert evaluate(y, scores).roc_auc == pytest.approx(5 / 6)


def test_pr_auc_matches_a_hand_computed_value() -> None:
    """Average precision, summed over the recall steps.

    Ranked by score: P(0.9), N(0.8), P(0.4), N(0.3), N(0.2). Average precision
    is the mean precision at each positive: 1/1 at the first, 2/3 at the
    second.
    """
    y = np.array([[1.0], [1.0], [0.0], [0.0], [0.0]])
    scores = np.array([[0.9], [0.4], [0.8], [0.3], [0.2]])
    assert evaluate(y, scores).pr_auc == pytest.approx((1.0 + 2 / 3) / 2)


def test_a_one_dimensional_label_vector_is_treated_as_one_task() -> None:
    result = evaluate(np.array([0.0, 0.0, 1.0, 1.0]), np.array([0.1, 0.2, 0.8, 0.9]))
    assert len(result.per_task) == 1
    assert result.roc_auc == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Masking — the part that would silently invalidate the comparison
# --------------------------------------------------------------------------- #


def test_missing_labels_are_excluded_from_their_assay() -> None:
    """Adding unscreened molecules must not move the score at all."""
    y = np.array([[0.0], [0.0], [1.0], [1.0]])
    scores = np.array([[0.1], [0.2], [0.8], [0.9]])
    baseline = evaluate(y, scores).roc_auc

    # The same four molecules, plus three that were never screened. Their
    # predictions are deliberately terrible: if they leaked into the score,
    # it would collapse.
    padded_y = np.array([[0.0], [np.nan], [0.0], [1.0], [np.nan], [1.0], [np.nan]])
    padded_scores = np.array([[0.1], [0.99], [0.2], [0.8], [0.99], [0.9], [0.0]])

    assert evaluate(padded_y, padded_scores).roc_auc == pytest.approx(baseline)


def test_treating_missing_as_negative_would_change_the_score() -> None:
    """Positive control for the test above.

    This is the wrong implementation the proposal warns about. If it produced
    the same number, the masking test would be vacuous.
    """
    padded_y = np.array([[0.0], [np.nan], [0.0], [1.0], [np.nan], [1.0], [np.nan]])
    padded_scores = np.array([[0.1], [0.99], [0.2], [0.8], [0.99], [0.9], [0.0]])

    masked = evaluate(padded_y, padded_scores).roc_auc
    as_negative = evaluate(np.nan_to_num(padded_y, nan=0.0), padded_scores).roc_auc

    assert masked == pytest.approx(1.0)
    assert not math.isclose(masked, as_negative), (
        "masking makes no difference on this example, so it proves nothing"
    )


def test_missing_labels_are_masked_per_assay_not_per_molecule() -> None:
    """A molecule missing assay B still counts towards assay A.

    Dropping whole rows would discard most of Tox21, so this distinguishes
    per-assay masking from per-molecule filtering.
    """
    y = np.array([[1.0, np.nan], [0.0, 0.0], [1.0, 1.0], [0.0, np.nan]])
    scores = np.array([[0.9, 0.5], [0.1, 0.2], [0.8, 0.9], [0.2, 0.4]])
    result = evaluate(y, scores, tasks=["a", "b"])

    by_name = {t.task: t for t in result.per_task}
    assert by_name["a"].n_labelled == 4  # every molecule has an A label
    assert by_name["b"].n_labelled == 2  # only two have a B label
    assert by_name["a"].roc_auc == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Unscoreable assays
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("labels", "reason"),
    [
        ([1.0, 1.0, 1.0], "no negative molecules"),
        ([0.0, 0.0, 0.0], "no positive molecules"),
        ([np.nan, np.nan, np.nan], "no labelled molecules"),
    ],
)
def test_an_unscoreable_assay_is_skipped_and_named(labels, reason) -> None:
    """Scoring it as zero would blame the model for the data."""
    y = np.array(labels).reshape(-1, 1)
    result = evaluate(y, np.array([[0.2], [0.5], [0.8]]), tasks=["lonely"])

    assert result.roc_auc is None
    assert result.skipped_tasks == {"lonely": reason}
    assert result.scored_tasks == ()


def test_a_skipped_assay_does_not_drag_the_macro_average_down() -> None:
    """One unscoreable assay beside one perfect assay averages to 1.0, not 0.5."""
    y = np.array([[1.0, 1.0], [0.0, 1.0], [1.0, 1.0], [0.0, 1.0]])
    scores = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.1], [0.2, 0.9]])
    result = evaluate(y, scores, tasks=["good", "all_positive"])

    assert result.scored_tasks == ("good",)
    assert result.roc_auc == pytest.approx(1.0)


def test_macro_average_weights_assays_equally_not_by_size() -> None:
    """A big assay must not outvote a small one."""
    # Assay A: 8 molecules, perfectly ranked. Assay B: 2 molecules, inverted.
    y = np.full((8, 2), np.nan)
    y[:, 0] = [0, 0, 0, 0, 1, 1, 1, 1]
    y[:2, 1] = [1, 0]
    scores = np.zeros((8, 2))
    scores[:, 0] = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]
    scores[:2, 1] = [0.1, 0.9]

    result = evaluate(y, scores, tasks=["big", "small"])
    assert result.roc_auc == pytest.approx((1.0 + 0.0) / 2)


def test_per_task_table_keeps_each_assay_separate_from_the_macro_average(
    tmp_path,
) -> None:
    """One perfect assay and one chance assay must stay distinguishable."""
    dataset_dir = tmp_path / "toy"
    dataset_dir.mkdir()
    labels = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0]])
    scores = np.array([[0.1, 0.5], [0.2, 0.5], [0.8, 0.5], [0.9, 0.5]])
    test_metrics = evaluate(labels, scores, tasks=["easy", "hard"]).as_dict()

    for seed in (0, 1):
        payload = {
            "run": f"toy_rf_random_frac100_seed{seed}",
            "dataset": "toy",
            "model": "rf",
            "split": "random",
            "seed": seed,
            "train_fraction": 1.0,
            "metrics": {"test": test_metrics},
            "split_sizes": {"train": 4, "valid": 2, "test": 2},
        }
        (dataset_dir / f"toy_rf_random_frac100_seed{seed}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    rows = per_task_table(tmp_path)
    by_task = {row.task: row for row in rows}
    assert by_task["easy"].roc_auc_mean == pytest.approx(1.0)
    assert by_task["hard"].roc_auc_mean == pytest.approx(0.5)
    assert by_task["easy"].n_seeds == 2

    records, summaries, _ = aggregate(tmp_path, write=True)
    assert records
    with (tmp_path / "per_task.csv").open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert [(row["task"], int(row["n_seeds"])) for row in csv_rows] == [
        ("easy", 2),
        ("hard", 2),
    ]
    summary = summaries[0]
    assert summary.roc_auc_mean == pytest.approx(0.75)
    assert by_task["easy"].roc_auc_mean != summary.roc_auc_mean


# --------------------------------------------------------------------------- #
# Reporting and validation
# --------------------------------------------------------------------------- #


def test_per_task_counts_and_positive_rate_are_reported() -> None:
    y = np.array([[1.0], [0.0], [0.0], [np.nan]])
    result = evaluate(y, np.array([[0.9], [0.2], [0.1], [0.5]]), tasks=["assay"])
    task = result.per_task[0]

    assert (task.n_labelled, task.n_positive) == (3, 1)
    assert task.positive_rate == pytest.approx(1 / 3)
    assert task.scored


def test_result_is_json_serialisable_and_indexable_by_metric() -> None:
    import json

    y = np.array([[1.0], [0.0]])
    result = evaluate(y, np.array([[0.9], [0.1]]), tasks=["assay"])

    assert result[PRIMARY_METRIC] == pytest.approx(1.0)
    json.dumps(result.as_dict())  # must not raise

    with pytest.raises(KeyError, match="unknown metric"):
        result["accuracy"]


def test_every_declared_metric_is_present_on_the_result() -> None:
    result = evaluate(np.array([[1.0], [0.0]]), np.array([[0.9], [0.1]]))
    for metric in METRICS:
        assert getattr(result, metric) is not None


@pytest.mark.parametrize(
    ("y", "scores", "message"),
    [
        (np.zeros((3, 1)), np.zeros((4, 1)), "same shape"),
        (np.zeros((0, 1)), np.zeros((0, 1)), "empty"),
        (np.array([[0.0], [1.0]]), np.array([[0.5], [np.nan]]), "NaN or infinity"),
        (np.array([[0.0], [2.0]]), np.array([[0.5], [0.5]]), "only 0, 1 or NaN"),
    ],
)
def test_invalid_input_is_rejected(y, scores, message) -> None:
    with pytest.raises(ValueError, match=message):
        evaluate(y, scores)


def test_wrong_number_of_task_names_is_rejected() -> None:
    with pytest.raises(ValueError, match="task name"):
        evaluate(np.zeros((2, 2)), np.zeros((2, 2)), tasks=["only_one"])


def test_evaluation_result_is_frozen() -> None:
    result = EvaluationResult(roc_auc=1.0, pr_auc=1.0)
    with pytest.raises(Exception):
        result.roc_auc = 0.0  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# The loss must mask the same way the metrics do
# --------------------------------------------------------------------------- #


def test_masked_loss_ignores_missing_labels() -> None:
    """The GNN's loss follows the same rule as the metrics (proposal §6)."""
    torch = pytest.importorskip("torch")
    from admet.models.gnn import masked_bce_loss

    logits = torch.tensor([[2.0, -1.0], [0.5, 3.0]])
    observed = torch.tensor([[1.0, 0.0], [1.0, 1.0]])

    # Blanking an entry must give the loss over the remaining three.
    partial = torch.tensor([[1.0, float("nan")], [1.0, 1.0]])
    kept = torch.tensor([2.0, 0.5, 3.0])
    kept_targets = torch.tensor([1.0, 1.0, 1.0])
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        kept, kept_targets
    )

    assert float(masked_bce_loss(logits, partial)) == pytest.approx(
        float(expected), abs=1e-6
    )
    # Positive control: the masked loss differs from the unmasked one.
    assert not math.isclose(
        float(masked_bce_loss(logits, partial)),
        float(masked_bce_loss(logits, observed)),
    )


def test_masked_loss_of_an_all_missing_batch_is_zero_and_finite() -> None:
    """An empty mask must contribute nothing, not NaN through every weight."""
    torch = pytest.importorskip("torch")
    from admet.models.gnn import masked_bce_loss

    logits = torch.tensor([[1.0, -2.0]], requires_grad=True)
    targets = torch.full((1, 2), float("nan"))

    loss = masked_bce_loss(logits, targets)
    assert float(loss.detach()) == 0.0
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_masked_loss_gradient_does_not_reach_a_missing_label() -> None:
    torch = pytest.importorskip("torch")
    from admet.models.gnn import masked_bce_loss

    logits = torch.zeros((1, 2), requires_grad=True)
    targets = torch.tensor([[1.0, float("nan")]])

    masked_bce_loss(logits, targets).backward()
    assert logits.grad[0, 1] == 0.0
    assert logits.grad[0, 0] != 0.0
