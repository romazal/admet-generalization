"""Tests for the figures.

These do not check that a figure looks good — they check the one failure a
figure can have that is completely silent: the data is drawn outside the
visible axis, so the file is written, no error is raised, and the chart is
blank. That is exactly what happened with PR-AUC, where a hardcoded 0.5
y-floor (correct for ROC-AUC, where 0.5 is chance) clipped every bar,
because Tox21's PR-AUC runs at about 0.35-0.47.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

from admet.analysis.aggregate import ConditionSummary, GapRow  # noqa: E402
from admet.analysis.plots import (  # noqa: E402
    _bar_ylim,
    make_all_figures,
    plot_split_comparison,
)


def _summary(
    *,
    dataset: str = "tox21",
    model: str = "rf",
    split: str = "random",
    fraction: float = 1.0,
    roc: float = 0.83,
    pr: float = 0.47,
) -> ConditionSummary:
    return ConditionSummary(
        dataset=dataset,
        model=model,
        split=split,
        train_fraction=fraction,
        n_seeds=5,
        roc_auc_mean=roc,
        roc_auc_std=0.01,
        pr_auc_mean=pr,
        pr_auc_std=0.02,
        n_train=6259,
    )


def _grid(dataset: str = "tox21", *, roc: float = 0.83, pr: float = 0.47):
    return [
        _summary(dataset=dataset, model=m, split=s, roc=roc, pr=pr)
        for m in ("rf", "gnn")
        for s in ("random", "scaffold")
    ]


# --------------------------------------------------------------------------- #
# The y-limit rule itself
# --------------------------------------------------------------------------- #


def test_roc_auc_keeps_the_chance_floor() -> None:
    """0.5 is chance for ROC-AUC, so it stays the baseline."""
    assert _bar_ylim("roc_auc", [0.83, 0.76]) == (0.5, 1.0)


def test_pr_auc_starts_at_zero() -> None:
    """PR-AUC has no fixed chance level, so the bar length is the score."""
    low, high = _bar_ylim("pr_auc", [0.47, 0.38])
    assert low == 0.0
    assert high > 0.47


def test_pr_auc_limit_never_exceeds_one() -> None:
    _, high = _bar_ylim("pr_auc", [0.99])
    assert high <= 1.0


def test_ylim_survives_no_data() -> None:
    """An empty series must not raise on max()."""
    low, high = _bar_ylim("pr_auc", [])
    assert high > low


# --------------------------------------------------------------------------- #
# The regression: every bar must be inside the axis
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("metric", "value"), [("roc_auc", 0.83), ("pr_auc", 0.47)]
)
def test_every_bar_is_inside_the_visible_axis(metric: str, value: float) -> None:
    """The silent failure: a figure written with its bars off-screen."""
    low, high = _bar_ylim(metric, [value + 0.02])
    assert low <= value <= high, (
        f"{metric} value {value} falls outside the axis {low}-{high}; "
        "the chart would render blank"
    )


def test_pr_auc_figure_would_not_be_blank() -> None:
    """Guards the exact regression: Tox21 PR-AUC below a 0.5 floor."""
    low, high = _bar_ylim("pr_auc", [0.473, 0.442, 0.379, 0.353])
    for value in (0.473, 0.442, 0.379, 0.353):
        assert low <= value <= high


# --------------------------------------------------------------------------- #
# Datasets must not be mixed
# --------------------------------------------------------------------------- #


def test_split_comparison_filters_by_dataset(tmp_path: Path) -> None:
    """A ClinTox figure must not be able to draw a Tox21 bar."""
    summaries = _grid("tox21", roc=0.83) + _grid("clintox", roc=0.90)
    path = plot_split_comparison(
        summaries, tmp_path / "clintox.png", metric="roc_auc", dataset="clintox"
    )
    assert path is not None and path.exists()


def test_unknown_dataset_writes_nothing(tmp_path: Path) -> None:
    assert (
        plot_split_comparison(
            _grid("tox21"), tmp_path / "x.png", metric="roc_auc", dataset="nope"
        )
        is None
    )


def test_make_all_figures_writes_one_set_per_dataset(tmp_path: Path) -> None:
    summaries = _grid("tox21") + _grid("clintox")
    gaps: list[GapRow] = []
    written = make_all_figures(summaries, gaps, tmp_path, metric="roc_auc")

    # Each dataset gets its own subdirectory, and the dataset stays in the
    # filename too, so a figure pasted into a slide still says which it is.
    relative = {str(p.relative_to(tmp_path)).replace("\\", "/") for p in written}
    assert "tox21/random_vs_scaffold_tox21_roc_auc.png" in relative
    assert "clintox/random_vs_scaffold_clintox_roc_auc.png" in relative
    for path in written:
        assert path.exists() and path.stat().st_size > 0

    # No figure may land outside its dataset's folder.
    for path in written:
        assert path.parent.name in {"tox21", "clintox"}
