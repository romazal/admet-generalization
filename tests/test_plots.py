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
from admet.analysis import plots  # noqa: E402
from admet.analysis.plots import (  # noqa: E402
    GAP_COLOR,
    MODEL_COLORS,
    _bar_ylim,
    make_all_figures,
    plot_gap,
    plot_learning_curves,
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


# --------------------------------------------------------------------------- #
# Panel layout
#
# The learning-curve and gap figures used to put four and two series on one
# pair of axes. Four curves, told apart by colour *and* line style and
# crossing repeatedly, is not readable, so both are now faceted by split.
#
# These inspect the real figure rather than a rebuilt imitation: the plot
# functions close their figure on save, so `_save` is patched to capture it
# first. A test that redraws the chart itself would pass while the shipped
# figure was wrong.
# --------------------------------------------------------------------------- #


@pytest.fixture
def capture(monkeypatch):
    """Capture the Figure each plot function saves, without writing a file."""
    held: dict[str, object] = {}

    def fake_save(figure, path, *, bottom: float = 0.0):
        held["figure"] = figure
        return Path(path)

    monkeypatch.setattr(plots, "_save", fake_save)
    return held


def _curve(dataset: str = "tox21"):
    return [
        _summary(
            dataset=dataset,
            model=m,
            split=s,
            fraction=f,
            roc=0.70 + 0.03 * i,
            pr=0.30 + 0.02 * i,
        )
        for m in ("rf", "gnn")
        for s in ("random", "scaffold")
        for i, f in enumerate((0.1, 0.25, 0.5, 1.0))
    ]


def _gaps(dataset: str = "tox21"):
    return [
        GapRow(
            dataset=dataset,
            split=s,
            train_fraction=f,
            metric="roc_auc",
            baseline_mean=0.80,
            gnn_mean=0.79,
            gap=-0.01,
            gap_std=0.02,
            n_seeds=5,
        )
        for s in ("random", "scaffold")
        for f in (0.1, 0.25, 0.5, 1.0)
    ]


def test_learning_curves_draw_one_panel_per_split(capture, tmp_path: Path) -> None:
    plot_learning_curves(_curve(), tmp_path / "c.png", dataset="tox21")
    axes = capture["figure"].get_axes()
    assert len(axes) == 2
    assert [ax.get_title(loc="left") for ax in axes] == ["Random split", "Scaffold split"]


def test_each_learning_curve_panel_holds_both_models(capture, tmp_path: Path) -> None:
    """Two lines per panel, in the two model colours.

    If a panel ever carries all four series again, or only one model, this
    fails — that regression is invisible in a rendered PNG at a glance.
    """
    plot_learning_curves(_curve(), tmp_path / "c.png", dataset="tox21")
    for ax in capture["figure"].get_axes():
        # errorbar() puts the label on the container, not the Line2D, so the
        # colours are the identity to check here. Named explicitly rather than
        # compared against every registered colour: the dummy is registered
        # too but has only the 100% point, so it draws no curve.
        colors = {line.get_color() for line in ax.get_lines()}
        assert colors == {MODEL_COLORS["rf"], MODEL_COLORS["gnn"]}


def test_every_learning_curve_point_is_inside_its_panel(
    capture, tmp_path: Path
) -> None:
    """The blank-chart failure, checked per panel rather than per figure."""
    plot_learning_curves(_curve(), tmp_path / "c.png", dataset="tox21")
    for ax in capture["figure"].get_axes():
        bottom, top = ax.get_ylim()
        for line in ax.get_lines():
            for y in line.get_ydata():
                assert bottom <= y <= top


def test_gap_draws_one_panel_per_split(capture, tmp_path: Path) -> None:
    plot_gap(_gaps(), tmp_path / "g.png", dataset="tox21")
    axes = capture["figure"].get_axes()
    assert len(axes) == 2
    assert [ax.get_title(loc="left") for ax in axes] == ["Random split", "Scaffold split"]


def test_gap_never_paints_a_series_in_a_model_colour(capture, tmp_path: Path) -> None:
    """The gap series must not reuse a model's hue.

    It used to: ``split_colors`` mapped random to MODEL_COLORS["rf"] and
    scaffold to MODEL_COLORS["gnn"], so blue meant "random forest" in the bar
    chart and the learning curve but "random split" in the gap plot. The gap
    is a quantity about both models, so it gets its own neutral.
    """
    plot_gap(_gaps(), tmp_path / "g.png", dataset="tox21")
    model_hues = {c.lower() for c in MODEL_COLORS.values()}
    for ax in capture["figure"].get_axes():
        for line in ax.get_lines():
            assert str(line.get_color()).lower() not in model_hues


def test_gap_panels_show_the_zero_line(capture, tmp_path: Path) -> None:
    """Zero is the claim boundary, so every panel needs it, not just the first."""
    plot_gap(_gaps(), tmp_path / "g.png", dataset="tox21")
    for ax in capture["figure"].get_axes():
        assert any(
            line.get_ydata()[0] == 0.0 for line in ax.get_lines() if len(line.get_ydata())
        )


def test_bar_value_labels_clear_the_error_bar_caps(capture, tmp_path: Path) -> None:
    """The printed value sits above the error-bar cap, not above the bar.

    With an SD of 0.01 on a 0.83 bar the cap lands where the text used to be
    drawn, and the two overlapped in every committed bar chart.
    """
    plot_split_comparison(_grid(), tmp_path / "b.png", dataset="tox21")
    ax = capture["figure"].get_axes()[0]
    # .xy is the anchor point; .xyann is the text offset in points.
    labels = {a.get_text(): a.xy[1] for a in ax.texts}
    assert labels, "no value labels were drawn"
    # Every label is anchored at mean + SD (0.83 + 0.01), never at the mean.
    for text, y in labels.items():
        assert y == pytest.approx(float(text) + 0.01, abs=1e-9)


def test_figures_state_what_the_error_bars_are(capture, tmp_path: Path) -> None:
    """An unexplained interval reads as a drawing artefact, not as uncertainty."""
    plot_learning_curves(_curve(), tmp_path / "c.png", dataset="tox21")
    note = " ".join(t.get_text() for t in capture["figure"].texts)
    assert "±1 SD" in note and "seeds" in note


# --------------------------------------------------------------------------- #
# The dummy floor: a bar sitting exactly on the baseline
# --------------------------------------------------------------------------- #


def _grid_with_dummy(dataset: str = "tox21"):
    """The real Tox21 shape: a dummy at exactly 0.5 ROC-AUC, beside two models."""
    rows = _grid(dataset)
    rows += [
        _summary(dataset=dataset, model="dummy", split="random", roc=0.5, pr=0.077),
        _summary(dataset=dataset, model="dummy", split="scaffold", roc=0.5, pr=0.100),
    ]
    return rows


def test_a_bar_on_the_chance_floor_is_still_visible() -> None:
    """The dummy scores exactly 0.5, which is the ROC-AUC baseline.

    With the floor pinned at 0.5 the bar has zero height: the PNG is written,
    nothing raises, and the reader sees a floating "0.500" label over an
    unexplained gap between the other two bars. The same silent-blank failure
    as the PR-AUC one above, at the other end of the axis.
    """
    low, high = _bar_ylim("roc_auc", [0.85, 0.5], [0.83, 0.5])
    assert low < 0.5, "a bar at 0.5 would render with zero height"
    assert high == 1.0


def test_the_chance_floor_is_kept_when_nothing_sits_on_it() -> None:
    """Without a dummy the axis must not drift off 0.5 — chance is the baseline."""
    assert _bar_ylim("roc_auc", [0.85, 0.78], [0.83, 0.76]) == (0.5, 1.0)


def test_every_dummy_bar_has_height_in_the_real_figure(capture, tmp_path: Path) -> None:
    """Checks the drawn figure, not the helper: every bar must be visible."""
    for metric in ("roc_auc", "pr_auc"):
        plot_split_comparison(
            _grid_with_dummy(), tmp_path / f"{metric}.png",
            metric=metric, dataset="tox21",
        )
        ax = capture["figure"].get_axes()[0]
        floor, _ = ax.get_ylim()
        drawn = [p for p in ax.patches if p.get_width() > 0]
        assert drawn, f"{metric}: no bars drawn at all"
        for patch in drawn:
            visible = patch.get_y() + patch.get_height() - floor
            assert visible > 0, (
                f"{metric}: a bar of height {patch.get_height()} sits on the "
                f"axis floor {floor} and renders as nothing"
            )


def test_the_dummy_is_not_painted_in_a_model_colour(capture, tmp_path: Path) -> None:
    """The floor is not a third competing approach, so it gets a neutral."""
    plot_split_comparison(
        _grid_with_dummy(), tmp_path / "c.png", metric="pr_auc", dataset="tox21"
    )
    assert MODEL_COLORS["dummy"] not in (MODEL_COLORS["rf"], MODEL_COLORS["gnn"])
    ax = capture["figure"].get_axes()[0]
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == [
        "Dummy (class prior)",
        "Fingerprint + random forest",
        "Graph neural network",
    ], "the floor should read first, before the two approaches it bounds"
