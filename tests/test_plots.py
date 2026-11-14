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

from admet.analysis.aggregate import (  # noqa: E402
    ConditionSummary,
    GapRow,
    PerTaskRow,
    RunRecord,
)
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
    yield held
    # The real `_save` closes the figure; this one cannot, because the test
    # still needs it. Close on teardown instead, or the suite leaks a figure
    # per test and matplotlib starts warning about it.
    plots.plt.close("all")


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


# --------------------------------------------------------------------------- #
# Train/test similarity — the one figure that plots data, not scores
# --------------------------------------------------------------------------- #

# Twelve scaffolds of uneven size (1 to 10 molecules). Equal-sized groups are
# no good here: the randomized splitter packs whole groups into whichever part
# is furthest below its target, so five groups of eight fill train and
# validation and leave the test set empty, and there is nothing to measure.
_SIMILARITY_CORES = (
    "c1ccccc1", "c1ccncc1", "C1CCCCC1", "c1ccc2ccccc2c1", "c1ccsc1", "c1ccoc1",
    "C1CCNCC1", "c1cncnc1", "c1ccc2[nH]ccc2c1", "C1CCCC1", "c1cc[nH]c1", "C1COCCN1",
)
_SIMILARITY_SUBS = ("C", "CC", "CCC", "OC", "NC", "CCO", "CCCC", "CCN", "OCC", "C(C)C")
_SIMILARITY_SMILES = [
    _SIMILARITY_SUBS[j] + core
    for i, core in enumerate(_SIMILARITY_CORES)
    for j in range(1 + (i * 7) % 10)
]


def test_similarity_matches_an_independent_tanimoto_computation() -> None:
    """Recomputed naively from the same fingerprints and the same split.

    The matrix form (one product, union from popcounts) is the kind of
    rewrite that is easy to get subtly wrong and still produce a plausible
    distribution, so it is checked against the definition.
    """
    import numpy as np

    from admet.analysis.plots import max_tanimoto_to_train
    from admet.data.splits import make_split
    from admet.features.fingerprints import featurize_smiles

    got = max_tanimoto_to_train(_SIMILARITY_SMILES, kind="scaffold", seed=0)

    split = make_split("scaffold", _SIMILARITY_SMILES, seed=0)
    fps = featurize_smiles(_SIMILARITY_SMILES).astype(bool)
    expected = []
    for i in split.test:
        best = 0.0
        for j in split.train:
            inter = int((fps[i] & fps[j]).sum())
            union = int((fps[i] | fps[j]).sum())
            best = max(best, inter / union if union else 0.0)
        expected.append(best)

    assert np.allclose(got, expected)
    assert got.size == len(split.test)
    assert ((got >= 0.0) & (got <= 1.0)).all()


def test_scaffold_test_molecules_are_less_like_the_training_set() -> None:
    """The claim the whole figure exists to show, asserted as a number."""
    import numpy as np

    from admet.analysis.plots import max_tanimoto_to_train

    random_sim = max_tanimoto_to_train(_SIMILARITY_SMILES, kind="random", seed=0)
    scaffold_sim = max_tanimoto_to_train(_SIMILARITY_SMILES, kind="scaffold", seed=0)
    assert np.median(scaffold_sim) < np.median(random_sim)


def test_similarity_figure_draws_both_splits_in_neither_model_colour(
    capture, tmp_path: Path
) -> None:
    """Split is the colour channel here, so it must not borrow a model's hue.

    Blue means "random forest" and orange means "GNN" in every other figure;
    reusing either for a split is the confusion `plot_gap` already had to have
    undone once.
    """
    from admet.analysis.plots import SPLIT_COLORS, plot_train_test_similarity

    plot_train_test_similarity(_SIMILARITY_SMILES, tmp_path / "s.png", dataset="tox21")
    ax = capture["figure"].get_axes()[0]

    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert len(labels) == 2
    assert any("Random" in t for t in labels) and any("Scaffold" in t for t in labels)
    assert set(SPLIT_COLORS.values()).isdisjoint(set(MODEL_COLORS.values()))
    assert ax.get_xlim() == (0.0, 1.0)


def test_similarity_figure_returns_none_when_a_split_has_no_test_set() -> None:
    """One molecule cannot be split, and must not leave a misleading file."""
    from admet.analysis.plots import plot_train_test_similarity

    assert plot_train_test_similarity(["CCO"], Path("unused.png")) is None


# --------------------------------------------------------------------------- #
# The four figures added for TODO §1 (ideas b-e)
#
# Each is checked the way the section asks for: the right number of drawn
# artists, and every value actually inside the axis it is drawn on. A figure
# can be wrong in a way no exception reports - the PR-AUC bars once rendered
# six blank charts against a hardcoded ylim - so "it saved a PNG" is not a
# test.
# --------------------------------------------------------------------------- #


def _per_task(dataset: str = "tox21", n_tasks: int = 4):
    return [
        PerTaskRow(
            dataset=dataset,
            model=m,
            split=s,
            train_fraction=1.0,
            task=f"ASSAY-{i}",
            n_seeds=5,
            roc_auc_mean=0.70 + 0.02 * i + (0.05 if s == "random" else 0.0),
            roc_auc_std=0.01,
            pr_auc_mean=0.30 + 0.01 * i,
            pr_auc_std=0.02,
        )
        for m in ("dummy", "rf", "gnn")
        for s in ("random", "scaffold")
        for i in range(n_tasks)
    ]


def _records(dataset: str = "tox21", n_seeds: int = 5):
    return [
        RunRecord(
            run=f"{dataset}_{m}_{s}_frac100_seed{seed}",
            dataset=dataset,
            model=m,
            split=s,
            seed=seed,
            train_fraction=1.0,
            test_roc_auc=0.80 + 0.01 * seed + (0.004 if m == "gnn" else 0.0),
            test_pr_auc=0.40 + 0.01 * seed,
            n_train=6259,
            n_valid=782,
            n_test=782,
            path=f"{dataset}/{dataset}_{m}_{s}_frac100_seed{seed}.json",
        )
        for m in ("rf", "gnn")
        for s in ("random", "scaffold")
        for seed in range(n_seeds)
    ]


# -- b. the per-assay heatmap ----------------------------------------------


def test_heatmap_draws_one_cell_per_assay_and_condition(
    capture, tmp_path: Path
) -> None:
    plots.plot_per_assay_heatmap(_per_task(n_tasks=4), tmp_path / "h.png")
    (image,) = capture["figure"].axes[0].get_images()

    # 4 assays x 2 models x 2 splits. The dummy is excluded, so 3 models in
    # equals 2 columns' worth of models out.
    assert image.get_array().shape == (4, 4)


def test_heatmap_excludes_the_dummy(capture, tmp_path: Path) -> None:
    """A constant 0.500 column would drag vmin to the floor and flatten the ramp."""
    plots.plot_per_assay_heatmap(_per_task(), tmp_path / "h.png")
    ax = capture["figure"].axes[0]

    labels = " ".join(t.get_text() for t in ax.get_xticklabels())
    assert "Dummy" not in labels
    assert ax.get_images()[0].get_array().min() > 0.5


def test_heatmap_colour_range_covers_the_values(capture, tmp_path: Path) -> None:
    """Positive control: no cell may fall outside the ramp it is coloured by."""
    plots.plot_per_assay_heatmap(_per_task(), tmp_path / "h.png")
    (image,) = capture["figure"].axes[0].get_images()

    values = image.get_array()
    low, high = image.get_clim()
    assert low <= values.min() and values.max() <= high


def test_heatmap_declines_a_single_assay_dataset(tmp_path: Path) -> None:
    """ClinTox scores one assay; a one-row heatmap is a bar chart with steps."""
    rows = _per_task(dataset="clintox", n_tasks=1)
    assert plots.plot_per_assay_heatmap(rows, tmp_path / "h.png") is None


# -- c. the paired per-seed scatter ----------------------------------------


def test_paired_scatter_draws_one_point_per_seed(capture, tmp_path: Path) -> None:
    plots.plot_paired_seeds(_records(n_seeds=5), tmp_path / "p.png")
    collections = capture["figure"].axes[0].collections

    assert len(collections) == 2  # one per split
    assert sum(len(c.get_offsets()) for c in collections) == 10  # 2 splits x 5


def test_every_paired_point_is_inside_the_axes(capture, tmp_path: Path) -> None:
    plots.plot_paired_seeds(_records(), tmp_path / "p.png")
    ax = capture["figure"].axes[0]
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()

    for collection in ax.collections:
        for x, y in collection.get_offsets():
            assert x0 <= x <= x1 and y0 <= y <= y1


def test_paired_axes_share_one_scale(capture, tmp_path: Path) -> None:
    """A 45° line only means equality if the axes match in range *and* aspect.

    Equal limits alone are not enough: on a non-square box the line of
    equality is drawn at some other angle, and a reader judging points
    against "the diagonal" reads the wrong comparison. An earlier version of
    this test checked only the limits and a mutant that deleted the aspect
    survived it.
    """
    plots.plot_paired_seeds(_records(), tmp_path / "p.png")
    ax = capture["figure"].axes[0]

    assert ax.get_xlim() == ax.get_ylim()
    assert ax.get_aspect() == 1.0


def test_paired_scatter_never_paints_a_split_in_a_model_colour(
    capture, tmp_path: Path
) -> None:
    """Colour means split here; the model hues mean model everywhere else."""
    from matplotlib.colors import to_hex

    plots.plot_paired_seeds(_records(), tmp_path / "p.png")
    model_hues = {c.lower() for c in plots.MODEL_COLORS.values()}

    for collection in capture["figure"].axes[0].collections:
        for rgba in collection.get_facecolor():
            assert to_hex(rgba).lower() not in model_hues


# -- d. every gap in one figure --------------------------------------------


def test_all_gaps_draws_every_row(capture, tmp_path: Path) -> None:
    rows = _gaps("tox21") + _gaps("clintox")
    plots.plot_all_gaps(rows, tmp_path / "g.png")
    ax = capture["figure"].axes[0]

    assert len(ax.get_yticklabels()) == len(rows)


def test_all_gaps_labels_name_both_datasets(capture, tmp_path: Path) -> None:
    """This is the one figure allowed to mix datasets, so each row must say which."""
    plots.plot_all_gaps(_gaps("tox21") + _gaps("clintox"), tmp_path / "g.png")
    labels = [t.get_text() for t in capture["figure"].axes[0].get_yticklabels()]

    assert any("Tox21" in text for text in labels)
    assert any("ClinTox" in text for text in labels)
    assert all("%" in text for text in labels)


def test_all_gaps_shows_the_zero_line_and_contains_every_interval(
    capture, tmp_path: Path
) -> None:
    rows = _gaps("tox21") + _gaps("clintox")
    plots.plot_all_gaps(rows, tmp_path / "g.png")
    ax = capture["figure"].axes[0]

    assert any(line.get_xdata()[0] == 0.0 for line in ax.lines)
    low, high = ax.get_xlim()
    for row in rows:
        assert low <= row.gap - (row.gap_std or 0) and row.gap + (row.gap_std or 0) <= high


# -- e. the dataset overview ------------------------------------------------


def _labels_and_smiles(n_tasks: int = 3):
    import numpy as np

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=(40, n_tasks)).astype(float)
    y[0, 0] = np.nan  # one unscreened cell, so density is not trivially 1.0
    tasks = [f"ASSAY-{i}" for i in range(n_tasks)]
    smiles = ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O", "CCN(CC)CC"] * 10
    return y, tasks, smiles


def test_overview_draws_two_bars_per_assay(capture, tmp_path: Path) -> None:
    y, tasks, smiles = _labels_and_smiles(n_tasks=3)
    plots.plot_dataset_overview(y, tasks, smiles, tmp_path / "o.png")
    left = capture["figure"].axes[0]

    # One "screened" bar and one "positive" bar for each assay.
    assert len(left.containers) == 2
    assert all(len(container) == 3 for container in left.containers)


def test_overview_positives_never_exceed_the_screened_count(
    capture, tmp_path: Path
) -> None:
    """The positive bar is drawn over the screened one, so it must fit inside it."""
    y, tasks, smiles = _labels_and_smiles()
    plots.plot_dataset_overview(y, tasks, smiles, tmp_path / "o.png")
    screened, positive = capture["figure"].axes[0].containers

    for total, part in zip(screened, positive):
        assert part.get_width() <= total.get_width()


def test_overview_bars_are_inside_the_visible_axis(capture, tmp_path: Path) -> None:
    y, tasks, smiles = _labels_and_smiles()
    plots.plot_dataset_overview(y, tasks, smiles, tmp_path / "o.png")
    left = capture["figure"].axes[0]
    _, high = left.get_xlim()

    for container in left.containers:
        for bar in container:
            assert bar.get_width() <= high


def test_overview_refuses_an_empty_label_matrix(tmp_path: Path) -> None:
    import numpy as np

    assert (
        plots.plot_dataset_overview(
            np.empty((0, 0)), [], [], tmp_path / "o.png"
        )
        is None
    )


def test_every_registered_dataset_has_a_display_label() -> None:
    """A missing label silently titles the figure with the raw registry key.

    BBBP shipped one round of figures titled "bbbp" because nothing caught it:
    `_label` falls back to the key rather than raising, which is right at run
    time and invisible in review.
    """
    from admet.data.loaders import DATASETS

    missing = [n for n in DATASETS if n not in plots.DATASET_LABELS]
    assert not missing, f"no DATASET_LABELS entry for {missing}"
