"""Figures for the report and the slides, drawn from the aggregated tables.

Three figures, matching the three claims the project makes (proposal §5):

``random_vs_scaffold`` — the main experiment. Both models under both splits, so
the drop from random to scaffold is visible per model and H1 can be read off
directly.

``learning_curves`` — the additional experiment. Test score against training-set
size, one line per model and split, which is where H2 lives.

``gnn_minus_rf_gap`` — the quantity both hypotheses are actually stated in,
plotted against training-set size with a zero line. H1 says the scaffold line
sits below the random line; H2 says both rise with training data.

Everything is regenerated from ``results/`` by
:func:`admet.analysis.aggregate.aggregate`, so no figure can disagree with the
numbers in the tables, and no figure survives a change to the results without
being redrawn (proposal §7 — no manual steps).

Encoding
--------
**Model is colour, split is line style** — the baseline is blue and the GNN is
orange throughout all three figures, while random is solid with a round marker
and scaffold is dashed with a square one. Identity is therefore never carried
by colour alone, which keeps the figures readable in greyscale print and under
colour-vision deficiency. The two hues are slots 1 and 2 of the reference
categorical palette, used unmodified.

Error bars are the standard deviation over seeds. A condition with one seed has
no defined deviation and is drawn without a bar rather than with a zero-length
one.

Output is PNG for a written report and slides, so there is no hover layer;
every figure carries a legend and selective direct value labels instead, and
the same numbers are available as CSV beside the figures.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

# A non-interactive backend, so figures render identically in a terminal, in
# CI and over SSH, with no display attached.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from admet.analysis.aggregate import ConditionSummary, GapRow  # noqa: E402

__all__ = [
    "MODEL_COLORS",
    "MODEL_LABELS",
    "DATASET_LABELS",
    "SPLIT_STYLES",
    "plot_split_comparison",
    "plot_learning_curves",
    "plot_gap",
    "make_all_figures",
]

# -- Design tokens ---------------------------------------------------------- #
# Slots 1 and 2 of the reference categorical palette, light mode.

_SURFACE = "#fcfcfb"
_INK = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_GRID = "#dcdbd6"

#: Model is the colour channel.
MODEL_COLORS: dict[str, str] = {"rf": "#2a78d6", "gnn": "#eb6834"}

MODEL_LABELS: dict[str, str] = {
    "rf": "Fingerprint + random forest",
    "gnn": "Graph neural network",
}

#: Split is the line-style / marker channel — the secondary encoding.
SPLIT_STYLES: dict[str, dict[str, object]] = {
    "random": {"linestyle": "-", "marker": "o"},
    "scaffold": {"linestyle": "--", "marker": "s"},
}

DATASET_LABELS: dict[str, str] = {"tox21": "Tox21", "clintox": "ClinTox"}

SPLIT_LABELS: dict[str, str] = {
    "random": "Random split",
    "scaffold": "Scaffold split",
}

_LINE_WIDTH = 2.0
_MARKER_SIZE = 7.0


def _label(mapping: dict[str, str], key: str) -> str:
    """Human-readable name, falling back to the raw key for a new model."""
    return mapping.get(key, key)


def _bar_ylim(metric: str, tops: Sequence[float]) -> tuple[float, float]:
    """Y-limits for the bar chart, by metric.

    A bar chart's baseline has to be a value the reader can treat as "none of
    it". For ROC-AUC that is 0.5: chance, not zero, so the axis starts there
    and the bar length is the part above chance.

    PR-AUC has no such fixed level — its chance value is the positive rate,
    which differs per assay and per dataset — so the baseline is 0 and the
    bar length is the score itself.

    This used to be a hardcoded ``(0.5, 1.0)``, which silently clipped every
    PR-AUC figure: Tox21's PR-AUC is about 0.35-0.47, entirely below the
    floor, so the chart rendered with no bars at all.
    """
    if metric == "roc_auc":
        return 0.5, 1.0
    headroom = max(tops, default=1.0) * 1.18
    return 0.0, min(1.0, max(headroom, 0.1))


def _suffix(dataset: str | None) -> str:
    """Name the dataset in the title, so no figure is ambiguous about its source."""
    return f" - {DATASET_LABELS.get(dataset, dataset)}" if dataset else ""


def _color(model: str) -> str:
    """Colour for a model, defaulting to a neutral for an unregistered one."""
    return MODEL_COLORS.get(model, _INK_SECONDARY)


def _style_axes(ax: plt.Axes, *, title: str, xlabel: str, ylabel: str) -> None:
    """Recessive grid and axes, ink-coloured text."""
    ax.set_title(title, color=_INK, fontsize=12, pad=12, loc="left")
    ax.set_xlabel(xlabel, color=_INK_SECONDARY, fontsize=10)
    ax.set_ylabel(ylabel, color=_INK_SECONDARY, fontsize=10)
    ax.set_facecolor(_SURFACE)
    ax.grid(True, color=_GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_GRID)
    ax.tick_params(colors=_INK_SECONDARY, labelsize=9)


def _new_figure(size: tuple[float, float] = (7.5, 4.6)) -> tuple[plt.Figure, plt.Axes]:
    figure, ax = plt.subplots(figsize=size)
    figure.patch.set_facecolor(_SURFACE)
    return figure, ax


def _save(figure: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=200, facecolor=_SURFACE)
    plt.close(figure)
    return path


# --------------------------------------------------------------------------- #
# Figure 1 — the main experiment
# --------------------------------------------------------------------------- #


def plot_split_comparison(
    summaries: Sequence[ConditionSummary],
    path: str | Path,
    *,
    metric: str = "roc_auc",
    train_fraction: float = 1.0,
    dataset: str | None = None,
) -> Path | None:
    """Grouped bars: both models under both splits, at full training data.

    ``dataset`` restricts the figure to one dataset. It is not optional in
    practice: two datasets share the same (model, split) keys, so without it
    the bar drawn for a cell would be whichever dataset happened to sort
    first. :func:`make_all_figures` always passes it.

    Returns the written path, or ``None`` if no condition matched — a sweep
    that has not reached this cell yet should not leave a stale figure.
    """
    rows = [
        s
        for s in summaries
        if s.train_fraction == train_fraction
        and getattr(s, f"{metric}_mean") is not None
        and (dataset is None or s.dataset == dataset)
    ]
    if not rows:
        return None

    splits = sorted({s.split for s in rows})
    models = sorted({s.model for s in rows}, key=lambda m: m != "rf")
    # Thin marks: the group occupies a little over half the slot, leaving the
    # groups clearly separated rather than filling the axis.
    width = 0.55 / max(len(models), 1)

    figure, ax = _new_figure()
    tops: list[float] = []
    for index, model in enumerate(models):
        positions, heights, errors = [], [], []
        for group, split in enumerate(splits):
            match = next(
                (s for s in rows if s.split == split and s.model == model), None
            )
            if match is None:
                continue
            # A 2px surface gap between adjacent bars.
            positions.append(group + (index - (len(models) - 1) / 2) * width)
            heights.append(getattr(match, f"{metric}_mean"))
            errors.append(getattr(match, f"{metric}_std") or 0.0)
            tops.append(heights[-1] + errors[-1])

        bars = ax.bar(
            positions,
            heights,
            width=width * 0.92,
            label=_label(MODEL_LABELS, model),
            color=_color(model),
            yerr=errors if any(errors) else None,
            capsize=4,
            ecolor=_INK_SECONDARY,
            edgecolor=_SURFACE,
            linewidth=1.5,
        )
        # Direct value labels: few enough bars that every one can carry its
        # number without crowding, and they satisfy the relief rule.
        for bar, height in zip(bars, heights):
            ax.annotate(
                f"{height:.3f}",
                (bar.get_x() + bar.get_width() / 2, height),
                textcoords="offset points",
                xytext=(0, 4),
                ha="center",
                fontsize=9,
                color=_INK,
            )

    ax.set_xticks(range(len(splits)))
    ax.set_xticklabels([_label(SPLIT_LABELS, s) for s in splits])
    ax.set_ylim(*_bar_ylim(metric, tops))
    _style_axes(
        ax,
        title=f"Test {metric.replace('_', '-').upper()} by model and split{_suffix(dataset)}",
        xlabel="",
        ylabel=f"Test {metric.replace('_', '-').upper()} (mean ± SD over seeds)",
    )
    ax.legend(frameon=False, fontsize=9, labelcolor=_INK_SECONDARY)
    return _save(figure, Path(path))


# --------------------------------------------------------------------------- #
# Figure 2 — the additional experiment
# --------------------------------------------------------------------------- #


def plot_learning_curves(
    summaries: Sequence[ConditionSummary],
    path: str | Path,
    *,
    metric: str = "roc_auc",
    dataset: str | None = None,
) -> Path | None:
    """Test score against training-set fraction, per model and split.

    ``dataset`` restricts the figure to one dataset; see
    :func:`plot_split_comparison` for why mixing two is wrong.
    """
    rows = [
        s
        for s in summaries
        if getattr(s, f"{metric}_mean") is not None
        and (dataset is None or s.dataset == dataset)
    ]
    if not rows:
        return None

    figure, ax = _new_figure()
    drawn = 0
    for model in sorted({s.model for s in rows}, key=lambda m: m != "rf"):
        for split in sorted({s.split for s in rows}):
            series = sorted(
                (s for s in rows if s.model == model and s.split == split),
                key=lambda s: s.train_fraction,
            )
            if len(series) < 2:
                continue  # a single point is not a curve
            style = SPLIT_STYLES.get(split, {"linestyle": "-", "marker": "o"})
            ax.errorbar(
                [s.train_fraction * 100 for s in series],
                [getattr(s, f"{metric}_mean") for s in series],
                yerr=[getattr(s, f"{metric}_std") or 0.0 for s in series],
                label=f"{_label(MODEL_LABELS, model)} — {_label(SPLIT_LABELS, split).lower()}",
                color=_color(model),
                linewidth=_LINE_WIDTH,
                markersize=_MARKER_SIZE,
                capsize=3,
                elinewidth=1.0,
                **style,
            )
            drawn += 1

    if not drawn:
        plt.close(figure)
        return None

    ax.set_xscale("log")
    # Ticks come from the fractions actually run, so a partial sweep does not
    # advertise points that do not exist.
    ax.set_xticks(sorted({int(round(s.train_fraction * 100)) for s in rows}))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    _style_axes(
        ax,
        title=(
            f"Learning curves: test {metric.replace('_', '-').upper()} "
            f"vs training-set size{_suffix(dataset)}"
        ),
        xlabel="Share of the training set used (%)",
        ylabel=f"Test {metric.replace('_', '-').upper()} (mean ± SD over seeds)",
    )
    ax.legend(frameon=False, fontsize=9, labelcolor=_INK_SECONDARY, loc="lower right")
    return _save(figure, Path(path))


# --------------------------------------------------------------------------- #
# Figure 3 — the quantity the hypotheses are stated in
# --------------------------------------------------------------------------- #


def plot_gap(
    gaps: Sequence[GapRow],
    path: str | Path,
    *,
    metric: str = "roc_auc",
    dataset: str | None = None,
) -> Path | None:
    """GNN-minus-baseline against training-set size, one line per split.

    ``dataset`` restricts the figure to one dataset; see
    :func:`plot_split_comparison` for why mixing two is wrong.
    """
    rows = [
        g
        for g in gaps
        if g.metric == metric
        and g.gap is not None
        and (dataset is None or g.dataset == dataset)
    ]
    if not rows:
        return None

    figure, ax = _new_figure()
    # Zero is the claim boundary: above it the GNN wins, below it the baseline
    # does, so it gets a reference line rather than being left implicit.
    ax.axhline(0.0, color=_INK_SECONDARY, linewidth=1.2, zorder=1)

    # Here the split is the only distinction, so it takes the colour channel
    # as well as the line style.
    split_colors = {"random": MODEL_COLORS["rf"], "scaffold": MODEL_COLORS["gnn"]}
    for split in sorted({g.split for g in rows}):
        series = sorted(
            (g for g in rows if g.split == split), key=lambda g: g.train_fraction
        )
        style = SPLIT_STYLES.get(split, {"linestyle": "-", "marker": "o"})
        ax.errorbar(
            [g.train_fraction * 100 for g in series],
            [g.gap for g in series],
            yerr=[g.gap_std or 0.0 for g in series],
            label=_label(SPLIT_LABELS, split),
            color=split_colors.get(split, _INK_SECONDARY),
            linewidth=_LINE_WIDTH,
            markersize=_MARKER_SIZE,
            capsize=3,
            elinewidth=1.0,
            zorder=2,
            **style,
        )

    ax.set_xscale("log")
    ax.set_xticks(sorted({int(round(g.train_fraction * 100)) for g in rows}))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    _style_axes(
        ax,
        title=f"GNN minus baseline ({metric.replace('_', '-').upper()}){_suffix(dataset)}",
        xlabel="Share of the training set used (%)",
        ylabel="Difference in test score (GNN − baseline)",
    )
    ax.legend(frameon=False, fontsize=9, labelcolor=_INK_SECONDARY)
    return _save(figure, Path(path))


# --------------------------------------------------------------------------- #
# Everything at once
# --------------------------------------------------------------------------- #


def make_all_figures(
    summaries: Sequence[ConditionSummary],
    gaps: Sequence[GapRow],
    figures_dir: str | Path,
    *,
    metric: str = "roc_auc",
) -> list[Path]:
    """Draw all three figures per dataset, returning the paths actually written.

    One subdirectory per dataset, mirroring ``results/``:
    ``figures/tox21/random_vs_scaffold_tox21_roc_auc.png``. The dataset stays
    in the filename as well as the folder, because a figure is usually read
    after being pasted into a report or a slide, where the folder is gone and
    the filename is the only thing left saying which dataset it is.

    Two datasets must never share an axis here: their class balances and
    test-set sizes differ, so a shared y-axis would invite a comparison the
    numbers do not support. The cross-dataset comparison belongs in the
    tables.
    """
    directory = Path(figures_dir)
    datasets = sorted({s.dataset for s in summaries} | {g.dataset for g in gaps})

    written: list[Path | None] = []
    for dataset in datasets:
        into = directory / dataset
        written += [
            plot_split_comparison(
                summaries,
                into / f"random_vs_scaffold_{dataset}_{metric}.png",
                metric=metric,
                dataset=dataset,
            ),
            plot_learning_curves(
                summaries,
                into / f"learning_curves_{dataset}_{metric}.png",
                metric=metric,
                dataset=dataset,
            ),
            plot_gap(
                gaps,
                into / f"gnn_minus_rf_gap_{dataset}_{metric}.png",
                metric=metric,
                dataset=dataset,
            ),
        ]
    return [p for p in written if p is not None]
