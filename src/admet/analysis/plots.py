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
import numpy as np  # noqa: E402

from admet.analysis.aggregate import ConditionSummary, GapRow  # noqa: E402

__all__ = [
    "MODEL_COLORS",
    "MODEL_LABELS",
    "DATASET_LABELS",
    "SPLIT_STYLES",
    "SPLIT_COLORS",
    "max_tanimoto_to_train",
    "plot_train_test_similarity",
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
#:
#: The dummy is deliberately a neutral grey rather than a third hue: it is
#: the floor the two approaches are measured against, not a third approach.
MODEL_COLORS: dict[str, str] = {
    "rf": "#2a78d6",
    "gnn": "#eb6834",
    "dummy": "#8a8784",
}

#: Colour for the gap series, deliberately *not* a model hue.
#:
#: The gap is GNN-minus-baseline — a quantity about both models, so painting
#: it in either one's colour would imply it belongs to that model. This used
#: to reuse MODEL_COLORS for the two splits, which meant blue read as "random
#: forest" in two figures and "random split" in a third.
GAP_COLOR: str = "#3d3c3a"

MODEL_LABELS: dict[str, str] = {
    "dummy": "Dummy (class prior)",
    "rf": "Fingerprint + random forest",
    "gnn": "Graph neural network",
}

#: Left-to-right bar order: the floor, then the baseline, then the advanced
#: model, so a group reads in increasing order of what it is claiming.
MODEL_ORDER: tuple[str, ...] = ("dummy", "rf", "gnn")

#: Colours for the one figure whose *subject* is the split rather than the
#: model, so split has to become the colour channel there.
#:
#: Deliberately neither model hue. Blue and orange mean "random forest" and
#: "GNN" in every other figure, and reusing them for the two splits is the
#: exact confusion that had to be undone in `plot_gap`. These are slots 7 and
#: 6 of the same reference palette: the highest-separating pair available in
#: which *both* colours also clear 3:1 contrast on the surface (violet 5.6,
#: green 4.8), at CVD ΔE 28.8 and normal-vision ΔE 35.5 against thresholds of
#: 8 and 15. The yellow and aqua pairings separate slightly better but fall
#: under the contrast floor. Split is additionally encoded by line style
#: below, so identity never rests on colour alone.
SPLIT_COLORS: dict[str, str] = {"random": "#6250d6", "scaffold": "#008300"}

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

#: Smallest bar height, in axis units, that still reads as a bar rather than
#: as a gap between its neighbours. Used to keep a bar sitting on the ROC-AUC
#: chance floor visible.
_MIN_VISIBLE_BAR = 0.02


def _label(mapping: dict[str, str], key: str) -> str:
    """Human-readable name, falling back to the raw key for a new model."""
    return mapping.get(key, key)


def _model_sort_key(model: str) -> tuple[int, str]:
    """Order models by :data:`MODEL_ORDER`; unregistered ones sort last.

    Sorting by name is what put ``dummy`` between the baseline and the GNN,
    which reads as a third competing approach rather than as the floor.
    """
    if model in MODEL_ORDER:
        return MODEL_ORDER.index(model), model
    return len(MODEL_ORDER), model


def _bar_ylim(
    metric: str, tops: Sequence[float], heights: Sequence[float] = ()
) -> tuple[float, float]:
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

    ``heights`` is the bar heights without their error bars, and exists for
    the other end of the same failure. A bar *at* the baseline has zero
    height and disappears, leaving only a floating value label and an
    unexplained gap between its neighbours. The dummy model sits at exactly
    0.5 ROC-AUC by construction, so this is not a corner case: the floor
    drops just below the shortest bar whenever one would otherwise vanish.
    """
    if metric == "roc_auc":
        shortest = min(heights, default=1.0)
        if shortest > 0.5 + _MIN_VISIBLE_BAR:
            return 0.5, 1.0
        return max(0.0, shortest - _MIN_VISIBLE_BAR), 1.0
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


def _new_panels(
    n: int, size: tuple[float, float] = (9.5, 4.6)
) -> tuple[plt.Figure, list[plt.Axes]]:
    """One panel per split, sharing a y-axis.

    Small multiples rather than more series on one axis. Two models x two
    splits on a single pair of axes needs colour *and* line style to be read
    at once, and the four curves cross repeatedly; split out, each panel
    carries two curves told apart by colour alone.

    The y-axis is shared because the panels are the same measurement under
    two conditions — the drop from one panel to the next is the largest
    effect in the study, and an independent axis per panel would hide it.
    """
    figure, axes = plt.subplots(1, n, figsize=size, sharey=True)
    figure.patch.set_facecolor(_SURFACE)
    return figure, list(np.atleast_1d(axes))


def _seed_note(figure: plt.Figure, seeds: Sequence[int]) -> None:
    """State what the error bars are, where a reader will actually see it.

    The y-axis label says "mean ± SD over seeds", which is easy to miss and
    does not say how many seeds. An unexplained interval is worse than none:
    it reads as a drawing artefact rather than as the uncertainty that, in
    several of these figures, is the entire finding.
    """
    counts = sorted({s for s in seeds if s})
    if not counts:
        return
    n = f"{counts[0]}" if len(counts) == 1 else f"{min(counts)}–{max(counts)}"
    figure.text(
        0.01,
        0.015,
        f"Error bars: ±1 SD over {n} seeds.",
        fontsize=8,
        color=_INK_SECONDARY,
    )


def _save(figure: plt.Figure, path: Path, *, bottom: float = 0.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(rect=(0, bottom, 1, 1))
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
    models = sorted({s.model for s in rows}, key=_model_sort_key)
    # Thin marks: the group occupies under half the slot, leaving the groups
    # clearly separated rather than filling the axis with ink.
    width = 0.40 / max(len(models), 1)

    figure, ax = _new_figure()
    tops: list[float] = []
    all_heights: list[float] = []
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
            all_heights.append(heights[-1])

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
        #
        # Anchored to the top of the *error bar*, not the top of the bar, or
        # the two collide: with an SD of 0.009 on a 0.834 bar the cap lands
        # exactly where the text sits.
        for bar, height, error in zip(bars, heights, errors):
            ax.annotate(
                f"{height:.3f}",
                (bar.get_x() + bar.get_width() / 2, height + error),
                textcoords="offset points",
                xytext=(0, 5),
                ha="center",
                fontsize=9,
                color=_INK,
            )

    ax.set_xticks(range(len(splits)))
    ax.set_xticklabels([_label(SPLIT_LABELS, s) for s in splits])
    ax.set_ylim(*_bar_ylim(metric, tops, all_heights))
    _style_axes(
        ax,
        title=f"Test {metric.replace('_', '-').upper()} by model and split{_suffix(dataset)}",
        xlabel="",
        ylabel=f"Test {metric.replace('_', '-').upper()} (mean ± SD over seeds)",
    )
    ax.legend(frameon=False, fontsize=9, labelcolor=_INK_SECONDARY)
    _seed_note(figure, [s.n_seeds for s in rows])
    return _save(figure, Path(path), bottom=0.04)


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

    splits = sorted({s.split for s in rows})
    models = sorted({s.model for s in rows}, key=_model_sort_key)
    ticks = sorted({int(round(s.train_fraction * 100)) for s in rows})
    name = metric.replace("_", "-").upper()

    figure, axes = _new_panels(len(splits))
    drawn = 0
    for ax, split in zip(axes, splits):
        for model in models:
            series = sorted(
                (s for s in rows if s.model == model and s.split == split),
                key=lambda s: s.train_fraction,
            )
            if len(series) < 2:
                continue  # a single point is not a curve
            ax.errorbar(
                [s.train_fraction * 100 for s in series],
                [getattr(s, f"{metric}_mean") for s in series],
                yerr=[getattr(s, f"{metric}_std") or 0.0 for s in series],
                label=_label(MODEL_LABELS, model),
                color=_color(model),
                linestyle="-",
                marker="o",
                linewidth=_LINE_WIDTH,
                markersize=_MARKER_SIZE,
                capsize=3,
                elinewidth=1.0,
            )
            drawn += 1

        ax.set_xscale("log")
        # Ticks come from the fractions actually run, so a partial sweep does
        # not advertise points that do not exist.
        ax.set_xticks(ticks)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        _style_axes(
            ax,
            title=_label(SPLIT_LABELS, split),
            xlabel="Share of the training set used (%)",
            ylabel=f"Test {name} (mean ± SD over seeds)" if ax is axes[0] else "",
        )

    if not drawn:
        plt.close(figure)
        return None

    figure.suptitle(
        f"Learning curves: test {name} vs training-set size{_suffix(dataset)}",
        color=_INK,
        fontsize=12,
        x=0.01,
        ha="left",
    )
    # One legend for the whole figure, below the panels rather than inside
    # them: in-panel legends sat on top of the ClinTox curves.
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        frameon=False,
        fontsize=9,
        labelcolor=_INK_SECONDARY,
        loc="lower center",
        ncol=max(len(labels), 1),
        bbox_to_anchor=(0.5, 0.0),
    )
    _seed_note(figure, [s.n_seeds for s in rows])
    return _save(figure, Path(path), bottom=0.09)


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

    splits = sorted({g.split for g in rows})
    ticks = sorted({int(round(g.train_fraction * 100)) for g in rows})
    name = metric.replace("_", "-").upper()

    figure, axes = _new_panels(len(splits))
    for ax, split in zip(axes, splits):
        # Zero is the claim boundary: above it the GNN wins, below it the
        # baseline does, so it gets a reference line rather than being left
        # implicit.
        ax.axhline(0.0, color=_INK_SECONDARY, linewidth=1.2, zorder=1)

        series = sorted(
            (g for g in rows if g.split == split), key=lambda g: g.train_fraction
        )
        ax.errorbar(
            [g.train_fraction * 100 for g in series],
            [g.gap for g in series],
            yerr=[g.gap_std or 0.0 for g in series],
            color=GAP_COLOR,
            linestyle="-",
            marker="o",
            linewidth=_LINE_WIDTH,
            markersize=_MARKER_SIZE,
            capsize=3,
            elinewidth=1.0,
            zorder=2,
        )

        ax.set_xscale("log")
        ax.set_xticks(ticks)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        _style_axes(
            ax,
            title=_label(SPLIT_LABELS, split),
            xlabel="Share of the training set used (%)",
            ylabel="Difference in test score (GNN − baseline)"
            if ax is axes[0]
            else "",
        )

    figure.suptitle(
        f"GNN minus baseline ({name}){_suffix(dataset)}",
        color=_INK,
        fontsize=12,
        x=0.01,
        ha="left",
    )
    # One series per panel, so the panel title names it and no legend is
    # needed (a legend box for a single line is noise).
    _seed_note(figure, [g.n_seeds for g in rows])
    return _save(figure, Path(path), bottom=0.05)


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


# --------------------------------------------------------------------------- #
# Figure 4 — the mechanism behind the headline result
# --------------------------------------------------------------------------- #


def max_tanimoto_to_train(
    smiles: Sequence[str],
    *,
    kind: str,
    seed: int = 0,
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
    n_bits: int = 2048,
    radius: int = 2,
) -> np.ndarray:
    """For each test molecule, its Tanimoto similarity to its nearest neighbour
    in the training set.

    This is the quantity the scaffold split is *for*. The report says test
    molecules are structurally unlike the training molecules and then shows a
    0.073 ROC-AUC drop; this measures the first claim directly instead of
    inferring it from the second.

    Computed as a dense matrix: Tanimoto on bit vectors is
    ``|a & b| / |a | b|``, and with popcounts ``pa``, ``pb`` the union is
    ``pa + pb - intersection``, so one matrix product gives every pair. Tox21
    is 782 test molecules against 6258 training ones — a 20 MB float32 block.
    """
    from admet.data.splits import make_split
    from admet.features.fingerprints import featurize_smiles

    split = make_split(
        kind,
        smiles,
        frac_train=fractions[0],
        frac_valid=fractions[1],
        frac_test=fractions[2],
        seed=seed,
    )
    if not split.train or not split.test:
        return np.empty(0, dtype=float)

    matrix = featurize_smiles(smiles, radius=radius, n_bits=n_bits).astype(np.float32)
    train = matrix[list(split.train)]
    test = matrix[list(split.test)]

    popcount_train = train.sum(axis=1)
    popcount_test = test.sum(axis=1)
    intersection = test @ train.T
    union = popcount_test[:, None] + popcount_train[None, :] - intersection
    # A molecule whose fingerprint is all zeros gives a zero union; it has no
    # bits to share, so its similarity to anything is 0 rather than undefined.
    with np.errstate(divide="ignore", invalid="ignore"):
        tanimoto = np.where(union > 0, intersection / union, 0.0)
    return np.asarray(tanimoto.max(axis=1), dtype=float)


def plot_train_test_similarity(
    smiles: Sequence[str],
    path: str | Path,
    *,
    dataset: str | None = None,
    seed: int = 0,
    bins: int = 40,
) -> Path | None:
    """Nearest-training-neighbour similarity, random split versus scaffold split.

    The two distributions on one axis are the mechanism behind the main
    result: under a scaffold split the curve sits to the left, which is what
    "the test molecules are structurally different" means in numbers. Every
    other figure in this set plots a model score; this one plots the data, and
    it is the only one that would look the same if both models were deleted.

    Returns the written path, or ``None`` if the dataset is too small to split.
    """
    target = Path(path)
    series = {
        kind: max_tanimoto_to_train(smiles, kind=kind, seed=seed)
        for kind in ("random", "scaffold")
    }
    if any(values.size == 0 for values in series.values()):
        return None

    edges = np.linspace(0.0, 1.0, bins + 1)
    figure, ax = _new_figure()
    for kind, values in series.items():
        counts, _ = np.histogram(values, bins=edges)
        share = counts / counts.sum()
        style = SPLIT_STYLES[kind]
        ax.stairs(
            share,
            edges,
            color=SPLIT_COLORS[kind],
            linewidth=_LINE_WIDTH,
            linestyle=str(style["linestyle"]),
            fill=False,
            label=f"{_label(SPLIT_LABELS, kind)} (median {np.median(values):.2f})",
        )
        ax.stairs(share, edges, color=SPLIT_COLORS[kind], fill=True, alpha=0.12)

    _style_axes(
        ax,
        title=f"Similarity of each test molecule to its nearest "
        f"training molecule{_suffix(dataset)}",
        xlabel="Maximum Tanimoto similarity to any training molecule",
        ylabel="Share of test molecules",
    )
    ax.set_xlim(0.0, 1.0)
    ax.legend(frameon=False, fontsize=9, labelcolor=_INK_SECONDARY)
    figure.text(
        0.01,
        0.015,
        f"Morgan fingerprints, radius 2, 2048 bits. One seed (seed {seed}).",
        fontsize=8,
        color=_INK_SECONDARY,
    )
    return _save(figure, target, bottom=0.05)
