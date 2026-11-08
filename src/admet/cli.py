"""Command-line entry point: ``admet <command>``.

Four commands cover the whole workflow:

``admet run <config>``        one configuration, over its seeds
``admet sweep <config>``      the learning-curve grid (proposal §3.3)
``admet aggregate``           results/*.json into summary tables
``admet plots``               the figures, from those tables

A run is fully specified by a config file plus a seed (proposal §7), so the
commands are thin: they parse arguments, call into :mod:`admet.training` or
:mod:`admet.analysis`, and print progress. Nothing here decides anything
scientific — overrides exist for convenience, and the configs remain the
record of what was actually run.

Every command is re-runnable. ``run`` and ``sweep`` skip a cell whose result
file already exists unless ``--overwrite`` is given, so an interrupted sweep
is resumed by issuing the same command again.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Sequence

from admet.training import load_config, run_experiment, run_name

__all__ = ["main", "build_parser"]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = _REPO_ROOT / "configs"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _resolve_config(value: str, *, relative_to: Path | None = None) -> Path:
    """Accept a path, or a bare config name resolved against known locations.

    ``relative_to`` is the directory of the config that named this one, which
    is tried first. A sweep config therefore refers to its children the same
    way ``extends:`` refers to its parent — by a name relative to itself —
    so a sweep outside ``configs/`` still finds its own siblings.
    """
    candidate = Path(value)
    if candidate.exists():
        return candidate

    guesses: list[Path] = []
    if relative_to is not None:
        guesses += [relative_to / value, relative_to / f"{value}.yaml"]
    guesses += [_CONFIG_DIR / value, _CONFIG_DIR / f"{value}.yaml"]
    for guess in guesses:
        if guess.exists():
            return guess

    searched = ", ".join(str(p) for p in {relative_to, _CONFIG_DIR} if p)
    raise SystemExit(
        f"config not found: {value}\n"
        f"Looked in the current directory and in: {searched}."
    )


def _results_dir(config: dict[str, Any], override: str | None) -> Path:
    if override:
        return Path(override)
    configured = Path(config["output"]["results_dir"])
    return configured if configured.is_absolute() else _REPO_ROOT / configured


def _figures_dir(config: dict[str, Any], override: str | None) -> Path:
    if override:
        return Path(override)
    configured = Path(config["output"]["figures_dir"])
    return configured if configured.is_absolute() else _REPO_ROOT / configured


def _execute(
    config: dict[str, Any],
    *,
    seed: int,
    train_fraction: float,
    results_dir: Path,
    overwrite: bool,
    index: str,
) -> bool:
    """Run one cell. Returns ``True`` if it completed, ``False`` on failure."""
    name = run_name(config, seed, train_fraction)
    target = results_dir / f"{name}.json"
    if target.exists() and not overwrite:
        print(f"  [{index}] {name}: already done, skipping")
        return True

    print(f"  [{index}] {name}: running...", flush=True)
    started = time.perf_counter()
    try:
        result = run_experiment(
            config,
            seed=seed,
            train_fraction=train_fraction,
            results_dir=results_dir,
            save=True,
        )
    except Exception:  # noqa: BLE001 - one bad cell must not kill a sweep
        print(f"  [{index}] {name}: FAILED", file=sys.stderr)
        traceback.print_exc()
        return False

    test = result["metrics"]["test"]
    roc, pr = test.get("roc_auc"), test.get("pr_auc")
    elapsed = time.perf_counter() - started
    print(
        f"  [{index}] {name}: "
        f"ROC-AUC {roc:.4f}  PR-AUC {pr:.4f}  [{elapsed:.1f}s]"
        if roc is not None and pr is not None
        else f"  [{index}] {name}: no assay could be scored  [{elapsed:.1f}s]"
    )
    return True


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #


def command_run(args: argparse.Namespace) -> int:
    """Run one config over its seeds and training fractions."""
    path = _resolve_config(args.config)
    config = load_config(path)

    seeds = args.seeds if args.seeds else config["experiment"]["seeds"]
    fractions = (
        args.train_fractions
        if args.train_fractions
        else config["experiment"]["train_fractions"]
    )
    results_dir = _results_dir(config, args.results_dir)

    cells = [(f, s) for f in fractions for s in seeds]
    print(
        f"{path.name}: {config['model']['name']} / {config['split']['kind']} split / "
        f"{config['dataset']['name']} - {len(cells)} run(s)"
    )

    failures = 0
    for position, (fraction, seed) in enumerate(cells, start=1):
        ok = _execute(
            config,
            seed=seed,
            train_fraction=fraction,
            results_dir=results_dir,
            overwrite=args.overwrite,
            index=f"{position}/{len(cells)}",
        )
        failures += not ok

    print(f"Done. Results in {results_dir}" + (f" ({failures} failed)" if failures else ""))
    return 1 if failures else 0


def command_sweep(args: argparse.Namespace) -> int:
    """Run the grid defined by a sweep config's ``sweep:`` block."""
    path = _resolve_config(args.config)
    config = load_config(path)

    sweep = config.get("sweep")
    if not sweep:
        raise SystemExit(
            f"{path.name} has no `sweep:` block. Use `admet run` for a single "
            f"config, or see configs/sweep_learning_curve.yaml."
        )

    child_names = sweep.get("configs") or []
    fractions = args.train_fractions or sweep.get("train_fractions", [1.0])
    seeds = args.seeds or sweep.get("seeds", config["experiment"]["seeds"])
    results_dir = _results_dir(config, args.results_dir)

    children = [
        (name, load_config(_resolve_config(name, relative_to=path.parent)))
        for name in child_names
    ]
    total = len(children) * len(fractions) * len(seeds)
    print(
        f"{path.name}: {len(children)} config(s) x {len(fractions)} fraction(s) "
        f"x {len(seeds)} seed(s) = {total} run(s)"
    )

    failures = 0
    position = 0
    for name, child in children:
        print(f"\n{name}")
        for fraction in fractions:
            for seed in seeds:
                position += 1
                ok = _execute(
                    child,
                    seed=seed,
                    train_fraction=fraction,
                    results_dir=results_dir,
                    overwrite=args.overwrite,
                    index=f"{position}/{total}",
                )
                failures += not ok

    print(f"\nDone. Results in {results_dir}" + (f" ({failures} failed)" if failures else ""))
    return 1 if failures else 0


def command_aggregate(args: argparse.Namespace) -> int:
    """Build ``runs.csv``, ``summary.csv`` and ``gap.csv`` from ``results/``."""
    from admet.analysis.aggregate import aggregate

    config = load_config(_resolve_config(args.config))
    results_dir = _results_dir(config, args.results_dir)

    records, summaries, gaps = aggregate(results_dir, write=True)
    if not records:
        print(f"No result files in {results_dir}. Run an experiment first.")
        return 1

    print(f"{len(records)} run(s) -> {len(summaries)} condition(s)\n")
    header = f"{'model':<6} {'split':<9} {'frac':>5} {'seeds':>5} {'ROC-AUC':>16} {'PR-AUC':>16}"
    print(header)
    print("-" * len(header))
    for summary in summaries:
        roc = _format_pm(summary.roc_auc_mean, summary.roc_auc_std)
        pr = _format_pm(summary.pr_auc_mean, summary.pr_auc_std)
        print(
            f"{summary.model:<6} {summary.split:<9} "
            f"{summary.train_fraction:>5.2f} {summary.n_seeds:>5} {roc:>16} {pr:>16}"
        )

    if gaps:
        print("\nGNN minus baseline (ROC-AUC):")
        for row in gaps:
            if row.metric != "roc_auc" or row.gap is None:
                continue
            print(
                f"  {row.split:<9} frac {row.train_fraction:>4.2f}: "
                f"{row.gap:+.4f}" + (f" +/- {row.gap_std:.4f}" if row.gap_std else "")
            )

    print(f"\nTables written to {results_dir}")
    return 0


def _format_pm(mean: float | None, std: float | None) -> str:
    if mean is None:
        return "n/a"
    return f"{mean:.4f} +/- {std:.4f}" if std is not None else f"{mean:.4f}"


def command_plots(args: argparse.Namespace) -> int:
    """Redraw every figure from the aggregated tables."""
    from admet.analysis.aggregate import aggregate
    from admet.analysis.plots import make_all_figures

    config = load_config(_resolve_config(args.config))
    results_dir = _results_dir(config, args.results_dir)
    figures_dir = _figures_dir(config, args.figures_dir)

    records, summaries, gaps = aggregate(results_dir, write=False)
    if not records:
        print(f"No result files in {results_dir}. Run an experiment first.")
        return 1

    written = []
    for metric in args.metrics:
        written.extend(make_all_figures(summaries, gaps, figures_dir, metric=metric))

    if not written:
        print("Nothing to plot yet - no condition had a scoreable result.")
        return 1
    for path in written:
        print(f"wrote {path}")
    return 0


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    """The ``admet`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="admet",
        description=(
            "ADMET generalization study: when does a GNN beat a fingerprint "
            "model, and does the benefit survive a structural shift?"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument(
            "--results-dir",
            default=None,
            help="override output.results_dir from the config",
        )

    run = subparsers.add_parser("run", help="run one config over its seeds")
    run.add_argument("config", help="config file, or a name under configs/")
    run.add_argument(
        "--seeds", type=int, nargs="+", default=None, help="override the config's seeds"
    )
    run.add_argument(
        "--train-fractions",
        type=float,
        nargs="+",
        default=None,
        help="override the config's training fractions",
    )
    run.add_argument(
        "--overwrite", action="store_true", help="redo runs that already have a result"
    )
    add_common(run)
    run.set_defaults(handler=command_run)

    sweep = subparsers.add_parser("sweep", help="run a grid of configs")
    sweep.add_argument("config", help="a config with a `sweep:` block")
    sweep.add_argument("--seeds", type=int, nargs="+", default=None)
    sweep.add_argument("--train-fractions", type=float, nargs="+", default=None)
    sweep.add_argument("--overwrite", action="store_true")
    add_common(sweep)
    sweep.set_defaults(handler=command_sweep)

    aggregate_cmd = subparsers.add_parser(
        "aggregate", help="summarize results/ into CSV tables"
    )
    aggregate_cmd.add_argument("--config", default="base.yaml")
    add_common(aggregate_cmd)
    aggregate_cmd.set_defaults(handler=command_aggregate)

    plots = subparsers.add_parser("plots", help="redraw the figures")
    plots.add_argument("--config", default="base.yaml")
    plots.add_argument(
        "--figures-dir", default=None, help="override output.figures_dir"
    )
    plots.add_argument(
        "--metrics",
        nargs="+",
        default=["roc_auc", "pr_auc"],
        help="metrics to draw figures for",
    )
    add_common(plots)
    plots.set_defaults(handler=command_plots)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and dispatch. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
