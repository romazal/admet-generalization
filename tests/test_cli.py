"""Tests for the command-line entry point.

`cli.py` is the layer every workflow in the README goes through, and it was
the last module with no test file of its own. These do not re-test the
pipeline underneath — `aggregate`, `plots` and the splitters have their own
suites. What is checked here is the wiring: that a command exists, is
reachable through `main`, returns the exit code a shell would act on, and
prints what the README says it prints.
"""

from __future__ import annotations

import pytest

from admet.cli import build_parser, main


def test_version_exits_cleanly_and_prints_the_package_version(capsys) -> None:
    """`--version` leaves through argparse's SystemExit, on the success path."""
    from admet import __version__

    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_datasets_lists_every_registered_dataset(capsys) -> None:
    """The command's whole purpose: naming what exists without reading source."""
    from admet.data.loaders import DATASETS

    assert main(["datasets"]) == 0
    out = capsys.readouterr().out
    for name in DATASETS:
        assert name in out, f"{name} missing from `admet datasets` output"


def test_datasets_reports_parsed_counts_not_registry_guesses(capsys) -> None:
    """The figure printed must be what the loader actually parsed.

    The registry says what a dataset should contain; the loader says what
    survived unreadable SMILES and duplicate merging. Printing the first
    while implying the second is how a stale number reaches a slide.
    """
    from admet.data.loaders import load_dataset

    main(["datasets"])
    out = capsys.readouterr().out
    for name in ("tox21", "clintox"):
        dataset = load_dataset(name)
        assert str(len(dataset)) in out
        assert str(dataset.n_tasks) in out


def test_datasets_needs_no_arguments() -> None:
    """It is the one command you can run before knowing anything else."""
    args = build_parser().parse_args(["datasets"])
    assert args.handler.__name__ == "command_datasets"


@pytest.mark.parametrize(
    ("command", "argv"),
    [
        ("run", ["run", "tox21_rf_random.yaml"]),
        ("sweep", ["sweep", "sweep_learning_curve.yaml"]),
        ("aggregate", ["aggregate"]),
        ("plots", ["plots"]),
        ("datasets", ["datasets"]),
    ],
)
def test_every_subcommand_parses_and_carries_a_handler(
    command: str, argv: list[str]
) -> None:
    """A subparser registered without a handler fails only when someone runs it.

    Parsing only — nothing is executed, so this stays fast and touches no
    results directory.
    """
    args = build_parser().parse_args(argv)
    assert getattr(args, "handler", None) is not None, f"{command} has no handler"
    assert callable(args.handler)


def test_an_unknown_command_is_rejected() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["definitely-not-a-command"])
    assert excinfo.value.code != 0
