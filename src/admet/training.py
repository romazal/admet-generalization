"""Run one experiment: config plus seed in, a result file out.

One configuration file and one seed fully specify a run (proposal §7), and a
run is the unit everything else is built from — the learning-curve sweep is
just many runs, and ``analysis/`` only ever reads the files this module
writes.

The trainer is deliberately model-agnostic: it talks to
:class:`admet.models.base.Model` and never learns whether it is running a
random forest or a GNN, which is what lets a third model be added without
touching it.

What one run does
-----------------
1. Seed :mod:`random`, NumPy and torch from the run seed.
2. Load the dataset and build the split, applying the learning-curve training
   fraction to the **training set only**.
3. Re-check scaffold disjointness at run time for a scaffold split (§3.4) —
   the unit tests cover the splitter, this covers the actual run.
4. Tune: draw ``tuning.budget`` hyperparameter settings and score each on the
   validation set, which is of the **same split type as the test set**, so
   hyperparameters are never selected under an easier regime than the one
   they are judged in.
5. Evaluate the best model on the test set and write a result file holding the
   config, the seed, the split, every trial's score, the metrics, library
   versions and timings.

The selected model is the one fitted on the training set alone; it is not
refitted on train plus validation. Refitting would change the amount of
training data between the tuning and final stages, which would muddle H2,
whose whole subject is how performance depends on training-set size.
"""

from __future__ import annotations

import json
import math
import os
import platform
import random
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from admet.data.loaders import Dataset, load_dataset
from admet.data.splits import Split, assert_scaffold_disjoint, make_split
from admet.evaluation import METRICS, PRIMARY_METRIC, evaluate
from admet.models.base import Model, build_model, sample_configs

__all__ = [
    "DEFAULT_CONFIG",
    "deep_merge",
    "load_config",
    "run_name",
    "run_experiment",
    "set_global_seeds",
]


#: Defaults applied under every config, so ``configs/base.yaml`` only has to
#: state what the project actually chooses and an experiment config only has
#: to state what differs from the base.
DEFAULT_CONFIG: dict[str, Any] = {
    "dataset": {"name": "tox21", "tasks": None},
    "split": {
        "kind": "scaffold",
        "fractions": [0.8, 0.1, 0.1],
        "params": {},
    },
    "model": {"name": "rf", "params": {}},
    "tuning": {"budget": 8, "metric": PRIMARY_METRIC},
    "experiment": {"seeds": [0, 1, 2], "train_fractions": [1.0]},
    "output": {"results_dir": "results", "figures_dir": "figures"},
}


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` onto ``base``, returning a new dict.

    Nested mappings are merged key by key; anything else is replaced outright,
    so a config can override one tuning setting without restating the others.
    """
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config, merged onto its base and onto :data:`DEFAULT_CONFIG`.

    A config may name its parent with a top-level ``extends:`` key, resolved
    relative to the config's own directory. This is how the experiment configs
    share ``configs/base.yaml`` instead of copying it (proposal §5).

    Raises:
        FileNotFoundError: if the config or a parent it extends is missing.
        ValueError: if the file does not contain a mapping, or ``extends``
            forms a cycle.
    """
    import yaml

    def _read(target: Path, seen: tuple[Path, ...]) -> dict[str, Any]:
        target = target.resolve()
        if target in seen:
            chain = " -> ".join(p.name for p in (*seen, target))
            raise ValueError(f"`extends` forms a cycle: {chain}")
        if not target.exists():
            raise FileNotFoundError(f"config not found: {target}")

        with target.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        if not isinstance(raw, Mapping):
            raise ValueError(f"{target} must contain a mapping at the top level")

        raw = dict(raw)
        parent_name = raw.pop("extends", None)
        if parent_name is None:
            return raw
        parent = _read(target.parent / str(parent_name), (*seen, target))
        return deep_merge(parent, raw)

    return deep_merge(DEFAULT_CONFIG, _read(Path(path), ()))


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #


def set_global_seeds(seed: int) -> None:
    """Seed every global RNG a run can touch, and pin torch to deterministic
    kernels.

    The splitters seed themselves (see :mod:`admet.data.splits`), so the
    seeding here is for everything else: scikit-learn, torch initialisation
    and shuffling.

    Seeding alone is not enough on a GPU, and this was measured rather than
    assumed. Before the determinism settings below, three runs of the *same*
    config, seed, machine and torch build gave test ROC-AUCs of 0.7369,
    0.7413 and 0.7690 — a spread of 0.032, wider than the 0.011 across-seed
    SD the report quotes as uncertainty, and wide enough to change which
    hyperparameters won the tuning search. The cause is float addition order:
    the scatter-add in message passing (`index_add`) and several cuBLAS
    kernels pick an order per launch, so "same seed" did not mean "same
    numbers". Pinning the kernels removes it; two runs now agree bit for bit.

    ``CUBLAS_WORKSPACE_CONFIG`` has to be in the environment before cuBLAS
    initialises, which is why it is set here rather than in a shell script —
    this runs before any model is built.
    """
    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dependency
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - no GPU in CI
        torch.cuda.manual_seed_all(seed)

    if os.environ.get("ADMET_NONDETERMINISTIC") == "1":
        # An escape hatch, not a convenience. `use_deterministic_algorithms`
        # raises on any op with no deterministic kernel, so a future torch or
        # a different GPU could stop this pipeline dead; the variable lets
        # someone finish a run and report the breakage. A result produced
        # with it set records `deterministic: false` and is not comparable
        # with the committed ones.
        return
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def deterministic_mode() -> bool:
    """Whether this process pinned torch to deterministic kernels."""
    if os.environ.get("ADMET_NONDETERMINISTIC") == "1":
        return False
    try:
        import torch
    except ImportError:  # pragma: no cover
        return True
    return bool(torch.are_deterministic_algorithms_enabled())


def _library_versions() -> dict[str, str]:
    """Versions of everything whose behaviour could change a number."""
    import importlib

    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for module in ("numpy", "sklearn", "rdkit", "torch", "yaml"):
        try:
            versions[module] = getattr(
                importlib.import_module(module), "__version__", "unknown"
            )
        except ImportError:
            versions[module] = "not installed"

    # The compute device, not just the library version. A torch build string
    # says what *could* run on a GPU; this says what did. Without it a result
    # file cannot explain why two runs with the same seed and config differ,
    # and the CPU and CUDA runs of this project do differ.
    try:
        import torch

        if torch.cuda.is_available():
            versions["torch_device"] = f"cuda:{torch.cuda.get_device_name(0)}"
        else:
            versions["torch_device"] = "cpu"
    except Exception:  # noqa: BLE001 - provenance must never fail a run
        versions["torch_device"] = "unknown"

    return versions


def _git_commit() -> str | None:
    """Current commit, so a result can be traced back to the code."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=Path(__file__).resolve().parents[2],
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


# --------------------------------------------------------------------------- #
# Naming
# --------------------------------------------------------------------------- #


def run_name(config: Mapping[str, Any], seed: int, train_fraction: float) -> str:
    """Stable identifier for a run, used as the result filename stem."""
    percent = int(round(train_fraction * 100))
    return (
        f"{config['dataset']['name']}"
        f"_{config['model']['name']}"
        f"_{config['split']['kind']}"
        f"_frac{percent:03d}"
        f"_seed{seed}"
    )


# --------------------------------------------------------------------------- #
# Tuning
# --------------------------------------------------------------------------- #


def _featurize_cached(
    model: Model, smiles: Sequence[str], cache: dict[tuple, Any], part: str
) -> Any:
    """Featurize ``smiles``, reusing the result across trials where possible."""
    key = (*model.featurization_key(), part)
    if key not in cache:
        cache[key] = model.featurize(smiles)
    return cache[key]


def _tune(
    *,
    model_name: str,
    base_params: Mapping[str, Any],
    dataset: Dataset,
    split: Split,
    seed: int,
    budget: int,
    metric: str,
) -> tuple[Model, dict[str, Any], list[dict[str, Any]]]:
    """Fit every sampled configuration and keep the best on validation.

    Returns the fitted best model, its hyperparameters, and one record per
    trial, so the result file shows the whole search and not only its winner.
    """
    if metric not in METRICS:
        raise ValueError(
            f"unknown tuning metric {metric!r}; expected one of {', '.join(METRICS)}"
        )

    model_class = type(build_model(model_name, n_tasks=dataset.n_tasks))
    candidates = sample_configs(model_class.search_space(), budget, seed=seed)

    train_smiles = [dataset.smiles[i] for i in split.train]
    valid_smiles = [dataset.smiles[i] for i in split.valid]
    train_y = dataset.y[list(split.train)]
    valid_y = dataset.y[list(split.valid)]
    has_validation = len(split.valid) > 0

    cache: dict[tuple, Any] = {}
    trials: list[dict[str, Any]] = []
    best_model: Model | None = None
    best_params: dict[str, Any] = {}
    best_score = -math.inf

    for index, candidate in enumerate(candidates):
        params = {**base_params, **candidate}
        model = build_model(
            model_name, n_tasks=dataset.n_tasks, seed=seed, **params
        )

        started = time.perf_counter()
        train_features = _featurize_cached(model, train_smiles, cache, "train")
        valid_features = (
            _featurize_cached(model, valid_smiles, cache, "valid")
            if has_validation
            else None
        )
        model.fit(
            train_features,
            train_y,
            valid_features=valid_features,
            valid_y=valid_y if has_validation else None,
        )

        if has_validation:
            scored = evaluate(valid_y, model.predict_proba(valid_features), tasks=dataset.tasks)
            value = scored[metric]
        else:
            value = None
        elapsed = time.perf_counter() - started

        trials.append(
            {
                "trial": index,
                "params": candidate,
                f"valid_{metric}": value,
                "seconds": round(elapsed, 2),
            }
        )

        # Without a validation set there is nothing to select on, so the first
        # (and, with an empty search space, only) candidate is kept.
        rank = -math.inf if value is None else value
        if best_model is None or rank > best_score:
            best_model, best_params, best_score = model, params, rank

    assert best_model is not None  # sample_configs always returns >= 1 config
    return best_model, best_params, trials


# --------------------------------------------------------------------------- #
# One run
# --------------------------------------------------------------------------- #


def run_experiment(
    config: Mapping[str, Any],
    *,
    seed: int = 0,
    train_fraction: float = 1.0,
    results_dir: str | Path | None = None,
    save: bool = True,
) -> dict[str, Any]:
    """Execute one run and return (and optionally save) its result.

    Args:
        config: a merged config, as returned by :func:`load_config`.
        seed: the run seed; drives the split, the tuning draw and model init.
        train_fraction: learning-curve fraction, applied to the training set
            only (proposal §3.3).
        results_dir: where to write the result file; defaults to the config's
            ``output.results_dir``, resolved from the repository root.
        save: set ``False`` to compute a result without writing it.

    Returns:
        The result dictionary, which is also what gets written as JSON.
    """
    started = time.perf_counter()
    set_global_seeds(seed)

    dataset_config = config["dataset"]
    split_config = config["split"]
    model_config = config["model"]
    tuning_config = config["tuning"]

    dataset = load_dataset(dataset_config["name"], tasks=dataset_config.get("tasks"))

    frac_train, frac_valid, frac_test = split_config["fractions"]
    split = make_split(
        split_config["kind"],
        dataset.smiles,
        frac_train=frac_train,
        frac_valid=frac_valid,
        frac_test=frac_test,
        seed=seed,
        train_fraction=train_fraction,
        **split_config.get("params", {}),
    )

    # §3.4: the unit tests prove the splitter is correct; this proves the
    # split this run actually used is clean.
    leakage_checked = split_config["kind"] == "scaffold"
    if leakage_checked:
        assert_scaffold_disjoint(dataset.smiles, split)

    tuning_started = time.perf_counter()
    model, best_params, trials = _tune(
        model_name=model_config["name"],
        base_params=model_config.get("params", {}) or {},
        dataset=dataset,
        split=split,
        seed=seed,
        budget=int(tuning_config["budget"]),
        metric=tuning_config.get("metric", PRIMARY_METRIC),
    )
    tuning_seconds = time.perf_counter() - tuning_started

    test_smiles = [dataset.smiles[i] for i in split.test]
    test_features = model.featurize(test_smiles)
    test_scores = model.predict_proba(test_features)
    test_result = evaluate(dataset.y[list(split.test)], test_scores, tasks=dataset.tasks)

    valid_result = None
    if split.valid:
        valid_features = model.featurize([dataset.smiles[i] for i in split.valid])
        valid_result = evaluate(
            dataset.y[list(split.valid)],
            model.predict_proba(valid_features),
            tasks=dataset.tasks,
        )

    name = run_name(config, seed, train_fraction)
    result: dict[str, Any] = {
        "run": name,
        "dataset": dataset_config["name"],
        "model": model_config["name"],
        "split": split_config["kind"],
        "seed": seed,
        "train_fraction": train_fraction,
        "metrics": {
            "test": test_result.as_dict(),
            "valid": valid_result.as_dict() if valid_result else None,
        },
        "split_sizes": {
            "train": len(split.train),
            "valid": len(split.valid),
            "test": len(split.test),
        },
        "scaffold_disjointness_checked": leakage_checked,
        "deterministic": deterministic_mode(),
        "best_params": best_params,
        "tuning": {
            "budget": int(tuning_config["budget"]),
            "metric": tuning_config.get("metric", PRIMARY_METRIC),
            "n_trials": len(trials),
            "trials": trials,
        },
        "dataset_summary": dataset.summary(),
        "config": dict(config),
        "versions": _library_versions(),
        "git_commit": _git_commit(),
        "timing_seconds": {
            "tuning": round(tuning_seconds, 2),
            "total": round(time.perf_counter() - started, 2),
        },
    }

    # Model-specific diagnostics worth keeping with the numbers they explain.
    degenerate = getattr(model, "degenerate_tasks", None)
    if degenerate:
        result["degenerate_tasks"] = {
            dataset.tasks[i]: reason for i, reason in degenerate.items()
        }
    if getattr(model, "best_epoch", None) is not None:
        result["best_epoch"] = model.best_epoch

    if save:
        base = Path(results_dir) if results_dir else _results_dir(config)
        # One subdirectory per dataset. 136 files in a flat directory is hard
        # to look through, and the dataset is the first thing a reader wants
        # to separate on. The aggregation recurses, so nothing else changes.
        directory = base / config["dataset"]["name"]
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{name}.json"
        path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        result["result_path"] = str(path)

    return result


def _results_dir(config: Mapping[str, Any]) -> Path:
    """Resolve ``output.results_dir`` against the repository root."""
    configured = Path(config["output"]["results_dir"])
    if configured.is_absolute():
        return configured
    return Path(__file__).resolve().parents[2] / configured
