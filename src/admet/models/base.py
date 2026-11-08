"""The shared model interface, and the tuning budget both models share.

Every model exposes the same three operations — :meth:`Model.featurize`,
:meth:`Model.fit`, :meth:`Model.predict_proba` — so ``training.py`` never
learns whether it is running a random forest or a GNN, and a new model (the
pretrained stretch goal of proposal §3.2) can be added without touching the
trainer.

Featurization belongs to the model rather than to the trainer because the two
models consume different things: the baseline wants a fingerprint matrix and
the GNN wants molecular graphs. The trainer only ever passes SMILES.

Equal tuning budgets
--------------------
Proposal §3.4 requires that both models receive the *same* hyperparameter
tuning budget — otherwise a win could just mean one model was tuned harder.
:func:`sample_configs` is the single mechanism for that: each model declares a
:meth:`Model.search_space`, the trainer draws the same number of candidates
from it, and each candidate is scored on a validation split **of the same type
as the test split** (a scaffold validation set for a scaffold run), so
hyperparameters are not selected on an easier regime than the one they are
judged in.

Sampling is random search rather than a grid: with a fixed budget, random
search covers the parameters that matter better than an exhaustive grid
truncated at the same count. Draws are deduplicated, and a space smaller than
the budget is enumerated in full instead of resampling the same point.
"""

from __future__ import annotations

import itertools
import random
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, Callable

import numpy as np

__all__ = [
    "Model",
    "MODELS",
    "register_model",
    "build_model",
    "sample_configs",
]


# --------------------------------------------------------------------------- #
# Interface
# --------------------------------------------------------------------------- #


class Model(ABC):
    """A multitask binary classifier over molecules.

    Subclasses are constructed with ``n_tasks`` plus their own
    hyperparameters, and must tolerate ``NaN`` in the label matrix: a molecule
    with a missing label for an assay has to be excluded from *that assay
    only*, in the loss exactly as in the metrics (proposal §6).
    """

    #: Registry key, as written in a config file.
    name: str = "model"

    def __init__(self, n_tasks: int, **hyperparameters: Any) -> None:
        if n_tasks <= 0:
            raise ValueError(f"n_tasks must be positive, got {n_tasks!r}")
        self.n_tasks = n_tasks
        self.hyperparameters = dict(hyperparameters)

    # -- representation ---------------------------------------------------- #

    @abstractmethod
    def featurize(self, smiles: Sequence[str]) -> Any:
        """Convert SMILES into whatever this model consumes.

        The return value is passed back to :meth:`fit` and
        :meth:`predict_proba` unchanged, so its type is the model's own
        business.
        """

    def featurization_key(self) -> tuple:
        """Identifies what :meth:`featurize` would produce, for caching.

        Two models with equal keys must produce interchangeable features. The
        trainer featurizes once per distinct key instead of once per tuning
        trial, which matters for the baseline: most of its search space leaves
        the fingerprints untouched, so recomputing them every trial would be
        pure waste. The default assumes featurization depends only on the
        model class; a model whose search space includes representation
        settings must override this to include them.
        """
        return (type(self).__name__,)

    # -- training ---------------------------------------------------------- #

    @abstractmethod
    def fit(
        self,
        features: Any,
        y: np.ndarray,
        *,
        valid_features: Any | None = None,
        valid_y: np.ndarray | None = None,
    ) -> "Model":
        """Fit on ``features``/``y``, returning ``self``.

        Args:
            features: output of :meth:`featurize` for the training molecules.
            y: ``(n_molecules, n_tasks)`` labels with ``NaN`` for *not screened*.
            valid_features, valid_y: the validation part, for models that use
                it (early stopping). Models that do not may ignore it.
        """

    @abstractmethod
    def predict_proba(self, features: Any) -> np.ndarray:
        """Predicted positive-class probabilities.

        Returns:
            An ``(n_molecules, n_tasks)`` array of finite scores. A prediction
            is owed for every molecule and every assay, including assays the
            model saw no positives for during training.
        """

    # -- tuning ------------------------------------------------------------ #

    @classmethod
    def search_space(cls) -> dict[str, Sequence[Any]]:
        """Hyperparameters to sample during tuning.

        Returns a mapping of name to candidate values. An empty space means
        the model has nothing to tune and is fitted once with its defaults.
        """
        return {}

    def __repr__(self) -> str:
        settings = ", ".join(f"{k}={v!r}" for k, v in sorted(self.hyperparameters.items()))
        return f"{type(self).__name__}(n_tasks={self.n_tasks}, {settings})"


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

#: Models selectable by name from a config file.
MODELS: dict[str, type[Model]] = {}


def register_model(cls: type[Model]) -> type[Model]:
    """Class decorator adding ``cls`` to :data:`MODELS` under its ``name``."""
    if not cls.name or cls.name == "model":
        raise ValueError(f"{cls.__name__} must set a distinct class-level `name`")
    if cls.name in MODELS and MODELS[cls.name] is not cls:
        raise ValueError(f"a different model is already registered as {cls.name!r}")
    MODELS[cls.name] = cls
    return cls


def build_model(name: str, n_tasks: int, **hyperparameters: Any) -> Model:
    """Construct a registered model by name.

    Raises:
        ValueError: if ``name`` is not registered.
    """
    # Importing the implementations here keeps the registry populated without
    # making this module import torch at module scope.
    from admet.models import gnn as _gnn  # noqa: F401
    from admet.models import rf as _rf  # noqa: F401

    try:
        cls = MODELS[name]
    except KeyError:
        known = ", ".join(sorted(MODELS))
        raise ValueError(
            f"unknown model {name!r}; expected one of: {known}"
        ) from None
    return cls(n_tasks=n_tasks, **hyperparameters)


# --------------------------------------------------------------------------- #
# Tuning budget
# --------------------------------------------------------------------------- #


def sample_configs(
    space: Mapping[str, Sequence[Any]],
    budget: int,
    *,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Draw up to ``budget`` distinct hyperparameter settings from ``space``.

    Both models are sampled through this one function with the same ``budget``,
    which is what makes the tuning effort comparable (proposal §3.4).

    Args:
        space: hyperparameter name to candidate values. An empty mapping, or a
            space in which every parameter has a single value, yields exactly
            one configuration.
        budget: maximum number of configurations to return; must be positive.
        seed: seeds the draw, so a run's tuning is reproducible.

    Returns:
        A list of ``1 <= k <= budget`` distinct configurations. The model's
        default configuration is **not** included automatically — put the
        defaults in the space if they should be candidates.

    Raises:
        ValueError: if ``budget`` is not positive, or a parameter has no
            candidate values.
    """
    if budget <= 0:
        raise ValueError(f"budget must be positive, got {budget!r}")

    names = sorted(space)
    options = []
    for name in names:
        values = list(space[name])
        if not values:
            raise ValueError(f"hyperparameter {name!r} has no candidate values")
        options.append(values)

    if not names:
        return [{}]

    total = 1
    for values in options:
        total *= len(values)

    rng = random.Random(seed)

    # Small space: enumerate it and take a random subset, so the budget is
    # never spent redrawing points already seen.
    if total <= budget:
        grid = [dict(zip(names, combo)) for combo in itertools.product(*options)]
        rng.shuffle(grid)
        return grid

    seen: set[tuple] = set()
    configs: list[dict[str, Any]] = []
    # Bounded so a space of near-duplicate points cannot spin forever.
    for _ in range(budget * 50):
        if len(configs) >= budget:
            break
        choice = tuple(rng.choice(values) for values in options)
        if choice in seen:
            continue
        seen.add(choice)
        configs.append(dict(zip(names, choice)))
    return configs
