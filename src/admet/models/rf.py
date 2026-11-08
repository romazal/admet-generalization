"""The baseline: Morgan fingerprints into a random forest, one forest per assay.

This is the classical model the GNN has to beat (proposal §3.2). It is
deliberately a *strong* baseline rather than a straw man: it tunes its own
representation (fingerprint radius and length) alongside the forest, and it
draws from :func:`admet.models.base.sample_configs` under the same budget as
the GNN, because a comparison where only the advanced model was tuned would
answer nothing.

Multitask by independent forests
--------------------------------
A random forest has no natural multitask head, so the baseline fits one
classifier per assay. Each forest is trained **only on the molecules that have
a label for its assay**, which is how the masking rule of
:mod:`admet.evaluation` is honoured on this side of the comparison: a missing
label removes a molecule from that assay's training set and nothing else.

Not sharing information between assays is exactly the asymmetry the study is
about — the GNN's multitask head does share it — so it is a property of the
baseline, not a defect in it.

Degenerate assays
-----------------
Under a scaffold split with a small training fraction, an assay can end up with
no positives among the labelled training molecules. Scikit-learn would refuse
to fit, so such an assay falls back to a constant prediction at the observed
class rate. The fallback is recorded in :attr:`RandomForestModel.degenerate_tasks`
and surfaces in the result file, since a constant prediction scores ROC-AUC 0.5
and should be visible rather than read as a model that learned nothing.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from admet.features.fingerprints import DEFAULT_N_BITS, DEFAULT_RADIUS
from admet.features.fingerprints import featurize_smiles as morgan_matrix
from admet.models.base import Model, register_model

__all__ = ["RandomForestModel"]


@register_model
class RandomForestModel(Model):
    """Morgan fingerprints + one :class:`RandomForestClassifier` per assay."""

    name = "rf"

    def __init__(
        self,
        n_tasks: int,
        *,
        seed: int = 0,
        n_estimators: int = 500,
        max_depth: int | None = None,
        min_samples_leaf: int = 1,
        max_features: str | float = "sqrt",
        class_weight: str | None = None,
        fp_radius: int = DEFAULT_RADIUS,
        fp_n_bits: int = DEFAULT_N_BITS,
        fp_counts: bool = False,
        n_jobs: int = -1,
    ) -> None:
        super().__init__(
            n_tasks,
            seed=seed,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight=class_weight,
            fp_radius=fp_radius,
            fp_n_bits=fp_n_bits,
            fp_counts=fp_counts,
        )
        self.seed = seed
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.class_weight = class_weight
        self.fp_radius = fp_radius
        self.fp_n_bits = fp_n_bits
        self.fp_counts = fp_counts
        self.n_jobs = n_jobs

        #: Per-assay fitted forest, or ``None`` where the fallback was used.
        self.estimators_: list[RandomForestClassifier | None] = []
        #: Constant score used for assays that could not be fitted.
        self.fallbacks_: list[float] = []
        #: Assays that fell back to a constant prediction, and why.
        self.degenerate_tasks: dict[int, str] = {}

    # -- representation ---------------------------------------------------- #

    def featurize(self, smiles: Sequence[str]) -> np.ndarray:
        """Morgan fingerprints as an ``(n_molecules, fp_n_bits)`` matrix."""
        return morgan_matrix(
            smiles,
            radius=self.fp_radius,
            n_bits=self.fp_n_bits,
            use_counts=self.fp_counts,
        )

    def featurization_key(self) -> tuple:
        """Fingerprints depend on these three settings, all of them tunable."""
        return ("rf", self.fp_radius, self.fp_n_bits, self.fp_counts)

    # -- training ---------------------------------------------------------- #

    def _new_forest(self, task: int) -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            class_weight=self.class_weight,
            n_jobs=self.n_jobs,
            # Offset per assay so the 12 forests are not identically seeded,
            # while the whole model stays a deterministic function of `seed`.
            random_state=self.seed * 1000 + task,
        )

    def fit(
        self,
        features: np.ndarray,
        y: np.ndarray,
        *,
        valid_features: np.ndarray | None = None,
        valid_y: np.ndarray | None = None,
    ) -> "RandomForestModel":
        """Fit one forest per assay on that assay's labelled molecules.

        The validation arguments are accepted for interface compatibility and
        ignored: a forest needs no early stopping, and the validation set is
        used by the trainer for hyperparameter selection instead.
        """
        X = np.asarray(features)
        labels = np.asarray(y, dtype=float)
        if labels.ndim == 1:
            labels = labels.reshape(-1, 1)
        if X.shape[0] != labels.shape[0]:
            raise ValueError(
                f"{X.shape[0]} feature rows but {labels.shape[0]} label rows"
            )
        if labels.shape[1] != self.n_tasks:
            raise ValueError(
                f"model has n_tasks={self.n_tasks} but y has "
                f"{labels.shape[1]} column(s)"
            )

        self.estimators_ = []
        self.fallbacks_ = []
        self.degenerate_tasks = {}

        for task in range(self.n_tasks):
            column = labels[:, task]
            mask = ~np.isnan(column)  # this assay's labelled molecules only.
            targets = column[mask]
            classes = np.unique(targets)

            if targets.size == 0 or classes.size < 2:
                # Nothing to learn: predict the observed rate (0.5 if nothing
                # was observed at all), and record why.
                rate = float(targets.mean()) if targets.size else 0.5
                self.estimators_.append(None)
                self.fallbacks_.append(rate)
                self.degenerate_tasks[task] = (
                    "no labelled training molecules"
                    if targets.size == 0
                    else f"only one class present (label {classes[0]:.0f})"
                )
                continue

            forest = self._new_forest(task)
            forest.fit(X[mask], targets.astype(int))
            self.estimators_.append(forest)
            self.fallbacks_.append(float("nan"))

        return self

    # -- prediction -------------------------------------------------------- #

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Positive-class probability per molecule and assay."""
        if not self.estimators_:
            raise RuntimeError("predict_proba() called before fit()")

        X = np.asarray(features)
        scores = np.empty((X.shape[0], self.n_tasks), dtype=float)
        for task, forest in enumerate(self.estimators_):
            if forest is None:
                scores[:, task] = self.fallbacks_[task]
                continue
            probabilities = forest.predict_proba(X)
            # Column of the positive class; `classes_` is sorted, so with both
            # classes present the positive class is the last column.
            positive = int(np.where(forest.classes_ == 1)[0][0])
            scores[:, task] = probabilities[:, positive]
        return scores

    # -- tuning ------------------------------------------------------------ #

    @classmethod
    def search_space(cls) -> dict[str, Sequence[Any]]:
        """Forest settings and the fingerprint settings that feed them."""
        return {
            "n_estimators": (300, 500),
            "max_depth": (None, 12, 24),
            "min_samples_leaf": (1, 2, 5),
            "max_features": ("sqrt", "log2", 0.1),
            "class_weight": (None, "balanced", "balanced_subsample"),
            "fp_radius": (2, 3),
            "fp_n_bits": (1024, 2048),
            "fp_counts": (False, True),
        }
