"""A dummy baseline that always predicts the observed class prior.

This is the floor every score should beat: the model ignores molecule identity
and answers with the assay's training-set prevalence. It is intentionally
simple, and the only useful question is whether the real models rise above it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.dummy import DummyClassifier

from admet.models.base import Model, register_model

__all__ = ["DummyModel"]


@register_model
class DummyModel(Model):
    """Per-assay class-prior predictor, one dummy per assay."""

    name = "dummy"

    def __init__(self, n_tasks: int, *, strategy: str = "prior", **hyperparameters: Any) -> None:
        super().__init__(n_tasks, strategy=strategy, **hyperparameters)
        self.strategy = strategy
        self.estimators_: list[DummyClassifier | None] = []
        self.fallbacks_: list[float] = []
        self.degenerate_tasks: dict[int, str] = {}

    def featurize(self, smiles: Sequence[str]) -> np.ndarray:
        """The model ignores chemistry and only needs a valid row count."""
        n_molecules = len(smiles)
        return np.zeros((n_molecules, 1), dtype=float)

    def featurization_key(self) -> tuple:
        return ("dummy", self.strategy)

    def fit(
        self,
        features: np.ndarray,
        y: np.ndarray,
        *,
        valid_features: np.ndarray | None = None,
        valid_y: np.ndarray | None = None,
    ) -> "DummyModel":
        """Fit one prior-based dummy per assay on that assay's labelled molecules."""
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
            mask = ~np.isnan(column)
            targets = column[mask]
            classes = np.unique(targets)

            if targets.size == 0 or classes.size < 2:
                rate = float(targets.mean()) if targets.size else 0.5
                self.estimators_.append(None)
                self.fallbacks_.append(rate)
                self.degenerate_tasks[task] = (
                    "no labelled training molecules"
                    if targets.size == 0
                    else f"only one class present (label {classes[0]:.0f})"
                )
                continue

            classifier = DummyClassifier(strategy=self.strategy)
            classifier.fit(X[mask], targets.astype(int))
            self.estimators_.append(classifier)
            self.fallbacks_.append(float("nan"))

        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Positive-class probability per molecule and assay."""
        if not self.estimators_:
            raise RuntimeError("predict_proba() called before fit()")

        X = np.asarray(features)
        scores = np.empty((X.shape[0], self.n_tasks), dtype=float)
        for task, classifier in enumerate(self.estimators_):
            if classifier is None:
                scores[:, task] = self.fallbacks_[task]
                continue
            probabilities = classifier.predict_proba(X)
            if probabilities.shape[1] == 1:
                scores[:, task] = probabilities[:, 0]
                continue
            positive = int(np.where(classifier.classes_ == 1)[0][0])
            scores[:, task] = probabilities[:, positive]
        return scores

    @classmethod
    def search_space(cls) -> dict[str, Sequence[Any]]:
        """A prior model has no hyperparameters to tune."""
        return {}
