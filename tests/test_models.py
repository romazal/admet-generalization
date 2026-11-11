"""Tests for the small model implementations.

The dummy baseline is intentionally simple: it should be registered and should
produce a chance-level ROC-AUC on balanced data, because it ignores the input
features and predicts the class prior.
"""

from __future__ import annotations

import numpy as np
import pytest

from admet.evaluation import evaluate
from admet.models import MODELS
from admet.models.base import build_model


def test_dummy_model_is_registered() -> None:
    assert "dummy" in MODELS
    assert build_model("dummy", n_tasks=1).__class__.__name__ == "DummyModel"


def test_dummy_model_scores_at_chance() -> None:
    model = build_model("dummy", n_tasks=1)
    features = np.array([[0.0], [1.0], [0.0], [1.0]], dtype=float)
    labels = np.array([[0.0], [1.0], [1.0], [0.0]], dtype=float)

    model.fit(features, labels)
    scores = model.predict_proba(features)

    assert scores.shape == (4, 1)
    assert evaluate(labels, scores).roc_auc == pytest.approx(0.5)
