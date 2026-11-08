"""Models behind a shared interface: the fingerprint baseline and the GNN.

Importing this package registers both models in
:data:`admet.models.base.MODELS`, so a config can select either by name.
"""

from __future__ import annotations

from admet.models.base import MODELS, Model, build_model, sample_configs
from admet.models.gnn import GNNModel
from admet.models.rf import RandomForestModel

__all__ = [
    "MODELS",
    "Model",
    "build_model",
    "sample_configs",
    "GNNModel",
    "RandomForestModel",
]
