"""The advanced model: a message-passing GNN with a masked multitask head.

The network is a GIN with edge features (GINE): every atom starts from its own
feature vector, and at each layer it receives a message along each bond from
its neighbour, combining the neighbour's current embedding with the bond's
features. After ``n_layers`` rounds an atom's embedding summarises its
neighbourhood out to that many bonds; pooling the atoms gives a molecule
vector, and one linear head predicts all assays at once (proposal §3.2).

Why this is the interesting half of the comparison: the fingerprint baseline
hashes a *fixed* set of substructures chosen before any label is seen, while
these embeddings are learned from the assay data. H1 asks whether that
advantage survives when the test molecules carry ring systems the model never
trained on.

Masked multitask loss
---------------------
The head emits one logit per assay, and the loss is a binary cross-entropy
averaged over **observed labels only**: an entry that is ``NaN`` contributes no
gradient. This is the training-side counterpart of the masking rule in
:mod:`admet.evaluation`, and it is what lets all 12 assays share one encoder
despite none of them being measured on every molecule. Getting this wrong —
treating missing as negative — is on the proposal's list of errors that would
invalidate the comparison (§6).

Implementation note
-------------------
Message passing is written directly in PyTorch, using ``index_add_`` for the
neighbour sum, instead of depending on PyTorch Geometric. Proposal §3.2 allows
"a GIN in PyTorch Geometric"; this is the same model without the extra
dependency, which does not install cleanly on the development machine. The
scatter-add is a handful of lines and is covered by the tests. See
``AGENT_LOG.md``.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from admet.evaluation import evaluate
from admet.features.graphs import (
    BatchedGraph,
    MolGraph,
    atom_feature_dim,
    batch_graphs,
    bond_feature_dim,
)
from admet.features.graphs import featurize_smiles as graph_featurize
from admet.models.base import Model, register_model

__all__ = ["GINLayer", "GINNetwork", "GNNModel", "masked_bce_loss"]


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #


def masked_bce_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """Binary cross-entropy over observed labels only.

    Args:
        logits: ``(batch, n_tasks)`` raw scores.
        targets: ``(batch, n_tasks)`` labels in ``{0, 1}`` with ``NaN`` for
            *not screened*.

    Returns:
        A scalar: the mean loss over observed entries, or ``0.0`` when the
        batch happens to contain none (possible for a small batch on a sparse
        assay matrix — contributing nothing is correct, and avoids the ``NaN``
        that dividing by an empty count would send through every weight).
    """
    mask = ~torch.isnan(targets)
    if not bool(mask.any()):
        return logits.sum() * 0.0
    return nn.functional.binary_cross_entropy_with_logits(
        logits[mask], targets[mask]
    )


# --------------------------------------------------------------------------- #
# Network
# --------------------------------------------------------------------------- #


class GINLayer(nn.Module):
    """One round of message passing (GIN with edge features).

    ``h_i <- MLP((1 + eps) * h_i + sum_j ReLU(h_j + W_e e_ji))``

    where ``j`` runs over the neighbours of atom ``i``. ``eps`` is learned, so
    the layer can weigh an atom's own state against its neighbourhood.
    """

    def __init__(self, hidden_dim: int, edge_dim: int) -> None:
        super().__init__()
        self.edge_projection = nn.Linear(edge_dim, hidden_dim)
        self.eps = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.BatchNorm1d(2 * hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )

    def forward(
        self, h: Tensor, edge_index: Tensor, edge_features: Tensor
    ) -> Tensor:
        """Update every atom embedding once."""
        aggregated = torch.zeros_like(h)
        if edge_index.numel():
            source, target = edge_index[0], edge_index[1]
            messages = torch.relu(h[source] + self.edge_projection(edge_features))
            # Sum each atom's incoming messages; atoms with no bonds (a
            # single-atom molecule) correctly keep a zero neighbourhood.
            aggregated = aggregated.index_add(0, target, messages)
        return self.mlp((1.0 + self.eps) * h + aggregated)


class GINNetwork(nn.Module):
    """Atom encoder, ``n_layers`` of message passing, pooling, multitask head."""

    def __init__(
        self,
        *,
        atom_dim: int,
        bond_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 3,
        n_tasks: int = 1,
        dropout: float = 0.1,
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError(f"n_layers must be at least 1, got {n_layers!r}")
        if pooling not in {"mean", "sum", "max"}:
            raise ValueError(
                f"unknown pooling {pooling!r}; expected 'mean', 'sum' or 'max'"
            )

        self.pooling = pooling
        self.atom_encoder = nn.Linear(atom_dim, hidden_dim)
        self.layers = nn.ModuleList(
            GINLayer(hidden_dim, bond_dim) for _ in range(n_layers)
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_tasks),
        )

    def _pool(self, h: Tensor, node_to_graph: Tensor, n_graphs: int) -> Tensor:
        """Reduce atom embeddings to one vector per molecule."""
        hidden = h.shape[1]
        if self.pooling == "max":
            pooled = h.new_full((n_graphs, hidden), float("-inf"))
            pooled = pooled.index_reduce(
                0, node_to_graph, h, "amax", include_self=True
            )
            # A molecule with no atoms cannot occur, but guard the -inf anyway.
            return torch.nan_to_num(pooled, neginf=0.0)

        summed = h.new_zeros((n_graphs, hidden)).index_add(0, node_to_graph, h)
        if self.pooling == "sum":
            return summed
        counts = h.new_zeros(n_graphs).index_add(
            0, node_to_graph, torch.ones_like(node_to_graph, dtype=h.dtype)
        )
        return summed / counts.clamp(min=1.0).unsqueeze(1)

    def forward(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        edge_features: Tensor,
        node_to_graph: Tensor,
        n_graphs: int,
    ) -> Tensor:
        """Return ``(n_graphs, n_tasks)`` logits."""
        h = self.atom_encoder(node_features)
        for layer in self.layers:
            h = torch.relu(layer(h, edge_index, edge_features))
            h = self.dropout(h)
        return self.head(self._pool(h, node_to_graph, n_graphs))


# --------------------------------------------------------------------------- #
# Model wrapper
# --------------------------------------------------------------------------- #


def _to_tensors(
    batch: BatchedGraph, device: torch.device
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Move one batched graph onto ``device``."""
    return (
        torch.from_numpy(batch.node_features).to(device),
        torch.from_numpy(batch.edge_index).to(device),
        torch.from_numpy(batch.edge_features).to(device),
        torch.from_numpy(batch.node_to_graph).to(device),
    )


@register_model
class GNNModel(Model):
    """A GINE message-passing network with a masked multitask head.

    Training uses Adam, and stops early on the validation ROC-AUC when a
    validation set is supplied, restoring the best-scoring weights. Without one
    it simply trains for ``epochs``.
    """

    name = "gnn"

    def __init__(
        self,
        n_tasks: int,
        *,
        seed: int = 0,
        hidden_dim: int = 128,
        n_layers: int = 3,
        dropout: float = 0.1,
        pooling: str = "mean",
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int = 64,
        epochs: int = 50,
        patience: int = 10,
        device: str | None = None,
    ) -> None:
        super().__init__(
            n_tasks,
            seed=seed,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
            pooling=pooling,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
        )
        self.seed = seed
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.pooling = pooling
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.network: GINNetwork | None = None
        #: Epoch whose weights were restored, and its validation score.
        self.best_epoch: int | None = None
        self.best_valid_score: float | None = None
        #: Mean training loss per epoch, for sanity-checking a run.
        self.history: list[float] = []

    # -- representation ---------------------------------------------------- #

    def featurize(self, smiles: Sequence[str]) -> list[MolGraph]:
        """Molecular graphs, one per SMILES."""
        return graph_featurize(smiles)

    # -- internals --------------------------------------------------------- #

    def _build(self) -> GINNetwork:
        torch.manual_seed(self.seed)
        return GINNetwork(
            atom_dim=atom_feature_dim(),
            bond_dim=bond_feature_dim(),
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            n_tasks=self.n_tasks,
            dropout=self.dropout,
            pooling=self.pooling,
        ).to(self.device)

    def _forward_indices(
        self, graphs: Sequence[MolGraph], indices: Sequence[int]
    ) -> Tensor:
        """Logits for the molecules at ``indices``."""
        batch = batch_graphs([graphs[i] for i in indices])
        nodes, edge_index, edges, membership = _to_tensors(batch, self.device)
        assert self.network is not None
        return self.network(nodes, edge_index, edges, membership, batch.n_graphs)

    def _batches(self, n: int, order: Sequence[int]) -> list[list[int]]:
        """Split ``order`` into mini-batches.

        A trailing batch of one is merged into the previous batch: BatchNorm
        cannot compute a batch statistic from a single sample and would raise
        in training mode.
        """
        chunks = [
            list(order[i : i + self.batch_size])
            for i in range(0, n, self.batch_size)
        ]
        if len(chunks) > 1 and len(chunks[-1]) == 1:
            chunks[-2].extend(chunks.pop())
        return chunks

    def _validation_score(
        self, graphs: Sequence[MolGraph], y: np.ndarray
    ) -> float | None:
        """Mean ROC-AUC on the validation set, or ``None`` if unscoreable."""
        scores = self.predict_proba(graphs)
        result = evaluate(y, scores)
        return result.roc_auc

    # -- training ---------------------------------------------------------- #

    def fit(
        self,
        features: Sequence[MolGraph],
        y: np.ndarray,
        *,
        valid_features: Sequence[MolGraph] | None = None,
        valid_y: np.ndarray | None = None,
    ) -> "GNNModel":
        """Train the network, early-stopping on validation ROC-AUC if given."""
        graphs = list(features)
        labels = np.asarray(y, dtype=float)
        if labels.ndim == 1:
            labels = labels.reshape(-1, 1)
        if len(graphs) != labels.shape[0]:
            raise ValueError(
                f"{len(graphs)} graphs but {labels.shape[0]} label rows"
            )
        if labels.shape[1] != self.n_tasks:
            raise ValueError(
                f"model has n_tasks={self.n_tasks} but y has "
                f"{labels.shape[1]} column(s)"
            )
        if not graphs:
            raise ValueError("cannot fit on an empty training set")

        self.network = self._build()
        optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        targets = torch.from_numpy(labels.astype(np.float32)).to(self.device)

        use_early_stopping = (
            valid_features is not None
            and valid_y is not None
            and len(list(valid_features)) > 0
        )
        best_state: dict[str, Tensor] | None = None
        best_score = -math.inf
        epochs_without_improvement = 0
        self.history = []
        self.best_epoch = None
        self.best_valid_score = None

        generator = torch.Generator().manual_seed(self.seed)
        n = len(graphs)

        for epoch in range(1, self.epochs + 1):
            self.network.train()
            order = torch.randperm(n, generator=generator).tolist()
            epoch_loss = 0.0
            n_batches = 0

            for chunk in self._batches(n, order):
                optimizer.zero_grad(set_to_none=True)
                logits = self._forward_indices(graphs, chunk)
                loss = masked_bce_loss(logits, targets[chunk])
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach())
                n_batches += 1

            self.history.append(epoch_loss / max(n_batches, 1))

            if not use_early_stopping:
                continue

            score = self._validation_score(list(valid_features), np.asarray(valid_y))
            if score is None:
                # No assay in the validation set could be scored; there is
                # nothing to select on, so keep training rather than stopping
                # on a score that does not exist.
                continue
            if score > best_score:
                best_score = score
                best_state = copy.deepcopy(self.network.state_dict())
                self.best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    break

        if best_state is not None:
            self.network.load_state_dict(best_state)
            self.best_valid_score = float(best_score)

        return self

    # -- prediction -------------------------------------------------------- #

    @torch.no_grad()
    def predict_proba(self, features: Sequence[MolGraph]) -> np.ndarray:
        """Positive-class probability per molecule and assay."""
        if self.network is None:
            raise RuntimeError("predict_proba() called before fit()")
        graphs = list(features)
        if not graphs:
            raise ValueError("cannot predict on an empty set of molecules")

        self.network.eval()
        blocks = []
        for start in range(0, len(graphs), self.batch_size):
            indices = range(start, min(start + self.batch_size, len(graphs)))
            logits = self._forward_indices(graphs, list(indices))
            blocks.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(blocks, axis=0).astype(float)

    # -- tuning ------------------------------------------------------------ #

    @classmethod
    def search_space(cls) -> dict[str, Sequence[Any]]:
        """Capacity, regularisation and optimisation settings.

        ``epochs`` is not tuned: early stopping on the validation set already
        selects the effective number of epochs, so spending budget on it would
        tune the same thing twice.
        """
        return {
            "hidden_dim": (64, 128, 256),
            "n_layers": (2, 3, 4),
            "dropout": (0.0, 0.1, 0.3),
            "learning_rate": (3e-4, 1e-3, 3e-3),
            "batch_size": (32, 64, 128),
            "pooling": ("mean", "sum"),
        }
