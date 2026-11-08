"""SMILES to molecular graphs — the GNN's molecular representation.

A molecule becomes a graph whose nodes are atoms and whose edges are bonds
(proposal §3.2). Where a fingerprint hashes fixed substructures decided before
training, a graph keeps the connectivity intact and lets the model learn what
to make of it; this module is also the content of the lecture part on moving
from SMILES to graphs.

Representation
--------------
Each atom becomes a concatenation of one-hot blocks and binary flags — element,
degree, formal charge, hybridization, attached hydrogens, aromaticity, ring
membership — and each bond becomes its type, conjugation and ring membership.
Every categorical block carries a trailing *other* slot, so an element or
hybridization outside the vocabulary degrades to a generic value instead of
raising mid-run or, worse, silently producing an all-zero block that the model
would read as a meaningful feature pattern.

Bonds are stored **in both directions**. Message passing has to send
information each way along a bond, so a chemical bond becomes two directed
edges that share their feature vector.

The output is plain NumPy, with no torch import, so the featurizer is testable
on its own; :mod:`admet.models.gnn` converts batches to tensors.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

__all__ = [
    "ATOM_ELEMENTS",
    "MolGraph",
    "BatchedGraph",
    "atom_feature_dim",
    "bond_feature_dim",
    "mol_to_graph",
    "featurize_smiles",
    "batch_graphs",
]

#: Elements given their own one-hot slot. These cover essentially all of
#: Tox21; anything else lands in the trailing *other* slot.
ATOM_ELEMENTS: tuple[str, ...] = (
    "C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B", "Si", "Se", "Na", "K",
    "Ca", "Mg", "Zn", "Fe", "Cu", "Mn", "Hg", "As", "Sn", "Cr", "Ni", "Pt",
)

_DEGREES: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
_CHARGES: tuple[int, ...] = (-2, -1, 0, 1, 2)
_NUM_HS: tuple[int, ...] = (0, 1, 2, 3, 4)
_HYBRIDIZATIONS: tuple[str, ...] = ("SP", "SP2", "SP3", "SP3D", "SP3D2", "S")
_BOND_TYPES: tuple[str, ...] = ("SINGLE", "DOUBLE", "TRIPLE", "AROMATIC")


def _one_hot(value: object, vocabulary: Sequence[object]) -> list[float]:
    """One-hot over ``vocabulary`` with a trailing slot for anything else."""
    encoding = [0.0] * (len(vocabulary) + 1)
    try:
        encoding[vocabulary.index(value)] = 1.0
    except ValueError:
        encoding[-1] = 1.0
    return encoding


def atom_feature_dim() -> int:
    """Length of one atom's feature vector."""
    return (
        len(ATOM_ELEMENTS) + 1
        + len(_DEGREES) + 1
        + len(_CHARGES) + 1
        + len(_NUM_HS) + 1
        + len(_HYBRIDIZATIONS) + 1
        + 2  # aromatic, in a ring
    )


def bond_feature_dim() -> int:
    """Length of one bond's feature vector."""
    return len(_BOND_TYPES) + 1 + 2  # type one-hot, conjugated, in a ring


# --------------------------------------------------------------------------- #
# Containers
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MolGraph:
    """One molecule as a graph.

    Attributes:
        node_features: ``(n_atoms, atom_feature_dim())`` float32.
        edge_index: ``(2, n_edges)`` int64 of ``[source, target]`` rows, with
            both directions of every bond present, so ``n_edges`` is twice the
            number of chemical bonds.
        edge_features: ``(n_edges, bond_feature_dim())`` float32; the two
            directions of a bond carry identical features.
        smiles: the molecule this came from, kept for error messages.
    """

    node_features: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray
    smiles: str = ""

    def __post_init__(self) -> None:
        if self.node_features.ndim != 2:
            raise ValueError("node_features must be 2-dimensional")
        if self.edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape (2, n_edges)")
        if self.edge_index.shape[1] != self.edge_features.shape[0]:
            raise ValueError(
                f"{self.edge_index.shape[1]} edges but "
                f"{self.edge_features.shape[0]} edge feature rows"
            )
        if self.edge_index.size and int(self.edge_index.max()) >= self.n_atoms:
            raise ValueError("edge_index refers to an atom outside the molecule")

    @property
    def n_atoms(self) -> int:
        """Number of atoms (graph nodes)."""
        return int(self.node_features.shape[0])

    @property
    def n_edges(self) -> int:
        """Number of directed edges — twice the number of chemical bonds."""
        return int(self.edge_index.shape[1])


@dataclass(frozen=True)
class BatchedGraph:
    """Several molecules as one disconnected graph.

    Stacking molecules lets the GNN process a batch with a single pass of
    message passing. ``node_to_graph`` says which molecule each atom belongs
    to, which is what the readout pools over.
    """

    node_features: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray
    node_to_graph: np.ndarray  # (n_atoms,) int64, values in [0, n_graphs)
    n_graphs: int

    @property
    def n_atoms(self) -> int:
        """Total atoms across the batch."""
        return int(self.node_features.shape[0])

    @property
    def n_edges(self) -> int:
        """Total directed edges across the batch."""
        return int(self.edge_index.shape[1])


# --------------------------------------------------------------------------- #
# Featurization
# --------------------------------------------------------------------------- #


def _atom_features(atom) -> list[float]:
    """Feature vector for one RDKit atom."""
    return [
        *_one_hot(atom.GetSymbol(), ATOM_ELEMENTS),
        *_one_hot(atom.GetDegree(), _DEGREES),
        *_one_hot(atom.GetFormalCharge(), _CHARGES),
        *_one_hot(atom.GetTotalNumHs(), _NUM_HS),
        *_one_hot(str(atom.GetHybridization()), _HYBRIDIZATIONS),
        float(atom.GetIsAromatic()),
        float(atom.IsInRing()),
    ]


def _bond_features(bond) -> list[float]:
    """Feature vector for one RDKit bond."""
    return [
        *_one_hot(str(bond.GetBondType()), _BOND_TYPES),
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
    ]


def mol_to_graph(smiles: str) -> MolGraph:
    """Convert one SMILES string into a :class:`MolGraph`.

    Args:
        smiles: a SMILES string RDKit can parse.

    Returns:
        The molecule as a graph.

    Raises:
        TypeError: if ``smiles`` is not a string.
        ValueError: if RDKit cannot parse it. Dropping unparseable molecules is
            ``loaders.py``'s job, so reaching this point is a bug upstream.
    """
    from rdkit import Chem

    if not isinstance(smiles, str):
        raise TypeError(f"expected a SMILES string, got {type(smiles).__name__}")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")

    n_atoms = mol.GetNumAtoms()
    if n_atoms == 0:
        raise ValueError(f"molecule has no atoms: {smiles!r}")

    nodes = np.array(
        [_atom_features(atom) for atom in mol.GetAtoms()], dtype=np.float32
    )

    sources: list[int] = []
    targets: list[int] = []
    edges: list[list[float]] = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        features = _bond_features(bond)
        # Both directions, sharing the bond's features.
        sources.extend((i, j))
        targets.extend((j, i))
        edges.extend((features, features))

    if edges:
        edge_index = np.array([sources, targets], dtype=np.int64)
        edge_features = np.array(edges, dtype=np.float32)
    else:
        # A single-atom molecule has no bonds; keep the shapes well-formed so
        # batching and message passing need no special case.
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_features = np.zeros((0, bond_feature_dim()), dtype=np.float32)

    return MolGraph(
        node_features=nodes,
        edge_index=edge_index,
        edge_features=edge_features,
        smiles=smiles,
    )


def featurize_smiles(smiles: Sequence[str]) -> list[MolGraph]:
    """Convert a sequence of SMILES into graphs, in the same order.

    Raises:
        ValueError: naming the dataset row that could not be converted.
    """
    rows = list(smiles)
    if not rows:
        raise ValueError("cannot featurize an empty sequence of molecules")

    graphs: list[MolGraph] = []
    for index, smi in enumerate(rows):
        try:
            graphs.append(mol_to_graph(smi))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"at dataset index {index}: {exc}") from exc
    return graphs


def batch_graphs(graphs: Sequence[MolGraph]) -> BatchedGraph:
    """Stack molecules into one disconnected graph.

    Atom indices of the ``k``-th molecule are shifted by the number of atoms in
    the molecules before it, so no edge can ever connect two different
    molecules.

    Raises:
        ValueError: if ``graphs`` is empty.
    """
    items = list(graphs)
    if not items:
        raise ValueError("cannot batch an empty list of graphs")

    node_blocks = [g.node_features for g in items]
    edge_blocks = [g.edge_features for g in items]
    index_blocks = []
    membership = []

    offset = 0
    for graph_id, graph in enumerate(items):
        index_blocks.append(graph.edge_index + offset)
        membership.append(np.full(graph.n_atoms, graph_id, dtype=np.int64))
        offset += graph.n_atoms

    return BatchedGraph(
        node_features=np.concatenate(node_blocks, axis=0),
        edge_index=np.concatenate(index_blocks, axis=1),
        edge_features=np.concatenate(edge_blocks, axis=0),
        node_to_graph=np.concatenate(membership),
        n_graphs=len(items),
    )
