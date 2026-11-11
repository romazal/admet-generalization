"""Tests for :mod:`admet.features` — fingerprints and molecular graphs.

Proposal §5 asks for graph and fingerprint construction checked on known
molecules and for invalid SMILES to be handled. The molecules used are the
three the proposal names — ethanol, benzene, aspirin — whose atom and bond
counts can be read straight off the structure, so a failure points at the
featurizer rather than at a reference value nobody can check.

The featurizers are the two halves of the comparison, so an error here would
not announce itself: a subtly wrong graph still trains, it just trains on the
wrong molecule.
"""

from __future__ import annotations

import numpy as np
import pytest

from admet.features.fingerprints import (
    DEFAULT_N_BITS,
    featurize_smiles as fingerprint_matrix,
    maccs_fingerprint,
    morgan_fingerprint,
)
from admet.features.graphs import (
    ATOM_ELEMENTS,
    MolGraph,
    atom_feature_dim,
    batch_graphs,
    bond_feature_dim,
    featurize_smiles as graph_list,
    mol_to_graph,
)
from admet.models.base import sample_configs
from admet.models.rf import RandomForestModel

ETHANOL = "CCO"
BENZENE = "c1ccccc1"
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
CYCLOHEXANE = "C1CCCCC1"


# --------------------------------------------------------------------------- #
# Fingerprints
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("smiles", [ETHANOL, BENZENE, ASPIRIN])
def test_fingerprint_has_the_requested_shape_and_is_binary(smiles: str) -> None:
    fingerprint = morgan_fingerprint(smiles)
    assert fingerprint.shape == (DEFAULT_N_BITS,)
    assert set(np.unique(fingerprint)) <= {0, 1}
    assert fingerprint.sum() > 0, "a real molecule must set at least one bit"


def test_fingerprint_length_follows_n_bits() -> None:
    assert morgan_fingerprint(ASPIRIN, n_bits=512).shape == (512,)


def test_fingerprint_is_deterministic() -> None:
    assert np.array_equal(morgan_fingerprint(ASPIRIN), morgan_fingerprint(ASPIRIN))


def test_different_molecules_give_different_fingerprints() -> None:
    assert not np.array_equal(morgan_fingerprint(ETHANOL), morgan_fingerprint(BENZENE))


def test_a_larger_radius_adds_environments() -> None:
    """Radius 3 sees everything radius 2 sees, plus wider environments.

    Bits are hashed, so a collision could in principle lose one; aspirin is
    small enough that none occur, which is why the superset check is safe
    here and would not be on an arbitrary molecule.
    """
    small = morgan_fingerprint(ASPIRIN, radius=2, n_bits=4096)
    large = morgan_fingerprint(ASPIRIN, radius=3, n_bits=4096)
    assert large.sum() > small.sum()
    assert np.all(large[small.astype(bool)] == 1)


def test_count_fingerprints_record_repeats() -> None:
    """Benzene's six equivalent aromatic carbons collapse to one bit, count 6."""
    bits = morgan_fingerprint(BENZENE, use_counts=False)
    counts = morgan_fingerprint(BENZENE, use_counts=True)
    assert counts.max() > 1
    assert bits.max() == 1
    assert np.array_equal(counts.astype(bool), bits.astype(bool))


def test_fingerprint_matrix_rows_follow_input_order() -> None:
    matrix = fingerprint_matrix([ETHANOL, BENZENE, ASPIRIN])
    assert matrix.shape == (3, DEFAULT_N_BITS)
    for row, smiles in enumerate([ETHANOL, BENZENE, ASPIRIN]):
        assert np.array_equal(matrix[row], morgan_fingerprint(smiles))


def test_fingerprint_rejects_an_unparseable_smiles() -> None:
    with pytest.raises(ValueError, match="could not parse"):
        morgan_fingerprint("not_a_molecule")


def test_fingerprint_matrix_names_the_offending_row() -> None:
    """``loaders.py`` drops bad molecules, so reaching here is a bug upstream."""
    with pytest.raises(ValueError, match="at dataset index 1"):
        fingerprint_matrix([ETHANOL, "C(C(C", BENZENE])


def test_fingerprint_rejects_non_strings_and_bad_sizes() -> None:
    with pytest.raises(TypeError):
        morgan_fingerprint(42)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="n_bits"):
        morgan_fingerprint(ETHANOL, n_bits=0)
    with pytest.raises(ValueError, match="empty"):
        fingerprint_matrix([])


def test_maccs_fingerprint_has_167_bits_and_distinguishes_aromaticity() -> None:
    benzene = maccs_fingerprint(BENZENE)
    cyclohexane = maccs_fingerprint(CYCLOHEXANE)
    assert benzene.shape == (167,)
    assert cyclohexane.shape == (167,)
    assert not np.array_equal(benzene, cyclohexane)


def test_random_forest_can_select_maccs_fingerprints() -> None:
    model = RandomForestModel(n_tasks=1, fp_kind="maccs")
    features = model.featurize([BENZENE, CYCLOHEXANE])
    assert features.shape == (2, 167)
    assert model.featurization_key() != RandomForestModel(
        n_tasks=1, fp_kind="morgan"
    ).featurization_key()


def test_random_forest_rejects_an_unknown_fingerprint_kind() -> None:
    with pytest.raises(ValueError, match="fp_kind"):
        RandomForestModel(n_tasks=1, fp_kind="unknown")


def test_fp_kind_stays_out_of_the_tuned_search_space() -> None:
    """Guards the committed results against a plausible-looking one-line edit.

    `sample_configs` sorts parameter names before drawing, so adding `fp_kind`
    shifts every later parameter in the draw and changes *all* eight sampled
    configurations, Morgan-only ones included. The baseline in the 68 committed
    runs would stop reproducing, and with it the control argument in REPORT
    §5b. It would also spend part of the baseline's budget on a second
    representation while the GNN keeps all of its own on one, which biases the
    GNN-minus-baseline gap that H1 is measured on.

    MACCS is reachable through `model.params.fp_kind` in a config instead.
    """
    assert "fp_kind" not in RandomForestModel.search_space()


def test_adding_fp_kind_to_the_space_would_change_every_sampled_config() -> None:
    """The reason the test above exists, demonstrated rather than asserted."""
    space = dict(RandomForestModel.search_space())
    widened = {**space, "fp_kind": ("morgan", "maccs")}

    drop_kind = lambda config: tuple(  # noqa: E731
        sorted((k, v) for k, v in config.items() if k != "fp_kind")
    )
    current = {drop_kind(c) for c in sample_configs(space, 8, seed=0)}
    after = {drop_kind(c) for c in sample_configs(widened, 8, seed=0)}

    assert len(current) == len(after) == 8
    assert current.isdisjoint(after)


# --------------------------------------------------------------------------- #
# Graphs — known molecules
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("smiles", "n_atoms", "n_bonds"),
    [
        (ETHANOL, 3, 2),  # C-C-O, a chain
        (BENZENE, 6, 6),  # a ring: as many bonds as atoms
        (CYCLOHEXANE, 6, 6),
        (ASPIRIN, 13, 13),  # 13 heavy atoms, one ring
    ],
)
def test_graph_has_the_right_atoms_and_bonds(
    smiles: str, n_atoms: int, n_bonds: int
) -> None:
    """Hydrogens are implicit, so only heavy atoms become nodes."""
    graph = mol_to_graph(smiles)
    assert graph.n_atoms == n_atoms
    # Every bond is stored in both directions for message passing.
    assert graph.n_edges == 2 * n_bonds


def test_feature_dimensions_match_the_declared_sizes() -> None:
    graph = mol_to_graph(ASPIRIN)
    assert graph.node_features.shape == (13, atom_feature_dim())
    assert graph.edge_features.shape == (graph.n_edges, bond_feature_dim())
    assert graph.node_features.dtype == np.float32
    assert graph.edge_index.dtype == np.int64


def test_each_edge_direction_shares_the_bonds_features() -> None:
    graph = mol_to_graph(ETHANOL)
    for i in range(0, graph.n_edges, 2):
        assert np.array_equal(graph.edge_features[i], graph.edge_features[i + 1])
        assert graph.edge_index[0, i] == graph.edge_index[1, i + 1]
        assert graph.edge_index[1, i] == graph.edge_index[0, i + 1]


def test_aromaticity_distinguishes_benzene_from_cyclohexane() -> None:
    """Same atom and bond count, different chemistry — the features must differ."""
    benzene = mol_to_graph(BENZENE)
    cyclohexane = mol_to_graph(CYCLOHEXANE)
    assert benzene.n_atoms == cyclohexane.n_atoms
    assert not np.array_equal(benzene.node_features, cyclohexane.node_features)


def test_ring_membership_is_recorded() -> None:
    """Ethanol has no ring; benzene is all ring."""
    assert mol_to_graph(ETHANOL).node_features[:, -1].sum() == 0
    assert mol_to_graph(BENZENE).node_features[:, -1].sum() == 6


def test_every_atom_gets_exactly_one_element_slot() -> None:
    """Each one-hot block must sum to 1, or a feature is silently missing."""
    element_block = len(ATOM_ELEMENTS) + 1
    nodes = mol_to_graph(ASPIRIN).node_features
    assert np.all(nodes[:, :element_block].sum(axis=1) == 1.0)


def test_an_unknown_element_falls_into_the_other_slot() -> None:
    """A rare element must degrade, not raise or produce an all-zero block."""
    graph = mol_to_graph("[U]")
    element_block = len(ATOM_ELEMENTS) + 1
    assert graph.node_features[0, element_block - 1] == 1.0


def test_a_single_atom_molecule_has_no_edges_but_valid_shapes() -> None:
    graph = mol_to_graph("C")
    assert graph.n_atoms == 1
    assert graph.n_edges == 0
    assert graph.edge_index.shape == (2, 0)
    assert graph.edge_features.shape == (0, bond_feature_dim())


def test_graph_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="could not parse"):
        mol_to_graph("not_a_molecule")
    with pytest.raises(TypeError):
        mol_to_graph(None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at dataset index 1"):
        graph_list([ETHANOL, "C(C(C", BENZENE])
    with pytest.raises(ValueError, match="empty"):
        graph_list([])


def test_malformed_graphs_are_rejected_on_construction() -> None:
    """A :class:`MolGraph` cannot carry an edge to a nonexistent atom."""
    with pytest.raises(ValueError, match="outside the molecule"):
        MolGraph(
            node_features=np.zeros((2, atom_feature_dim()), dtype=np.float32),
            edge_index=np.array([[0], [5]], dtype=np.int64),
            edge_features=np.zeros((1, bond_feature_dim()), dtype=np.float32),
        )
    with pytest.raises(ValueError, match="edge feature rows"):
        MolGraph(
            node_features=np.zeros((2, atom_feature_dim()), dtype=np.float32),
            edge_index=np.array([[0], [1]], dtype=np.int64),
            edge_features=np.zeros((2, bond_feature_dim()), dtype=np.float32),
        )


# --------------------------------------------------------------------------- #
# Batching
# --------------------------------------------------------------------------- #


def test_batching_concatenates_atoms_and_shifts_edge_indices() -> None:
    molecules = [ETHANOL, BENZENE, "C"]
    graphs = graph_list(molecules)
    batch = batch_graphs(graphs)

    assert batch.n_graphs == 3
    assert batch.n_atoms == 3 + 6 + 1
    assert batch.n_edges == 4 + 12 + 0
    assert list(np.bincount(batch.node_to_graph)) == [3, 6, 1]
    assert batch.edge_index.max() < batch.n_atoms


def test_batching_never_connects_two_molecules() -> None:
    """The whole point of the index shift: molecules stay disconnected."""
    graphs = graph_list([ETHANOL, BENZENE, ASPIRIN])
    batch = batch_graphs(graphs)
    source_graph = batch.node_to_graph[batch.edge_index[0]]
    target_graph = batch.node_to_graph[batch.edge_index[1]]
    assert np.array_equal(source_graph, target_graph)


def test_a_batch_of_one_matches_the_single_graph() -> None:
    graph = mol_to_graph(ASPIRIN)
    batch = batch_graphs([graph])
    assert np.array_equal(batch.node_features, graph.node_features)
    assert np.array_equal(batch.edge_index, graph.edge_index)
    assert np.all(batch.node_to_graph == 0)


def test_batching_rejects_an_empty_list() -> None:
    with pytest.raises(ValueError, match="empty"):
        batch_graphs([])
