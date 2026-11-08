"""Tests for :mod:`admet.data.splits`.

Proposal §3.4 and §5 make this file a safeguard rather than a convenience: a
scaffold leaking from the training set into the test set would silently
invalidate H1, because the "structural shift" regime would no longer involve a
structural shift. The tests therefore cover three things:

1. **No scaffold is shared** between train, validation and test under the
   scaffold split. :func:`test_random_split_does_leak_scaffolds` is the positive
   control for this: it shows that the check *can* fail, so passing it under the
   scaffold split means something.
2. **Sizes are correct** — exactly, for the random split; and within the bound
   implied by indivisible scaffold groups, for the scaffold split.
3. **Seeds behave as expected** — the same seed reproduces a split, different
   seeds give different splits, and no splitter depends on the global
   :mod:`random` state.

Everything runs on synthetic SMILES built from known scaffolds, so the suite
needs no download and can run before ``loaders.py`` exists.
"""

from __future__ import annotations

import random
import warnings

import pytest

from admet.data.splits import (
    DEFAULT_FRACTIONS,
    SPLITTERS,
    Split,
    assert_scaffold_disjoint,
    make_split,
    murcko_scaffold,
    random_split,
    scaffold_groups,
    scaffold_overlap,
    scaffold_split,
    subsample_train,
)

# --------------------------------------------------------------------------- #
# Synthetic dataset
# --------------------------------------------------------------------------- #

# Ring systems with a known Bemis-Murcko scaffold. Prefixing any of the
# substituents below adds a side chain only, so every molecule built from a core
# keeps that core as its scaffold.
_CORES = [
    "c1ccccc1",  # benzene
    "c1ccncc1",  # pyridine
    "c1ccc2ccccc2c1",  # naphthalene
    "c1ccc2ncccc2c1",  # quinoline
    "c1ccoc1",  # furan
    "c1ccsc1",  # thiophene
    "c1cncnc1",  # pyrimidine
    "C1CCCCC1",  # cyclohexane
    "C1CCNCC1",  # piperidine
    "c1ccc2[nH]ccc2c1",  # indole
    "c1ccc(-c2ccccc2)cc1",  # biphenyl
    "c1cnc[nH]1",  # imidazole
    "c1cc[nH]c1",  # pyrrole
    "C1CCCC1",  # cyclopentane
    "C1COCCN1",  # morpholine
    "c1ccc2occc2c1",  # benzofuran
]

_SUBSTITUENTS = ["C", "CC", "CCC", "OC", "NC", "CCO", "C(C)C", "CCCC", "CCN", "OCC"]

# Deliberately uneven group sizes (1 to 10 molecules per scaffold), which is
# what the packing bound has to cope with on real data.
_GROUP_SIZES = [1 + (i * 7) % 10 for i in range(len(_CORES))]

_ACYCLIC = ["CCO", "CCCCCC", "CC(C)CO", "CCCCO", "CC(=O)O"]


def _build_dataset() -> list[str]:
    smiles = [
        substituent + core
        for core, size in zip(_CORES, _GROUP_SIZES)
        for substituent in _SUBSTITUENTS[:size]
    ]
    # Interleave the scaffolds so that a group's indices are scattered through
    # the dataset; a splitter that relied on same-scaffold rows being adjacent
    # would pass on sorted input and fail here.
    random.Random(20261107).shuffle(smiles)
    return smiles


@pytest.fixture(scope="module")
def dataset() -> list[str]:
    """86 synthetic molecules spread over 16 scaffolds of size 1 to 10."""
    return _build_dataset()


@pytest.fixture(scope="module")
def scaffolds(dataset: list[str]) -> list[str]:
    """The Murcko scaffold of each molecule in ``dataset``, in row order."""
    return [murcko_scaffold(smi) for smi in dataset]


def _parts_as_sets(split: Split) -> list[set[int]]:
    return [set(split.train), set(split.valid), set(split.test)]


def _assert_is_partition(split: Split, n: int) -> None:
    train, valid, test = _parts_as_sets(split)
    assert train | valid | test == set(range(n)), "some rows are unassigned"
    assert train & valid == set()
    assert train & test == set()
    assert valid & test == set()
    assert len(split) == n
    assert split.covers(n)


# --------------------------------------------------------------------------- #
# Fixture sanity — if these fail, every other failure below is suspect
# --------------------------------------------------------------------------- #


def test_dataset_has_the_expected_shape(dataset: list[str], scaffolds: list[str]) -> None:
    assert len(dataset) == sum(_GROUP_SIZES) == 86
    assert len(set(dataset)) == len(dataset), "dataset contains duplicate SMILES"
    assert len(set(scaffolds)) == len(_CORES) == 16
    assert min(_GROUP_SIZES) == 1 and max(_GROUP_SIZES) == 10


# --------------------------------------------------------------------------- #
# Split container
# --------------------------------------------------------------------------- #


def test_split_normalises_parts_to_sorted_tuples() -> None:
    split = Split(train=[3, 1, 2], valid=(5,), test=[4])
    assert split.train == (1, 2, 3)
    assert isinstance(split.valid, tuple)
    assert split.sizes == (3, 1, 1)
    assert len(split) == 5
    assert list(split.parts) == ["train", "valid", "test"]
    assert split.as_dict() == {"train": [1, 2, 3], "valid": [5], "test": [4]}


def test_split_rejects_overlapping_parts() -> None:
    with pytest.raises(ValueError, match="train and test overlap"):
        Split(train=[0, 1], valid=[2], test=[1, 3])


def test_split_rejects_duplicate_indices_within_a_part() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        Split(train=[0, 0, 1], valid=[2], test=[3])


def test_split_rejects_negative_indices() -> None:
    with pytest.raises(ValueError, match="negative"):
        Split(train=[-1], valid=[], test=[0])


def test_split_covers_is_false_for_an_incomplete_partition() -> None:
    assert not Split(train=[0, 1], valid=[2], test=[]).covers(5)


# --------------------------------------------------------------------------- #
# Scaffolds
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("smiles", "expected"),
    [
        ("c1ccccc1", "c1ccccc1"),  # benzene is its own scaffold
        ("Cc1ccccc1", "c1ccccc1"),  # toluene: the side chain is stripped
        ("CC(=O)Oc1ccccc1C(=O)O", "c1ccccc1"),  # aspirin
        ("c1ccc2ccccc2c1", "c1ccc2ccccc2c1"),  # naphthalene
        ("CCO", ""),  # acyclic molecules have no scaffold
    ],
)
def test_murcko_scaffold_on_known_molecules(smiles: str, expected: str) -> None:
    assert murcko_scaffold(smiles) == expected


def test_murcko_scaffold_rejects_unparseable_smiles() -> None:
    with pytest.raises(ValueError, match="could not parse"):
        murcko_scaffold("this is not a smiles")


def test_murcko_scaffold_rejects_non_strings() -> None:
    with pytest.raises(TypeError):
        murcko_scaffold(None)  # type: ignore[arg-type]


def test_scaffold_groups_partition_the_dataset(dataset: list[str]) -> None:
    groups = scaffold_groups(dataset)
    assigned = [index for indices in groups.values() for index in indices]
    assert sorted(assigned) == list(range(len(dataset)))
    assert len(groups) == len(_CORES)
    assert sorted(len(indices) for indices in groups.values()) == sorted(_GROUP_SIZES)


def test_scaffold_groups_put_shared_scaffolds_together() -> None:
    smiles = ["Cc1ccccc1", "CCO", "CCc1ccncc1", "CCCc1ccccc1", "CCCCCC"]
    groups = scaffold_groups(smiles)
    assert groups["c1ccccc1"] == [0, 3]
    assert groups["c1ccncc1"] == [2]
    # Every acyclic molecule shares the empty scaffold.
    assert groups[""] == [1, 4]


def test_scaffold_groups_report_the_offending_row(dataset: list[str]) -> None:
    broken = list(dataset)
    broken[7] = "N(((C"
    with pytest.raises(ValueError, match="at dataset index 7"):
        scaffold_groups(broken)


def test_include_chirality_splits_a_scaffold_in_two() -> None:
    # cis- and trans-decalin: the stereocentres are the ring-fusion atoms, so
    # they survive the removal of side chains and the two molecules share a
    # scaffold only when chirality is ignored.
    smiles = ["C1CC[C@@H]2CCCC[C@@H]2C1", "C1CC[C@@H]2CCCC[C@H]2C1"]
    assert len(scaffold_groups(smiles, include_chirality=False)) == 1
    assert len(scaffold_groups(smiles, include_chirality=True)) == 2


# --------------------------------------------------------------------------- #
# Random split
# --------------------------------------------------------------------------- #


def test_random_split_is_a_partition(dataset: list[str]) -> None:
    _assert_is_partition(random_split(dataset, seed=0), len(dataset))


def test_random_split_sizes_are_exact(dataset: list[str]) -> None:
    # n = 86, so round(0.8 * 86) = 69 train, round(0.1 * 86) = 9 valid and the
    # remaining 8 rows are the test set.
    assert random_split(dataset, seed=0).sizes == (69, 9, 8)


def test_random_split_honours_custom_fractions() -> None:
    split = random_split(
        ["C"] * 100, frac_train=0.5, frac_valid=0.25, frac_test=0.25, seed=3
    )
    assert split.sizes == (50, 25, 25)


def test_random_split_is_reproducible(dataset: list[str]) -> None:
    assert random_split(dataset, seed=7) == random_split(dataset, seed=7)


def test_random_split_changes_with_the_seed(dataset: list[str]) -> None:
    splits = {random_split(dataset, seed=seed) for seed in range(5)}
    assert len(splits) == 5, "different seeds produced the same random split"


def test_random_split_ignores_the_global_random_state(dataset: list[str]) -> None:
    random.seed(1)
    first = random_split(dataset, seed=0)
    random.seed(999)
    [random.random() for _ in range(10)]
    second = random_split(dataset, seed=0)
    assert first == second


def test_random_split_does_leak_scaffolds(dataset: list[str]) -> None:
    """Positive control for the scaffold-disjointness assertion.

    If this ever stops failing, the leakage check below has become vacuous and
    the scaffold split is no longer testing a structural shift.
    """
    for seed in range(5):
        split = random_split(dataset, seed=seed)
        assert scaffold_overlap(dataset, split), f"seed {seed} happened not to leak"
        with pytest.raises(AssertionError, match="scaffold leakage"):
            assert_scaffold_disjoint(dataset, split)


# --------------------------------------------------------------------------- #
# Scaffold split — the leakage test
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", range(5))
def test_scaffold_split_is_a_partition(dataset: list[str], seed: int) -> None:
    _assert_is_partition(scaffold_split(dataset, seed=seed), len(dataset))


@pytest.mark.parametrize("seed", range(5))
def test_no_scaffold_is_shared_between_parts(
    dataset: list[str], scaffolds: list[str], seed: int
) -> None:
    """The core safeguard of proposal §3.4."""
    split = scaffold_split(dataset, seed=seed)
    by_part = {
        name: {scaffolds[i] for i in part} for name, part in split.parts.items()
    }
    assert by_part["train"] & by_part["valid"] == set()
    assert by_part["train"] & by_part["test"] == set()
    assert by_part["valid"] & by_part["test"] == set()
    assert scaffold_overlap(dataset, split) == {}
    assert_scaffold_disjoint(dataset, split)


def test_scaffold_split_keeps_whole_groups_together(dataset: list[str]) -> None:
    split = scaffold_split(dataset, seed=2)
    groups = scaffold_groups(dataset)
    for scaffold, indices in groups.items():
        homes = {
            name for name, part in split.parts.items() if set(indices) & set(part)
        }
        assert len(homes) == 1, f"scaffold {scaffold!r} was split across {homes}"


@pytest.mark.parametrize("seed", range(5))
def test_size_ordered_scaffold_split_respects_the_priority_cutoffs(
    dataset: list[str], seed: int
) -> None:
    """``order="size"`` keeps the hard MoleculeNet/DeepChem invariants.

    Its packer never lets a part cross its cutoff, so overflow accumulates in
    the test set. The lower bounds hold because the loop keeps trying every
    later group against train (then validation) too, so a part can only stop
    short of its cutoff by less than the largest group.
    """
    n = len(dataset)
    frac_train, frac_valid, _ = DEFAULT_FRACTIONS
    train_cutoff = frac_train * n
    valid_cutoff = (frac_train + frac_valid) * n
    largest_group = max(len(g) for g in scaffold_groups(dataset).values())

    n_train, n_valid, n_test = scaffold_split(dataset, seed=seed, order="size").sizes

    assert n_train <= train_cutoff
    assert n_train + n_valid <= valid_cutoff
    assert n_train > train_cutoff - largest_group
    assert n_train + n_valid > valid_cutoff - largest_group
    assert n_valid > 0 and n_test > 0


@pytest.mark.parametrize("seed", range(5))
def test_randomized_scaffold_split_sizes_stay_near_their_targets(
    dataset: list[str], seed: int
) -> None:
    """``order="random"`` packs by deficit, so every part lands near target.

    This is the bound that matters for the experiment: a part may overshoot
    its target (unlike ``order="size"``), but only by the group that pushed it
    over, and no part may be starved.
    """
    n = len(dataset)
    largest_group = max(len(g) for g in scaffold_groups(dataset).values())
    sizes = scaffold_split(dataset, seed=seed).sizes

    for size, fraction, name in zip(sizes, DEFAULT_FRACTIONS, ("train", "valid", "test")):
        target = fraction * n
        assert abs(size - target) <= largest_group, (
            f"{name} has {size} molecules, target {target:.0f}, "
            f"largest scaffold group {largest_group}"
        )
        assert size > 0, f"{name} came out empty"


def test_randomized_packing_survives_a_group_larger_than_a_part() -> None:
    """A group too big for validation or test must not empty either part.

    This is the Tox21 failure in miniature. Tox21 has one group of 1775
    acyclic molecules and one of 1474 sharing a ring system, against a 10%
    part of ~782. Packing by train/validation/test priority sent such a group
    into test the moment it did not fit validation, producing splits from
    ``(6048, 0, 1775)`` to ``(6258, 1474, 91)`` across seeds. Here one
    scaffold holds 40% of a dataset whose validation and test targets are 10%
    each.
    """
    dataset = ["c1ccccc1" + "C" * i for i in range(40)]  # one shared scaffold
    dataset += [f"C1CCNCC1{'C' * i}" for i in range(30)]  # another
    dataset += [sub + core for core in _CORES[2:] for sub in _SUBSTITUENTS[:2]]

    for seed in range(10):
        n_train, n_valid, n_test = scaffold_split(dataset, seed=seed).sizes
        assert n_valid > 0, f"seed {seed} left validation empty"
        assert n_test > 0, f"seed {seed} left test empty"
        assert n_train > 0, f"seed {seed} left train empty"


def test_scaffold_split_is_reproducible(dataset: list[str]) -> None:
    assert scaffold_split(dataset, seed=4) == scaffold_split(dataset, seed=4)


def test_scaffold_split_changes_with_the_seed(dataset: list[str]) -> None:
    """A *randomized* scaffold split: each seed must give a different split."""
    splits = {scaffold_split(dataset, seed=seed) for seed in range(5)}
    assert len(splits) > 1, "the scaffold split ignored the seed"


def test_scaffold_split_ignores_the_global_random_state(dataset: list[str]) -> None:
    random.seed(1)
    first = scaffold_split(dataset, seed=0)
    random.seed(999)
    [random.random() for _ in range(10)]
    second = scaffold_split(dataset, seed=0)
    assert first == second


def test_size_ordered_scaffold_split_is_seed_independent(dataset: list[str]) -> None:
    splits = {scaffold_split(dataset, seed=seed, order="size") for seed in range(5)}
    assert len(splits) == 1
    split = splits.pop()
    _assert_is_partition(split, len(dataset))
    assert_scaffold_disjoint(dataset, split)
    # Largest groups first means the biggest scaffold is in the training set.
    biggest = max(scaffold_groups(dataset).values(), key=len)
    assert set(biggest) <= set(split.train)


def test_scaffold_split_rejects_an_unknown_order(dataset: list[str]) -> None:
    with pytest.raises(ValueError, match="unknown scaffold group order"):
        scaffold_split(dataset, order="clustered")  # type: ignore[arg-type]


def test_scaffold_split_warns_when_a_part_comes_out_empty() -> None:
    """A dataset with one indivisible scaffold group cannot be split.

    The result is still a valid partition, but an empty part must not reach a
    training run unnoticed. Deficit packing sends the single group to train —
    the part with by far the largest deficit — so it is validation and test
    that come out empty, which is the better of two unusable outcomes: with
    the old priority packing everything landed in *test* and the run trained
    on nothing at all.
    """
    same_scaffold = ["C" * k + "c1ccccc1" for k in range(50)]
    with pytest.warns(RuntimeWarning, match="valid and test empty"):
        split = scaffold_split(same_scaffold, seed=0)
    _assert_is_partition(split, len(same_scaffold))
    assert split.sizes == (50, 0, 0)


def test_scaffold_split_does_not_warn_on_a_normal_dataset(dataset: list[str]) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        scaffold_split(dataset, seed=0)


def test_scaffold_split_handles_acyclic_molecules(dataset: list[str]) -> None:
    mixed = dataset + _ACYCLIC
    split = scaffold_split(mixed, seed=0)
    _assert_is_partition(split, len(mixed))
    assert_scaffold_disjoint(mixed, split)
    # All acyclic molecules share the empty scaffold, so they stay together.
    acyclic_rows = set(range(len(dataset), len(mixed)))
    homes = {name for name, part in split.parts.items() if acyclic_rows & set(part)}
    assert len(homes) == 1


# --------------------------------------------------------------------------- #
# Shared splitter behaviour
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("splitter", list(SPLITTERS.values()), ids=list(SPLITTERS))
def test_splitters_reject_fractions_that_do_not_sum_to_one(splitter) -> None:
    with pytest.raises(ValueError, match="must sum to 1.0"):
        splitter(["c1ccccc1"] * 10, frac_train=0.8, frac_valid=0.3, frac_test=0.1)


@pytest.mark.parametrize("splitter", list(SPLITTERS.values()), ids=list(SPLITTERS))
def test_splitters_reject_fractions_outside_the_unit_interval(splitter) -> None:
    with pytest.raises(ValueError, match="must lie in"):
        splitter(["c1ccccc1"] * 10, frac_train=1.2, frac_valid=-0.1, frac_test=-0.1)


@pytest.mark.parametrize("splitter", list(SPLITTERS.values()), ids=list(SPLITTERS))
def test_splitters_reject_an_empty_dataset(splitter) -> None:
    with pytest.raises(ValueError, match="empty dataset"):
        splitter([])


# --------------------------------------------------------------------------- #
# Learning-curve subsampling
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("fraction", [0.1, 0.25, 0.5])
def test_subsample_touches_the_training_set_only(
    dataset: list[str], fraction: float
) -> None:
    full = scaffold_split(dataset, seed=0)
    reduced = subsample_train(full, fraction, seed=0)
    assert reduced.valid == full.valid
    assert reduced.test == full.test
    assert set(reduced.train) < set(full.train)
    assert len(reduced.train) == round(fraction * len(full.train))


def test_subsample_fractions_are_nested(dataset: list[str]) -> None:
    """10% must be inside 25%, inside 50%, inside 100%.

    Without this, every point on a learning curve would use a freshly resampled
    training set, and a dip in the curve could be a resampling artefact rather
    than an effect of data size.
    """
    full = scaffold_split(dataset, seed=0)
    previous: set[int] = set()
    for fraction in (0.1, 0.25, 0.5, 1.0):
        current = set(subsample_train(full, fraction, seed=0).train)
        assert previous <= current, f"the {fraction:.0%} subset is not nested"
        previous = current
    assert previous == set(full.train)


def test_subsample_is_reproducible_and_seed_dependent(dataset: list[str]) -> None:
    full = random_split(dataset, seed=0)
    assert subsample_train(full, 0.25, seed=1) == subsample_train(full, 0.25, seed=1)
    assert subsample_train(full, 0.25, seed=1) != subsample_train(full, 0.25, seed=2)


def test_subsample_of_the_full_fraction_is_a_no_op(dataset: list[str]) -> None:
    full = random_split(dataset, seed=0)
    assert subsample_train(full, 1.0, seed=0) == full


def test_subsample_keeps_at_least_one_training_molecule() -> None:
    split = Split(train=[0, 1, 2], valid=[3], test=[4])
    assert len(subsample_train(split, 0.01, seed=0).train) == 1


@pytest.mark.parametrize("fraction", [0.0, -0.5, 1.5])
def test_subsample_rejects_fractions_outside_the_half_open_unit_interval(
    fraction: float,
) -> None:
    split = Split(train=[0, 1], valid=[2], test=[3])
    with pytest.raises(ValueError, match=r"must lie in \(0, 1\]"):
        subsample_train(split, fraction)


def test_subsample_preserves_scaffold_disjointness(dataset: list[str]) -> None:
    full = scaffold_split(dataset, seed=0)
    assert_scaffold_disjoint(dataset, subsample_train(full, 0.25, seed=0))


def test_split_method_matches_the_function(dataset: list[str]) -> None:
    full = scaffold_split(dataset, seed=0)
    assert full.subsample_train(0.5, seed=3) == subsample_train(full, 0.5, seed=3)


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("kind", sorted(SPLITTERS))
def test_make_split_dispatches_by_name(dataset: list[str], kind: str) -> None:
    expected = SPLITTERS[kind](dataset, seed=1)
    assert make_split(kind, dataset, seed=1) == expected


def test_make_split_rejects_an_unknown_kind(dataset: list[str]) -> None:
    with pytest.raises(ValueError, match="unknown split kind 'cluster'"):
        make_split("cluster", dataset)


def test_make_split_applies_the_training_fraction(dataset: list[str]) -> None:
    full = make_split("scaffold", dataset, seed=1)
    reduced = make_split("scaffold", dataset, seed=1, train_fraction=0.25)
    assert reduced.valid == full.valid
    assert reduced.test == full.test
    assert len(reduced.train) == round(0.25 * len(full.train))
    assert reduced == subsample_train(full, 0.25, seed=1)


def test_make_split_forwards_splitter_keywords(dataset: list[str]) -> None:
    assert make_split("scaffold", dataset, seed=0, order="size") == scaffold_split(
        dataset, seed=0, order="size"
    )


def test_make_split_keeps_the_evaluation_sets_fixed_across_a_learning_curve(
    dataset: list[str],
) -> None:
    """The whole point of the additional experiment in proposal §3.3."""
    for kind in sorted(SPLITTERS):
        splits = [
            make_split(kind, dataset, seed=0, train_fraction=fraction)
            for fraction in (0.1, 0.25, 0.5, 1.0)
        ]
        assert len({s.valid for s in splits}) == 1, f"{kind}: validation set moved"
        assert len({s.test for s in splits}) == 1, f"{kind}: test set moved"
        assert [len(s.train) for s in splits] == sorted(
            len(s.train) for s in splits
        ), f"{kind}: training set did not grow with the fraction"
