"""Train / validation / test splitters for the ADMET generalization study.

These splitters are the centre of both hypotheses in the proposal:

* **H1 (generalization)** needs a *random* split and a *randomized scaffold*
  split that differ only in how molecules are assigned, so that the
  GNN-minus-baseline gap can be compared between the two regimes.
* **H2 (data size)** needs learning curves, which are produced by subsampling
  the **training set only** (:func:`subsample_train`). The validation and test
  sets must stay byte-identical across training fractions, otherwise the points
  of a learning curve are not comparable with one another.

Design notes
------------
*Scaffold-disjointness.* The scaffold splitter groups molecules by their
Bemis-Murcko scaffold and assigns **whole groups** to a part, so no scaffold can
ever appear in more than one of train / validation / test. This is the property
that makes the scaffold split a genuine structural-shift test, and it is
asserted by ``tests/test_splits.py``; :func:`assert_scaffold_disjoint` is
available so that ``training.py`` can re-check it at run time.

*Randomization.* Proposal §3.3 asks for a *randomized* scaffold split: the
scaffold groups are shuffled with a seeded RNG, so every seed yields a different
split and the spread over seeds reflects split variance as well as model
variance. ``order="size"`` reproduces the deterministic, seed-independent
largest-group-first ordering used by DeepChem/MoleculeNet, for comparison with
published numbers.

*Group packing.* Because whole groups are indivisible, realised sizes cannot
match the requested fractions exactly, and *how* the groups are packed decides
how badly. Two packers are used, one per ordering:

``order="size"`` uses the MoleculeNet/DeepChem rule — a group goes to train if
it still fits under the train cutoff, else to validation if it fits under the
cumulative validation cutoff, else to test. That rule relies on its ordering:
with the largest groups placed first, the big ones land in train and only small
ones are left to top up validation and test.

``order="random"`` cannot rely on that, so it packs by **largest deficit**:
each group goes to whichever part is currently furthest below its target size.
The priority rule above is actively wrong under a shuffled order, because a
group too large for the remaining validation room spills into test no matter
how big it is. On Tox21 that is not hypothetical — the dataset has one group of
1775 acyclic molecules and one of 1474 sharing a single ring system, each far
larger than a 10% part — and the priority rule produced splits ranging from
``(6048, 0, 1775)`` to ``(6258, 1474, 91)`` across seeds, i.e. an empty
validation set on some seeds and a 91-molecule test set on others. Deficit
packing puts both giants in train (the only part with room for them) and
distributes the rest, keeping every part within one group-size of its target.

*Determinism.* Every function takes an explicit ``seed`` and uses a private
:class:`random.Random` instance. Nothing here reads or mutates the global
``random`` state, so a split depends on its ``seed`` argument alone.

Only a random and a randomized scaffold splitter are implemented. The
cluster-based splitter mentioned parenthetically in proposal §5 is optional and
is the second item on the scope-reduction list in §8; new splitters can be added
by registering them in :data:`SPLITTERS`.
"""

from __future__ import annotations

import hashlib
import random
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable, Literal

__all__ = [
    "Split",
    "SPLITTERS",
    "DEFAULT_FRACTIONS",
    "ScaffoldOrder",
    "murcko_scaffold",
    "scaffold_groups",
    "random_split",
    "scaffold_split",
    "subsample_train",
    "make_split",
    "scaffold_overlap",
    "assert_scaffold_disjoint",
]

try:  # RDKit is a hard dependency of the project, but importing it lazily
    # keeps ``random_split`` and the dataclass usable (and this module
    # importable) in environments where RDKit is not installed.
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
except ImportError as _exc:  # pragma: no cover - depends on the environment
    _RDKIT_IMPORT_ERROR: ImportError | None = _exc
    Chem = None  # type: ignore[assignment]
    MurckoScaffold = None  # type: ignore[assignment]
else:
    _RDKIT_IMPORT_ERROR = None

#: Default train / validation / test fractions (the MoleculeNet convention).
DEFAULT_FRACTIONS: tuple[float, float, float] = (0.8, 0.1, 0.1)

#: Ordering applied to scaffold groups before they are packed into parts.
ScaffoldOrder = Literal["random", "size"]

_FRACTION_TOLERANCE = 1e-6


# --------------------------------------------------------------------------- #
# Split container
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Split:
    """An index partition of a dataset into train, validation and test.

    The three parts hold **row indices into the dataset**, not SMILES. Each part
    is normalised to a sorted tuple on construction, and the three parts are
    checked to be pairwise disjoint, so a :class:`Split` cannot silently carry a
    leak introduced by a caller.
    """

    train: tuple[int, ...]
    valid: tuple[int, ...]
    test: tuple[int, ...]

    def __post_init__(self) -> None:
        for name in ("train", "valid", "test"):
            raw = getattr(self, name)
            try:
                part = tuple(sorted(int(i) for i in raw))
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"{name!r} must be an iterable of integer indices"
                ) from exc
            if any(i < 0 for i in part):
                raise ValueError(f"{name!r} contains a negative index")
            if len(set(part)) != len(part):
                raise ValueError(f"{name!r} contains duplicate indices")
            object.__setattr__(self, name, part)

        pairs = (("train", "valid"), ("train", "test"), ("valid", "test"))
        for left, right in pairs:
            shared = set(getattr(self, left)) & set(getattr(self, right))
            if shared:
                raise ValueError(
                    f"{left} and {right} overlap on "
                    f"{len(shared)} index/indices, e.g. {sorted(shared)[:5]}"
                )

    # -- convenience ------------------------------------------------------- #

    def __len__(self) -> int:
        """Total number of indices held by the split."""
        return len(self.train) + len(self.valid) + len(self.test)

    @property
    def sizes(self) -> tuple[int, int, int]:
        """``(n_train, n_valid, n_test)``."""
        return (len(self.train), len(self.valid), len(self.test))

    @property
    def parts(self) -> dict[str, tuple[int, ...]]:
        """The three parts keyed by name, in train/valid/test order."""
        return {"train": self.train, "valid": self.valid, "test": self.test}

    def covers(self, n: int) -> bool:
        """Whether the split is an exact partition of ``range(n)``."""
        return set(self.train) | set(self.valid) | set(self.test) == set(range(n))

    def as_dict(self) -> dict[str, list[int]]:
        """JSON-serialisable form, for storing the split inside a result file."""
        return {name: list(part) for name, part in self.parts.items()}

    def subsample_train(self, fraction: float, seed: int = 0) -> "Split":
        """Shorthand for :func:`subsample_train`."""
        return subsample_train(self, fraction, seed=seed)


# --------------------------------------------------------------------------- #
# Scaffolds
# --------------------------------------------------------------------------- #


def _require_rdkit() -> None:
    if _RDKIT_IMPORT_ERROR is not None:  # pragma: no cover
        raise ImportError(
            "RDKit is required for scaffold-based splitting. Install the "
            "project dependencies (`pip install -e .`) and try again."
        ) from _RDKIT_IMPORT_ERROR


def murcko_scaffold(smiles: str, *, include_chirality: bool = False) -> str:
    """Return the canonical Bemis-Murcko scaffold SMILES of ``smiles``.

    The Bemis-Murcko scaffold is the union of a molecule's ring systems and the
    linkers between them, with all side chains removed. Molecules that share a
    scaffold are the ones a scaffold split must keep on the same side of the
    train/test boundary.

    Acyclic molecules have no ring system and therefore an **empty** scaffold
    (``""``). They are all grouped together, which follows the DeepChem /
    MoleculeNet convention; see :func:`scaffold_groups`.

    Raises:
        ValueError: if RDKit cannot parse ``smiles``. Dropping unparseable
            SMILES is the job of ``loaders.py``, so by the time a molecule
            reaches a splitter it is expected to be valid.
    """
    _require_rdkit()
    if not isinstance(smiles, str):
        raise TypeError(f"expected a SMILES string, got {type(smiles).__name__}")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol, includeChirality=include_chirality
        )
    except Exception as exc:  # pragma: no cover - RDKit internal failure
        raise ValueError(
            f"RDKit could not compute a Murcko scaffold for {smiles!r}"
        ) from exc


def scaffold_groups(
    smiles: Sequence[str], *, include_chirality: bool = False
) -> dict[str, list[int]]:
    """Group dataset indices by Bemis-Murcko scaffold.

    Args:
        smiles: dataset SMILES, in dataset row order.
        include_chirality: keep stereochemistry in the scaffold, which makes
            scaffolds finer-grained (and the split therefore easier).

    Returns:
        A mapping ``scaffold SMILES -> sorted row indices``. Insertion order
        follows first occurrence in ``smiles``, so the result is deterministic
        for a given input. All acyclic molecules share the key ``""``.
    """
    groups: dict[str, list[int]] = {}
    for index, smi in enumerate(smiles):
        try:
            scaffold = murcko_scaffold(smi, include_chirality=include_chirality)
        except ValueError as exc:
            raise ValueError(f"at dataset index {index}: {exc}") from exc
        groups.setdefault(scaffold, []).append(index)
    return groups


# --------------------------------------------------------------------------- #
# Splitters
# --------------------------------------------------------------------------- #


def _derive_seed(*parts: object) -> int:
    """A stable integer seed derived from ``parts``.

    ``hash()`` is salted per process for strings, so it cannot be used to derive
    reproducible sub-seeds; BLAKE2b can.
    """
    digest = hashlib.blake2b(digest_size=8)
    for part in parts:
        digest.update(repr(part).encode("utf-8"))
        digest.update(b"\x00")
    return int.from_bytes(digest.digest(), "big")


def _check_fractions(frac_train: float, frac_valid: float, frac_test: float) -> None:
    for name, value in (
        ("frac_train", frac_train),
        ("frac_valid", frac_valid),
        ("frac_test", frac_test),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must lie in [0, 1], got {value!r}")
    total = frac_train + frac_valid + frac_test
    if abs(total - 1.0) > _FRACTION_TOLERANCE:
        raise ValueError(
            "frac_train + frac_valid + frac_test must sum to 1.0, "
            f"got {total!r}"
        )


def _check_dataset_size(n: int) -> None:
    if n == 0:
        raise ValueError("cannot split an empty dataset")


def random_split(
    smiles: Sequence[str],
    *,
    frac_train: float = DEFAULT_FRACTIONS[0],
    frac_valid: float = DEFAULT_FRACTIONS[1],
    frac_test: float = DEFAULT_FRACTIONS[2],
    seed: int = 0,
) -> Split:
    """Split dataset rows uniformly at random.

    This is the *optimistic* regime of H1: test molecules are drawn from the
    same structural distribution as the training molecules, so scaffolds are
    expected to be shared across the parts.

    Only ``len(smiles)`` is used, so any sequence of the right length works; the
    argument is named ``smiles`` to give every splitter the same signature.

    Sizes are exact: ``n_train`` and ``n_valid`` are the rounded fractions and
    the test set takes the remainder, so the three parts always cover every row
    of the dataset exactly once.
    """
    _check_fractions(frac_train, frac_valid, frac_test)
    n = len(smiles)
    _check_dataset_size(n)

    n_train = int(round(frac_train * n))
    n_valid = int(round(frac_valid * n))
    # Rounding can overshoot by one; the test set absorbs the remainder.
    n_train = min(n_train, n)
    n_valid = min(n_valid, n - n_train)

    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    return Split(
        train=indices[:n_train],
        valid=indices[n_train : n_train + n_valid],
        test=indices[n_train + n_valid :],
    )


def _pack_by_priority(
    groups: list[list[int]], frac_train: float, frac_valid: float, n: int
) -> tuple[list[int], list[int], list[int]]:
    """MoleculeNet/DeepChem packing: train, then validation, then overflow.

    Correct only when ``groups`` is ordered largest-first; see *Group packing*
    in the module docstring for what goes wrong otherwise.
    """
    train_cutoff = frac_train * n
    valid_cutoff = (frac_train + frac_valid) * n

    train: list[int] = []
    valid: list[int] = []
    test: list[int] = []
    for group in groups:
        if len(train) + len(group) <= train_cutoff:
            train.extend(group)
        elif len(train) + len(valid) + len(group) <= valid_cutoff:
            valid.extend(group)
        else:
            test.extend(group)
    return train, valid, test


def _pack_by_deficit(
    groups: list[list[int]], fractions: tuple[float, float, float], n: int
) -> tuple[list[int], list[int], list[int]]:
    """Give each group to the part currently furthest below its target size.

    This is what makes a *shuffled* scaffold split usable: a group larger than
    a whole part can only go where there is room for it, and every other group
    goes wherever it is most needed, so no part is starved by the order the
    groups happen to arrive in. Ties go to the earlier part (train, then
    validation, then test), which keeps the packing a deterministic function
    of the shuffled order.
    """
    targets = [fraction * n for fraction in fractions]
    parts: tuple[list[int], list[int], list[int]] = ([], [], [])
    for group in groups:
        deficits = [targets[i] - len(parts[i]) for i in range(3)]
        # max() returns the first maximum, so ties already favour train.
        chosen = max(range(3), key=deficits.__getitem__)
        parts[chosen].extend(group)
    return parts


def scaffold_split(
    smiles: Sequence[str],
    *,
    frac_train: float = DEFAULT_FRACTIONS[0],
    frac_valid: float = DEFAULT_FRACTIONS[1],
    frac_test: float = DEFAULT_FRACTIONS[2],
    seed: int = 0,
    order: ScaffoldOrder = "random",
    include_chirality: bool = False,
) -> Split:
    """Split by Bemis-Murcko scaffold, keeping whole scaffold groups together.

    This is the *structural-shift* regime of H1: every scaffold occurs in
    exactly one part, so test molecules are guaranteed to carry ring systems the
    model never saw during training.

    Args:
        smiles: dataset SMILES, in dataset row order.
        frac_train, frac_valid, frac_test: requested sizes; must sum to 1.
        seed: seeds the shuffle of scaffold groups when ``order="random"``.
        order: ``"random"`` (default) shuffles the scaffold groups, so each seed
            gives a different split — the randomized scaffold split of proposal
            §3.3. ``"size"`` packs the largest groups first, which is the
            deterministic DeepChem/MoleculeNet ordering and ignores ``seed``.
        include_chirality: passed through to :func:`murcko_scaffold`.

    Returns:
        A :class:`Split` that is an exact partition of ``range(len(smiles))``.

    Note:
        Because whole groups are indivisible, realised sizes deviate from the
        requested fractions by at most the size of the largest scaffold group.
        The two orderings reach that bound differently — see *Group packing*
        in the module docstring. Under ``order="size"`` overflow accumulates in
        the test set and two invariants hold exactly:
        ``n_train <= frac_train * n`` and
        ``n_train + n_valid <= (frac_train + frac_valid) * n``. Under
        ``order="random"`` a part may instead overshoot its target, by at most
        the size of the group that pushed it over.

    Warns:
        RuntimeWarning: if a part comes out empty. That can only happen when the
            dataset has very few, very large scaffold groups, in which case the
            requested fractions are simply not reachable with indivisible
            groups. It is not a problem on Tox21, which has thousands of
            scaffolds, but an empty training or test set would otherwise sail
            silently into a run and produce a meaningless score.
    """
    _check_fractions(frac_train, frac_valid, frac_test)
    n = len(smiles)
    _check_dataset_size(n)

    groups = list(scaffold_groups(smiles, include_chirality=include_chirality).values())
    if order == "random":
        # Shuffle for the randomization, then order by descending size. The
        # sort is stable, so equal-sized groups keep their shuffled order and
        # the split still varies with the seed — but the few huge groups are
        # placed while every part is still empty, which is the only moment
        # they can be placed well. Shuffling alone is not enough: a giant
        # group arriving late lands in whichever part happens to have room.
        random.Random(seed).shuffle(groups)
        groups.sort(key=len, reverse=True)
        train, valid, test = _pack_by_deficit(
            groups, (frac_train, frac_valid, frac_test), n
        )
    elif order == "size":
        # Largest first, ties broken by the group's smallest index so that the
        # ordering is fully determined by the data.
        groups.sort(key=lambda group: (-len(group), group[0]))
        train, valid, test = _pack_by_priority(groups, frac_train, frac_valid, n)
    else:
        raise ValueError(
            f"unknown scaffold group order {order!r}; expected 'random' or 'size'"
        )

    empty = [
        name
        for name, part in (("train", train), ("valid", valid), ("test", test))
        if not part
    ]
    if empty:
        warnings.warn(
            f"scaffold split left {' and '.join(empty)} empty: the dataset has "
            f"only {len(groups)} scaffold group(s), and whole groups cannot be "
            f"divided to reach the requested fractions "
            f"({frac_train}/{frac_valid}/{frac_test}).",
            RuntimeWarning,
            stacklevel=2,
        )

    return Split(train=train, valid=valid, test=test)


#: Splitter registry, so a config can select a splitter by name.
SPLITTERS: dict[str, Callable[..., Split]] = {
    "random": random_split,
    "scaffold": scaffold_split,
}


# --------------------------------------------------------------------------- #
# Learning-curve subsampling
# --------------------------------------------------------------------------- #


def subsample_train(split: Split, fraction: float, seed: int = 0) -> Split:
    """Shrink the **training set only**, for the learning-curve experiment.

    Validation and test are returned unchanged, so every point of a learning
    curve is scored on exactly the same held-out molecules (proposal §3.3; this
    is also why the fraction applies to the training set and not to the whole
    dataset).

    The subsets are **nested**: the ordering of the training indices depends on
    ``seed`` but not on ``fraction``, so for a fixed split and seed the 10%
    training set is a subset of the 25% one, which is a subset of the 50% one,
    and so on. A learning curve therefore measures the effect of adding data
    rather than the effect of resampling it.

    Subsampling a subset of the training set can only remove molecules, so it
    cannot introduce scaffold leakage into a scaffold split.

    Args:
        split: the split to shrink.
        fraction: share of the training set to keep, in ``(0, 1]``.
        seed: seeds the training-set ordering.

    Returns:
        A new :class:`Split`; ``split`` itself is returned when ``fraction`` is
        1.0. At least one training molecule is always kept.
    """
    if not isinstance(split, Split):
        raise TypeError(f"expected a Split, got {type(split).__name__}")
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must lie in (0, 1], got {fraction!r}")
    if fraction == 1.0:
        return split
    if not split.train:
        raise ValueError("cannot subsample an empty training set")

    order = list(split.train)
    # Seeded independently of the splitter so that the training subsample is not
    # correlated with the train/valid/test assignment.
    random.Random(_derive_seed(seed, "subsample_train")).shuffle(order)

    n_keep = max(1, int(round(fraction * len(order))))
    return Split(train=order[:n_keep], valid=split.valid, test=split.test)


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #


def make_split(
    kind: str,
    smiles: Sequence[str],
    *,
    frac_train: float = DEFAULT_FRACTIONS[0],
    frac_valid: float = DEFAULT_FRACTIONS[1],
    frac_test: float = DEFAULT_FRACTIONS[2],
    seed: int = 0,
    train_fraction: float = 1.0,
    **splitter_kwargs: object,
) -> Split:
    """Build a split by name, as named in a config file.

    Args:
        kind: a key of :data:`SPLITTERS` (``"random"`` or ``"scaffold"``).
        smiles: dataset SMILES, in dataset row order.
        frac_train, frac_valid, frac_test: requested sizes; must sum to 1.
        seed: the run seed; drives both the split and the subsample.
        train_fraction: learning-curve fraction, applied to the training set
            only via :func:`subsample_train`.
        **splitter_kwargs: forwarded to the splitter (e.g. ``order`` or
            ``include_chirality`` for the scaffold splitter).
    """
    try:
        splitter = SPLITTERS[kind]
    except KeyError:
        known = ", ".join(sorted(SPLITTERS))
        raise ValueError(f"unknown split kind {kind!r}; expected one of: {known}") from None

    split = splitter(
        smiles,
        frac_train=frac_train,
        frac_valid=frac_valid,
        frac_test=frac_test,
        seed=seed,
        **splitter_kwargs,
    )
    return subsample_train(split, train_fraction, seed=seed)


# --------------------------------------------------------------------------- #
# Leakage checks
# --------------------------------------------------------------------------- #


def scaffold_overlap(
    smiles: Sequence[str], split: Split, *, include_chirality: bool = False
) -> dict[tuple[str, str], set[str]]:
    """Scaffolds shared between pairs of parts of ``split``.

    Returns a mapping ``(part_a, part_b) -> shared scaffold SMILES``; pairs with
    no overlap are omitted, so an empty result means the split is
    scaffold-disjoint. A random split is expected to have overlap here — that is
    exactly what distinguishes it from a scaffold split.
    """
    scaffolds = [
        murcko_scaffold(smi, include_chirality=include_chirality) for smi in smiles
    ]
    by_part = {
        name: {scaffolds[i] for i in part} for name, part in split.parts.items()
    }
    overlap: dict[tuple[str, str], set[str]] = {}
    for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
        shared = by_part[left] & by_part[right]
        if shared:
            overlap[(left, right)] = shared
    return overlap


def assert_scaffold_disjoint(
    smiles: Sequence[str], split: Split, *, include_chirality: bool = False
) -> None:
    """Raise :class:`AssertionError` if any scaffold spans two parts.

    Scaffold leakage between train and test would silently invalidate H1, so
    this is cheap enough to call at run time as well as in the tests
    (proposal §3.4).
    """
    overlap = scaffold_overlap(smiles, split, include_chirality=include_chirality)
    if overlap:
        details = "; ".join(
            f"{left}/{right}: {len(shared)} scaffold(s), "
            f"e.g. {sorted(shared)[:3]}"
            for (left, right), shared in overlap.items()
        )
        raise AssertionError(f"scaffold leakage between split parts — {details}")
