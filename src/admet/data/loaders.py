"""Dataset registry: a name from a config to SMILES plus a masked label matrix.

A loader's job is to hand the rest of the pipeline one thing — a
:class:`Dataset` of parsed, deduplicated SMILES in a fixed row order, with an
``(n_molecules, n_tasks)`` label matrix in which ``NaN`` means *not screened*
(see :mod:`admet.evaluation` for why that distinction is load-bearing).

Two cleaning steps happen here rather than downstream, because both would
otherwise corrupt the comparison:

*Unparseable SMILES are dropped.* A molecule RDKit cannot read has neither a
scaffold nor a fingerprint nor a graph, so it cannot reach a splitter
(proposal §3.1).

*Duplicate molecules are merged.* Tox21 ships the same structure more than
once, and the copies are not always labelled for the same assays. Left alone,
one copy can land in train and another in test — leakage of exactly the kind
the scaffold split exists to rule out, and it would flatter both models.
Copies are therefore collapsed onto their canonical SMILES before the data
ever reaches a split, with per-assay labels merged.

Downloads are cached on disk, so the first run needs the network and later runs
do not. Set ``ADMET_DATA_DIR`` to move the cache; it defaults to ``data/`` at
the repository root, which ``.gitignore`` excludes.

ClinTox: read this before reporting anything from it
----------------------------------------------------
ClinTox is registered as the optional smaller second dataset, and the figures
below were measured from the raw file on 2026-11-08 (they had previously been
taken from loader documentation and never checked). Every one of them is a
reason to treat a ClinTox result as a sanity check on dataset size, not as
evidence for or against H1 and H2:

* **It is tiny and extremely imbalanced.** 1484 raw rows; 1461 molecules after
  cleaning. Raw minority counts are ``FDA_APPROVED = 0``: 94 and
  ``CT_TOX = 1``: 112 — about 7%.
* **A scaffold test split leaves 10-17 positives** (measured over seeds 0-4,
  test part of 146). The seed-to-seed swing in a score computed on ~13
  positives is larger than the GNN-minus-baseline gap the study is defined to
  measure, so a ClinTox comparison is very likely inconclusive.
* **H2's 10% training point has 5 positives.** The low end of the learning
  curve is not measurable here.
* **The two labels are one label.** ``CT_TOX`` equals ``NOT FDA_APPROVED`` for
  98.8% of molecules (all but 18). ClinTox is effectively single-task, so the
  multitask head and the masked loss — the whole point of the Tox21 setup —
  have nothing to do. Its label matrix is also fully observed (density 1.0),
  so nothing is masked.
* **19 duplicate structures carried contradictory labels.** Merging produced
  38 label conflicts across 19 duplicate groups, i.e. every one of them
  disagreed on both labels: the same structure appears both as approved and as
  having failed for toxicity. This is consistent with the dataset being
  assembled from two sources, and is the kind of provenance artefact that can
  make a model look unexpectedly good.

**Score ``CT_TOX`` only.** PR-AUC is the right metric under this imbalance,
but only for the assay whose *positive* class is the rare one.
``FDA_APPROVED`` is positive for 91% of the test part, so its PR-AUC runs at
~0.98 where a constant predictor already scores ~0.91; macro-averaging it with
``CT_TOX`` lifted the headline PR-AUC from 0.38 to 0.68 in a measured run and
hid the real signal. ``configs/clintox_*.yaml`` therefore set
``tasks: [CT_TOX]``. Treat a ROC-AUC near 0.98 as suspected provenance leakage
rather than success.

Tox21 remains the primary dataset and that is settled.

.. note::
   CSV reading uses the standard library rather than pandas. Proposal §5 lists
   pandas as a dependency, but its compiled extensions are blocked by the
   Windows Application Control policy on the development machine, and the
   loader needs only ``csv`` and :mod:`numpy`. See ``AGENT_LOG.md``.
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

__all__ = [
    "Dataset",
    "DATASETS",
    "DatasetSpec",
    "default_data_dir",
    "load_dataset",
    "read_csv",
]

_DOWNLOAD_TIMEOUT = 120  # seconds
_MOLECULENET = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets"


# --------------------------------------------------------------------------- #
# Containers
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Dataset:
    """Parsed molecules and their masked multitask labels.

    Attributes:
        name: registry key the dataset was loaded under.
        smiles: canonical SMILES, one per molecule, in row order.
        y: ``(n_molecules, n_tasks)`` float array of ``0.0``/``1.0``/``NaN``.
        tasks: assay names, matching the columns of ``y``.
        n_unparseable: molecules dropped because RDKit could not read them.
        n_duplicates: rows removed by merging duplicate structures.
        n_label_conflicts: assay labels that disagreed between merged copies.
    """

    name: str
    smiles: tuple[str, ...]
    y: np.ndarray
    tasks: tuple[str, ...]
    n_unparseable: int = 0
    n_duplicates: int = 0
    n_label_conflicts: int = 0

    def __post_init__(self) -> None:
        if self.y.ndim != 2:
            raise ValueError(f"y must be 2-dimensional, got {self.y.ndim}-D")
        if len(self.smiles) != self.y.shape[0]:
            raise ValueError(
                f"{len(self.smiles)} SMILES but {self.y.shape[0]} label rows"
            )
        if len(self.tasks) != self.y.shape[1]:
            raise ValueError(
                f"{len(self.tasks)} task names but {self.y.shape[1]} label columns"
            )

    def __len__(self) -> int:
        """Number of molecules."""
        return len(self.smiles)

    @property
    def n_tasks(self) -> int:
        """Number of assays."""
        return len(self.tasks)

    @property
    def label_density(self) -> float:
        """Share of the label matrix that is observed rather than ``NaN``."""
        if self.y.size == 0:
            return 0.0
        return float(np.count_nonzero(~np.isnan(self.y)) / self.y.size)

    def positives_per_task(self) -> dict[str, int]:
        """Observed positives per assay."""
        return {
            task: int(np.count_nonzero(self.y[:, i] == 1.0))
            for i, task in enumerate(self.tasks)
        }

    def subset(self, indices: Iterable[int]) -> "Dataset":
        """A new dataset holding only ``indices``, in the order given."""
        rows = list(indices)
        return Dataset(
            name=self.name,
            smiles=tuple(self.smiles[i] for i in rows),
            y=self.y[rows],
            tasks=self.tasks,
            n_unparseable=self.n_unparseable,
            n_duplicates=self.n_duplicates,
            n_label_conflicts=self.n_label_conflicts,
        )

    def summary(self) -> dict[str, object]:
        """Provenance and shape, for the ``dataset`` block of a result file."""
        return {
            "name": self.name,
            "n_molecules": len(self),
            "n_tasks": self.n_tasks,
            "tasks": list(self.tasks),
            "label_density": self.label_density,
            "positives_per_task": self.positives_per_task(),
            "n_unparseable_dropped": self.n_unparseable,
            "n_duplicates_merged": self.n_duplicates,
            "n_label_conflicts": self.n_label_conflicts,
        }


@dataclass(frozen=True)
class DatasetSpec:
    """How to fetch and read one dataset."""

    name: str
    url: str
    filename: str
    smiles_column: str
    #: Label columns, or ``None`` to use every column except those excluded.
    task_columns: tuple[str, ...] | None = None
    exclude_columns: tuple[str, ...] = ()
    description: str = ""


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

#: Datasets selectable by name from a config file.
DATASETS: dict[str, DatasetSpec] = {
    "tox21": DatasetSpec(
        name="tox21",
        url=f"{_MOLECULENET}/tox21.csv.gz",
        filename="tox21.csv.gz",
        smiles_column="smiles",
        exclude_columns=("mol_id", "smiles"),
        description=(
            "Tox21: ~7.8k molecules screened against 12 nuclear-receptor (NR-) "
            "and stress-response (SR-) toxicity assays, with many missing "
            "labels. The primary dataset (proposal §3.1)."
        ),
    ),
    "clintox": DatasetSpec(
        name="clintox",
        url=f"{_MOLECULENET}/clintox.csv.gz",
        filename="clintox.csv.gz",
        smiles_column="smiles",
        task_columns=("FDA_APPROVED", "CT_TOX"),
        description=(
            "ClinTox: ~1.5k drug candidates with two labels — FDA_APPROVED "
            "(1 = approved) and CT_TOX (1 = failed clinical trials for "
            "toxicity). The optional smaller second dataset (proposal §3.1, "
            "'if time allows'); Tox21 remains primary. Read the caveats in "
            "the module docstring before reporting anything from it."
        ),
    ),
}


def default_data_dir() -> Path:
    """Directory used to cache raw downloads.

    ``$ADMET_DATA_DIR`` if set, else ``data/`` at the repository root.
    """
    override = os.environ.get("ADMET_DATA_DIR")
    if override:
        return Path(override).expanduser()
    # .../src/admet/data/loaders.py -> repository root is four levels up.
    return Path(__file__).resolve().parents[3] / "data"


# --------------------------------------------------------------------------- #
# Raw file access
# --------------------------------------------------------------------------- #


def read_csv(path: str | Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read a CSV (optionally gzipped) into a header and a list of row dicts.

    Returns:
        ``(header, rows)``. Empty files yield ``([], [])``.
    """
    path = Path(path)
    # `dtype=str, keep_default_na=False` is not a detail. Left to itself
    # pandas infers dtypes and turns blank cells into NaN, which happens to
    # coincide with our masking semantics — but only by coincidence, and it
    # would also rewrite a label column of 0/1 into floats and a SMILES
    # column into something type-inferred. Reading everything as text keeps
    # the cells exactly as the file has them and leaves the conversion to
    # `_to_label`, which is what every committed result was produced with.
    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError:
        return [], []
    header = [str(column) for column in frame.columns]
    return header, frame.to_dict(orient="records")


def _download(spec: DatasetSpec, cache_dir: Path) -> Path:
    """Return the cached raw file, downloading it if it is not there yet."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / spec.filename
    if target.exists() and target.stat().st_size > 0:
        return target

    partial = target.with_suffix(target.suffix + ".part")
    try:
        with urllib.request.urlopen(spec.url, timeout=_DOWNLOAD_TIMEOUT) as response:
            partial.write_bytes(response.read())
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"could not download the {spec.name} dataset from {spec.url}: {exc}. "
            f"Place the file at {target} by hand and re-run to work offline."
        ) from exc
    # Rename only once the body is complete, so an interrupted download cannot
    # leave a truncated file that later runs would happily read from cache.
    partial.replace(target)
    return target


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #


def _canonicalize(values: Iterable[str | None]) -> list[str | None]:
    """Canonical SMILES per row, or ``None`` where RDKit cannot parse."""
    from rdkit import Chem, RDLogger

    # RDKit logs a warning per unparseable molecule; we count them instead.
    RDLogger.DisableLog("rdApp.*")
    try:
        out: list[str | None] = []
        for value in values:
            if not isinstance(value, str) or not value.strip():
                out.append(None)
                continue
            mol = Chem.MolFromSmiles(value)
            out.append(None if mol is None else Chem.MolToSmiles(mol))
        return out
    finally:
        RDLogger.EnableLog("rdApp.*")


def _to_label(raw: str | None) -> float:
    """Parse one cell into ``0.0``, ``1.0`` or ``NaN`` for *not screened*."""
    if raw is None:
        return float("nan")
    text = raw.strip()
    if not text or text.upper() in {"NA", "NAN", "NULL", "NONE"}:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _merge_duplicates(
    smiles: list[str], labels: np.ndarray
) -> tuple[list[str], np.ndarray, int, int]:
    """Collapse rows sharing a canonical SMILES.

    For each assay the merged label is the first observed (non-``NaN``) value
    among the copies. Where copies disagree the first is kept and the
    disagreement is counted, so a silently contradictory dataset shows up in
    the result file instead of being averaged away.

    Returns ``(smiles, labels, n_rows_removed, n_label_conflicts)`` with rows
    in first-occurrence order.
    """
    first_row: dict[str, int] = {}
    order: list[str] = []
    members: dict[str, list[int]] = {}
    for row, smi in enumerate(smiles):
        if smi not in first_row:
            first_row[smi] = row
            order.append(smi)
            members[smi] = []
        members[smi].append(row)

    if len(order) == len(smiles):
        return smiles, labels, 0, 0

    conflicts = 0
    merged = np.full((len(order), labels.shape[1]), np.nan, dtype=float)
    for position, smi in enumerate(order):
        rows = members[smi]
        for task in range(labels.shape[1]):
            column = labels[rows, task]
            observed = column[~np.isnan(column)]
            if observed.size == 0:
                continue
            merged[position, task] = observed[0]
            if np.unique(observed).size > 1:
                conflicts += 1

    return order, merged, len(smiles) - len(order), conflicts


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def load_dataset(
    name: str,
    *,
    cache_dir: str | Path | None = None,
    tasks: Iterable[str] | None = None,
) -> Dataset:
    """Load a registered dataset, downloading and caching it on first use.

    Args:
        name: a key of :data:`DATASETS`, as written in a config file.
        cache_dir: where raw downloads live; defaults to :func:`default_data_dir`.
        tasks: restrict to these assays, in this order. Defaults to all of them.

    Returns:
        A cleaned :class:`Dataset`.

    Raises:
        ValueError: for an unknown dataset name, an unknown assay name, or a
            raw file whose columns do not match the spec.
        RuntimeError: if the download fails and no cached copy exists.
    """
    try:
        spec = DATASETS[name]
    except KeyError:
        known = ", ".join(sorted(DATASETS))
        raise ValueError(
            f"unknown dataset {name!r}; expected one of: {known}"
        ) from None

    directory = Path(cache_dir).expanduser() if cache_dir else default_data_dir()
    path = _download(spec, directory)

    header, rows = read_csv(path)
    if spec.smiles_column not in header:
        raise ValueError(
            f"{path} has no {spec.smiles_column!r} column; found {header}"
        )
    if not rows:
        raise ValueError(f"{path} contains no data rows")

    if spec.task_columns is not None:
        task_names = list(spec.task_columns)
    else:
        task_names = [c for c in header if c not in spec.exclude_columns]
    missing = [c for c in task_names if c not in header]
    if missing:
        raise ValueError(f"{path} is missing label column(s): {missing}")

    if tasks is not None:
        requested = list(tasks)
        unknown = [t for t in requested if t not in task_names]
        if unknown:
            raise ValueError(
                f"unknown assay(s) for {name}: {unknown}; available: {task_names}"
            )
        task_names = requested

    canonical = _canonicalize(row.get(spec.smiles_column) for row in rows)
    keep = [i for i, smi in enumerate(canonical) if smi is not None]
    n_unparseable = len(rows) - len(keep)
    if not keep:
        raise ValueError(f"no parseable molecules in {path}")

    kept_smiles = [canonical[i] for i in keep]
    labels = np.array(
        [[_to_label(rows[i].get(task)) for task in task_names] for i in keep],
        dtype=float,
    )

    kept_smiles, labels, n_duplicates, n_conflicts = _merge_duplicates(
        kept_smiles, labels
    )

    return Dataset(
        name=name,
        smiles=tuple(kept_smiles),
        y=labels,
        tasks=tuple(task_names),
        n_unparseable=n_unparseable,
        n_duplicates=n_duplicates,
        n_label_conflicts=n_conflicts,
    )


#: Loader registry, kept parallel to ``splits.SPLITTERS`` so a config can
#: select either by name. A new dataset is added by registering a
#: :class:`DatasetSpec` in :data:`DATASETS`.
LOADERS: dict[str, Callable[..., Dataset]] = {
    name: (lambda n=name, **kw: load_dataset(n, **kw)) for name in DATASETS
}
