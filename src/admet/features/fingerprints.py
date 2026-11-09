"""Molecular fingerprints used by the random-forest baseline.

Morgan fingerprints describe a molecule by its circular atom environments:
for every atom, the substructure reachable within ``radius`` bonds is hashed
into one of ``n_bits`` positions. MACCS keys encode a fixed set of predefined
substructures. Both are fixed-length bit vectors suitable for a random forest.

These are the representations the GNN is being compared *against* (proposal
§3.2), so the comparison is only fair if the baseline gets a competent version
of them. The Morgan defaults are the ones the ECFP4 literature and MoleculeNet
use — radius 2, 2048 bits — and are exposed in the configs so the baseline's
representation is part of its tuning budget rather than a fixed handicap.

What a fingerprint cannot do is the point of the study: its representation is
fixed before any label is seen, so unlike the GNN's learned embeddings it
cannot adapt to the assay. Morgan counting bits (``use_counts=True``) keeps
how *often* an environment occurs instead of only whether it occurs.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

import numpy as np

__all__ = [
    "DEFAULT_RADIUS",
    "DEFAULT_N_BITS",
    "MACCS_N_BITS",
    "morgan_fingerprint",
    "maccs_fingerprint",
    "featurize_smiles",
]

#: ECFP4-equivalent radius, the MoleculeNet convention.
DEFAULT_RADIUS = 2

#: Fingerprint length in bits.
DEFAULT_N_BITS = 2048

#: RDKit reserves bit zero in its 167-bit MACCS key vector.
MACCS_N_BITS = 167


@lru_cache(maxsize=8)
def _generator(radius: int, n_bits: int):
    """Cached RDKit Morgan generator.

    Building a generator is not free, and the trainer featurizes several times
    per run (train, validation, test) with identical settings.
    """
    from rdkit.Chem import rdFingerprintGenerator

    return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)


def _parse(smiles: str):
    """Parse one SMILES, raising a useful error rather than returning ``None``."""
    from rdkit import Chem

    if not isinstance(smiles, str):
        raise TypeError(f"expected a SMILES string, got {type(smiles).__name__}")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
    return mol


def morgan_fingerprint(
    smiles: str,
    *,
    radius: int = DEFAULT_RADIUS,
    n_bits: int = DEFAULT_N_BITS,
    use_counts: bool = False,
) -> np.ndarray:
    """Return the Morgan fingerprint of one molecule as a 1-D array.

    Args:
        smiles: a SMILES string RDKit can parse.
        radius: how many bonds out each atom environment reaches.
        n_bits: length of the output vector.
        use_counts: return occurrence counts instead of presence bits.

    Returns:
        A ``(n_bits,)`` array — ``uint8`` of 0/1 bits, or ``int32`` counts.

    Raises:
        ValueError: if ``radius``/``n_bits`` are not positive, or the SMILES
            cannot be parsed.
    """
    if radius < 0:
        raise ValueError(f"radius must be non-negative, got {radius!r}")
    if n_bits <= 0:
        raise ValueError(f"n_bits must be positive, got {n_bits!r}")

    mol = _parse(smiles)
    generator = _generator(radius, n_bits)
    if use_counts:
        fingerprint = generator.GetCountFingerprintAsNumPy(mol)
        return np.asarray(fingerprint, dtype=np.int32)
    fingerprint = generator.GetFingerprintAsNumPy(mol)
    return np.asarray(fingerprint, dtype=np.uint8)


def maccs_fingerprint(smiles: str) -> np.ndarray:
    """Return the 167-bit MACCS key fingerprint of one molecule.

    RDKit reserves bit zero, so the returned vector has 167 entries even
    though the keys are conventionally numbered 1 through 166.
    """
    from rdkit import DataStructs
    from rdkit.Chem import rdMolDescriptors

    mol = _parse(smiles)
    fingerprint = rdMolDescriptors.GetMACCSKeysFingerprint(mol)
    result = np.zeros((MACCS_N_BITS,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fingerprint, result)
    return result


def featurize_smiles(
    smiles: Sequence[str],
    *,
    radius: int = DEFAULT_RADIUS,
    n_bits: int = DEFAULT_N_BITS,
    use_counts: bool = False,
) -> np.ndarray:
    """Fingerprint a sequence of molecules into one design matrix.

    Args:
        smiles: SMILES in dataset row order.
        radius, n_bits, use_counts: passed to :func:`morgan_fingerprint`.

    Returns:
        An ``(n_molecules, n_bits)`` array whose rows line up with ``smiles``.

    Raises:
        ValueError: if ``smiles`` is empty, or a molecule cannot be parsed —
            the error names the offending row, since by this point unparseable
            molecules should already have been dropped by ``loaders.py``.
    """
    rows = list(smiles)
    if not rows:
        raise ValueError("cannot featurize an empty sequence of molecules")

    dtype = np.int32 if use_counts else np.uint8
    matrix = np.empty((len(rows), n_bits), dtype=dtype)
    for index, smi in enumerate(rows):
        try:
            matrix[index] = morgan_fingerprint(
                smi, radius=radius, n_bits=n_bits, use_counts=use_counts
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"at dataset index {index}: {exc}") from exc
    return matrix
