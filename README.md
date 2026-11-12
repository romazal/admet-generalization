# ADMET Generalization

**When does a graph neural network outperform a fingerprint-based model for
ADMET property prediction, and does that benefit survive when the test
molecules are structurally different from the training molecules?**

A reproducible pipeline comparing a classical baseline (Morgan fingerprints +
random forest) against a message-passing graph neural network on Tox21 and
ClinTox, under a random split and a scaffold split. The focus is on *when* the
advanced method helps — not on which model reaches the highest number.

Course: Biomedical Artificial Intelligence / Drug discovery and development
(flipped-classroom lecture), Group 2.

---

## Quick start

### 1. Install

Requires Python >= 3.12 (developed on 3.14.4; RDKit, PyTorch, scikit-learn and
matplotlib all publish wheels for 3.12–3.14).

```bash
python -m venv .venv
source .venv/bin/activate        # .venv\Scripts\activate on Windows
pip install -e ".[dev]"
make test                        # 192 tests, ~4 s
```

No download, and a GPU is optional (it makes the GNN ~2.5x faster; see
*CPU and GPU*): both raw datasets are committed under
`data/` (123 KB Tox21, 20 KB ClinTox), so a fresh clone runs offline. Set
`ADMET_DATA_DIR` to move the cache elsewhere.

### 2. Read the results without running anything

Everything is already committed, so the finished work is readable straight
from a clone:

| What | Where |
|---|---|
| The full written report | **[REPORT.md](REPORT.md)** |
| Headline numbers | *[Summary of results](#summary-of-results)* below |
| Per-condition means and SDs | `results/summary.csv` |
| Per-assay means and SDs | `results/per_task.csv` |
| GNN-minus-baseline gaps | `results/gap.csv` |
| One row per run | `results/runs.csv` |
| Every individual run | `results/tox21/*.json`, `results/clintox/*.json` |
| Figures | `figures/tox21/*.png`, `figures/clintox/*.png` |
| Small tasks open for contributors | **[TODO.md](TODO.md)** |

### 3. Reproduce it

A run is fully specified by **one config file plus one seed**, and writes one
JSON holding the config, the seed, the split sizes, every tuning trial, the
metrics, the library versions and the git commit.

```bash
make main        # Tox21: both models x both splits, 5 seeds     (~1.1 h)
make sweep       # Tox21 learning curves: 10/25/50/100% training (~40 min)
make clintox     # ClinTox: the same grid, 5 seeds x 4 fractions (~17 min)

make aggregate   # results/**/*.json -> runs.csv, summary.csv, gap.csv, per_task.csv
make plots       # -> figures/*.png
```

`make all` runs the lot in order (~2.0 h on a CUDA GPU, ~3.8 h on CPU).
Every command skips a cell
that already has a result file, so an interrupted pass resumes by reissuing
the same command; pass `--overwrite` to redo them. To reproduce a single cell:

```bash
admet run configs/tox21/gnn_scaffold.yaml --seeds 3 --train-fractions 0.25
```

### Runtime and hardware

Measured, not estimated: every figure below comes from the `timing_seconds`
field of the committed result files. Windows desktop, 16 threads, 31 GB RAM,
GNN on an **RTX 5070 Ti**; each result file records the device it used in
`versions.torch_device`.

| Step | Time |
|---|---|
| Test suite | ~3 s |
| Parse a dataset (no download — both are committed) | ~3 s |
| Baseline, one Tox21 run at 100% (8 tuning trials, 1 seed) | 3.0 min |
| GNN, one Tox21 run at 100% (8 tuning trials, 1 seed) | 11.2 min |
| Baseline, one ClinTox run at 100% | 7 s |
| GNN, one ClinTox run at 100% | 36 s |
| `make main` (4 configs x 5 seeds) | ~2.4 h |
| `make sweep` (the 36 cells below 100%) | ~1.5 h |
| `make clintox` (4 configs x 5 seeds x 4 fractions) | ~18 min |
| **Everything: 156 runs** | **4.19 h** |

The GNN costs about 3.8x the baseline on Tox21 and 5.5x on ClinTox, with
deterministic kernels pinned (*Reproducibility* has what that costs). Early
stopping ends a GNN trial at a mean best epoch of 34.4 (Tox21) and 25.8
(ClinTox) against a 50-epoch ceiling, so the ceiling rarely binds. To shorten
a pass, lower `tuning.budget` in `configs/base.yaml` — it applies to **both**
models at once, which is the point (see *Safeguards*).

These are indicative of relative cost rather than a benchmark: the machine was
in ordinary use and the baseline uses all cores via `n_jobs=-1`.

### CPU and GPU

Nothing in the code needs changing to switch — `GNNModel` picks CUDA whenever
`torch.cuda.is_available()` is true, and falls back to CPU otherwise. What is
committed is the GPU grid with deterministic kernels pinned.

Two caveats before the numbers. `versions.torch_device` records what torch
could *see*, not what the model used, so it reads `cuda` even on baseline runs
that are scikit-learn and never touch the card — read the `model` column
instead. And the CPU pass predates the determinism fix and used a different
wheel (2.15.0+cpu against 2.11.0+cu128), so that row is an order of magnitude
rather than a controlled comparison.

| Tox21 GNN at 100%, per run | Time |
|---|---|
| CPU, non-deterministic (torch 2.15.0+cpu) | 555 s |
| GPU, non-deterministic (torch 2.11.0+cu128) | 221 s |
| **GPU, deterministic — what is committed** | **673 s** |

The card is worth about 2.5x with the brakes off, and determinism costs 3.0x
on top of that, which leaves the committed configuration slower per run than
the CPU pass it replaced. That is the trade this project made deliberately: a
number that repeats beats a number that arrives sooner. Only the GNN is
affected either way — the baseline runs on cores at 178 s per Tox21 run at
100%, and the GPU's remaining value is for the larger datasets in
[TODO.md](TODO.md), where ToxCast and HIV are expensive.

To install a CUDA build (the PyPI default on Windows is CPU-only):

```bash
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.cuda.is_available())"   # expect True
```

Match the `cu…` version to your driver; recent NVIDIA cards need cu128 or
later. **The device changes the GNN's scores slightly** — see *Reproducibility*
below for how much, why the baseline is unaffected, and why the GNN needed
deterministic kernels pinned before any of it could be measured.

---

## Summary of results

156 runs: both models and both splits on Tox21 (five seeds at the full
training set, three along the learning curve) and the same grid on ClinTox
(five seeds throughout), plus a dummy floor on both. 4.19 h of compute, GNN on a GPU
with deterministic kernels pinned (see *Reproducibility*). Full tables and
threats to validity are in **[REPORT.md](REPORT.md)**; the Tox21 detail is in
*[Findings](#findings)* below.

- **The split type is the whole story.** Scaffold splitting costs the baseline
  0.073 ROC-AUC and the GNN 0.075 — roughly ten times the difference between
  the two models in any condition. When the test molecules stop resembling the
  training molecules, both methods degrade far more than either gains from its
  architecture.
- **H2 (data size): supported, with one caveat.** The fingerprint baseline
  beats the GNN with little data and the advantage closes as data is added —
  in PR-AUC the gap runs from −0.057 at 10% of the training set to −0.025 at
  100% under a scaffold split, and from −0.059 to −0.011 under a random split,
  monotonically in both. An earlier pass had the random-split ordering break;
  that turned out to be the GNN's own run-to-run noise (see *Reproducibility*).
- **H1 (generalization): not supported, because its premise fails.** At the
  full training set there is no GNN advantage to shrink: +0.005 ± 0.019 under
  the random split, +0.003 ± 0.016 under the scaffold split, both within one
  standard deviation of zero and of each other.
- **Headline numbers** (Tox21 test ROC-AUC, five seeds): baseline
  **0.834 ± 0.009** random / **0.760 ± 0.013** scaffold; GNN
  **0.838 ± 0.016** random / **0.763 ± 0.010** scaffold.
- **ClinTox is inconclusive, and informative anyway.** Every GNN-minus-baseline
  gap on it is no larger than its own SD (+0.091 ± 0.088 at best), so it settles
  nothing about H1 or H2. But it scores *higher* than Tox21 on a fifth of the
  data and the baseline loses only 0.021 ROC-AUC to a scaffold split, where
  Tox21 loses 0.073–0.075. Together with 19 duplicate structures carrying
  contradictory labels, that points at dataset provenance rather than at model
  quality.
- **The GNN costs 3.8x the baseline on Tox21 and 5.5x on ClinTox** at the full
  training set, on the GPU, for no measurable gain. Pinning deterministic
  kernels tripled its cost (see *Reproducibility*).
- **A methodological result worth the same weight.** An intermediate three-seed
  pass put the scaffold gap at −0.008 ± 0.011, which reads as a real effect at
  four times its own SD; at five seeds it is +0.003 ± 0.016 and the sign flips.
  The proposal's "target 5, minimum 3" is doing real work — three-seed numbers
  are directional only.

The honest one-line answer to the research question: *on this dataset, at this
tuning budget, the GNN and the fingerprint baseline are indistinguishable — and
the structural shift hurt both about equally.*

---

## Hypotheses

Both are stated so that the data can reject them, so either outcome is
informative.

**H1 (generalization).** The advantage of the GNN over the random forest is
smaller, or absent, under a scaffold split than under a random split.

**H2 (data size).** The advantage of the GNN depends on the amount of training
data: with little data the simpler baseline is competitive or better.

The quantity both hypotheses are stated in is the **GNN-minus-baseline gap**
per condition, not either model's absolute score. This links to the question
that motivates the lecture: molecules worth finding are, by definition, unlike
the training data.

## Findings

Tox21, full tuning budget (8 trials per model). Main experiment: **5 seeds**
(the proposal's target). Learning curves: **3 seeds** (its stated minimum — the
grid is 16× larger). 56 Tox21 runs. Every number below comes from
`results/summary.csv` and `results/gap.csv`; none is typed by hand.

### Main experiment (100% of the training set, 5 seeds)

| Metric | Split | Dummy (floor) | Baseline (Morgan + RF) | GNN (GINE) | GNN − baseline |
|---|---|---|---|---|---|
| ROC-AUC | Random | 0.500 ± 0.000 | 0.834 ± 0.009 | 0.838 ± 0.016 | **+0.005 ± 0.019** |
| ROC-AUC | Scaffold | 0.500 ± 0.000 | 0.760 ± 0.013 | 0.763 ± 0.010 | **+0.003 ± 0.016** |
| PR-AUC | Random | 0.077 ± 0.002 | 0.473 ± 0.023 | 0.462 ± 0.018 | −0.011 ± 0.029 |
| PR-AUC | Scaffold | 0.100 ± 0.007 | 0.379 ± 0.019 | 0.354 ± 0.017 | −0.025 ± 0.025 |

The dummy ignores the molecule and predicts each assay's training-set
positive rate. It is there to make the PR-AUC column readable: 0.473 looks
weak in isolation and is about six times chance on a dataset this imbalanced.

### Learning curves (3 seeds) — GNN minus baseline

| Training set | ROC-AUC, random | ROC-AUC, scaffold | PR-AUC, random | PR-AUC, scaffold |
|---|---|---|---|---|
| 10% | −0.014 ± 0.023 | +0.001 ± 0.014 | **−0.059 ± 0.024** | **−0.057 ± 0.024** |
| 25% | +0.006 ± 0.014 | −0.006 ± 0.012 | −0.052 ± 0.021 | −0.029 ± 0.025 |
| 50% | −0.007 ± 0.011 | −0.002 ± 0.024 | −0.042 ± 0.017 | −0.027 ± 0.028 |
| 100% | +0.005 ± 0.019 | +0.003 ± 0.016 | −0.011 ± 0.029 | −0.025 ± 0.025 |

Four things follow, in order of how well the data supports them.

**1. The split type dominates everything else.** Moving from a random to a
scaffold split costs the baseline 0.073 ROC-AUC and the GNN 0.075 — an order of
magnitude more than the difference between the two models in any condition.
Whatever is gained by choosing an architecture here is small next to what is
lost when the test molecules stop resembling the training molecules. This is
the result the research question was asked to expose, and it is unambiguous.

**2. H2 is supported, with a caveat on how cleanly.** The baseline is clearly
better than the GNN with little data, and the advantage closes as data is
added. The cleanest evidence is PR-AUC, where the 10% gap (−0.059 random,
−0.057 scaffold) is more than twice its own SD and shrinks monotonically to
−0.011 and −0.025 at 100%. An earlier, non-deterministic pass had the random
split break monotonicity and the two splits disagree about which was smooth;
that was the GNN's own run-to-run spread, not a property of the data. The
*endpoints* carry the claim either way — treat H2 as supported by the 10%
versus 100% contrast rather than by the shape in between.

**3. H1 is not supported, because its premise does not hold.** H1 predicts a
GNN advantage that shrinks under structural shift. At 100% there is no GNN
advantage to shrink in either split: +0.005 ± 0.019 random, +0.003 ± 0.016
scaffold, both inside one SD of zero and indistinguishable from each other. The
honest statement is *no GNN advantage was detected under either split*. That
outcome is compatible with H1 and equally compatible with a GNN that is simply
undertrained at 8 tuning trials; this design cannot separate those.

**4. Three seeds were not enough, and the data shows it.** At 3 seeds the
scaffold ROC-AUC gap read −0.008 ± 0.011 — four times its own SD, which reads
as a real baseline win; at 5 it reads +0.003 ± 0.016 and the sign flips. The
apparent effect was seed noise. Treat every 3-seed row above as directional
only.

Three sanity checks passed: the baseline's 0.834 under a random split sits in
the range published for fingerprint models on Tox21; the scaffold-disjointness
check is recorded `true` in all 73 scaffold result files and `false` in all 73
random ones; and the dummy's PR-AUC reproduces the measured test-set positive
rate to three decimals (0.077 vs 0.0768 random, 0.100 vs 0.0997 scaffold),
which is what a class-prior model must score if the metric is masking missing
labels correctly.

Figures: `figures/tox21/random_vs_scaffold_tox21_roc_auc.png` (H1),
`figures/tox21/learning_curves_tox21_roc_auc.png` and
`figures/tox21/gnn_minus_rf_gap_tox21_roc_auc.png` (H2), plus the PR-AUC
counterparts and the same three under `figures/clintox/`.

`figures/tox21/train_test_similarity_tox21.png` shows the mechanism behind
all of it: the Tanimoto similarity of each test molecule to its nearest
training molecule, under each split. The median falls from 0.583 to 0.419,
and the share of test molecules that have a *fingerprint-identical* training
molecule falls from 7.8% to 0.6% — which is what the scaffold split is for,
and a reminder of how much a random split flatters a model on this data.

ClinTox results and the cross-dataset comparison are in
[REPORT.md](REPORT.md) §4.

## Reproducibility

`make aggregate && make plots` on a fresh clone regenerates `runs.csv`,
`summary.csv`, `gap.csv`, `per_task.csv` and all 24 figures with **zero
diff** — every table and figure is a pure function of the committed result
files.

Re-running the *training* reproduces too. It did not at first, and the way
that was found is worth recording.

**What went wrong.** An earlier pass ran the whole grid twice, once with the
GNN on CPU and once on a GPU, and read the difference between them as a device
effect. It was not one. Repeating a single Tox21 scaffold run three times on
the *same* card, with the same torch wheel, the same seed and the same data,
gave test ROC-AUC 0.7369 / 0.7413 / 0.7690 — a spread of 0.032, three times
the seed-to-seed SD of 0.011, and enough to change which hyperparameter
configuration the tuning selected. The CPU-versus-GPU deltas reported at the
time (−0.002 and −0.001) were far smaller than the noise inside the
measurement, so they were not measuring the device at all.

**Why.** The GNN sums each atom's incoming messages with a scatter-add, and on
a GPU the order in which those partial sums land is not fixed from run to run;
cuBLAS picks kernels the same opportunistic way. Float addition is not
associative, so neither is repeatable. Neither is a bug — both are the normal
speed-for-ordering trade — but both make a result impossible to repeat.

**The fix** is three lines in [`src/admet/training.py`](src/admet/training.py):
pin `CUBLAS_WORKSPACE_CONFIG=:4096:8`, turn off `cudnn.benchmark`, and call
`torch.use_deterministic_algorithms(True)`. Every result file now records
`"deterministic": true`, and all 156 files in `results/` come from a single
commit with it on. Re-running any of them reproduces the file bit for bit,
every tuning-trial score included. `ADMET_NONDETERMINISTIC=1` opts out.

**It is not free.** Determinism costs the Tox21 GNN **3.0× its runtime** —
221 s to 673 s per run at the full training set, same card, same wheel, only
the flag differing. (The earlier CPU pass took 555 s per run, but on a
different torch version, so treat that as an order of magnitude rather than a
clean comparison.) On graphs this small the GPU's margin does not survive the
requirement to be repeatable.

**The baseline was never the problem**, and it stays the control. Across all
68 random forest runs the re-run reproduced every test metric bit-identically;
one validation ROC-AUC moved by 8e-7, from thread-order float summation, and
it did not change which tuning trial won. scikit-learn, fixed seeds, no GPU,
no drift — which is what licenses attributing any GNN movement to the GNN
rather than to the data, the splits or the harness.

**What it cost in findings: nothing load-bearing.** Every number in this README
is the deterministic one. Against the earlier pass the GNN's Tox21 means moved
by at most 0.008 ROC-AUC at the full training set, 0.019 along the learning
curve and 0.025 on ClinTox, and each conclusion survived: the split type still
dominates, both headline gaps still sit inside one SD of zero, and the
10%-to-100% narrowing still holds. What moved is the finer structure the
earlier pass had already flagged as fragile — the PR-AUC learning curve is now
monotone under *both* splits, and the 100% scaffold gap changed sign
(−0.005 → +0.003). Both were always inside their SDs. That is the same lesson
the three-seed result teaches from the other direction: a quantity reported
with an SD as large as its mean was never a finding.

## Data

Both raw files are committed under `data/`, so nothing is downloaded on a
fresh clone.

**Tox21** (primary): 7823 molecules after cleaning (8 unparseable SMILES
dropped, 0 duplicates), screened against 12 nuclear-receptor and
stress-response toxicity assays, from the public MoleculeNet distribution.
It is multitask with **many missing labels** — 82.9% of the label matrix is
observed — which is why masking is treated as a correctness concern
throughout. 186–942 positives per assay. Split 6259 / 782 / 782 (scaffold;
the random split gives 6258 / 782 / 783).

**ClinTox** (second dataset): 1461 molecules after cleaning, two labels —
`FDA_APPROVED` and `CT_TOX`. Included to compare dataset *sizes* against
Tox21 (proposal §3.1). Split 1169 / 146 / 146. Tox21 is the primary dataset
and that is settled.

Measured from the raw file rather than taken from documentation, because
every figure is a reason for caution:

| | ClinTox |
|---|---|
| Molecules (raw → cleaned) | 1484 → 1461 (4 unparseable, 19 duplicates merged) |
| Minority class, raw | `FDA_APPROVED = 0`: 94 · `CT_TOX = 1`: 112 (~7%) |
| `CT_TOX` positives after cleaning | 103 |
| Positives in a scaffold test split | **10–17** over seeds 0–4 (test part of 146) |
| Positives at H2's 10% training point | **5** |
| `CT_TOX` == `NOT FDA_APPROVED` | **98.8%** of molecules |
| Label density | 1.0 — nothing is missing, so nothing is masked |
| Duplicates with contradictory labels | 19 structures, 38 conflicting labels |

So: the seed-to-seed noise on ~13 positives exceeds the GNN-minus-baseline gap
the study is defined to measure; the two labels are effectively one, which
leaves the multitask head and masked loss with nothing to do; and 19 structures
appear both as approved and as failed-for-toxicity, consistent with the dataset
being assembled from two sources. The measured results bear this out — see
[REPORT.md](REPORT.md) §4.

**Score `CT_TOX` only** — `configs/clintox/*.yaml` set `tasks: [CT_TOX]`.
`FDA_APPROVED` is positive for 91% of the test part, so its PR-AUC runs at
~0.98 where a constant predictor scores ~0.91; averaging it in lifted the
headline PR-AUC from 0.38 to 0.68 in a measured run and hid the real signal.

```bash
make clintox     # all four ClinTox configs, 5 seeds x 4 training fractions
```

Treat the outcome as a comparison of dataset *sizes* against Tox21, not as a
second test of H1 and H2 — it came out inconclusive, as expected.

## Experimental design

**Main experiment.** Both models under a random split and a *randomized*
scaffold split (Bemis-Murcko scaffolds, whole scaffold groups assigned to one
part only, shuffled per seed so each seed gives a different split).

**Additional experiment.** Learning curves at 10%, 25%, 50% and 100% **of the
training set**. The fraction applies to the training set alone — validation and
test stay identical across the four points, so a change along a curve is the
effect of adding training data and not of resampling the test set. The subsets
are nested (10% ⊂ 25% ⊂ 50% ⊂ 100%).

**Metrics.** ROC-AUC (primary) and PR-AUC (secondary, because of class
imbalance), macro-averaged over the scored assays (12 on Tox21, 1 on ClinTox)
and reported as mean and standard deviation over seeds — **target 5 seeds,
minimum 3**.

### Safeguards for a fair comparison

These are the things that, if wrong, would invalidate the comparison rather
than merely weaken it:

- **Equal tuning budget.** Both models draw the same number of hyperparameter
  settings from `sample_configs()`, set once in `configs/base.yaml`. The
  baseline tunes its own representation (fingerprint radius, length, counts)
  alongside the forest, so it is not handicapped by a fixed featurization. A
  second fingerprint, MACCS, is implemented and selectable per-run
  (`configs/tox21/rf_maccs_*.yaml`) but deliberately **not** in the tuned
  space: spending part of the baseline's budget on a second representation
  while the GNN spends all of its own on one would quietly unbalance the
  comparison in the GNN's favour.
- **Validation matches test.** Hyperparameters are selected on a validation set
  of the *same split type* as the test set, so they are never chosen under an
  easier regime than the one they are judged in.
- **No scaffold spans two parts.** Enforced by the splitter, asserted by the
  test suite, and re-checked at run time on the actual split each scaffold run
  uses. `test_random_split_does_leak_scaffolds` is the positive control: it
  asserts a random split *does* share scaffolds, so the scaffold-split
  assertion cannot pass trivially.
- **Masking is tested on both sides.** A molecule with a missing label for an
  assay is excluded from that assay only — never read as a negative, never
  dropped wholesale — in the metrics *and* in the GNN's loss. Both have tests
  with positive controls showing the wrong implementation gives a different
  answer.
- **Duplicates merged before splitting.** Structures are canonicalized and
  deduplicated in the loader, so the same molecule cannot appear in train and
  test.
- **Seeds, config and library versions are stored in every result file.**

## Repository structure

```text
admet-generalization/
├── pyproject.toml          # pinned dependencies, `admet` entry point
├── Makefile                # make test / main / sweep / clintox / aggregate / plots
├── README.md
├── REPORT.md               # the results report: tables, runtimes, validity
├── TODO.md                 # small contributions open to the team
├── AGENT_LOG.md            # record of the AI-assisted development
├── configs/                     # one directory per dataset; base.yaml is shared
│   ├── base.yaml                   # shared defaults incl. the tuning budget
│   ├── tox21/
│   │   ├── {rf,gnn}_{random,scaffold}.yaml
│   │   ├── dummy_{random,scaffold}.yaml        # the class-prior floor
│   │   ├── rf_maccs_{random,scaffold}.yaml     # MACCS side comparison
│   │   └── sweep_learning_curve.yaml           # the learning-curve grid
│   └── clintox/
│       └── {rf,gnn}_{random,scaffold}.yaml
├── src/admet/
│   ├── cli.py              # admet run / sweep / aggregate / plots
│   ├── training.py         # one run: config + seed -> result file
│   ├── evaluation.py       # ROC-AUC / PR-AUC with masked multitask labels
│   ├── data/
│   │   ├── loaders.py      # dataset registry, cleaning, download cache
│   │   └── splits.py       # random + randomized scaffold split, leakage check
│   ├── features/
│   │   ├── fingerprints.py # Morgan / ECFP bit vectors, MACCS keys
│   │   └── graphs.py       # SMILES -> atom/bond graphs
│   ├── models/
│   │   ├── base.py         # shared interface + equal-budget sampler
│   │   ├── rf.py           # baseline: one forest per assay
│   │   ├── gnn.py          # GINE message passing, masked multitask head
│   │   └── dummy.py        # class-prior floor, ignores the molecule
│   └── analysis/
│       ├── aggregate.py    # results/**/*.json -> summary + gap tables
│       └── plots.py        # the three figures, one set per dataset
├── tests/                  # 192 tests
├── data/                   # raw datasets, committed so a clone runs offline
├── results/
│   ├── tox21/              # 66 run files (56 main grid + 10 dummy)
│   ├── clintox/            # 90 run files (80 main grid + 10 dummy)
│   └── {runs,summary,gap,per_task}.csv
└── figures/
    ├── all_gaps_{roc,pr}_auc.png   # the only figure spanning both datasets
    ├── tox21/              # 12 PNGs
    └── clintox/            # 10 PNGs (no per-assay heatmap: one assay)
```

## Figures

Regenerated from `results/` by `make plots`; none is drawn by hand. One
subdirectory per dataset, mirroring `results/`, as
`figures/<dataset>/<figure>_<dataset>_<metric>.png` — 24 in all. The dataset
is in the filename as well as the folder, because a figure is usually read
after being pasted into a report or a slide, where the folder is gone:

- `random_vs_scaffold_*.png` — both models under both splits (H1).
- `learning_curves_*.png` — test score against training-set size (H2).
- `gnn_minus_rf_gap_*.png` — the gap itself, with a zero reference line.
- `per_assay_*.png` — every Tox21 assay separately, as a heatmap. The
  macro-average is one number standing for twelve endpoints, and it hides
  that SR-MMP reaches 0.90 while NR-ER sits at 0.71. Tox21 only: ClinTox
  scores one assay, where a heatmap would be a bar chart with extra steps.
- `paired_seeds_*.png` — one point per seed, baseline on x against GNN on y,
  with the line of equality. Points on both sides of the diagonal *are* the
  null result, which a difference of means with an SD beside it states but
  does not show.
- `train_test_similarity_*.png` — the chemistry of the split rather than any
  score; drawn once per dataset, not once per metric.
- `dataset_overview_*.png` — the data before any model: positives against
  molecules screened per assay, and the molecule-size distribution.

**`figures/all_gaps_<metric>.png`** sits at the top rather than in a dataset
folder, and is the one figure that puts both datasets on one axis. It may,
because it plots *differences*: +0.01 means the same thing on Tox21 as on
ClinTox, where a raw 0.85 does not. Every gap in the study with ±1 SD against
a line at zero — 14 of the 16 intervals cross it, which is the H1/H2 result in
one image.

Two datasets never share an axis anywhere else. Their class balances and
test-set sizes differ, so a shared y-axis would invite a comparison the
numbers do not support; the cross-dataset comparison is made in the tables.

Model is encoded as colour, always the same colour, and never anything else.
Split is encoded as **position**: the learning-curve and gap figures are
faceted into a random-split panel and a scaffold-split panel sharing one
y-axis, so each panel carries at most two lines and the drop between panels
is the split's effect. Every figure states what its error bars are.

## Literature anchors

- **DeepTox** — the early multitask deep-learning approach to toxicity
  prediction on Tox21; the historical argument for neural models here.
- **MoleculeNet** — the benchmark conventions this project follows:
  the Tox21 distribution, the 80/10/10 split, scaffold splitting, and ROC-AUC
  as the primary metric.
- **BOOM** (NeurIPS) — out-of-distribution molecular property prediction;
  motivates the structural-shift test in the main experiment.
- **Nature Reviews Drug Discovery review** — broader context for the
  introduction and the critical discussion.

Results will be interpreted against these in the final discussion.

## Known deviations from the proposal

All are recorded in `AGENT_LOG.md` with the reasoning.

- **ClinTox as the second dataset, not BBBP.** §3.1 says "for example BBBP",
  so the slot was open; the team chose ClinTox. Its caveats are in *Data*
  above and in [REPORT.md](REPORT.md) §4.
- **`data/`, `results/` and `figures/` are committed.** The §5 file table
  gives `.gitignore` the job of keeping raw data and large result files out of
  version control. They are tracked instead so a clone reproduces the report
  offline and every number can be traced to its run file. The whole lot is
  2.8 MB.
- **Compute was never the constraint.** §8 budgets a 16 GB GPU and "a few GPU
  hours". The committed results use a GPU (RTX 5070 Ti) and take 4.19 h, inside
  that, though deterministic kernels tripled the GNN's share of it. An earlier
  pass ran the grid end to end on the CPU alone in 3.83 h — also inside the
  budget — so a teammate without a CUDA card can reproduce everything; see
  *CPU and GPU* and *Reproducibility* above. The code selects
  CUDA automatically when available (`models/gnn.py`); no flag to set.
- **No PyTorch Geometric.** Proposal §3.2 allows "a GIN in PyTorch Geometric";
  the same model is implemented directly in PyTorch (`index_add_` for the
  neighbour sum, ~10 lines) to avoid a dependency that does not install
  cleanly here. Covered by tests.

The optional cluster-based splitter and the pretrained-model stretch goal are
not implemented; they are the first two items on the proposal's
scope-reduction list (§8).

## Limitations

A Tox21-trained model inherits the assay's imperfect in vitro → human
concordance, and if the model's job is to catch what in vitro screening misses,
the residual signal is the least structure-predictable part. This belongs in
the limitations section of the lecture (Part 4), not in the choice of dataset.
