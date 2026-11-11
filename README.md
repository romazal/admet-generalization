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
make test                        # 168 tests, ~3 s
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
admet run configs/tox21_gnn_scaffold.yaml --seeds 3 --train-fractions 0.25
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
| Baseline, one Tox21 run at 100% (8 tuning trials, 1 seed) | 2.7 min |
| GNN, one Tox21 run at 100% (8 tuning trials, 1 seed) | 3.7 min |
| Baseline, one ClinTox run at 100% | 7 s |
| GNN, one ClinTox run at 100% | 38 s |
| `make main` (4 configs x 5 seeds) | ~1.1 h |
| `make sweep` (the 36 cells below 100%) | ~40 min |
| `make clintox` (4 configs x 5 seeds x 4 fractions) | ~17 min |
| **Everything: 136 runs** | **2.03 h** |

The GNN costs about 1.3x the baseline on Tox21 and 5x on ClinTox. Early
stopping ends a GNN trial at a mean best epoch of 35.4 (Tox21) and 26.2
(ClinTox) against a 50-epoch ceiling, so the ceiling rarely binds. To shorten
a pass, lower `tuning.budget` in `configs/base.yaml` — it applies to **both**
models at once, which is the point (see *Safeguards*).

These are indicative of relative cost rather than a benchmark: the machine was
in ordinary use and the baseline uses all cores via `n_jobs=-1`.

### CPU and GPU

Nothing in the code needs changing to switch — `GNNModel` picks CUDA whenever
`torch.cuda.is_available()` is true, and falls back to CPU otherwise. The
grid was run both ways; `versions.torch_device` in each result file says
which. On CPU the same 136 runs take **3.83 h** instead of 2.03 h:

| | CPU | GPU | Speedup |
|---|---|---|---|
| Tox21 GNN, mean per run | 302 s | 118 s | **2.6x** |
| Tox21 GNN at 100% | 555 s | 221 s | **2.5x** |
| ClinTox GNN, mean per run | 43 s | 19 s | **2.2x** |
| Baseline (scikit-learn, either way) | 119 s | 106 s | — |
| **All 136 runs** | **3.83 h** | **2.03 h** | **1.9x** |

Only the GNN is accelerated; the baseline is scikit-learn and never touches
the card, so its 119 → 106 s is machine noise, not a speedup. The overall
1.9x is therefore bounded by the baseline's share of the grid, and the 2.5x
on the heaviest cell is the figure that matters for scaling up — worth having
before attempting the larger datasets in [TODO.md](TODO.md), where ToxCast
and HIV are expensive.

To install a CUDA build (the PyPI default on Windows is CPU-only):

```bash
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.cuda.is_available())"   # expect True
```

Match the `cu…` version to your driver; recent NVIDIA cards need cu128 or
later. **The device changes the GNN's scores slightly** — see *Reproducibility*
below for how much, and why the baseline is unaffected.

---

## Summary of results

136 runs: both models and both splits on Tox21 (five seeds at the full
training set, three along the learning curve) and the same grid on ClinTox
(five seeds throughout). 2.03 h of compute, GNN on a GPU. Full tables and
threats to validity are in **[REPORT.md](REPORT.md)**; the Tox21 detail is in
*[Findings](#findings)* below.

- **The split type is the whole story.** Scaffold splitting costs the baseline
  0.073 ROC-AUC and the GNN 0.078 — roughly ten times the difference between
  the two models in any condition. When the test molecules stop resembling the
  training molecules, both methods degrade far more than either gains from its
  architecture.
- **H2 (data size): supported, with one caveat.** The fingerprint baseline
  beats the GNN with little data and the advantage closes as data is added —
  in PR-AUC the gap runs from −0.047 at 10% of the training set to −0.019 at
  100% under a scaffold split, monotonically. Under a random split it runs
  from −0.094 to −0.024 but not monotonically, and that ordering did not
  survive a rerun on different hardware (see *Reproducibility*).
- **H1 (generalization): not supported, because its premise fails.** At the
  full training set there is no GNN advantage to shrink: +0.000 ± 0.018 under
  the random split, −0.005 ± 0.017 under the scaffold split, both within one
  standard deviation of zero and of each other.
- **Headline numbers** (Tox21 test ROC-AUC, five seeds): baseline
  **0.834 ± 0.009** random / **0.760 ± 0.013** scaffold; GNN
  **0.834 ± 0.015** random / **0.755 ± 0.011** scaffold.
- **ClinTox is inconclusive, and informative anyway.** Every GNN-minus-baseline
  gap on it is smaller than its own SD (+0.083 ± 0.085 at best), so it settles
  nothing about H1 or H2. But it scores *higher* than Tox21 on a fifth of the
  data and the baseline loses only 0.021 ROC-AUC to a scaffold split, where
  Tox21 loses 0.073–0.078. Together with 19 duplicate structures carrying
  contradictory labels, that points at dataset provenance rather than at model
  quality.
- **The GNN costs 1.3x the baseline on Tox21 and 5x on ClinTox** with the GPU
  (2.5x and 12x without one), for no measurable gain.
- **A methodological result worth the same weight.** An intermediate three-seed
  pass put the scaffold gap at −0.016 ± 0.020, which reads as a real effect;
  at five seeds it is −0.005 ± 0.017. The proposal's "target 5, minimum 3" is
  doing real work — three-seed numbers are directional only.

The honest one-line answer to the research question: *on this dataset, at this
tuning budget, the GNN never outperformed the fingerprint baseline — and the
structural shift hurt it slightly more.*

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

| Metric | Split | Baseline (Morgan + RF) | GNN (GINE) | GNN − baseline |
|---|---|---|---|---|
| ROC-AUC | Random | 0.834 ± 0.009 | 0.834 ± 0.015 | **+0.000 ± 0.018** |
| ROC-AUC | Scaffold | 0.760 ± 0.013 | 0.755 ± 0.011 | **−0.005 ± 0.017** |
| PR-AUC | Random | 0.473 ± 0.023 | 0.449 ± 0.029 | −0.024 ± 0.037 |
| PR-AUC | Scaffold | 0.379 ± 0.019 | 0.360 ± 0.020 | −0.019 ± 0.028 |

### Learning curves (3 seeds) — GNN minus baseline

| Training set | ROC-AUC, random | ROC-AUC, scaffold | PR-AUC, random | PR-AUC, scaffold |
|---|---|---|---|---|
| 10% | −0.016 ± 0.020 | −0.018 ± 0.035 | **−0.094 ± 0.033** | **−0.047 ± 0.019** |
| 25% | +0.005 ± 0.015 | −0.003 ± 0.007 | −0.033 ± 0.018 | −0.039 ± 0.010 |
| 50% | −0.003 ± 0.013 | −0.001 ± 0.014 | −0.041 ± 0.020 | −0.034 ± 0.020 |
| 100% | +0.000 ± 0.018 | −0.005 ± 0.017 | −0.024 ± 0.037 | −0.019 ± 0.028 |

Four things follow, in order of how well the data supports them.

**1. The split type dominates everything else.** Moving from a random to a
scaffold split costs the baseline 0.073 ROC-AUC and the GNN 0.078 — an order of
magnitude more than the difference between the two models in any condition.
Whatever is gained by choosing an architecture here is small next to what is
lost when the test molecules stop resembling the training molecules. This is
the result the research question was asked to expose, and it is unambiguous.

**2. H2 is supported, with a caveat on how cleanly.** The baseline is clearly
better than the GNN with little data, and the advantage closes as data is
added. The cleanest evidence is PR-AUC, where the 10% gap (−0.094 random,
−0.047 scaffold) is larger than its own SD and falls to roughly −0.02 at 100%
in both splits. Under the scaffold split the decline is monotone; under the
random split it is not (the 25% point sits above the 50% one), and which of
the two is monotone changed between the CPU and GPU passes. The *endpoints*
are robust, the shape in between is not — treat H2 as supported by the 10%
versus 100% contrast rather than by a smooth curve.

**3. H1 is not supported, because its premise does not hold.** H1 predicts a
GNN advantage that shrinks under structural shift. At 100% there is no GNN
advantage to shrink in either split: +0.000 ± 0.018 random, −0.005 ± 0.017
scaffold, both inside one SD of zero and indistinguishable from each other. The
honest statement is *no GNN advantage was detected under either split*. That
outcome is compatible with H1 and equally compatible with a GNN that is simply
undertrained at 8 tuning trials; this design cannot separate those.

**4. Three seeds were not enough, and the data shows it.** At 3 seeds the
scaffold ROC-AUC gap read −0.016 ± 0.020; at 5 it reads −0.005 ± 0.017. The
apparent effect was seed noise. Treat every 3-seed row above as directional
only.

Two sanity checks passed: the baseline's 0.834 under a random split sits in the
range published for fingerprint models on Tox21, and the scaffold-disjointness
check is recorded `true` in all 68 scaffold result files and `false` in all 68
random ones.

Figures: `figures/tox21/random_vs_scaffold_tox21_roc_auc.png` (H1),
`figures/tox21/learning_curves_tox21_roc_auc.png` and
`figures/tox21/gnn_minus_rf_gap_tox21_roc_auc.png` (H2), plus the PR-AUC
counterparts and the same three under `figures/clintox/`.

ClinTox results and the cross-dataset comparison are in
[REPORT.md](REPORT.md) §4.

## Reproducibility

`make aggregate && make plots` on a fresh clone regenerates `runs.csv`,
`summary.csv`, `gap.csv`, `per_task.csv` and all 12 figures with **zero
diff** — every table and figure is a pure function of the committed result
files.

Re-running the *training* is reproducible to the extent the libraries are.
The whole grid was run twice, once with the GNN on CPU and once on a GPU,
which measures that directly:

| | Reproduced |
|---|---|
| Baseline test metrics, all 68 runs | **bit-identical** |
| Baseline validation metrics | 67/68 bit-identical; one drifted by 8e-7 |
| GNN test metrics | changed, within seed noise (see below) |

**The baseline is exactly reproducible and the GNN is not.** The baseline is
scikit-learn with fixed seeds and never touches the GPU, so it is the control:
its test metrics came back identical to the last decimal place, which is what
licenses attributing every GNN change to the GNN rather than to the data, the
splits or the harness. The one baseline value that moved is a validation
ROC-AUC, 8e-7, from non-deterministic float summation across threads; it did
not change which tuning trial won, so no reported number depends on it.

The GNN's scores moved more. Two things differed at once — the device, and
the torch version (2.15.0+cpu → 2.11.0+cu128, because no cu128 wheel exists
for 2.15 yet) — so the change cannot be attributed to the GPU alone:

| Tox21 GNN ROC-AUC, 100% | CPU | GPU | Δ | SD |
|---|---|---|---|---|
| Random | 0.836 | 0.834 | −0.002 | ± 0.015 |
| Scaffold | 0.756 | 0.755 | −0.001 | ± 0.011 |

Both shifts are an order of magnitude below the seed-to-seed SD, and **every
conclusion above survives**: the split type still dominates, both headline
gaps still sit inside one SD of zero, and the 10%-to-100% narrowing still
holds. What did *not* survive is the finer structure — the monotone ordering
of the PR-AUC learning curve swapped splits, and on ClinTox the GNN's random
split moved +0.037 and its scaffold drop went from 0.043 to 0.082. Those were
always inside the SDs; the rerun is a reminder that reading them as findings
would have been wrong.

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

**Score `CT_TOX` only** — `configs/clintox_*.yaml` set `tasks: [CT_TOX]`.
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
  (`configs/tox21_rf_maccs_*.yaml`) but deliberately **not** in the tuned
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
├── configs/
│   ├── base.yaml                   # shared defaults incl. the tuning budget
│   ├── tox21_{rf,gnn}_{random,scaffold}.yaml
│   ├── clintox_{rf,gnn}_{random,scaffold}.yaml
│   └── sweep_learning_curve.yaml   # the learning-curve grid
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
│   │   └── gnn.py          # GINE message passing, masked multitask head
│   └── analysis/
│       ├── aggregate.py    # results/**/*.json -> summary + gap tables
│       └── plots.py        # the three figures, one set per dataset
├── tests/                  # 168 tests
├── data/                   # raw datasets, committed so a clone runs offline
├── results/
│   ├── tox21/              # 56 run files
│   ├── clintox/            # 80 run files
│   └── {runs,summary,gap}.csv
└── figures/
    ├── tox21/              # 6 PNGs
    └── clintox/            # 6 PNGs
```

## Figures

Regenerated from `results/` by `make plots`; none is drawn by hand. One
subdirectory per dataset, mirroring `results/`, as
`figures/<dataset>/<figure>_<dataset>_<metric>.png` — 12 in all. The dataset
is in the filename as well as the folder, because a figure is usually read
after being pasted into a report or a slide, where the folder is gone:

- `random_vs_scaffold_*.png` — both models under both splits (H1).
- `learning_curves_*.png` — test score against training-set size (H2).
- `gnn_minus_rf_gap_*.png` — the gap itself, with a zero reference line.

Two datasets never share an axis. Their class balances and test-set sizes
differ, so a shared y-axis would invite a comparison the numbers do not
support; the cross-dataset comparison is made in the tables instead.

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
  hours". The committed results use a GPU (RTX 5070 Ti) and take 2.03 h, well
  inside that. The grid was also run end to end on the CPU alone, in 3.83 h —
  still inside it — so a teammate without a CUDA card can reproduce
  everything; see *CPU and GPU* and *Reproducibility* above. The code selects
  CUDA automatically when available (`models/gnn.py`); no flag to set.
- **No pandas.** Proposal §5 lists it, but its compiled extensions are blocked
  by the Windows Application Control policy on the development machine. The
  loader and the aggregation use the standard library's `csv` plus NumPy.
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
