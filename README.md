# ADMET Generalization

**When does a graph neural network outperform a fingerprint-based model for
ADMET property prediction, and does that benefit survive when the test
molecules are structurally different from the training molecules?**

A reproducible pipeline comparing a classical baseline (Morgan fingerprints +
random forest) against a message-passing graph neural network on Tox21, under
a random split and a scaffold split. The focus is on *when* the advanced method
helps — not on which model reaches the highest number.

Course: Biomedical Artificial Intelligence / Drug discovery and development
(flipped-classroom lecture), Group 2.

---

## Summary of results

56 runs on Tox21: both models, both splits, five seeds at the full training set
and three seeds along the learning curve. Full tables and reasoning in
*[Findings](#findings)*.

- **The split type is the whole story.** Scaffold splitting costs the baseline
  0.074 ROC-AUC and the GNN 0.080 — roughly ten times the difference between
  the two models in any condition. When the test molecules stop resembling the
  training molecules, both methods degrade far more than either gains from its
  architecture.
- **H2 (data size): supported.** The fingerprint baseline beats the GNN with
  little data and the advantage closes as data is added — in PR-AUC the gap
  runs from −0.086 at 10% of the training set to −0.031 at 100% under a random
  split, and from −0.058 to −0.026 under a scaffold split, monotonically in
  both.
- **H1 (generalization): not supported, because its premise fails.** At the
  full training set there is no GNN advantage to shrink: +0.003 ± 0.015 under
  the random split, −0.004 ± 0.022 under the scaffold split, both within one
  standard deviation of zero and of each other.
- **Headline numbers** (test ROC-AUC, five seeds): baseline **0.834 ± 0.009**
  random / **0.760 ± 0.013** scaffold; GNN **0.836 ± 0.011** random /
  **0.756 ± 0.018** scaffold.
- **A methodological result worth the same weight.** An intermediate three-seed
  pass put the scaffold gap at −0.016 ± 0.020, which reads as a real effect;
  at five seeds it is −0.004 ± 0.022. The proposal's "target 5, minimum 3" is
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
grid is 16× larger). 56 runs in total. Every number below comes from
`results/summary.csv` and `results/gap.csv`; none is typed by hand.

### Main experiment (100% of the training set, 5 seeds)

| Metric | Split | Baseline (Morgan + RF) | GNN (GINE) | GNN − baseline |
|---|---|---|---|---|
| ROC-AUC | Random | 0.834 ± 0.009 | 0.836 ± 0.011 | **+0.003 ± 0.015** |
| ROC-AUC | Scaffold | 0.760 ± 0.013 | 0.756 ± 0.018 | **−0.004 ± 0.022** |
| PR-AUC | Random | 0.473 ± 0.023 | 0.442 ± 0.029 | −0.031 ± 0.037 |
| PR-AUC | Scaffold | 0.379 ± 0.019 | 0.353 ± 0.019 | −0.026 ± 0.027 |

### Learning curves (3 seeds) — GNN minus baseline

| Training set | ROC-AUC, random | ROC-AUC, scaffold | PR-AUC, random | PR-AUC, scaffold |
|---|---|---|---|---|
| 10% | −0.016 ± 0.025 | +0.001 ± 0.014 | **−0.086 ± 0.054** | **−0.058 ± 0.030** |
| 25% | −0.008 ± 0.019 | −0.013 ± 0.017 | −0.063 ± 0.018 | −0.041 ± 0.019 |
| 50% | −0.001 ± 0.011 | −0.010 ± 0.018 | −0.028 ± 0.020 | −0.032 ± 0.033 |
| 100% | +0.003 ± 0.015 | −0.004 ± 0.022 | −0.031 ± 0.037 | −0.026 ± 0.027 |

Four things follow, in order of how well the data supports them.

**1. The split type dominates everything else.** Moving from a random to a
scaffold split costs the baseline 0.074 ROC-AUC and the GNN 0.080 — an order of
magnitude more than the difference between the two models in any condition.
Whatever is gained by choosing an architecture here is small next to what is
lost when the test molecules stop resembling the training molecules. This is
the result the research question was asked to expose, and it is unambiguous.

**2. H2 is supported.** The baseline is clearly better than the GNN with little
data, and the advantage closes as data is added. The cleanest evidence is
PR-AUC, where the 10% gap (−0.086 random, −0.058 scaffold) is larger than its
own SD and shrinks monotonically to roughly −0.03 at 100% in both splits. The
ROC-AUC curve under the random split shows the same shape and crosses zero near
50%. This is the proposal's H2 as stated.

**3. H1 is not supported, because its premise does not hold.** H1 predicts a
GNN advantage that shrinks under structural shift. At 100% there is no GNN
advantage to shrink in either split: +0.003 ± 0.015 random, −0.004 ± 0.022
scaffold, both inside one SD of zero and indistinguishable from each other. The
honest statement is *no GNN advantage was detected under either split*. That
outcome is compatible with H1 and equally compatible with a GNN that is simply
undertrained at 8 tuning trials; this design cannot separate those.

**4. Three seeds were not enough, and the data shows it.** At 3 seeds the
scaffold ROC-AUC gap read −0.016 ± 0.020; at 5 it reads −0.004 ± 0.022. The
apparent effect was seed noise. Treat every 3-seed row above as directional
only.

Two sanity checks passed: the baseline's 0.834 under a random split sits in the
range published for fingerprint models on Tox21, and the scaffold-disjointness
assertion ran on every scaffold split used.

Figures: `figures/random_vs_scaffold_tox21_roc_auc.png` (H1),
`figures/learning_curves_tox21_roc_auc.png` and
`figures/gnn_minus_rf_gap_tox21_roc_auc.png` (H2), plus the PR-AUC
counterparts and the same three for ClinTox.

## Data

**Tox21** (primary): ~7.8k molecules screened against 12 nuclear-receptor and
stress-response toxicity assays, loaded from the public MoleculeNet
distribution and cached under `data/`. It is multitask with **many missing
labels** — only 83% of the label matrix is observed — which is why masking is
treated as a correctness concern throughout.

As loaded: 7823 molecules (8 unparseable SMILES dropped), 12 assays, 186–942
positives per assay.

**ClinTox** (optional second): 1461 molecules after cleaning, two labels —
`FDA_APPROVED` and `CT_TOX`. Registered and downloadable, included to compare
dataset *sizes* (proposal §3.1, "if time allows"). Tox21 is the primary
dataset and that is settled.

Measured from the raw file rather than taken from documentation, because
every figure is a reason for caution:

| | ClinTox |
|---|---|
| Molecules (raw → cleaned) | 1484 → 1461 (4 unparseable, 19 duplicates merged) |
| Minority class, raw | `FDA_APPROVED = 0`: 94 · `CT_TOX = 1`: 112 (~7%) |
| Positives in a scaffold test split | **10–17** over seeds 0–4 (test part of 146) |
| Positives at H2's 10% training point | **5** |
| `CT_TOX` == `NOT FDA_APPROVED` | **98.8%** of molecules |
| Label density | 1.0 — nothing is missing, so nothing is masked |
| Duplicates with contradictory labels | 19 structures, 38 conflicting labels |

So: the seed-to-seed noise on ~13 positives exceeds the GNN-minus-baseline gap
the study is defined to measure; the two labels are effectively one, which
leaves the multitask head and masked loss with nothing to do; and 19 structures
appear both as approved and as failed-for-toxicity, consistent with the dataset
being assembled from two sources.

**Score `CT_TOX` only** — `configs/clintox_*.yaml` set `tasks: [CT_TOX]`.
`FDA_APPROVED` is positive for 91% of the test part, so its PR-AUC runs at
~0.98 where a constant predictor scores ~0.91; averaging it in lifted the
headline PR-AUC from 0.38 to 0.68 in a measured run and hid the real signal.
Read a ROC-AUC near 0.98 as suspected provenance leakage, not success.

```bash
admet run configs/clintox_rf_scaffold.yaml
admet run configs/clintox_gnn_scaffold.yaml
```

Treat the outcome as a comparison of dataset *sizes* against Tox21, not as a
second test of H1 and H2 — and expect it to be inconclusive.

## Install

Requires Python >= 3.12 (developed on 3.14; RDKit, PyTorch, scikit-learn and
matplotlib all publish wheels for 3.12–3.14).

```bash
python -m venv .venv
source .venv/bin/activate        # .venv\Scripts\activate on Windows
pip install -e ".[dev]"
```

A GPU is optional — see *Runtime* below. The first run downloads Tox21
(~3 MB) and caches it; set `ADMET_DATA_DIR` to move the cache.

## Reproducing each result

Every run is fully specified by **one config file plus one seed**, and writes
one JSON result file holding the config, the seed, the split sizes, every
tuning trial, the metrics, the library versions and the git commit.

```bash
make test        # 133 tests, ~3 s

# Main experiment: both models x both splits, 5 seeds each
make main
#   == admet run configs/tox21_rf_random.yaml
#      admet run configs/tox21_rf_scaffold.yaml
#      admet run configs/tox21_gnn_random.yaml
#      admet run configs/tox21_gnn_scaffold.yaml

# Additional experiment: learning curves, 10/25/50/100% of the TRAINING set
make sweep       # == admet sweep configs/sweep_learning_curve.yaml

make aggregate   # results/*.json -> runs.csv, summary.csv, gap.csv
make plots       # -> figures/*.png
```

Both commands skip cells that already have a result file, so an interrupted
run is resumed by reissuing the same command; pass `--overwrite` to redo them.
To reproduce a single cell:

```bash
admet run configs/tox21_gnn_scaffold.yaml --seeds 3 --train-fractions 0.25
```

### Runtime and hardware

Measured on a CPU-only Windows laptop (no GPU required; Tox21 is small enough
that the GNN trains on CPU in minutes).

| Step | Time |
|---|---|
| Test suite | ~3 s |
| Tox21 download + parse (first run only) | ~3 s |
| Baseline, one run (8 tuning trials, 1 seed) | ~2.5 min |
| GNN, one epoch (hidden 128, 3 layers, full training set) | ~1.9 s |
| GNN, one run (8 tuning trials x up to 50 epochs, 1 seed) | ~12 min |
| `make main` (4 configs x 5 seeds) | ~2.5 h |
| `make sweep` (4 configs x 4 fractions x 3 seeds = 48 runs) | ~3–4 h |

Early stopping usually ends a GNN trial well before 50 epochs, so these are
upper estimates. To shorten a pass, lower `tuning.budget` in
`configs/base.yaml` — it applies to **both** models at once, which is the
point (see below).

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
imbalance), macro-averaged over the 12 assays and reported as mean and standard
deviation over seeds — **target 5 seeds, minimum 3**.

### Safeguards for a fair comparison

These are the things that, if wrong, would invalidate the comparison rather
than merely weaken it:

- **Equal tuning budget.** Both models draw the same number of hyperparameter
  settings from `sample_configs()`, set once in `configs/base.yaml`. The
  baseline tunes its own representation (fingerprint radius, length, counts)
  alongside the forest, so it is not handicapped by a fixed featurization.
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
├── Makefile                # make test / main / sweep / aggregate / plots
├── README.md
├── AGENT_LOG.md            # record of the AI-assisted development
├── configs/
│   ├── base.yaml                   # shared defaults incl. the tuning budget
│   ├── tox21_{rf,gnn}_{random,scaffold}.yaml
│   └── sweep_learning_curve.yaml   # the learning-curve grid
├── src/admet/
│   ├── cli.py              # admet run / sweep / aggregate / plots
│   ├── training.py         # one run: config + seed -> result file
│   ├── evaluation.py       # ROC-AUC / PR-AUC with masked multitask labels
│   ├── data/
│   │   ├── loaders.py      # dataset registry, cleaning, download cache
│   │   └── splits.py       # random + randomized scaffold split, leakage check
│   ├── features/
│   │   ├── fingerprints.py # Morgan / ECFP bit vectors
│   │   └── graphs.py       # SMILES -> atom/bond graphs
│   ├── models/
│   │   ├── base.py         # shared interface + equal-budget sampler
│   │   ├── rf.py           # baseline: one forest per assay
│   │   └── gnn.py          # GINE message passing, masked multitask head
│   └── analysis/
│       ├── aggregate.py    # results/*.json -> summary + gap tables
│       └── plots.py        # the three figures
├── tests/                  # 133 tests
├── data/                   # cached raw downloads (gitignored)
├── results/                # one JSON per run, plus the CSV tables (gitignored)
└── figures/                # regenerated from results/ (gitignored)
```

## Figures

Regenerated from `results/` by `make plots`; none is drawn by hand.

Named `<figure>_<dataset>_<metric>.png`, one set per dataset:

- `random_vs_scaffold_*.png` — both models under both splits (H1).
- `learning_curves_*.png` — test score against training-set size (H2).
- `gnn_minus_rf_gap_*.png` — the gap itself, with a zero reference line.

Two datasets never share an axis. Their class balances and test-set sizes
differ, so a shared y-axis would invite a comparison the numbers do not
support; the cross-dataset comparison is made in the tables instead.

Model is encoded as colour and split as line style, so identity never depends
on colour alone.

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

Both are recorded in `AGENT_LOG.md` with the reasoning.

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
