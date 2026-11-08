# When does a GNN beat a fingerprint model for ADMET prediction?

Results report, Group 2 — Biomedical Artificial Intelligence / Drug discovery
and development.

Every number here is read from `results/summary.csv`, `results/gap.csv` and
the per-run JSON files in `results/`, all of which are committed. Nothing is
typed from memory, and the tables can be regenerated with
`make aggregate && make plots`.

---

## 1. Question and hypotheses

**When does a graph neural network outperform a fingerprint-based model for
ADMET property prediction, and does that benefit survive when the test
molecules are structurally different from the training molecules?**

**H1 (generalization).** The advantage of the GNN over the random forest is
smaller, or absent, under a scaffold split than under a random split.

**H2 (data size).** The advantage of the GNN depends on the amount of training
data: with little data the simpler baseline is competitive or better.

Both are stated in the **GNN-minus-baseline gap**, not in either model's
absolute score. Both are falsifiable as written, so either outcome informs the
lecture.

## 2. What was run

| | Tox21 (primary) | ClinTox (second) |
|---|---|---|
| Molecules after cleaning | 7823 | 1461 |
| Dropped as unparseable | 8 | 4 |
| Duplicate structures merged | 0 | 19 (all 19 with conflicting labels) |
| Assays scored | 12 | 1 (`CT_TOX`) |
| Label density | 0.829 | 1.000 |
| Positives per assay | 186–942 | 103 |
| Train / valid / test | 6259 / 782 / 782 | 1169 / 146 / 146 |

Both models, both splits, four training fractions. Tox21: 5 seeds at 100% and
3 along the learning curve (56 runs). ClinTox: 5 seeds throughout (80 runs).
**136 runs in total.**

`CT_TOX` is scored alone because ClinTox's second label is the same label
inverted — `CT_TOX == NOT FDA_APPROVED` for 98.8% of molecules — and
`FDA_APPROVED`'s positive class is the 91% *majority*, so its PR-AUC runs near
0.98 where a constant predictor scores 0.91. Macro-averaging the two lifted a
measured PR-AUC from 0.38 to 0.68, a near-doubling that is purely an averaging
artefact.

### Fairness safeguards

- **Equal tuning budget.** Both models draw 8 hyperparameter settings from
  their own search space. The baseline tunes its representation (fingerprint
  radius, length, counts) alongside the forest, so it is not handicapped by a
  fixed featurization.
- **Validation matches test.** Hyperparameters are selected on a validation set
  of the same split type as the test set.
- **No scaffold spans two parts.** Enforced by the splitter and re-checked at
  run time on the actual split of every scaffold run; `scaffold_disjointness_checked`
  is `true` in all 68 scaffold result files. A positive control asserts that a
  *random* split does share scaffolds, so the check cannot pass trivially.
- **Masking.** A missing label excludes a molecule from that assay only — never
  read as a negative, never dropped wholesale — in the metrics and in the GNN
  loss alike, both with positive-control tests.
- **Duplicates merged before splitting**, so one structure cannot sit in both
  train and test.

## 3. Results — Tox21

Test ROC-AUC, mean ± SD over 5 seeds, full training set:

| Split | Baseline (Morgan + RF) | GNN (GINE) | GNN − baseline |
|---|---|---|---|
| Random | 0.834 ± 0.009 | 0.836 ± 0.011 | **+0.003 ± 0.015** |
| Scaffold | 0.760 ± 0.013 | 0.756 ± 0.018 | **−0.004 ± 0.022** |

Test PR-AUC, same runs:

| Split | Baseline | GNN | GNN − baseline |
|---|---|---|---|
| Random | 0.473 ± 0.023 | 0.442 ± 0.029 | −0.031 ± 0.037 |
| Scaffold | 0.379 ± 0.019 | 0.353 ± 0.019 | −0.026 ± 0.027 |

Learning curves (3 seeds), GNN − baseline:

| Training set | ROC-AUC, random | ROC-AUC, scaffold | PR-AUC, random | PR-AUC, scaffold |
|---|---|---|---|---|
| 10% | −0.016 ± 0.025 | +0.001 ± 0.014 | **−0.086 ± 0.054** | **−0.058 ± 0.030** |
| 25% | −0.008 ± 0.019 | −0.013 ± 0.017 | −0.063 ± 0.018 | −0.041 ± 0.019 |
| 50% | −0.001 ± 0.011 | −0.010 ± 0.018 | −0.028 ± 0.020 | −0.032 ± 0.033 |
| 100% | +0.003 ± 0.015 | −0.004 ± 0.022 | −0.031 ± 0.037 | −0.026 ± 0.027 |

**The split type dominates.** Scaffold splitting costs the baseline 0.074
ROC-AUC and the GNN 0.080 — roughly ten times the difference between the two
models in any condition. This is the clearest result in the study.

**H2 is supported.** The baseline is better than the GNN with little data and
the advantage closes as data is added. The cleanest evidence is PR-AUC, where
the 10% gap exceeds its own SD in both splits and shrinks monotonically to
about −0.03 at 100%. The random-split ROC-AUC curve has the same shape and
crosses zero near 50%.

**H1 is not supported, because its premise fails.** At the full training set
there is no GNN advantage to shrink: +0.003 ± 0.015 and −0.004 ± 0.022, both
within one SD of zero and of each other. The accurate statement is *no GNN
advantage was detected under either split*. That is compatible with H1 and
equally compatible with a GNN undertrained at 8 tuning trials; this design
cannot separate the two.

## 4. Results — ClinTox, and the comparison

Test ROC-AUC, mean ± SD over 5 seeds, full training set:

| Split | Baseline | GNN | GNN − baseline |
|---|---|---|---|
| Random | 0.855 ± 0.077 | 0.901 ± 0.047 | +0.046 ± 0.090 |
| Scaffold | 0.833 ± 0.045 | 0.858 ± 0.061 | +0.025 ± 0.076 |

Taken at face value this reverses the Tox21 finding: the GNN leads on both
splits. It should not be taken at face value. **Every ClinTox gap is far
smaller than its own standard deviation** — ±0.09 on a +0.046 difference — and
the gap stays positive at every training fraction with SDs up to ±0.18. With
~146 test molecules carrying roughly 10–17 positives, a handful of molecules
moving across the decision boundary swings the score. This is the inconclusive
result the dataset decision predicted, and it is reported as inconclusive.

The informative comparison is the **drop from random to scaffold**:

| Dataset | Training molecules | Baseline drop | GNN drop |
|---|---|---|---|
| Tox21 | 6259 | **−0.074** | **−0.080** |
| ClinTox | 1169 | −0.022 | −0.043 |

ClinTox scores *higher* than Tox21 (0.90 vs 0.84) on a fifth of the data, and
barely loses anything when the test molecules are forced to be structurally
novel. Both facts point the same way: the ClinTox label is easier to predict
than clinical toxicity should be. ClinTox is assembled from two sources, and
the cleaning step found 19 duplicate structures carrying *contradictory*
labels — the same molecule appearing as approved and as failed-for-toxicity.
A scaffold shift that costs almost nothing suggests the model is partly
recovering which source a molecule came from rather than whether it is toxic.

**Read ClinTox as a cautionary comparison of dataset sizes, not as a second
test of H1 and H2.** A high score on a small, provenance-mixed dataset is the
failure mode the study was designed to expose, not a success.

## 5. Runtime and compute

Measured on a CPU-only Windows laptop with no GPU, from the `timing_seconds`
field of each result file. Total across all 136 runs: **3.83 h (230 min)** of
wall clock.

Per run, including the full 8-trial tuning search:

| Dataset | Model | Runs | Mean | Min | Max | Total |
|---|---|---|---|---|---|---|
| Tox21 | GNN | 28 | 302 s | 50 s | 666 s | 141 min |
| Tox21 | Baseline | 28 | 119 s | 60 s | 244 s | 56 min |
| ClinTox | GNN | 40 | 43 s | 11 s | 110 s | 29 min |
| ClinTox | Baseline | 40 | 6 s | 4 s | 9 s | 4 min |

At the full training set only:

| Dataset | Model | Random split | Scaffold split |
|---|---|---|---|
| Tox21 | GNN | 563 s | 547 s |
| Tox21 | Baseline | 190 s | 189 s |
| ClinTox | GNN | 91 s | 82 s |
| ClinTox | Baseline | 7 s | 8 s |

**The GNN costs 2.5× the baseline on Tox21 and 12× on ClinTox**, for no
measurable gain on Tox21 and an unresolvable one on ClinTox. That cost ratio
belongs in the discussion alongside the scores.

Early stopping on validation ROC-AUC ends a GNN trial at a mean best epoch of
34.9 on Tox21 and 22.8 on ClinTox, against a 50-epoch ceiling — so the ceiling
binds rarely and the budget is not obviously starving the model, though it
does not rule out that more tuning trials would help.

These timings are indicative of relative cost, not a benchmark: the machine
was in ordinary use, the baseline uses all cores via `n_jobs=-1`, and the GNN
runs single-threaded on CPU.

### Environment

| | |
|---|---|
| Python | 3.14.4 |
| Platform | Windows 11 (10.0.26300), CPU only |
| PyTorch | 2.15.0+cpu |
| RDKit | 2026.09.1 |
| scikit-learn | 1.9.1 |
| NumPy | 2.5.4 |

Each result file also records the git commit it was produced at.

## 6. Figures

In `figures/`, named `<figure>_<dataset>_<metric>.png`, regenerated by
`make plots`; none is drawn by hand.

- `random_vs_scaffold_*` — both models under both splits (H1).
- `learning_curves_*` — test score against training-set size (H2).
- `gnn_minus_rf_gap_*` — the gap itself, with a zero reference line.

Model is encoded as colour and split as line style, so identity never depends
on colour alone. The two datasets never share an axis: their class balances and
test-set sizes differ enough that a shared y-axis would invite a comparison the
numbers do not support.

## 7. Threats to validity

- **Three seeds is not enough, and we have the evidence.** An intermediate
  3-seed pass put the Tox21 scaffold gap at −0.016 ± 0.020, which reads as a
  real effect. At 5 seeds it is −0.004 ± 0.022. The apparent effect was seed
  noise. The learning-curve rows in §3 are 3-seed and should be read as
  directional only.
- **The GNN may simply be undertrained.** 8 tuning trials is equal across
  models, which makes the comparison fair, but not necessarily sufficient. A
  null result is evidence about *this* budget.
- **ClinTox is underpowered and probably leaky**, as set out in §4.
- **Tox21 is in vitro.** A model trained on it inherits the assay's imperfect
  in vitro → human concordance. If the model's job is to catch what in vitro
  screening misses, the residual signal is the least structure-predictable
  part. This belongs in the limitations part of the lecture.
- **One architecture, one baseline.** "GNNs" are not tested; one GINE
  implementation is.

## 8. Deviations from the proposal

- **No pandas** (§5 lists it). Its compiled extensions are blocked by the
  Windows Application Control policy on the development machine. The loader
  and the aggregation use the standard library's `csv` plus NumPy.
- **No PyTorch Geometric** (§3.2 allows it). The same GINE model is written
  directly in PyTorch, using `index_add_` for the neighbour sum, to avoid a
  dependency that does not install cleanly here. Covered by tests.
- The optional cluster-based splitter and the pretrained-model stretch goal
  are not implemented; they are the first two items on the proposal's
  scope-reduction list (§8).

## 9. Conclusion

On Tox21, at an equal tuning budget, **the GNN never outperformed the
fingerprint baseline** — and with little training data it was clearly worse.
What dominates both models is the structural shift: forcing the test molecules
to be scaffold-novel costs about 0.08 ROC-AUC, an order of magnitude more than
anything the choice of architecture bought.

For the lecture's framing, that is the point. Molecules worth finding are, by
definition, unlike the training data, and that is exactly the regime where both
methods degrade most and where their difference matters least. The GNN also
costs 2.5–12× more compute to reach the same place.

ClinTox neither confirms nor contradicts this; it is too small to resolve a
gap of this size, and its suspiciously small scaffold penalty is better
explained by dataset provenance than by model quality.
