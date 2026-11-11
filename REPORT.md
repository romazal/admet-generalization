# When does a GNN beat a fingerprint model for ADMET prediction?

Results report, Group 2 — Biomedical Artificial Intelligence / Drug discovery
and development.

Every number here is read from `results/summary.csv`, `results/gap.csv` and
the per-run JSON files in `results/tox21/` and `results/clintox/`, all of
which are committed. Nothing is
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
  fixed featurization. A second fingerprint (MACCS) is implemented and can be
  selected per-run, but is kept out of the tuned space on purpose: letting the
  baseline spend part of its 8 on a second representation while the GNN spends
  all 8 on one would thin the baseline's search and widen the gap below for a
  reason unrelated to the models.
- **Validation matches test.** Hyperparameters are selected on a validation set
  of the same split type as the test set.
- **No scaffold spans two parts.** Enforced by the splitter and re-checked at
  run time on the actual split of every scaffold run; `scaffold_disjointness_checked`
  is `true` in all 68 scaffold result files. A positive control asserts that a
  *random* split does share scaffolds, so the check cannot pass trivially.
- **Masking.** A missing label excludes a molecule from that assay only — never
  read as a negative, never dropped wholesale — from the moment the CSV cell is
  parsed, through the metrics and the GNN loss alike, each with positive-control
  tests.
- **Duplicates merged before splitting**, so one structure cannot sit in both
  train and test; covered by a loader test.

## 3. Results — Tox21

Test ROC-AUC, mean ± SD over 5 seeds, full training set:

| Split | Baseline (Morgan + RF) | GNN (GINE) | GNN − baseline |
|---|---|---|---|
| Random | 0.834 ± 0.009 | 0.834 ± 0.015 | **+0.000 ± 0.018** |
| Scaffold | 0.760 ± 0.013 | 0.755 ± 0.011 | **−0.005 ± 0.017** |

Test PR-AUC, same runs:

| Split | Baseline | GNN | GNN − baseline |
|---|---|---|---|
| Random | 0.473 ± 0.023 | 0.449 ± 0.029 | −0.024 ± 0.037 |
| Scaffold | 0.379 ± 0.019 | 0.360 ± 0.020 | −0.019 ± 0.028 |

Learning curves (3 seeds), GNN − baseline:

| Training set | ROC-AUC, random | ROC-AUC, scaffold | PR-AUC, random | PR-AUC, scaffold |
|---|---|---|---|---|
| 10% | −0.016 ± 0.020 | −0.018 ± 0.035 | **−0.094 ± 0.033** | **−0.047 ± 0.019** |
| 25% | +0.005 ± 0.015 | −0.003 ± 0.007 | −0.033 ± 0.018 | −0.039 ± 0.010 |
| 50% | −0.003 ± 0.013 | −0.001 ± 0.014 | −0.041 ± 0.020 | −0.034 ± 0.020 |
| 100% | +0.000 ± 0.018 | −0.005 ± 0.017 | −0.024 ± 0.037 | −0.019 ± 0.028 |

**The split type dominates.** Scaffold splitting costs the baseline 0.073
ROC-AUC and the GNN 0.078 — roughly ten times the difference between the two
models in any condition. This is the clearest result in the study, and the
only one that came back unchanged to three decimal places on a rerun (§5b).

**H2 is supported, with a caveat on how cleanly.** The baseline is better than
the GNN with little data and the advantage closes as data is added. The
cleanest evidence is PR-AUC, where the 10% gap exceeds its own SD in both
splits and falls to about −0.02 at 100%. The decline is monotone under the
scaffold split but not under the random one, and which split was the monotone
one differed between the CPU and GPU passes (§5b), so the endpoints carry the
claim and the shape in between should not be read as a trend.

**H1 is not supported, because its premise fails.** At the full training set
there is no GNN advantage to shrink: +0.000 ± 0.018 and −0.005 ± 0.017, both
within one SD of zero and of each other. The accurate statement is *no GNN
advantage was detected under either split*. That is compatible with H1 and
equally compatible with a GNN undertrained at 8 tuning trials; this design
cannot separate the two.

## 4. Results — ClinTox, and the comparison

Test ROC-AUC, mean ± SD over 5 seeds, full training set:

| Split | Baseline | GNN | GNN − baseline |
|---|---|---|---|
| Random | 0.855 ± 0.077 | 0.938 ± 0.037 | +0.083 ± 0.085 |
| Scaffold | 0.833 ± 0.045 | 0.855 ± 0.026 | +0.022 ± 0.052 |

Taken at face value this reverses the Tox21 finding: the GNN leads on both
splits. It should not be taken at face value. **Every ClinTox gap is far
smaller than its own standard deviation** — ±0.085 on a +0.083 difference — and
the gap stays positive at every training fraction with SDs up to ±0.18. With
~146 test molecules carrying roughly 10–17 positives, a handful of molecules
moving across the decision boundary swings the score. This is the inconclusive
result the dataset decision predicted, and it is reported as inconclusive.

The informative comparison is the **drop from random to scaffold**:

| Dataset | Training molecules | Baseline drop | GNN drop |
|---|---|---|---|
| Tox21 | 6259 | **−0.073** | **−0.078** |
| ClinTox | 1169 | −0.021 | −0.082 |

ClinTox scores *higher* than Tox21 (0.94 vs 0.83) on a fifth of the data: the
ClinTox label is easier to predict than clinical toxicity should be. ClinTox is
assembled from two sources, and the cleaning step found 19 duplicate structures
carrying *contradictory* labels — the same molecule appearing as approved and
as failed-for-toxicity. A label that is partly recoverable from which source a
molecule came from, rather than from whether it is toxic, would look exactly
like this.

The scaffold drop is the weaker half of that argument, and the rerun is why.
The baseline loses only 0.021 — a third of what it loses on Tox21, which is
the suspicious part. The GNN's drop, however, read 0.043 on the CPU pass and
0.082 on the GPU pass; at ±0.05–±0.06 per arm, both are consistent with the
same underlying value and neither is measurable. Only the baseline's small
drop should be cited.

**Read ClinTox as a cautionary comparison of dataset sizes, not as a second
test of H1 and H2.** A high score on a small, provenance-mixed dataset is the
failure mode the study was designed to expose, not a success.

## 5. Runtime and compute

Measured on a Windows desktop (16 threads, 31 GB RAM, RTX 5070 Ti) from the
`timing_seconds` field of each result file. Total across all 136 runs:
**2.03 h (122 min)** of wall clock with the GNN on the GPU, against 3.83 h
(230 min) for the same grid on the CPU (§5b).

Per run, including the full 8-trial tuning search:

| Dataset | Model | Runs | Mean | Min | Max | Total |
|---|---|---|---|---|---|---|
| Tox21 | GNN | 28 | 118 s | 24 s | 327 s | 55 min |
| Tox21 | Baseline | 28 | 106 s | 54 s | 217 s | 50 min |
| ClinTox | GNN | 40 | 19 s | 5 s | 60 s | 13 min |
| ClinTox | Baseline | 40 | 6 s | 4 s | 8 s | 4 min |

At the full training set only:

| Dataset | Model | Random split | Scaffold split |
|---|---|---|---|
| Tox21 | GNN | 228 s | 213 s |
| Tox21 | Baseline | 159 s | 170 s |
| ClinTox | GNN | 40 s | 35 s |
| ClinTox | Baseline | 7 s | 8 s |

**The GNN costs 1.3× the baseline on Tox21 and 5× on ClinTox**, for no
measurable gain on Tox21 and an unresolvable one on ClinTox. Without a GPU
those ratios are 2.5× and 12×. Either way the cost ratio belongs in the
discussion alongside the scores — and the direction of the argument does not
depend on the hardware, because the cheaper model is also the one that never
lost.

Early stopping on validation ROC-AUC ends a GNN trial at a mean best epoch of
35.4 on Tox21 and 26.2 on ClinTox, against a 50-epoch ceiling — so the ceiling
binds rarely and the budget is not obviously starving the model, though it
does not rule out that more tuning trials would help.

These timings are indicative of relative cost, not a benchmark: the machine
was in ordinary use and the baseline uses all cores via `n_jobs=-1`. The
baseline's own mean moved 119 → 106 s between the two passes without any
change in its arithmetic, which puts the noise floor on these figures at
roughly ±10%.

### Environment

| | |
|---|---|
| Python | 3.14.4 |
| Platform | Windows 11 (10.0.26300) |
| GPU | NVIDIA RTX 5070 Ti (sm_120), CUDA 12.8 |
| PyTorch | 2.11.0+cu128 (CPU pass: 2.15.0+cpu) |
| RDKit | 2026.09.1 |
| scikit-learn | 1.9.1 |
| NumPy | 2.5.4 |

Each result file also records the git commit it was produced at and, in
`versions.torch_device`, the device torch had available —
`cuda:NVIDIA GeForce RTX 5070 Ti` in all 136 committed runs. The field reports
what torch would use, so it is only meaningful for the GNN runs; the baseline
never calls torch.

## 5b. CPU vs GPU: speed, and a reproducibility check

The whole grid was run twice — once with the GNN on the CPU, once on the GPU.
The second pass had two purposes: measure the speedup, and use the repetition
as an independent check on the results.

### Speed

| | CPU | GPU | Speedup |
|---|---|---|---|
| Tox21 GNN, mean per run | 302 s | 118 s | **2.6×** |
| Tox21 GNN at 100% | 555 s | 221 s | **2.5×** |
| ClinTox GNN, mean per run | 43 s | 19 s | **2.2×** |
| Baseline (scikit-learn) | 119 s | 106 s | — (noise) |
| **All 136 runs** | **3.83 h** | **2.03 h** | **1.9×** |

Only the GNN is accelerated. The overall 1.9× is diluted by the baseline,
which is 45% of the grid's wall clock and never touches the card; 2.5× on the
heaviest cell is the figure to use when estimating a larger grid. The speedup
is modest for a GPU because these are small graphs — Tox21 molecules average
under 20 heavy atoms, so per-batch kernel launches and host-side Python
dominate and the card is never saturated.

### What the repetition shows

The two passes differ in device *and* in torch version (2.15.0+cpu →
2.11.0+cu128 — no cu128 wheel exists for 2.15 yet), so a difference cannot be
attributed to the GPU alone.

| | Reproduced |
|---|---|
| Baseline test metrics, all 68 runs | **bit-identical** |
| Baseline validation metrics | 67/68 bit-identical; one drifted 8e-7 |
| GNN test metrics | 67/68 changed |

**The baseline acts as the control.** It is scikit-learn with fixed seeds and
is unaffected by the torch change, so its test metrics coming back identical
to the last decimal place establishes that the data, the splits, the tuning
search and the metric code are deterministic — which is what licenses
attributing the GNN's movement to the GNN. The single baseline value that
moved is a validation ROC-AUC (`tox21_rf_random_frac100_seed2`, +8.3e-7),
from non-deterministic float summation across threads. It did not change
which tuning trial won, and no reported number depends on it.

The GNN's Tox21 scores moved well inside seed noise:

| Tox21 GNN ROC-AUC, 100% | CPU | GPU | Δ | Seed SD |
|---|---|---|---|---|
| Random | 0.836 | 0.834 | −0.002 | ± 0.015 |
| Scaffold | 0.756 | 0.755 | −0.001 | ± 0.011 |

**Every conclusion in §3 survives**, which is the main result of this check:
the split type still dominates (0.073 / 0.078, unchanged), both headline gaps
still sit inside one SD of zero, and the baseline still leads at 10% and not
at 100%.

**What did not survive is the finer structure**, and this is worth stating
plainly because it is a limit on the study rather than on the hardware:

- The PR-AUC learning curve's *monotone* decline held under the random split
  on CPU and under the scaffold split on GPU. The 10%-to-100% narrowing is
  robust; the ordering of the two middle points is not.
- Two 3-seed learning-curve gaps changed sign (Tox21 random at 25%,
  −0.008 → +0.005; scaffold at 10%, +0.001 → −0.018). Both are far inside
  their SDs. This is §3's three-seed finding arrived at independently.
- On ClinTox the GNN's random-split score moved +0.037 and its scaffold drop
  went 0.043 → 0.082. ClinTox was already reported as inconclusive; the rerun
  shows how little its numbers constrain anything.

Nothing above fell outside its stated uncertainty. The rerun did not overturn
a finding — it confirmed that the quantities reported with SDs larger than
their means deserved exactly that treatment.

## 6. Figures

In `figures/tox21/` and `figures/clintox/`, named
`<figure>_<dataset>_<metric>.png`, regenerated by `make plots`; none is
drawn by hand.

- `random_vs_scaffold_*` — both models under both splits (H1).
- `learning_curves_*` — test score against training-set size (H2).
- `gnn_minus_rf_gap_*` — the gap itself, with a zero reference line.

Model is encoded as colour, always the same colour. Split is encoded as
position: the learning-curve and gap figures are faceted into a random-split
and a scaffold-split panel on a shared y-axis, so no panel carries more than
two lines and the step between panels is the split's effect. Error bars are
±1 SD over seeds and every figure says so in a footnote. The two datasets
never share an axis: their class balances and test-set sizes differ enough
that a shared y-axis would invite a comparison the numbers do not support.

## 7. Threats to validity

- **Three seeds is not enough, and we have the evidence twice.** An
  intermediate 3-seed pass put the Tox21 scaffold gap at −0.016 ± 0.020,
  which reads as a real effect. At 5 seeds it is −0.005 ± 0.017. The apparent
  effect was seed noise. Independently, two 3-seed learning-curve gaps changed
  sign between the CPU and GPU passes (§5b). The learning-curve rows in §3 are
  3-seed and should be read as directional only.
- **The GNN's numbers are not bit-reproducible across environments.** Changing
  device and torch version moved the Tox21 GNN's ROC-AUC by up to 0.002 and
  ClinTox's by 0.037 (§5b). The baseline, as the deterministic control,
  reproduced exactly. No §3 conclusion depends on the difference, but a
  single GNN run quoted to three decimals is over-precise.
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

- **ClinTox as the second dataset, not BBBP.** §3.1 names BBBP as an example
  only, so the slot was open; the team chose ClinTox.
- **`data/`, `results/` and `figures/` are committed**, where the §5 file
  table gives `.gitignore` the job of excluding them. Tracking them lets a
  clone reproduce this report offline and lets any number be traced to its
  run file; the whole lot is 2.8 MB.
- **Compute was never the constraint.** §8 budgets a 16 GB GPU and "a few GPU
  hours". The committed results use an RTX 5070 Ti and take 2.03 h, inside
  that budget; the same grid also completed on the CPU alone in 3.83 h, also
  inside it, so the study does not require a GPU to reproduce. `models/gnn.py`
  selects CUDA automatically when it is available, so neither pass needed a
  code change — only a different torch wheel. See §5b.
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
costs more compute to reach the same place — 1.3–5× the baseline with a GPU,
2.5–12× without one.

ClinTox neither confirms nor contradicts this; it is too small to resolve a
gap of this size, and the baseline's suspiciously small scaffold penalty on it
is better explained by dataset provenance than by model quality.

Running the whole grid a second time on different hardware left every one of
these conclusions standing while moving several of the smaller numbers around
inside their error bars (§5b). That is the result we would want from such a
check, and it is also the clearest argument for having reported SDs
throughout rather than point estimates.
