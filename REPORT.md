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
**136 runs in total**, and with the dummy floor below, **156 result files**.

A dummy class-prior model was run separately on each dataset, both splits, 5
seeds at 100% (20 runs), to establish the floor in §3 and §4. It is not part
of the 136; it contributes 35 s to the §5 timings and nothing to the §5b
comparison, both of which describe the main grid.

Every one of the 156 files was produced with deterministic kernels pinned and
records `"deterministic": true`; the 146 main-grid and Tox21-dummy files come
from a single commit, and the ten ClinTox dummy runs were added afterwards.
§5b says why that mattered and what it cost.

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
  is `true` in all 73 scaffold result files. A positive control asserts that a
  *random* split does share scaffolds, so the check cannot pass trivially.
- **Masking.** A missing label excludes a molecule from that assay only — never
  read as a negative, never dropped wholesale — from the moment the CSV cell is
  parsed, through the metrics and the GNN loss alike, each with positive-control
  tests.
- **Duplicates merged before splitting**, so one structure cannot sit in both
  train and test; covered by a loader test.

## 3. Results — Tox21

Test ROC-AUC, mean ± SD over 5 seeds, full training set. The dummy is a
class-prior model that ignores the molecule entirely — the floor both real
models have to clear:

| Split | Dummy (floor) | Baseline (Morgan + RF) | GNN (GINE) | GNN − baseline |
|---|---|---|---|---|
| Random | 0.500 ± 0.000 | 0.834 ± 0.009 | 0.838 ± 0.016 | **+0.005 ± 0.019** |
| Scaffold | 0.500 ± 0.000 | 0.760 ± 0.013 | 0.763 ± 0.010 | **+0.003 ± 0.016** |

Test PR-AUC, same runs:

| Split | Dummy (floor) | Baseline | GNN | GNN − baseline |
|---|---|---|---|---|
| Random | 0.077 ± 0.002 | 0.473 ± 0.023 | 0.462 ± 0.018 | −0.011 ± 0.029 |
| Scaffold | 0.100 ± 0.007 | 0.379 ± 0.019 | 0.354 ± 0.017 | −0.025 ± 0.025 |

**Both models clear the floor by a wide margin.** The dummy scores exactly
0.500 ROC-AUC, as it must — it emits one constant per assay, so every
threshold ranks the molecules identically. The informative column is PR-AUC,
where the floor is the positive rate rather than a fixed 0.5: at 0.077 under
the random split, the baseline's 0.473 is about six times chance. Tox21 is
heavily imbalanced, and a PR-AUC of 0.47 is easy to read as weak until the
0.077 next to it shows what "knowing nothing" scores. The floor is higher
under the scaffold split (0.100 vs 0.077) because scaffold-disjoint test sets
happen to be slightly more positive, which is worth keeping in mind when
reading the scaffold PR-AUC drop: a little of it is absorbed by a higher
baseline rate.

Learning curves (3 seeds), GNN − baseline:

| Training set | ROC-AUC, random | ROC-AUC, scaffold | PR-AUC, random | PR-AUC, scaffold |
|---|---|---|---|---|
| 10% | −0.014 ± 0.023 | +0.001 ± 0.014 | **−0.059 ± 0.024** | **−0.057 ± 0.024** |
| 25% | +0.006 ± 0.014 | −0.006 ± 0.012 | −0.052 ± 0.021 | −0.029 ± 0.025 |
| 50% | −0.007 ± 0.011 | −0.002 ± 0.024 | −0.042 ± 0.017 | −0.027 ± 0.028 |
| 100% | +0.005 ± 0.019 | +0.003 ± 0.016 | −0.011 ± 0.029 | −0.025 ± 0.025 |

**The split type dominates.** Scaffold splitting costs the baseline 0.073
ROC-AUC and the GNN 0.075 — roughly ten times the difference between the two
models in any condition. This is the clearest result in the study, and the one
that has survived every rerun unchanged to three decimal places (§5b).

**H2 is supported, with a caveat on how cleanly.** The baseline is better than
the GNN with little data and the advantage closes as data is added. The
cleanest evidence is PR-AUC, where the 10% gap is more than twice its own SD
in both splits and shrinks monotonically to −0.011 (random) and −0.025
(scaffold) at 100%. Before the GNN was made deterministic the decline was
monotone under one split and not the other (§5b) — that was its own
run-to-run spread. The endpoints carry the claim either way, and the shape in
between is not worth reading as a trend at 3 seeds.

**H1 is not supported, because its premise fails.** At the full training set
there is no GNN advantage to shrink: +0.005 ± 0.019 and +0.003 ± 0.016, both
within one SD of zero and of each other. The accurate statement is *no GNN
advantage was detected under either split*. That is compatible with H1 and
equally compatible with a GNN undertrained at 8 tuning trials; this design
cannot separate the two.

## 4. Results — ClinTox, and the comparison

Test ROC-AUC, mean ± SD over 5 seeds, full training set:

| Split | Baseline | GNN | GNN − baseline |
|---|---|---|---|
| Random | 0.855 ± 0.077 | 0.921 ± 0.058 | +0.067 ± 0.096 |
| Scaffold | 0.833 ± 0.045 | 0.847 ± 0.029 | +0.014 ± 0.054 |

The dummy floor is 0.500 ROC-AUC on both splits, as it must be. In PR-AUC it
is 0.070 ± 0.013 random and 0.096 ± 0.023 scaffold, so the baseline's 0.494
and 0.417 are about seven and four times chance — ClinTox is imbalanced in the
same direction as Tox21 and its scores have to be read against that.

Taken at face value the table above reverses the Tox21 finding: the GNN leads
on both splits. It should not be taken at face value. **Every ClinTox gap is far
smaller than its own standard deviation** — ±0.096 on a +0.067 difference — and
while the ROC-AUC gap stays positive at every training fraction, its SDs run up
to ±0.16, and the PR-AUC gap changes sign twice. With
~146 test molecules carrying roughly 10–17 positives, a handful of molecules
moving across the decision boundary swings the score. This is the inconclusive
result the dataset decision predicted, and it is reported as inconclusive.

The informative comparison is the **drop from random to scaffold**:

| Dataset | Training molecules | Baseline drop | GNN drop |
|---|---|---|---|
| Tox21 | 6259 | **−0.073** | **−0.075** |
| ClinTox | 1169 | −0.021 | −0.074 |

ClinTox scores *higher* than Tox21 (0.92 vs 0.84) on a fifth of the data: the
ClinTox label is easier to predict than clinical toxicity should be. ClinTox is
assembled from two sources, and the cleaning step found 19 duplicate structures
carrying *contradictory* labels — the same molecule appearing as approved and
as failed-for-toxicity. A label that is partly recoverable from which source a
molecule came from, rather than from whether it is toxic, would look exactly
like this.

The scaffold drop is the weaker half of that argument, and the reruns are why.
The baseline loses only 0.021 — a third of what it loses on Tox21, which is
the suspicious part. The GNN's drop read 0.082 before the determinism fix and
0.074 after it; at ±0.05–±0.06 per arm both are consistent with one
underlying value and neither is measurable. Only the baseline's small drop
should be cited.

That small drop is more damning than it first looks, because the scaffold
split works *harder* on ClinTox than on Tox21. Measuring each test molecule's
Tanimoto similarity to its nearest training molecule, under both splits:

| Dataset | Median, random | Median, scaffold | Drop | Baseline ROC-AUC lost |
|---|---|---|---|---|
| Tox21 | 0.583 | 0.419 | −0.164 (28%) | **0.073** |
| ClinTox | 0.523 | 0.333 | −0.190 (36%) | **0.021** |

ClinTox's test molecules end up *less* like its training molecules than
Tox21's do, and its baseline still loses a third as much accuracy. If the
score were carried by structure, a larger structural shift could not cost
less. Something in the ClinTox label survives having the scaffolds pulled
apart — which is what a label partly recoverable from provenance would do, and
is the strongest version of the argument above. It is drawn from one seed of
each split (`figures/clintox/train_test_similarity_clintox.png`), so read the
medians rather than the shape of that histogram: with 146 test molecules its
bars stand on three or four molecules each.

**Read ClinTox as a cautionary comparison of dataset sizes, not as a second
test of H1 and H2.** A high score on a small, provenance-mixed dataset is the
failure mode the study was designed to expose, not a success.

## 5. Runtime and compute

Measured on a Windows desktop (16 threads, 31 GB RAM, RTX 5070 Ti) from the
`timing_seconds` field of each result file. Total across all 156 result files:
**4.19 h (251 min)** of wall clock, GNN on the GPU with deterministic kernels
pinned. The same grid took 2.03 h before they were pinned (§5b).

Per run, including the full 8-trial tuning search:

| Dataset | Model | Runs | Mean | Min | Max | Total |
|---|---|---|---|---|---|---|
| Tox21 | GNN | 28 | 384 s | 82 s | 1396 s | 179 min |
| Tox21 | Baseline | 28 | 115 s | 58 s | 231 s | 54 min |
| ClinTox | GNN | 40 | 21 s | 5 s | 48 s | 14 min |
| ClinTox | Baseline | 40 | 6 s | 4 s | 7 s | 4 min |

At the full training set only:

| Dataset | Model | Random split | Scaffold split |
|---|---|---|---|
| Tox21 | GNN | 763 s | 582 s |
| Tox21 | Baseline | 178 s | 179 s |
| ClinTox | GNN | 39 s | 34 s |
| ClinTox | Baseline | 7 s | 7 s |

**The GNN costs 3.8× the baseline on Tox21 and 5.5× on ClinTox** at the full
training set, for no measurable gain on Tox21 and an unresolvable one on
ClinTox. Deterministic kernels are most of that: the Tox21 ratio was 1.3×
before they were pinned (§5b). The cost ratio belongs in the discussion
alongside the scores — the cheaper model is also the one that never lost.

Early stopping on validation ROC-AUC ends a GNN trial at a mean best epoch of
34.4 on Tox21 and 25.8 on ClinTox, against a 50-epoch ceiling — so the ceiling
binds rarely and the budget is not obviously starving the model, though it
does not rule out that more tuning trials would help.

These timings are indicative of relative cost, not a benchmark: the machine
was in ordinary use and the baseline uses all cores via `n_jobs=-1`. The
baseline's own mean has read 106 and now 115 s across two runs of the grid
with bit-identical arithmetic both times, which puts the noise floor on these
figures at roughly ±10%.

### Environment

| | |
|---|---|
| Python | 3.14.4 |
| Platform | Windows 11 (10.0.26300) |
| GPU | NVIDIA RTX 5070 Ti (sm_120), CUDA 12.8 |
| PyTorch | 2.11.0+cu128 |
| RDKit | 2026.09.1 |
| scikit-learn | 1.9.1 |
| NumPy | 2.5.4 |

Each result file also records the git commit it was produced at and, in
`versions.torch_device`, the device torch had available —
`cuda:NVIDIA GeForce RTX 5070 Ti` in all 156 committed runs. The field reports
what torch could *see*, not what the model used, so it is only meaningful for
the GNN runs; the baseline never calls torch. Read the `model` column to know
what ran where.

## 5b. Determinism: what repeats, and what it costs

Everything in this report was produced on one GPU. The grid was run twice on
it: once before deterministic kernels were pinned, and once after. The second
pass exists because the first one could not be repeated.

### The problem

Repeating a single Tox21 scaffold run three times — same card, same wheel,
same seed, same data, nothing changed between them — gives:

| Repeat of one identical run | Test ROC-AUC |
|---|---|
| 1 | 0.7369 |
| 2 | 0.7413 |
| 3 | 0.7690 |

A spread of 0.032, three times the seed-to-seed SD of 0.011. It also changed
which of the 8 hyperparameter configurations the tuning selected, so the
variation is not confined to the final score: two runs of the same experiment
could choose different models. Against that, the 0.005 differences this study
reports between the two architectures are not measurable at all.

### Cause

Message passing sums each atom's incoming messages with a scatter-add, and on
a GPU the order in which those partial sums are applied is not fixed from run
to run; cuBLAS picks kernels opportunistically for the same reason. Float
addition is not associative, so neither is repeatable. Both are deliberate
speed-for-ordering trades in the libraries rather than bugs — but a study
whose unit of evidence is a 0.005 difference cannot afford 0.032 of
unexplained movement.

### The fix, and what it costs

Three lines in [`src/admet/training.py`](src/admet/training.py), applied
before any run: `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `cudnn.benchmark = False`,
and `torch.use_deterministic_algorithms(True)`. `ADMET_NONDETERMINISTIC=1`
opts out, and every result file records the outcome in a `deterministic`
field.

Verified by re-running `tox21_gnn_scaffold` seed 0 at 100% into a separate
results directory and comparing field by field: test ROC-AUC matches to all 17
significant digits (0.75171164118465583), as do test PR-AUC, the selected
hyperparameters, the best epoch, and all 8 tuning-trial scores. Only the
wall-clock fields differ.

It is not free. The same Tox21 GNN run at the full training set goes from
**221 s to 673 s — 3.0×** — on the same card and the same wheel, with only
the flag differing. That is most of why the grid takes 4.19 h, and it is the
trade this project made deliberately: a number that repeats is worth more
than a number that arrives sooner. Only the GNN is affected; the baseline is
scikit-learn and took 178 s per Tox21 run at 100%.

### What the baseline shows

The baseline is the control throughout, and it has never moved. Across all 68
random forest runs, the deterministic pass reproduced every **test** metric
bit-identically against the one before it. One validation ROC-AUC drifted by
8.3e-7 (`tox21_rf_random_frac100_seed2`), from non-deterministic float
summation across threads; it did not change which tuning trial won, and no
reported number depends on it. That the data, the splits, the tuning search
and the metric code all come back identical is what licenses attributing the
GNN's movement to the GNN.

### What changed in the findings

| | Before → after |
|---|---|
| Tox21 GNN means, 100% | moved ≤ 0.008 ROC-AUC |
| Tox21 GNN means, learning curve | moved ≤ 0.019 |
| ClinTox GNN means | moved ≤ 0.025 |
| Baseline, all 68 runs | unchanged |

**Every conclusion in §3 survives**: the split type still dominates
(0.073 / 0.075), both headline gaps still sit inside one SD of zero, and the
baseline still leads at 10% in PR-AUC and not at 100%.

What moved is the finer structure that was already flagged as fragile:

- The PR-AUC learning curve is now monotone under *both* splits. Before, it
  was monotone under one and not the other. The 10%-to-100% narrowing held
  either way.
- The 100% scaffold ROC-AUC gap changed sign, −0.005 → +0.003. It has sat
  inside one SD of zero throughout.
- On ClinTox the GNN's scaffold drop read 0.082 and now reads 0.074.

Nothing in that list ever fell outside its stated uncertainty, and that is the
point. The rerun did not overturn a single quantity reported with an SD
smaller than its mean, and it reshuffled almost every quantity that was not.
The fix does not make the conclusions truer; it makes them checkable, which is
the weaker claim this study is actually entitled to make.

## 6. Figures

In `figures/tox21/` and `figures/clintox/`, named
`<figure>_<dataset>_<metric>.png`, regenerated by `make plots`; none is
drawn by hand.

- `random_vs_scaffold_*` — both models under both splits (H1).
- `learning_curves_*` — test score against training-set size (H2).
- `gnn_minus_rf_gap_*` — the gap itself, with a zero reference line.
- `per_assay_*` — Tox21's twelve assays separately, as a heatmap. §3 reports
  a macro-average, and this is what it averages over: SR-MMP reaches 0.90
  under a random split where NR-ER manages 0.71, and the scaffold penalty is
  not uniform across endpoints either: SR-ATAD5 loses 0.16 while NR-ER loses
  nothing at all (−0.01 — marginally *better* scaffold-split than random).
- `paired_seeds_*` — baseline against GNN, one point per seed, with the line
  of equality. §3's headline gaps are differences of means with an SD beside
  them; this is the same fact as a picture, and points falling on both sides
  of the diagonal are what "no detectable difference" looks like.
- `train_test_similarity_*` — the mechanism, measured: nearest-training-
  neighbour similarity under each split. §4 now leans on it.
- `dataset_overview_*` — the data before any model.

`figures/all_gaps_<metric>.png` is the exception to the rule below: it puts
both datasets on one axis, which it may because it plots differences rather
than scores. Every gap in the study with ±1 SD against a line at zero; 14 of
16 intervals include zero.

Model is encoded as colour, always the same colour. Split is encoded as
position: the learning-curve and gap figures are faceted into a random-split
and a scaffold-split panel on a shared y-axis, so no panel carries more than
two lines and the step between panels is the split's effect. The per-assay
heatmap is the only figure using a colour *ramp*; it is a single hue, and a
hue used nowhere else, because every other hue in the set already means a
model or a split. Error bars are ±1 SD over seeds and every figure says so in
a footnote. Two datasets never share an axis for raw scores: their class
balances and test-set sizes differ enough that it would invite a comparison
the numbers do not support.

## 7. Threats to validity

- **Three seeds is not enough, and we have the evidence twice.** The same
  Tox21 scaffold gap reads −0.008 ± 0.011 at 3 seeds — four times its own SD,
  which looks like a real baseline win — and +0.003 ± 0.016 at 5 seeds, where
  it flips sign. The apparent effect was seed noise. Independently, several
  3-seed learning-curve gaps changed sign between passes (§5b). The
  learning-curve rows in §3 are 3-seed and are directional only.
- **The GNN's numbers were not reproducible at all until they were made so.**
  Repeating one identical run moved its ROC-AUC by 0.032, three times the
  seed SD, and changed the selected hyperparameters (§5b). Deterministic
  kernels are now pinned and the committed grid re-runs bit for bit, at 3×
  the GNN's runtime. The risk this leaves is a reading one: every GNN number
  here comes from one fixed kernel ordering on one card, so a single run
  quoted to three decimals is still over-precise on any other machine.
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
  hours". The committed results use an RTX 5070 Ti and take 4.19 h, inside
  that budget, with deterministic kernels accounting for most of the increase
  over the 2.03 h the same grid took without them. `models/gnn.py` selects
  CUDA automatically when it is available and falls back to CPU otherwise, so
  the study does not require a card to re-run, only patience. See §5b.
- **No PyTorch Geometric** (§3.2 allows it). The same GINE model is written
  directly in PyTorch, using `index_add_` for the neighbour sum, to avoid a
  dependency that does not install cleanly here. Covered by tests.
- The optional cluster-based splitter and the pretrained-model stretch goal
  are not implemented; they are the first two items on the proposal's
  scope-reduction list (§8).

## 9. Conclusion

On Tox21, at an equal tuning budget, **the GNN never measurably outperformed
the fingerprint baseline** — the two are separated by less than one SD in
every condition at the full training set, and with little training data the
baseline is clearly ahead. What dominates both models is the structural shift:
forcing the test molecules to be scaffold-novel costs about 0.07 ROC-AUC, an
order of magnitude more than anything the choice of architecture bought.

For the lecture's framing, that is the point. Molecules worth finding are, by
definition, unlike the training data, and that is exactly the regime where both
methods degrade most and where their difference matters least. The GNN also
costs more compute to reach the same place — 3.8–5.5× the baseline at the full
training set, on a GPU, with deterministic kernels pinned.

ClinTox neither confirms nor contradicts this; it is too small to resolve a
gap of this size, and the baseline's suspiciously small scaffold penalty on it
is better explained by dataset provenance than by model quality.

The grid has been run twice, before and after the GNN was made
deterministic, and every one of these conclusions stood through both while
the smaller numbers moved around inside their error bars (§5b). Finding that
the GNN's own scores moved by 0.032 between identical runs — six times the
difference it was being credited with over the baseline — is the strongest
argument in this report for having reported SDs throughout rather than point
estimates, and for the three-line change that makes the grid repeatable.
