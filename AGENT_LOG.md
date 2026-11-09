# AI coding agent log

Record of how the coding agent was used, as the assignment requires
(proposal §5, §6). The pattern throughout: we specify the module and its
constraints, the agent drafts it, and we then read, correct and test it before
it is kept. Bugs the agent introduced are recorded here too.

Entries are in the order the work happened. The *Instruction* line summarises
what the agent was asked for; it is a paraphrase, not a transcript.

## Project scaffold

*Instruction:* check the repository layout against §4 of the proposal and
report any gaps.

The agent compared the directory tree with the PDF and reported the gaps; the
structure was then written into the README. No implementation at this stage —
every `.py` and `.yaml` file was still empty.

## Data splits

*Instruction:* implement `src/admet/data/splits.py` and
`tests/test_splits.py` to the specification in the proposal, working from the
PDF pages in `pdf_pages/` and the handover notes.

**Added**

- `splits.py` — `random_split`; `scaffold_split` (randomized Bemis-Murcko, with
  `order="size"` for the deterministic MoleculeNet ordering); `subsample_train`
  for the learning curves; a `SPLITTERS` registry with a `make_split`
  dispatcher for config-driven runs; and `assert_scaffold_disjoint` as a
  run-time leakage check for §3.4.
- `tests/test_splits.py` — 72 tests on synthetic SMILES, so they run before any
  download works.
- `pyproject.toml` — pytest `pythonpath = ["src"]`, so the package imports in a
  test session without an install.

**Design points**

- Whole scaffold groups move together, so realised sizes cannot match the
  requested fractions exactly. The resulting bound is asserted in the tests
  rather than left implicit.
- Learning-curve subsets are nested (10% ⊂ 25% ⊂ 50% ⊂ 100%), and validation
  and test never move between fractions, so a dip in a curve cannot be a
  resampling artefact.
- Every function seeds its own `random.Random`; nothing reads or mutates global
  random state, so a split depends on its `seed` alone.
- The optional cluster splitter (§5) was left out; `SPLITTERS` is where it
  would go.

**What we checked, and what it caught**

- The 16 ring cores and 10 substituents used in the tests were verified against
  RDKit before being relied on.
- *Bug in the agent's test.* The chirality case used `C[C@H]1CCCCC1C`, whose
  stereocentre sits on a side-chain-bearing atom and is therefore stripped with
  the side chain, so `include_chirality=True` changed nothing. Replaced with
  cis-/trans-decalin, whose stereocentres are ring-fusion atoms and survive.
- *Gap the agent had missed.* With very few, very large scaffold groups a split
  is still a valid partition but can leave a part empty (one scaffold, 50
  molecules → sizes `(0, 0, 50)`). An empty training set must not reach a run
  silently, so `scaffold_split` now warns, with a test covering it. This cannot
  occur on Tox21.
- *We confirmed the leakage test is not vacuous.* Seven deliberately broken
  copies of `splits.py` were run against the suite; all seven were caught,
  while an unmodified copy passed. `test_random_split_does_leak_scaffolds`
  keeps this honest inside the suite: it asserts that a random split *does*
  leak, so the scaffold-split assertion cannot pass trivially.

All 72 tests pass.

## Environment check

*Instruction:* assess whether `requires-python` should be lowered from 3.14 to
3.13.

Wheel availability was checked on real 3.12, 3.13 and 3.14 interpreters: RDKit,
PyTorch and scikit-learn publish wheels for all three, so 3.14 is not short of
wheels and the downgrade was unnecessary. We kept `requires-python = ">=3.14"`
and corrected the README, which still claimed 3.12 — that clears the
README/`pyproject.toml` contradiction.

Two notes for the team: RDKit does not import under 3.13 on this machine (a
Windows Application Control policy blocks the DLL — not a code problem), and
the project `.venv` is empty and has no `pip`, so it must be recreated before
the suite can be run locally. Still open: `dependencies = []` and the missing
`[build-system]` table.

## Basic end-to-end implementation

*Instruction:* implement a working version of every module §5 specifies, so
the project runs end to end.

Everything in §5 that was still a 0-byte file, built in the order §6 asks for:
loaders and evaluation, fingerprints and graphs, the two models, the trainer
and configs, aggregation and plots, then the two remaining test files.

**Added**

- `data/loaders.py` — dataset registry (Tox21, BBBP), download cache,
  canonicalization, unparseable-SMILES drop, duplicate merging.
- `evaluation.py` — per-assay ROC-AUC/PR-AUC with masked multitask labels and
  macro-averaging; unscoreable assays are named, not scored as zero.
- `features/fingerprints.py`, `features/graphs.py` — Morgan bit/count vectors;
  atom/bond graphs with batching.
- `models/base.py`, `models/rf.py`, `models/gnn.py` — shared interface, the
  equal-budget sampler, one forest per assay, and a GINE network with a masked
  multitask head.
- `training.py`, `cli.py`, the five configs — one run = one config + one seed.
- `analysis/aggregate.py`, `analysis/plots.py` — summary and gap tables, three
  figures.
- `tests/test_evaluation.py`, `tests/test_features.py` — 55 new tests.
- `pyproject.toml` — pinned dependencies, `[build-system]`, `admet` entry
  point. `.gitignore` — `/data/`, `/results/`, `/figures/` now actually
  ignored (they were commented out). `Makefile`, `README.md`.

**What we checked, and what it caught**

- *Bug in the agent's splitter — the serious one.* `scaffold_split` packed
  groups by the MoleculeNet priority rule (train, else validation, else test).
  That rule only works when groups are placed largest-first, which the
  *randomized* order of proposal §3.3 does not do. Tox21 has one group of 1775
  acyclic molecules and one of 1474 sharing a ring system, against a 10% part
  of ~782, so a giant group spilled into whichever part it first failed to fit:
  seed 0 gave `(6048, 0, 1775)` — an **empty validation set**, which silently
  disables early stopping and leaves hyperparameter tuning nothing to select
  on — and seed 2 gave `(6258, 1474, 91)`, a 91-molecule test set. Two of five
  seeds were unusable and none were comparable.
  The agent's own module docstring had asserted this "is not a problem on
  Tox21, which has thousands of scaffolds"; the claim was never checked against
  the data, and the 72-test suite missed it because its synthetic dataset has
  no group larger than 10 molecules.
  Fixed by packing the randomized order by **largest deficit** — each group
  goes to the part currently furthest below its target. That alone was still
  not enough: a giant group arriving *late* still landed wherever there was
  room (seed 0: a 200-molecule test set). The groups are now sorted by
  descending size after the shuffle, so the giants are placed while every part
  is empty; the sort is stable, so equal-sized groups keep their shuffled order
  and the split still varies with the seed. All five seeds now give
  `(6259, 782, 782)` against a target of `(6258, 782, 782)`, with ~33% test-set
  overlap between consecutive seeds. `order="size"` keeps the original
  DeepChem-compatible packer, for comparison with published numbers.
  Two tests were updated to match (the old packing-bound test now covers
  `order="size"` only) and three added, including
  `test_randomized_packing_survives_a_group_larger_than_a_part`, which is the
  Tox21 failure reproduced in miniature.
- *Bug in the agent's CLI.* `admet sweep` resolved its child configs against
  `configs/` and the working directory but not against the sweep file's own
  directory, so a sweep config anywhere else could not find its siblings —
  unlike `extends:`, which does resolve relatively. Fixed.
- *Checked the baseline against published numbers* (the §6 human-verification
  task). Tox21, random split, full tuning budget, seed 0: ROC-AUC 0.838, which
  is in the published range for fingerprint models on this dataset. The loader
  also reproduces the expected shape — 7823 molecules after dropping 8
  unparseable SMILES, 12 assays, 83% label density, 186–942 positives per
  assay.
- *Checked the masking is not vacuous.* The metric and loss tests each carry a
  positive control asserting that the wrong implementation (missing label read
  as a negative) gives a **different** answer, so the masking tests cannot pass
  on code that ignores the mask.
- *Looked at the rendered figures,* which caught the learning-curve x-axis
  advertising 25% and 50% ticks when only 10% and 100% had been run.

**Environment findings**

- `requires-python` was `>=3.14`; lowered to `>=3.12` as a **floor**, not a
  pin. The environment-check entry above verified wheels exist for 3.14 and
  that remains true — this only stops teammates on 3.12/3.13 from being locked
  out. Revert if the team wants 3.14 enforced.
- RDKit, PyTorch 2.15, scikit-learn, matplotlib and PyYAML all install and
  import on 3.14 on this machine. The 3.13 RDKit DLL problem noted earlier did
  not recur.
- **pandas does not work here.** Its compiled extensions are blocked by the
  Windows Application Control policy, and inconsistently so — two successive
  imports failed on *different* DLLs (`offsets`, then `_ujson`). Rather than
  build the pipeline on a dependency that fails nondeterministically, pandas
  was dropped; the loader and aggregation use `csv` plus NumPy. This is a
  documented deviation from proposal §5.
- PyTorch Geometric was not used. Proposal §3.2 allows "a GIN in PyTorch
  Geometric"; the same model is ~10 lines of `index_add_` in plain PyTorch,
  and avoiding PyG removes an install risk on this machine.

**Still open**

- The full seed grid has not been run, so the README's *Findings* section is
  explicitly empty. `make main` then `make sweep` fills it.
- The GNN's hyperparameter defaults have had no real search behind them yet;
  the pilot numbers in the README are from a deliberately tiny budget and say
  nothing about H1.
- Optional and unimplemented, per the §8 scope-reduction order: the
  cluster-based splitter, the pretrained-model stretch, and BBBP as a second
  dataset (registered in the loader, never run).

## ClinTox replaces BBBP as the optional second dataset

*Instruction:* replace BBBP with ClinTox as the smaller second dataset, and
download the raw file.

The loader registry listed BBBP, following the "for example BBBP" wording of
proposal §3.1; the team had already settled on ClinTox as the optional second
dataset. Swapped, downloaded, and — since we had the raw file in hand — used
to **verify the ClinTox figures the handover notes listed as unverified**
("come from loader docs, never checked against the raw CSV").

**Verified against the raw CSV.** Both predicted counts were right:

| Claim, previously unverified | Measured |
|---|---|
| ~94 for `FDA_APPROVED = 0` | **94** ✓ |
| ~110–112 for `CT_TOX = 1` | **112** ✓ |
| scaffold test split leaves ~15 positives | **10–17** over seeds 0–4 ✓ |
| H2's 10% point would have ~7 positives | **5** — slightly worse ✓ |
| 2 near-duplicate labels, no missing values | `CT_TOX == NOT FDA_APPROVED` for **98.8%**; label density **1.0** ✓ |

So the argument for keeping Tox21 primary holds on measured numbers, not just
on estimates. 1484 raw rows → 1461 molecules (4 unparseable, 19 duplicates).

**Two things the team had not predicted**

- *19 duplicate structures carry contradictory labels.* Merging produced 38
  conflicts across 19 groups — every group disagreed on **both** labels, i.e.
  the same structure appears as approved and as failed-for-toxicity. This is
  the provenance artefact the team had suspected, now visible in the data
  rather than hypothesised.
- *A reporting trap that would have inflated the headline number.* The
  standing advice was "lead with PR-AUC given ~7% positives". That is right
  for `CT_TOX` and **wrong for `FDA_APPROVED`**, whose positive class is the
  91% majority: a measured run gave it PR-AUC 0.981 where a constant predictor
  scores ~0.911, and macro-averaging the two tasks lifted the reported PR-AUC
  from 0.379 to 0.680. Anyone quoting the macro figure would have reported a
  near-doubling that is entirely an artefact of averaging a trivial task with
  a hard one.
  `configs/clintox_{rf,gnn}_scaffold.yaml` therefore set `tasks: [CT_TOX]`,
  and the caveats are written into the `loaders.py` module docstring, which
  the registry entry points at.

**Checked**

- Both new configs run end to end at the full tuning budget (RF 9 s, GNN 77 s,
  seed 0): CT_TOX-only ROC-AUC 0.840 / 0.896, PR-AUC 0.487 / 0.477. One seed,
  so this says nothing about H1 — it confirms the 2-task-to-1-task path works,
  where Tox21 exercises 12.
- 133 tests still pass; no BBBP references remain outside the generated
  egg-info.

## Main experiment and learning curves executed

*Instruction:* run enough of the grid to produce real results.

Ran the full grid rather than a pilot: `tox21_{rf,gnn}_{random,scaffold}` at
the configured tuning budget of 8, seeds 0-4 (main experiment, the proposal's
target of 5 seeds), plus `sweep_learning_curve.yaml` at fractions
0.1/0.25/0.5/1.0 over seeds 0-2. 56 result files, 0 failures. 3.3 h of wall
clock on CPU; GNN runs dominate (~9 min each at 100%).

Aggregated with `admet aggregate` and redrew all six figures; the three ROC-AUC
figures were opened and inspected, not just written.

**Outcome.** H2 is supported (the baseline's advantage at 10% of the training
set is larger than its SD in PR-AUC and closes with data). H1 is not supported,
because no GNN advantage exists at 100% to shrink. Written into the README's
Findings section.

**Worth recording:** an intermediate 3-seed pass put the scaffold ROC-AUC gap at
-0.016 +/- 0.020; extending to 5 seeds moved it to -0.004 +/- 0.022. The
3-seed figure would have been reported as a real effect. The proposal's "target
5, minimum 3" is doing real work and the minimum should not be used for the
headline comparison.

## ClinTox grid, the report, and committing the artefacts

*Instruction:* run the ClinTox grid and compare it with Tox21; commit the
figures, the result tables and the raw data so teammates get them on clone;
add a report file covering the results and how long the models took to run;
then review the project against the proposal PDF, including the commit
history.

Ran the full ClinTox grid: 4 configs x 5 seeds x 4 training fractions, 80
runs, 0 failures. Added `configs/clintox_{rf,gnn}_random.yaml`, which were
missing - only the scaffold arm existed, so there was nothing to measure the
random-to-scaffold drop against on the smaller dataset.

**Bug found before it could mislead anyone.** `analysis/plots.py` never
looked at the `dataset` field although `aggregate.py` keys on it correctly.
`plot_split_comparison` took the first row matching `(model, split)`, so once
ClinTox results existed a Tox21 bar could be drawn into a ClinTox figure with
nothing on the figure to show it. Fixed: all three plot functions take a
`dataset` argument, `make_all_figures` emits one set per dataset, and the
dataset is in every filename and title. The two datasets deliberately never
share an axis.

**Correction made during the review against the proposal PDF.** The README's
runtime table still held pre-measurement estimates, including "GNN, one epoch
~1.9 s" from a single one-off measurement, after I had already said it was
replaced - it had only been fixed in REPORT.md. The table is now derived from
the `timing_seconds` field of the committed result files.

**Three deviations from the proposal were unrecorded** and are now listed in
both README and REPORT: ClinTox rather than §3.1's example BBBP; `data/`,
`results/` and `figures/` committed where §5 gives `.gitignore` the job of
excluding them; and CPU-only rather than §8's assumed 16 GB GPU - the full
136 runs took 3.83 h without one, so the feasibility estimate held.

**Verified rather than asserted:** `scaffold_disjointness_checked` is true in
68 of 68 scaffold result files and false in all 68 random ones.

**Outcome.** ClinTox is reported as inconclusive - every GNN-minus-baseline
gap on it is smaller than its own SD. The informative part is that it scores
higher than Tox21 on a fifth of the data (0.90 vs 0.84) and loses only
0.021-0.043 ROC-AUC to a scaffold split where Tox21 loses 0.073-0.080, which
together with its 19 contradictory duplicate structures points at dataset
provenance rather than model quality.

## Project-wide review before pushing

*Instruction:* review the whole project thoroughly, including every commit and
its message, and exclude anything that should not be pushed.

**Two bugs found by checking rather than reading.**

- *Every PR-AUC bar chart was blank.* `plot_split_comparison` hardcoded
  `ax.set_ylim(0.5, 1.0)`, which is right for ROC-AUC and wrong for PR-AUC:
  Tox21's PR-AUC values are 0.35–0.47, entirely below the floor, so the bars
  were drawn outside their own axis. Matplotlib does not warn. Two committed
  figures had been wrong since they were first produced and nobody had opened
  them. Fixed with a metric-aware limit, and `tests/test_plots.py` was written
  to assert that every plotted value falls inside the axis limits.
- *`results/runs.csv` leaked absolute paths* from the machine that produced it,
  so the committed file could not regenerate identically anywhere else. Found
  by cloning the repository fresh and regenerating all three tables:
  `summary.csv` and `gap.csv` matched byte for byte, `runs.csv` did not. Paths
  are now stored relative to the results directory, POSIX-style.

**Corrections to the documentation.** Four false statements in the README (a
download size, an artefact size, stale result paths, and a claim that ClinTox
scores are macro-averaged over 12 assays when it scores one) and three values
rounded wrong by rounding the operands before subtracting them. Results and
figures were moved into per-dataset subdirectories so the layout matches.

**A claim I had made and not checked.** The README stated the runs were CPU-only
because the machine had no GPU. The machine has an RTX 5070 Ti. The runs were
genuinely CPU-only, but because `pip install torch` takes the CPU-only wheel by
default on Windows — an environment default I had written up as a deliberate
choice.

**Commit history.** Squashed an added-then-rewritten pair, rewrote three commit
messages that used invented type prefixes, and corrected a size claim in a
fourth.

## GPU rerun of the full grid

*Instruction:* rerun the experiment on the GPU, record how much faster it is,
and use the rerun to check the results; keep the work on a separate branch.

A CUDA build of PyTorch was installed and the whole 136-run grid was rerun,
then merged to `master` as the primary results. `versions.torch_device` was
added to every result file first, so a run now records the device it used and
not only the library versions.

**Speed.** 2.03 h against 3.83 h. The GNN is 2.6x faster per Tox21 run and
2.5x at the full training set; the overall figure is lower only because the
baseline is 45% of the grid's wall clock and never touches the card. The
speedup is modest for a GPU because these graphs are small — under 20 heavy
atoms on average — so the card is never saturated.

**The rerun as a check, which is the more useful half.** The baseline is
scikit-learn with fixed seeds and is unaffected by the change, so it serves as
the control: its test metrics came back **bit-identical across all 68 runs**,
which is what licenses attributing the GNN's movement to the GNN rather than
to the data, the splits or the harness. One baseline *validation* ROC-AUC
drifted by 8.3e-7 from non-deterministic float summation across threads; it
did not change which tuning trial won.

The GNN's scores moved, by up to 0.002 on Tox21 and 0.037 on ClinTox, always
inside the seed SD. Device and torch version changed together (2.15.0+cpu →
2.11.0+cu128, as no cu128 wheel exists for 2.15), so the difference is not
attributable to the GPU alone.

**Every conclusion survived; three smaller claims were weakened and now say
so.** The split type still dominates and both headline gaps still sit inside
one SD of zero. But the monotone ordering of the PR-AUC learning curve swapped
splits between the two passes, so H2 now rests on the 10%-versus-100% contrast
rather than on a smooth curve; ClinTox's small scaffold drop is cited for the
baseline only, since the GNN's read 0.043 then 0.082; and the threats section
gained an entry noting that a single GNN run quoted to three decimals is
over-precise. Two 3-seed learning-curve gaps also changed sign, which is the
earlier three-seed finding arrived at independently.

Nothing fell outside its stated uncertainty. The rerun did not overturn a
finding — it confirmed that the quantities reported with SDs larger than their
means deserved exactly that treatment.
