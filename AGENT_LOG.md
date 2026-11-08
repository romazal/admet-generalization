# AI coding agent log

Record of how the coding agent was used, as the assignment requires
(proposal §5, §6). The pattern throughout: we specify the module and its
constraints, the agent drafts it, and we then read, correct and test it before
it is kept. Bugs the agent introduced are recorded here too.

## 2026-11-07 — project scaffold

*Prompt:* check the repository layout against §4 of the proposal.

The agent compared the directory tree with the PDF and reported the gaps; the
structure was then written into the README. No implementation at this stage —
every `.py` and `.yaml` file was still empty.

## 2026-11-07 — data splits

*Prompt:* "Read HANDOFF.md, then implement src/admet/data/splits.py and
tests/test_splits.py per the proposal in pdf_pages/."

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

## 2026-11-07 — environment check

*Prompt:* whether to lower `requires-python` from 3.14 to 3.13.

Wheel availability was checked on real 3.12, 3.13 and 3.14 interpreters: RDKit,
PyTorch and scikit-learn publish wheels for all three, so 3.14 is not short of
wheels and the downgrade was unnecessary. We kept `requires-python = ">=3.14"`
and corrected README line 66, which still claimed 3.12 — that clears the
README/`pyproject.toml` contradiction.

Two notes for the team: RDKit does not import under 3.13 on this machine (a
Windows Application Control policy blocks the DLL — not a code problem), and
the project `.venv` is empty and has no `pip`, so it must be recreated before
the suite can be run locally. Still open: `dependencies = []` and the missing
`[build-system]` table.

## 2026-11-08 — basic end-to-end implementation

*Prompt:* "can u implement a basic version of the project?"

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
  pin. The 2026-11-07 entry verified wheels exist for 3.14 and that remains
  true — this only stops teammates on 3.12/3.13 from being locked out. Revert
  if the team wants 3.14 enforced.
- RDKit, PyTorch 2.15, scikit-learn, matplotlib and PyYAML all install and
  import on 3.14 on this machine. The 3.13 RDKit DLL problem noted on
  2026-11-07 did not recur.
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

## 2026-11-08 — ClinTox replaces BBBP as the optional second dataset

*Prompt:* "instead of BBBP can u hace ClinTox as the smaller dataset? also
download it"

The loader registry listed BBBP, following the "for example BBBP" wording of
proposal §3.1; the team had already settled on ClinTox as the optional second
dataset on 2026-11-07. Swapped, downloaded, and — since we had the raw file in
hand — used to **verify the ClinTox figures that HANDOFF.md listed as
unverified** ("come from loader docs, never checked against the raw CSV").

**Verified against the raw CSV.** Both predicted counts were right:

| Claim (2026-11-07, unverified) | Measured |
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
  the provenance artefact the 2026-11-07 note suspected, now visible in the
  data rather than hypothesised.
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

## 2026-11-08 — Main experiment and learning curves executed

**Prompt:** "i need the rpoject to give some basic results at least"

Ran the full grid rather than a pilot: `tox21_{rf,gnn}_{random,scaffold}` at
the configured tuning budget of 8, seeds 0-4 (main experiment, the proposal's
target of 5 seeds), plus `sweep_learning_curve.yaml` at fractions
0.1/0.25/0.5/1.0 over seeds 0-2. 56 result files, 0 failures. Wall clock
roughly 5 h on CPU; GNN runs dominate (~8-11 min each at 100%).

Aggregated with `admet aggregate` and redrew all six figures; the three ROC-AUC
figures were opened and inspected, not just written.

**Outcome.** H2 is supported (the baseline's advantage at 10% of the training
set is larger than its SD in PR-AUC and closes monotonically with data). H1 is
not supported, because no GNN advantage exists at 100% to shrink. Written into
the README's Findings section.

**Worth recording:** an intermediate 3-seed pass put the scaffold ROC-AUC gap at
-0.016 +/- 0.020; extending to 5 seeds moved it to -0.004 +/- 0.022. The
3-seed figure would have been reported as a real effect. The proposal's "target
5, minimum 3" is doing real work and the minimum should not be used for the
headline comparison.

## ClinTox grid, the report, and committing the artefacts

**Prompts:** "now also do the clintox and compare with tox21" / "add and make
appropriate commits of figures and [the CSVs] and also add data so teammates
can load them upon cloning the repo" / "also u need to include a report file
in which the results (also time the models where running) and stuff like
that" / "check the pdf, commits and the whole project"

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
0.022-0.043 ROC-AUC to a scaffold split where Tox21 loses 0.074-0.080, which
together with its 19 contradictory duplicate structures points at dataset
provenance rather than model quality.
