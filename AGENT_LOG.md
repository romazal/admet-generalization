# AI coding agent log

Record of how the coding agent was used, as the assignment requires
(proposal §5, §6).

**How the work was divided.** We set the requirements, choose the design, read
what comes back and decide what is kept; the agent drafts to that
specification and carries the volume work — writing out the test suites,
running the 146-run grids, regenerating the tables and the twenty-four
figures,
and applying mechanical edits across the documentation. It is a fast pair of
hands, not a collaborator with judgement about this project: left to itself it
has asserted things it had not checked, and several of the bugs below are
ones it introduced.

So the useful question for each entry is not "what did the agent produce" but
"what did we ask for, what came back, and what did checking it reveal". The
entries are written that way. Where a problem was caught by one of us reading
output rather than by the agent, the entry says so, because that is the part
of the process the assignment is actually asking about.

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
~1.9 s" from a single one-off measurement, after the agent had reported it
replaced - it had only been fixed in REPORT.md, which is why the review
checked both files rather than taking the report at its word. The table is now derived from
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

**Two bugs, one spotted by eye and one by checking.**

- *Every PR-AUC bar chart was blank* — noticed by opening a committed figure
  and asking why it was empty, which no amount of reading the code had done.
  `plot_split_comparison` hardcoded `ax.set_ylim(0.5, 1.0)`, which is right
  for ROC-AUC and wrong for PR-AUC: Tox21's PR-AUC values are 0.35–0.47,
  entirely below the floor, so the bars were drawn outside their own axis.
  Matplotlib does not warn. Two committed figures had been wrong since they
  were first produced. The agent had written the limit, had regenerated those
  figures several times, and had not looked at them. Fixed with a
  metric-aware limit, and the agent was set to write `tests/test_plots.py`
  asserting that every plotted value falls inside the axis limits.
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

**A claim the agent had made and not checked**, caught by asking it why the
runs were on the CPU at all. The README stated it was because the machine had
no GPU. The machine has an RTX 5070 Ti. The runs were genuinely CPU-only, but
because `pip install torch` takes the CPU-only wheel by default on Windows —
an environment default the agent had inferred from the runs being slow and
written up as a deliberate decision.

**Commit history.** Squashed an added-then-rewritten pair, rewrote three commit
messages that used invented type prefixes, and corrected a size claim in a
fourth.

## GPU rerun of the full grid

*Instruction:* rerun the experiment on the GPU, record how much faster it is,
and use the rerun to check the results; keep the work on a separate branch.

We installed a CUDA build of PyTorch and confirmed the card was visible; the
agent then reran the whole 136-run grid, which is the kind of long mechanical
job it is good for. `versions.torch_device` was added to every result file
first, so a run records the device it used and not only the library versions.
Making the GPU pass the primary results, rather than keeping the CPU pass as
the submitted numbers, was our call — the agent had argued for it but the
trade-off (a newer torch on the CPU pass against an independently verified
GPU pass) was a judgement about what to submit.

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

## Contribution notes for the team

*Instruction:* write up small, self-contained tasks the other three group
members could take on with a free AI agent, so they also appear as
contributors; later, drop the difficulty ratings and time estimates, and add
a larger milestone.

`TODO.md`. The first draft came back as lecture-preparation work rather than
code contributions, which is not what had been asked for; redirected, it was
rewritten as small tasks each naming the file to add, the nearest existing
file to copy from, and what to test.

The star ratings and hour estimates in that draft were removed on review.
They were guesses presented as information: they ranked the tasks on behalf
of teammates better placed to judge, and a wrong estimate puts someone off a
task they would have enjoyed. The "suggested split" table, which assigned the
three tasks the agent rated highest to three unnamed people, went with them —
replaced by a note on which tasks share files, which is the part that is
actually verifiable.

Two errors in the first draft, both caught in review before it was committed:
it cited `references/palette.md`, which is a tool file and not part of this
repository, and it proposed adding assay names to `Dataset.summary()`, which
already returns them. Both are the same failure — describing the repository
from memory instead of reading it.

The dataset candidates were checked rather than recalled — all five filenames
resolve on the MoleculeNet mirror, and BBBP's header was fetched
(`num,name,p_np,smiles`, 2050 rows) so the task states the column names as
fact.

## Environment: the pandas blocker has lifted

*Instruction:* add restoring pandas to the task list, since the project
currently does without it.

Checked before writing the task, and the premise of the deviation has
changed: **pandas 3.1.1 is installed in the current `.venv` and imports
cleanly on repeated attempts.** The original failure was the Windows
Application Control policy blocking its compiled extensions intermittently —
two successive imports failed on *different* DLLs. The environment was
rebuilt when the CUDA build of torch was installed, which is the likely
explanation.

Two things were measured rather than assumed, because the committed CSVs must
keep regenerating byte-identically:

- pandas reproduces all three exactly, with
  `to_csv(index=False, lineterminator="\r\n")`. Float formatting already
  matches, since pandas and `csv` use the same shortest-repr. The line
  terminator is the only trap.
- The agent's first round-trip check reported all three as failing. The
  check was wrong, not the files: `read_text()` normalises line endings, so
  it compared `\n` against `\r\n`. The task tells whoever takes it to compare
  `read_bytes()` for that reason.

The deviation is still recorded in four places and stays there until the work
is done.

## Figure review

*Instruction:* check all twelve figures — the learning curves are hard to
read with four overlapping lines in two colours, and it is not clear what the
vertical bars are.

The complaint was the finding. Reviewing the figures against it turned up one
bug and three readability failures, none of which the agent had raised in the
several sessions it spent generating and regenerating these files.

- *Bug: colour meant two different things across the set.* `plot_gap` set
  `split_colors` to the two **model** hues, so blue read as "random forest"
  in the bar chart and the learning curve but "random split" in the gap plot.
  Reading the figures in order misleads you about one of them. The gap is
  GNN-minus-baseline, a quantity about both models, so it now has its own
  neutral.
- *Four series on one axis.* The learning curves encoded model as colour and
  split as line style, so two channels had to be read together while the four
  curves crossed repeatedly. Both the learning-curve and gap figures are now
  faceted into a random-split and a scaffold-split panel on a shared y-axis:
  two lines per panel, distinguished by colour alone, and the step between
  panels is the split effect — the largest result in the study, previously
  just one more gap between lines.
- *The ClinTox legend sat on top of its own curves.* One legend below the
  figure now.
- *Bar value labels collided with the error-bar caps.* The label was anchored
  to the top of the bar, which is where the cap already is: with an SD of
  0.009 on a 0.834 bar the two overlapped in every committed bar chart.

Also added a footnote to every figure stating that the error bars are ±1 SD
and over how many seeds. This was the question that prompted the review, and
it was fair: the only statement of it was in the y-axis label. An unexplained
interval reads as a drawing artefact rather than as uncertainty — and on
ClinTox, where it spans −0.20 to +0.37, it is the finding.

**Checked.** The agent wrote the eight new tests and regenerated the twelve
figures; the requirement it was held to was that a test must inspect the real
figure — `_save` is patched to capture it — rather than redrawing the chart
inside the test, because a test that redraws passes while the shipped figure
is wrong. That is precisely the failure mode that let the blank PR-AUC charts
survive a 143-test suite.

Each test was then run against a deliberately broken copy, since a figure
test that cannot fail is worth nothing: restoring the model colour, moving
the label back to the bar top and deleting the footnote each fail exactly one
test and nothing else. 156 tests pass. The twelve figures were opened and
looked at, which is the step that started this entry.

## Loader tests, contributed by Cas

*Instruction (Cas's, with his own agent):* task 7 of `TODO.md` — write
`tests/test_loaders.py`, offline, covering the cleaning that `loaders.py`
does.

The first contribution from someone outside the two of us who had been
driving the agent so far, and the first chance to see whether the task notes
in `TODO.md` are specific enough for a teammate to work from without reading
the whole codebase. They were: the branch arrived with the four cases the
task asked for and nothing else.

Four tests, all offline — a toy `DatasetSpec` is registered through
`monkeypatch.setitem(DATASETS, ...)` and a tiny CSV written to `tmp_path`, so
nothing downloads: an unparseable SMILES is dropped and counted in both
`n_unparseable` and `summary()["n_unparseable_dropped"]`; `CCO` and `OCC`
merge to one row; a blank cell stays `NaN` rather than becoming a zero; and
`label_density` matches a hand-computed 4/6.

**Checked.** Reviewed on a branch before merging: it touches no protected
file, adds no runs, and fast-forwards onto master with Cas's authorship
intact. 160 tests pass.

The task text asks the contributor to prove the tests are not vacuous, and
that claim is not visible in a diff, so it was re-done here against five
deliberately broken copies of `loaders.py`. All five are caught: reading a
blank cell as `0.0` instead of `NaN` fails two tests, disabling duplicate
merging fails all four, dropping unparseable rows without counting them fails
one, keeping them instead of dropping them fails one, and making
`label_density` ignore the mask fails one.

The blank-cell mutation is the one that matters. A missing Tox21 label
silently read as a negative would turn "not screened" into "screened,
inactive" across the whole matrix, inflate every baseline, and leave the
masking argument in §3.4 resting on nothing — and it would not fail any other
test in the suite. That gap is now closed, and it was the one module where it
was still open.

## Per-assay table and a dummy baseline, contributed by Jollly23

*Instruction (Jollly23's, with her own agent):* tasks 2 and 4 of `TODO.md` —
break the macro average down per assay, and add a null model to put a floor
under the scores.

Two additions. `per_task_table()` and `PerTaskRow` in `aggregate.py` collapse
across seeds the per-assay metrics each run already records, and `aggregate()`
writes `results/per_task.csv` beside the existing three tables; this is an
aggregation over stored values, not a recomputation from predictions, so it
cannot disagree with the headline numbers. `DummyModel` is a per-assay
class-prior predictor registered as `dummy`, with configs for both splits.

**Checked.** The committed `per_task.csv` was regenerated from the committed
run files and matches byte-for-byte, and after merging, `make aggregate` still
rewrites all four tables with zero diff — the property the whole
verify-by-reproduction argument rests on. The dummy was run on synthetic data
at 10% and 50% prevalence with some labels masked: constant predictions,
ROC-AUC exactly 0.5, PR-AUC equal to prevalence, masked labels excluded. That
is the claimed floor, confirmed rather than assumed. `gap_table()` takes
`baseline="rf"` and `advanced="gnn"` explicitly, so a third registered model
does not disturb the H1 gap. 163 tests pass.

**Not done, and it matters.** The dummy model exists but has never been run.
There are no dummy result files, so the floor is still absent from the report —
and the floor is the useful half of task 4. The PR-AUC figure is the one to
want: a Tox21 PR-AUC of 0.47 means little until the number a model that knows
nothing scores is next to it. Running the two configs is a few minutes of CPU.

**An authorship problem, recorded because it is an agent problem.** The branch
arrived as four commits: the same tree committed three times under three
different identities — `Copilot <copilot@example.com>`, `romazal` and
Jollly23 — plus one empty commit. Her agent had been reassigning the author
field, so on GitHub the work read as partly Copilot's and partly Roman's. The
code was unaffected, but the contribution record was wrong, and the assignment
asks who did what. Squashed to a single commit authored by Jollly23, verified
to produce a tree identical to a real merge of all four, and force-pushed over
the branch; the duplicate `feature/per-assay-results`, which pointed at the
same commit, was deleted. Worth watching for on other branches: an agent left
to manage git will happily sign work with whatever identity it finds.

## MACCS keys, contributed by Lukas — and a task instruction that was wrong

*Instruction (Lukas's, with his own agent):* task 5 of `TODO.md` — add MACCS
keys as a second fingerprint for the baseline, and let the random forest
choose between them.

The code came back correct and complete: `maccs_fingerprint()` using RDKit's
`GetMACCSKeysFingerprint`, an `fp_kind` option on `RandomForestModel` with
validation, the 167-bit length test and the benzene-vs-cyclohexane test, and
— the part the task flagged as the trap — `featurization_key()` returning
`("rf", "maccs")` so the two representations cannot share a feature cache. He
did what was asked.

**The task was wrong, and reviewing the branch is what surfaced it.** Task 5
told the contributor to add `fp_kind` to `search_space()`. That instruction
was ours, written without thinking the consequence through:

- `sample_configs` sorts parameter names before drawing, so inserting
  `fp_kind` shifts the position of every later parameter in the draw. With a
  budget of 8 that leaves **none** of the eight sampled configurations
  unchanged — Morgan-only ones included. Measured, not assumed: 1296 → 2592
  grid points, 0 of 8 configurations shared, 3 of the 8 slots taken by MACCS.
- So the baseline in all 68 committed runs would stop being the baseline the
  code produces. REPORT §5b leans on exactly that reproducing: the baseline's
  bit-identical test metrics across the CPU and GPU passes are the control
  that licenses attributing the GNN's movement to the GNN.
- It also unbalances §3.4 while appearing not to. The budget stays 8 for
  both, but the baseline would spread it over two representations while the
  GNN keeps all 8 on one. If MACCS is the weaker representation that inflates
  the GNN-minus-baseline gap — the quantity H1 is measured on — for a reason
  that has nothing to do with graph neural networks, and in the direction
  that favours our own hypothesis.

**What was changed.** `fp_kind` is out of `search_space()` and selected
per-run instead, through `model.params` in a config; `configs/tox21_rf_maccs_
random.yaml` and `..._scaffold.yaml` differ from the plain baseline configs in
that one setting, so the comparison is a deliberate side experiment rather
than a hidden degree of freedom. Lukas's commit is kept as his, with the
change made on top rather than folded into it.

**Checked.** The point of the exercise was that master stays authoritative, so
that was verified rather than argued: `tox21_rf_random` seed 0 was re-run on
the modified code and compared against its committed result file. The eight
tuning trials are the same eight with the same scores, and the test metrics
are bit-identical to the last decimal (ROC-AUC 0.838364862412147, PR-AUC
0.5036566674371696). Only `git_commit` and per-trial wall-clock seconds
differ. 168 tests pass.

Two tests now pin this. One asserts `fp_kind` is absent from the search space;
the other demonstrates why, by sampling both spaces and asserting the two sets
of eight configurations are disjoint. The second is the useful one — the
failure it guards against is a one-line edit that looks like an improvement
and silently detaches every committed baseline number from the code.

## Running the dummy baseline, and a figure it broke

*Instruction:* run the dummy baseline so the floor reaches the report — on
the GPU.

**The GPU does not apply here, and saying so was the useful part.** The GPU is
real and in use for the GNN (RTX 5070 Ti, torch 2.11.0+cu128), but `dummy.py`
imports no torch: it is `sklearn.dummy.DummyClassifier` emitting one constant
per assay. There is nothing to accelerate. Both configs, five seeds each, took
26 seconds of CPU. The `torch_device` field still records
`cuda:NVIDIA GeForce RTX 5070 Ti`, as it does in all 136 committed runs
including the scikit-learn baseline — it reports what torch can see, not what
the model used.

**Results.** ROC-AUC exactly 0.500 ± 0.000 under both splits, which is what a
constant predictor must score. PR-AUC 0.077 ± 0.002 random, 0.100 ± 0.007
scaffold. Those PR-AUC values reproduce the measured test-set positive rates
(0.0768 and 0.0997, computed independently from the splits) to three decimals,
which is a check on the masking as well as a floor: a class-prior model can
only score its own prevalence if missing labels are being excluded correctly.
This is now a third sanity check in the README. It is also the answer to the
question task 4 was really asking — a PR-AUC of 0.473 reads as weak until the
0.077 beside it shows what knowing nothing scores.

**A figure broke, and the tests did not catch it.** Adding a third model to
`summary.csv` put a dummy bar into the random-vs-scaffold charts. The PR-AUC
chart came out well. The ROC-AUC chart did not: its y-axis starts at 0.5
because 0.5 is chance, and the dummy scores exactly 0.5, so the bar had zero
height. The figure rendered as two coloured bars, an unexplained gap, and a
floating "0.500" label over nothing.

This is the same failure as the one in the figure-review entry above —
a chart written with its data outside the visible axis — at the other end of
the scale, and the test added for that one did not catch it. It asserted each
value lies *within* the limits, and 0.5 does lie within [0.5, 1.0]. A bar
exactly on the floor passes that check and still draws nothing.

Fixed in `plots.py`: `_bar_ylim` now takes the bar heights as well as their
tops and drops the ROC-AUC floor just below the shortest bar when one would
otherwise vanish, keeping 0.5 when nothing sits on it. The dummy also gained a
label (`Dummy (class prior)`) and a deliberately neutral grey, and bars are
ordered floor, baseline, GNN via `MODEL_ORDER` — sorting by name had put
`dummy` between the two models, where it reads as a third competing approach
rather than as the floor they are measured against.

Four tests added. The useful one inspects the drawn figure and asserts every
bar has height above the axis floor, which is the check that was missing
rather than a restatement of the one that passed. One existing test needed
amending: it compared the curve colours against every registered model colour,
which was the same thing as "both models" until a third model was registered.
172 tests pass, and `make aggregate && make plots` is idempotent — a second
run changes no byte of the four tables or the twelve figures.

## pandas restored — the blocker had lifted

*Instruction:* task 8 of `TODO.md` — put pandas back.

**The blocker was checked before anything was written, and repeatedly.** The
original failure was intermittent: two successive imports died on different
compiled extensions, which is why pandas was dropped in the first place. Ten
separate processes now import pandas 3.1.1 and complete a real
`read_csv`/`to_csv` round trip, ten times out of ten. The environment was
rebuilt when the CUDA torch wheel went in, which is the likely explanation.
This closes the only deviation from proposal §5 that existed for an
environment reason rather than a considered one.

**Hard dependency, not optional.** An optional import with the `csv` path as
a fallback would mean two writers for the committed tables, and the property
that matters is that those tables regenerate byte-for-byte. Two paths can
diverge by a digit and nobody would see it until a teammate's regeneration
came back dirty. One path cannot. The project already depends on RDKit,
PyTorch and scikit-learn, all with compiled extensions, so pandas adds no new
class of risk — and if the Application Control policy does block it on
someone's machine, task 8 is right that this should surface loudly rather
than be silently worked around.

**Two call sites, both with a trap the task had already identified.**

- Reading: `dtype=str, keep_default_na=False`. Left to infer, pandas turns
  blank cells into NaN, which happens to coincide with our masking — but only
  coincidentally, and it would also rewrite a 0/1 label column as floats.
  Reading everything as text leaves `_to_label` doing the conversion exactly
  as every committed result was produced. Verified after the change: 7823 and
  1461 molecules, 8 and 4 unparseable, 0 and 19 duplicates, density 0.8294 and
  1.0000, and the label matrix still holds nothing but 0.0, 1.0 and NaN.
- Writing: `index=False, lineterminator="\r\n"`. All four tables regenerate
  byte-identically — compared with `read_bytes`, since reading as text
  normalises line endings and would have reported success either way.

**A test that could not fail on this machine, found by mutating it.** Three
tests were added and then run against deliberately broken copies, as usual.
Dropping `index=False` was caught. Dropping `lineterminator="\r\n"` was
**not**: `to_csv` falls back to `os.linesep`, which on Windows *is* CRLF, so
the bytes are identical here and every byte-level assertion still passed. The
bug would appear on the first POSIX machine to run `make aggregate`, rewriting
all four committed tables with nothing visible in a text diff. A fourth test
now spies on the `to_csv` call and asserts the terminator is pinned — the only
check in the suite that fails on the machine the project is developed on. 175
tests pass.

Worth recording for whoever reads this next: a byte-comparison test is not
automatically a strong test. This one was only as strong as the platform it
ran on, and nothing but mutating it would have shown that.

## A batch of self-contained tasks, before touching the GNN

*Instruction:* do the TODO tasks that a GNN rerun could not invalidate, one
commit each, then move on to the GNN.

Five pieces, chosen because none of them touches a model result: a GNN rerun
would change the 68 GNN result files, the four tables and the twelve figures,
but no code, CLI, loader or test. Each is recorded below with what checking it
turned up, which in three cases was more than the task asked for.

**`admet datasets` and `--version` (tasks 6 and 9).** The command loads each
dataset rather than printing its registry entry: the registry says what a
dataset should contain, the loader says what survived unreadable SMILES and
duplicate merging, and the second number is the one every result rests on.
With it came `tests/test_cli.py` — `cli.py` was the last module with no test
file of its own, the same gap task 7 closed for `loaders.py`. The tests check
wiring, not the pipeline beneath it. Verified against broken copies: an
unwired handler, a hardcoded count and a wrong version string each fail.

**`CITATION.cff` (task 9).** All four contributors to master. Two fields were
left as explicit TODOs rather than filled in: commit authorship gives an email
and a GitHub handle, not always a legal name, and a guessed surname in a
citation file is worse than a blank one. No `license:` field either, because
the repository has no LICENSE — those two belong together.

**Warn on conflicting duplicates (task 9).** The count was already in every
result file, where nobody looks. The task says "19 on ClinTox"; that is the
*row* count. The conflict count is per assay and is 38, because every one of
those 19 merged rows disagrees on both assays. The warning gives both numbers,
since quoting one as the other understates it by half.

**`make figures-clean` (task 9), and a neighbour it exposed.** `clean-results`
globbed `results/*.json` and `figures/*.png`, but run files live in
`results/<dataset>/` and figures in `figures/<dataset>/`. The only thing it
ever deleted was the four aggregated tables — it reported success and left all
146 run files and all 12 figures in place, so anyone using it to reset was
working from stale results. Both targets now glob one level down, verified on
a copy of the tree rather than on the real one.

**The train/test similarity figure (task 1a).** For every test molecule, its
Tanimoto similarity to the nearest training molecule, drawn once per split.
This is the mechanism the report asserts in prose and then evidences with a
0.073 ROC-AUC drop; the figure measures the claim itself. On Tox21 the median
falls 0.583 → 0.419, and the share of test molecules with a
fingerprint-identical training molecule falls from 7.8% to 0.6% — a number
worth having, because it says how much a random split flatters a model here.

Two things about that figure are worth recording. It is the only one whose
subject is the split rather than the model, so split has to become the colour
channel — and the two model hues were therefore unavailable, since blue and
orange already mean "random forest" and "GNN" elsewhere and reusing them is
the exact confusion `plot_gap` had to have undone. The pair was chosen by
running the colour checks rather than by eye: of the reference palette's
remaining slots, violet and green are the best-separating pair in which
*both* colours also clear 3:1 contrast on the surface (CVD ΔE 28.8,
normal-vision ΔE 35.5, contrast 5.6 and 4.8, against thresholds of 8, 15 and
3). The yellow and aqua pairings separate slightly better and fail contrast.
Line style carries the split as well, so identity never rests on colour alone.

The second is that the first version of its test fixture was wrong in a way
that looked right: five scaffolds of eight molecules each. The randomized
splitter packs whole groups into whichever part is furthest below its target,
so those five groups filled train and validation and left the **test set
empty** — there was nothing to measure, and two tests failed with an empty
array rather than a wrong number. Twelve scaffolds of uneven size fixed it.
Worth knowing when writing any fixture for this splitter: equal-sized groups
are the pathological case, not the clean one.

## GNN determinism

*Instruction:* the GNN's numbers moved between the CPU and the GPU passes —
work out why, and if it can be fixed, fix it and re-run the grid so every
committed result comes from one version of the code.

**What it actually was.** §5b read the CPU-to-GPU movement as a device effect.
Asked to isolate the device, the agent repeated one Tox21 scaffold run three
times on the same card, the same wheel and the same seed: 0.7369 / 0.7413 /
0.7690. A spread of 0.032 against a seed SD of 0.011, with nothing changed
between the runs, and the tuning selected a different winning configuration
each time. The CPU-to-GPU deltas the report had been quoting were −0.002 and
−0.001 — well inside the noise of the measurement that produced them. That
comparison had never been measuring the device.

The cause is the scatter-add in message passing: on a GPU the order the
partial sums land in is not fixed between runs, float addition is not
associative, and cuBLAS picks kernels the same opportunistic way. The fix is
three lines applied before any run (`CUBLAS_WORKSPACE_CONFIG`,
`cudnn.benchmark = False`, `use_deterministic_algorithms`), with
`ADMET_NONDETERMINISTIC=1` to opt out and a `deterministic` field in every
result file, so the provenance is recorded rather than assumed.

**Re-running the grid was the expensive part, and half of it was wasted.** All
146 runs were redone with `--overwrite` so that every committed result comes
from one commit. Only the 68 GNN runs could change. The 68 baseline runs and
the 10 dummy runs came back with every test metric bit-identical — which is
the control doing its job, but it is also 78 runs of machine time spent to add
a provenance field to results that were already reproducible. Scoping the
re-run to the GNN would have been the better call, and the agent did not make
it or flag the cost before starting.

**Two other things it got wrong on the way.** It estimated 2.5–3 h for the
re-run and it took 4.18 h, because deterministic kernels cost the Tox21 GNN
3.0× its runtime (221 s → 673 s per run at the full training set) and that was
not priced in. And twice it reported the GPU as idle from `nvidia-smi` without
first checking which configuration was running — both times the grid was in a
scikit-learn phase, which is CPU by design.

**Verification.** `tox21_gnn_scaffold` seed 0 at 100% was re-run into a
separate results directory and compared field by field against the committed
file: test ROC-AUC identical to all 17 significant digits, same PR-AUC, same
selected hyperparameters, same best epoch, same eight tuning-trial scores,
only the wall-clock fields differing. A claim of reproducibility is worth
nothing without that check, and the check costs one run.

**What it changed in the findings: nothing load-bearing.** Every GNN number in
the README and the report was updated from the regenerated tables, and all 38
quoted values were then re-checked against `results/*.csv` by script rather
than by eye. The Tox21 GNN means moved by at most 0.008 ROC-AUC at the full
training set and 0.019 along the learning curve. Two smaller things did move:
the PR-AUC learning curve is now monotone under both splits, where the two
earlier passes each had it monotone under one split and disagreed about which;
and the 100% scaffold gap changed sign, −0.005 to +0.003. Both had always sat
inside one SD of zero. That is the point of the exercise rather than a
complication of it — the same lesson the three-seed finding teaches from the
other direction.

194 tests pass.

## Figure review, and configs by dataset

*Instruction:* check every figure, and move the config files into a directory
per dataset.

**Fourteen figures, all sound.** Axes, legends, error bars and the dummy floor
all render correctly, including the case that used to break — a dummy landing
exactly on the ROC axis floor of 0.500. Two things are worth acting on and are
written up in TODO §9: the ClinTox bar charts carry no class-prior floor,
because the dummy was only ever run on Tox21, which makes ClinTox's 0.855 look
comparable to Tox21's 0.834 with nothing to calibrate either; and the ClinTox
similarity histogram spreads ~146 test molecules over the same fixed bin count
as Tox21's ~783, so its spikes are three or four molecules each and mean
nothing. Its medians are fine.

**One figure turned out to carry an argument the report was not making.**
Reading the two similarity histograms side by side: the scaffold split cuts
ClinTox's median test-to-train similarity by 0.190 (36%) against Tox21's 0.164
(28%) — a *larger* structural shift — while ClinTox's baseline loses 0.021
ROC-AUC against Tox21's 0.073. A score carried by structure cannot lose less
accuracy to a bigger structural shift. That is now in §4 as the strongest form
of the provenance argument, which until then rested on the duplicate labels
and on a drop that was merely small.

**Configs now live in `configs/<dataset>/`**, with `base.yaml` shared at the
top and the dataset prefix dropped from each filename, since the directory
carries it. `extends:` became `../base.yaml`, the sweep's child list became
bare siblings, and the Makefile, the CLI's error text, two docstrings, two
tests and the README tree were repointed. `run_name()` is built from the
config's *contents*, not its path, so every one of the 146 result filenames is
unchanged — checked by loading all fourteen configs and comparing. A missing
config now prints what is actually available, since a bare `gnn_scaffold` is
ambiguous between two datasets and is deliberately not guessed.

**The move exposed a trap that was already there.** Because `run_name()` uses
dataset, model and split but not the featurization,
`configs/tox21/rf_maccs_random.yaml` and `configs/tox21/rf_random.yaml`
generate the *same* result filename. Running the MACCS comparison would either
silently skip as "already done" or, with `--overwrite`, destroy ten committed
baseline results. Recorded as a warning in TODO §5 rather than fixed: the fix
renames every result file.

194 tests pass.

## `n_valid`, and a TODO section that had stopped being true

*Instruction:* §9 of `TODO.md` claims to be done — is it?

Not quite, and part of that was the agent's doing. The section opened with
"four of the five are done", which was accurate until two figure items were
appended to it earlier the same day without touching the header, leaving a
list of seven under a count of five. Rewritten so status sits on each item
instead of in a paragraph that goes stale the moment anyone adds a bullet.

Checking each of the five rather than trusting the header: `admet --version`,
the duplicate-conflict warning and `make figures-clean` are genuinely done.
`CITATION.cff` exists but still contains two literal `TODO:` placeholders at
lines 40 and 48, so the citation GitHub currently offers for this project is
wrong, and no code change can fix it. `n_valid` was the one real remainder.

**`n_valid` is now done, and it needed no re-run** — which is what separated
it from everything open in §10. `split_sizes` was already in all 146 result
files; only `RunRecord` dropped the middle value. One field, one line to read
it, and the header follows the dataclass automatically.

It paid for itself on the first regeneration. The column shows validation and
test pinned at 782 and 783 across all four training fractions, which is the
learning-curve design's central claim — that only the *training* set varies
— and until now that claim could not be checked from the committed tables at
all, only from the individual run files.

The first test written for it would have passed against a wrong
implementation: the shared fixture gives validation and test the same size, so
an assertion could not tell `sizes["valid"]` from `sizes["test"]`. Rewritten
with three distinct sizes, and both tests were then run against two mutants
— one reading the wrong key, one dropping the column — and both mutants were
caught.

196 tests pass.

## Four more figures (task 1, ideas b-e)

*Instruction:* §1 of `TODO.md` — do it.

Four figures, each answering something the existing three could not.

**The per-assay heatmap** is the one that changed a sentence in the report.
§3 reports a macro-average over twelve Tox21 assays, and the heatmap is what
that average is averaging: SR-MMP reaches 0.90 under a random split where
NR-ER manages 0.71, and the scaffold penalty is not uniform either — SR-ATAD5
loses 0.16 and NR-ER loses nothing at all. The first version included the
dummy, which scores exactly 0.500 on every assay by construction; that put
two dead columns in the figure *and* pulled the colour scale's floor down to
0.5, squashing every real score into the top half of the ramp. Looking at the
PNG is what caught it, not a test.

**The paired per-seed scatter** makes the null result visible: baseline on x,
GNN on y, one point per seed, line of equality. Points on both sides of the
diagonal are what "no detectable difference" looks like, where the tables say
it as ±0.019 around +0.005.

**The all-gaps figure** is the only one allowed to put both datasets on one
axis, because it plots differences rather than scores — +0.01 means the same
thing on either, where a raw 0.85 does not. Fourteen of its sixteen intervals
cross zero.

**The dataset overview** has no score in it at all. On ClinTox it shows in one
bar what §4 argues in prose: `FDA_APPROVED` is positive for 1376 of 1461
molecules, which is why only `CT_TOX` is scored.

**What the tests caught, and what they did not.** Sixteen tests were written
and then run against six deliberate mutants. Five were caught. The sixth
survived: deleting `set_aspect("equal")` from the scatter left the test
passing, because it only checked that the two axes had the same *range*. On a
non-square box the line of equality is no longer at 45° and every point is
being judged against the wrong line — the exact class of defect this file
exists to catch. The assertion now checks the aspect too.

Three defects were found by looking at the PNGs rather than by any test: the
dummy columns above, a legend sitting on top of a panel title on ClinTox
(short figure, two assays, so the legend had nowhere to go — it is in the
figure footer now), and a molecule-size axis stretched to 136 heavy atoms by
seven outliers, where the cut is now made explicit on the figure.

While in the file, the ClinTox similarity histogram's bin count was fixed to
follow the test-set size: 146 molecules over Tox21's 40 bins meant every bar
stood on three or four molecules, and the leftward shift the figure exists to
show was buried in sampling noise.

211 tests pass. 24 figures.

## Closing §1 and §9

*Instruction:* §1 and §9 of `TODO.md` are not marked done and are not updated.

Correct on both counts, and the headings were the clearer complaint: every
other finished section in that file carries "— done" on its `##` line, and
these two did not, so the file read as though nothing in them had landed.

§1 was finished and simply unmarked. §9 was not: the ClinTox dummy floor was
still open, so marking it would have been a lie. It is now run — two configs,
ten runs, 35 s, no existing result touched. The ROC floor comes out at 0.500
as it must; the PR floor is 0.070 random and 0.096 scaffold, which is the
useful half, because the baseline's 0.494 reads as weak until the 0.070 next
to it says what knowing nothing scores. §4 of the report now makes that
comparison.

Running it exposed something small: **no Makefile target ever ran the dummy.**
The Tox21 dummy runs had only ever been launched by hand, which means
`make clean-results` could delete ten committed results that `make` had no way
to rebuild. There is a `make dummy` target now, covering both datasets.

Counts updated where the ten new files moved them: 146 → 156 result files,
4.18 → 4.19 h. `gap.csv` is unchanged, which is the right answer — the gap is
GNN-minus-baseline and the dummy is in neither arm. Only the two ClinTox bar
charts were redrawn, and a second regeneration is byte-identical.

Both headings now say what the sections are. §9 keeps one item that no code
change can close: `CITATION.cff` still has two literal `TODO:` placeholders,
so the citation GitHub offers for this project is wrong until a human types
two names in.

211 tests pass. 156 results, 24 figures.

## §3's instructions had gone stale

*Instruction:* the instructions in §3 of `TODO.md` look deprecated — do not
do the task, update them.

They were. The task itself is unchanged and still open; what had drifted was
every step after "add a `DatasetSpec`", because four things landed underneath
it. Following the old text would have produced a dataset that was half-wired
into the project:

- configs now live in `configs/<dataset>/` and inherit with
  `extends: ../base.yaml`, so "copy the four `configs/clintox_*.yaml` files"
  named files that no longer exist;
- there are now **six** files to copy, not four — the dummy pair is part of
  a dataset's setup, and a dataset without a floor repeats exactly the defect
  ClinTox had for a day;
- nothing said to add a Makefile target, which is how the Tox21 dummy ended
  up runnable only by hand;
- nothing said the runs have to come from the pinned deterministic path.

The cost guidance was the most misleading part, because it was qualitative.
"Slow" and "the largest by far" were written when a Tox21 GNN run took a
third of what it takes now, so the section now carries the one measurement
worth scaling from — 11.2 minutes per run, 3.9 h for the Tox21 grid — and
says HIV's GNN arm is an overnight job rather than an afternoon.

Two things the new figures imply were added as well: the per-assay heatmap
returns `None` below two assays, so BBBP, HIV and BACE will not produce one
and that is correct rather than broken, while ToxCast *would* produce a
617-row heatmap that nobody can read. And `all_gaps` is the figure that grows
with each dataset — 16 rows now, 24 with a third.

**One claim in the rewrite was wrong on first pass and had to be checked out
of it.** It said `admet datasets` reports `n_unparseable` and
`n_label_conflicts`. It does not: it prints molecules, assays and label
density, and the conflict count only reaches you as a warning. The numbers
live in `Dataset.summary()`, which is what `make data` prints. The corrected
step names both commands and says which gives what.

All fourteen factual claims in the rewritten section were then checked
against the repository by script rather than by memory — config counts,
`extends` paths, the 11.2-minute and 3.9-hour figures, the 16 gap rows, the
24 figures, which spec field each dataset uses, and the heatmap's guard.

## The CPU/GPU comparison comes out

*Instruction:* the CPU-versus-GPU comparison was never the point of the
experiment — it was my mistake. Remove the comparisons and report the GPU
time only.

Agreed, and it makes §5b stronger rather than weaker. Running the grid on two
devices was never in the design, and presenting it as a result is what let a
confound pose as a finding: the "device effect" of −0.002 and −0.001 was an
order of magnitude smaller than the GNN's own run-to-run spread, so those
numbers measured nothing. The determinism finding never needed the CPU pass —
it rests entirely on repeating one run three times on the same card.

§5b is now **"Determinism: what repeats, and what it costs"** and says so in
one line: everything was produced on one GPU, and the grid was run twice on
it, before and after the kernels were pinned. Removed from README and REPORT:
the three-row speed table, the 2.5× and 3.83 h figures, the two torch wheels,
the "three passes" framing, and the README subsection titled *CPU and GPU*,
now *Running on a GPU*.

**What was kept, and the reason, in case it should also go.** The 221 s →
673 s figure stays. It is not a device comparison — same card, same wheel,
only the determinism flag differing — and it is what explains why the grid
takes 4.19 h instead of 2.03 h. Dropping it would leave the 3× cost claim
unsupported.

**The historical entries above were left alone.** They describe a CPU pass
that did happen, and this file is the record of how the work went rather than
a claim about the experiment — a wrong turn belongs in it. The reports are
where the project states what it found, and the comparison is gone from both.

