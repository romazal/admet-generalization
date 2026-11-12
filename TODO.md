# TODO — things you can contribute

Self-contained additions to the codebase. Tasks 1–9 are each small enough to
finish in one sitting with a free AI agent (ChatGPT, Claude, Gemini, Copilot —
any free tier), and each ends in a commit that stands on its own. After them
there is **one larger milestone**, for anyone who wants something with more in
it or wants to work with someone else.

Pick whatever interests you. Most of these cannot break the experiment: the
results are already committed, and they do not touch the splitter, the
masking or the metrics. The two exceptions are task 8 and the milestone, which
both rewrite how data is read — each says what must not change.

---

## Before you start

**Set up once:**

```bash
git clone <repo-url> && cd admet-generalization
python -m venv .venv
source .venv/bin/activate        # .venv\Scripts\activate on Windows
pip install -e ".[dev]"
make test                        # must print 192 passed
```

**Three rules for working with a free agent:**

1. **Give it the real file.** Paste the actual contents of the file you are
   changing, and of the closest existing example, into the chat. The agent
   cannot see this repository and will otherwise invent an API that does not
   exist. Every task below names the file to copy from.
2. **`make test` must pass before you push.** If it does not, paste the whole
   error back to the agent. Never push red tests.
3. **Add a test for what you added.** Every task below says what to test. An
   addition without a test will be asked for one in review.

**Then:**

```bash
git checkout -b your-name/short-description
# ... make the change ...
make test
git add -A
git commit -m "feat(cli): add a datasets command"
git push -u origin your-name/short-description
```

---

## 1. More ways of showing the data and the results

We currently draw three figures per dataset per metric — a bar chart, a
learning curve and a gap plot — and they carry the argument adequately but
narrowly. Every one of them plots a *summary number*. Nothing shows the
underlying distributions, nothing shows the twelve Tox21 assays separately,
and nothing shows *why* the scaffold split is harder. There is a lot of room
here, and figures are what the audience actually remembers.

Pick one of the ideas below — or your own. Each is a standalone figure.

### a. How similar are the test molecules to the training molecules? — done

*Done 2026-11-11. `plot_train_test_similarity` in `plots.py`, drawn by
`make plots` into `figures/<dataset>/train_test_similarity_<dataset>.png`.
Tox21 median similarity falls 0.583 → 0.419 from random to scaffold. Ideas b
and c below are still open.*

This is the mechanism behind the headline result, and we assert it in prose
without ever showing it. For every test molecule, compute its **maximum
Tanimoto similarity to any training molecule**, then draw the distribution
once for a random split and once for a scaffold split, on the same axis.

The scaffold curve should sit visibly to the left. That single figure
explains the 0.073 ROC-AUC drop better than the sentence currently does.

Everything needed exists: `make_split("random", ...)` and
`make_split("scaffold", ...)` in `src/admet/data/splits.py` give you the
index partitions, and `morgan_fingerprint()` in
`src/admet/features/fingerprints.py` returns a `uint8` 0/1 array. Tanimoto
between two bit arrays is `(a & b).sum() / (a | b).sum()`. Do it on ClinTox
first — 1461 molecules is a small enough matrix to compute naively.

### b. Per-assay heatmap

Twelve Tox21 assays collapsed into one macro-average hides which endpoints
are predictable. A heatmap of assay (rows) against model-and-split (columns),
coloured by ROC-AUC, shows the whole picture at once.

The numbers are already in every result file under
`metrics.test.per_task` — nothing needs recomputing. Pairs naturally with
task 2, but can be done alone by reading the JSON directly.

Use a **sequential** single-hue scale for this, light to dark. Not a rainbow,
and not the two model colours — those mean "which model", and reusing them
for magnitude would be misleading.

### c. Paired per-seed scatter

Every gap we report is a difference of means with an SD attached, which is
honest but hard to feel. Plot one point per seed: baseline ROC-AUC on x, GNN
ROC-AUC on y, with a 45° line. Points scattered on both sides of the diagonal
*are* the "no detectable difference" conclusion, made visual.

Colour by split, use the existing `SPLIT_STYLES`, and read the rows from
`results/runs.csv`.

### d. A single figure for all the gaps

One horizontal plot: every (dataset, split, training fraction) gap as a point
with an error bar of ±1 SD, and a vertical line at zero. A reader sees in one
glance that nearly every interval crosses zero. This is the whole H1/H2 story
in one image, and it is the figure most likely to end up on a slide.

### e. Dataset characterisation

Not results — the data itself. Positives per assay for Tox21 (they range from
186 to 942), label density, molecule-size distributions. Useful for the part
of the lecture that explains what Tox21 *is* before any model appears.

### How, for any of them

Add a `plot_*()` function to `src/admet/analysis/plots.py`, copying the shape
of `plot_split_comparison()`, and call it from `make_all_figures()` so
`make plots` picks it up. Write to the per-dataset subfolder the other
figures use: `figures/<dataset>/<name>_<dataset>_<metric>.png`.

**Follow the conventions already in that file**, which exist for a reason:

- `MODEL_COLORS` — model is colour, always the same colour. Never recolour a
  model between figures, and never reuse a model's hue to mean something
  else: the gap plot did exactly that, so blue meant "random forest" in two
  figures and "random split" in a third.
- Split is position, not colour — `_new_panels()` facets a figure into a
  random-split and a scaffold-split panel on a shared y-axis. Prefer that to
  adding a third and fourth series to one axis.
- `_style_axes()`, `_seed_note()` and `_save()` — use them rather than
  styling by hand, so the new figure matches the other twelve and says what
  its error bars are.

**Test:** add to `tests/test_plots.py`. Read what is there first — the tests
in it exist because a hardcoded `ax.set_ylim(0.5, 1.0)` once pushed every
PR-AUC bar off the top of its own axis and produced six blank charts that
nobody noticed for days. Matplotlib will not warn you. So: assert your figure
has the expected number of drawn artists, and assert every value you plot
falls inside the axis limits.

**Then look at the PNG.** A passing test does not mean a readable figure.

---

## 2. Per-assay results, not just the average — done

*Done by Jollly23 (`Jollly23`, merged 2026-11-11). `per_task_table()` in
`aggregate.py`, written to `results/per_task.csv` by `make aggregate`. The
original task text is kept below as the record.*

Every score we report is macro-averaged over the 12 Tox21 assays, so a single
number hides which toxicity endpoints are easy and which are impossible.
"SR-MMP is predictable, NR-PPAR-gamma is not" is a genuinely interesting
finding and nobody has looked.

The data already exists — `evaluation.py` computes per-assay `TaskScore`
objects and `training.py` writes them into every result file. Nothing
recomputes; this is purely surfacing what is already there.

**Files:** add a `per_task_table()` to `src/admet/analysis/aggregate.py`
(copy `summarize()`), writing `results/per_task.csv` with one row per
(dataset, model, split, assay).

**Test:** extend `tests/test_evaluation.py` — build a two-assay case where
one assay is perfectly predicted and the other is at chance, and assert the
per-assay table separates them while the macro average hides it.

**Expect to find:** the assays with the fewest positives (NR-PPAR-gamma has
186) are probably the worst. If so, that is a data-size story inside a single
dataset, and it belongs in REPORT.md.

---

## 3. Add a third dataset

Adding a dataset is about fifteen lines, because `loaders.py` is a registry:
you describe the file and everything downstream — splitting, masking, tuning,
metrics, figures — works unchanged. H1 and H2 currently rest on one real
dataset (ClinTox was too small to resolve anything), so a third is the
cheapest way to make the conclusion more than a Tox21 anecdote.

### Hard constraint, read first

This pipeline does **masked multitask binary classification**. A dataset
whose labels are continuous numbers will not work without new code — so ESOL,
FreeSolv, Lipophilicity and QM7/QM9 are all **out**, however tempting their
ADMET relevance. Labels must be 0/1, with blanks for unknown.

### Candidates

All five filenames below were checked against the MoleculeNet mirror and
exist. The counts are what MoleculeNet documents; treat them as *expected*,
not verified — the loader prints what it actually parsed, and that is the
number to write down.

| Dataset | File | Molecules | Tasks | Why it is worth adding |
|---|---|---|---|---|
| **BBBP** | `BBBP.csv` | 2050 | 1 | Blood–brain-barrier penetration — the **D** in ADMET, and the proposal's own named example (§3.1). Small and fast. |
| **SIDER** | `sider.csv.gz` | ~1427 | 27 | Marketed drugs against 27 side-effect classes. The only candidate that genuinely exercises the multitask head the way Tox21 does. |
| **ToxCast** | `toxcast_data.csv.gz` | ~8575 | 617 | The real stress test for the masked loss — the label matrix is mostly empty. Slow; try one model first. |
| **HIV** | `HIV.csv` | ~41k | 1 | Large and very imbalanced (~3.5% positives), so PR-AUC finally matters more than ROC-AUC. The largest of these by far. |
| **BACE** | `bace.csv` | ~1513 | 1 | Clean and small, but it is target binding, not ADMET. Only interesting as a contrast. |

**BBBP is the natural first choice.** It is the dataset the proposal itself
named, it is an ADMET endpoint, it is quick to run, and it has a documented
quirk worth reporting — a number of its SMILES do not parse, which our loader
already counts and drops. **SIDER** is the one to take if you want the
multitask story.

### How

Add one `DatasetSpec` to `DATASETS` in `src/admet/data/loaders.py`, copying
the `clintox` entry. The base URL `_MOLECULENET` is already defined, so
`url=f"{_MOLECULENET}/BBBP.csv"`. The fields you need are `name`, `url`,
`filename`, `smiles_column` and `task_columns`.

**Get the column names right** — they differ per file, and guessing wastes an
hour. Download it once and look:

```bash
curl -sL https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv | head -3
```

For BBBP this returns `num,name,p_np,smiles` — so `smiles_column="smiles"`,
`task_columns=("p_np",)`, and `num`/`name` are simply left out. (Checked:
that is the real header, and the file has 2050 rows.) Then copy the four
`configs/clintox/*.yaml` files, change `dataset.name`, and drop the `tasks:`
line if you want every label column.

**Test:** add a case to `tests/test_loaders.py` (task 7) or
`tests/test_aggregate.py` asserting the new spec is in `DATASETS` and that
its `task_columns` do not include the SMILES column — a real and easy mistake
that silently trains the model to predict its own input.

**Then run it:** `admet run configs/bbbp/rf_scaffold.yaml`. Report what you
find even if it is dull. A third dataset where the scaffold split costs
~0.08 ROC-AUC would be the strongest single result in the project; one where
it does not is more interesting still.

**Watch for:** a suspiciously high score. BBBP is known to be easy, and
REPORT §4 already explains why that reads as a dataset problem rather than a
model success.

---

## 4. A `dummy` baseline — the floor every score should beat — done and run

*Done by Jollly23 (`Jollly23`, merged 2026-11-11). `src/admet/models/dummy.py`
plus `configs/tox21/dummy_random.yaml` and `configs/tox21/dummy_scaffold.yaml`.
Run on 2026-11-11, 5 seeds x 2 splits: ROC-AUC 0.500 both splits, PR-AUC 0.077
random and 0.100 scaffold. The floor is in README and REPORT §3. The original
task text is kept below as the record.*

Right now we compare a GNN against a random forest. Neither is compared
against *nothing*. A dummy model that ignores the molecule and predicts the
training-set positive rate gives the floor — ROC-AUC 0.5 by construction, and
a PR-AUC equal to the class balance. That last number is the one we actually
need: our Tox21 PR-AUC of 0.47 means little until you can see what a model
that knows nothing scores.

**Files:** new `src/admet/models/dummy.py`. Copy `src/admet/models/rf.py`
wholesale and strip it down — it already does the per-assay masked fitting
you need, and you keep the `@register_model` decorator, the `name` attribute
and the same four methods. Use scikit-learn's
`DummyClassifier(strategy="prior")`. `search_space()` returns `{}` — there is
nothing to tune, which is correct and worth a comment. Then add
`from admet.models import dummy as _dummy` to the imports inside
`build_model` in `src/admet/models/base.py`, or it will never be registered.

Add `configs/tox21/dummy_random.yaml` and `configs/tox21/dummy_scaffold.yaml`
by copying the `rf` ones and changing `model.name`.

**Test:** in a new `tests/test_models.py`, assert the dummy's ROC-AUC on a toy
set is ≈0.5 and that it is registered in `MODELS`.

**Then run it** and add the row to REPORT.md §3. A PR-AUC floor next to our
0.47 makes that number mean something.

---

## 5. MACCS keys as a second fingerprint — done, with one change

*Done by Lukas (`feature/maccs-fingerprints`, merged 2026-11-11).
`maccs_fingerprint()` in `fingerprints.py`, `fp_kind` on `RandomForestModel`,
and the length-167 and benzene-vs-cyclohexane tests, all as written below.*

***One instruction here was wrong and was not followed.** This task said to put
`fp_kind` in `search_space()`. Doing so changes all eight sampled
configurations — `sample_configs` sorts parameter names, so inserting one
shifts every later parameter in the draw — and the 68 committed baseline runs
would stop reproducing, taking the control argument in REPORT §5b with them.
It would also spend part of the baseline's budget on a second representation
while the GNN keeps all of its own on one. `fp_kind` is therefore selected
per-run through `model.params` instead, with the two
`configs/tox21/rf_maccs_*.yaml` configs; two tests pin this. The original task
text is kept below as the record.*

***Before anyone runs the MACCS comparison, read this.** `run_name()` is built
from dataset, model, split, fraction and seed — not from the config file or
from `fp_kind` — so `configs/tox21/rf_maccs_random.yaml` and
`configs/tox21/rf_random.yaml` produce **the same result filename**,
`tox21_rf_random_frac100_seed0.json`, which is a committed baseline result.
Without `--overwrite` the MACCS run silently skips as "already done"; with it,
the run **destroys ten committed baseline results**. Run it with
`--results-dir` pointing somewhere outside `results/` and compare by hand, or
give `run_name()` a way to distinguish the featurization first — which renames
every result file and is Roman's call, not a task to pick up unasked.*

The baseline only knows Morgan/ECFP fingerprints. MACCS keys are 166
predefined substructure patterns — chemically interpretable in a way Morgan
bits are not, which makes them a nice contrast for a biomedical audience.

**Files:** add a `maccs_fingerprint()` function to
`src/admet/features/fingerprints.py` next to `morgan_fingerprint()`, using
RDKit's `rdMolDescriptors.GetMACCSKeysFingerprint`. Then let
`RandomForestModel` choose between them — add an `fp_kind` entry to its
`search_space()` in `src/admet/models/rf.py` and include it in
`featurization_key()`, **which matters**: that key is what stops two different
featurizations sharing a cache.

**Test:** in `tests/test_features.py`, assert the MACCS vector is length 167
(RDKit indexes from 1, so index 0 is unused — a classic trap) and that
benzene and cyclohexane give different vectors.

---

## 6. `admet datasets` — list what is available — done

*Done 2026-11-11. `command_datasets` in `cli.py`, plus `tests/test_cli.py` —
cli.py had been the last module with no test file of its own. The original
task text is kept below as the record.*

There is no way to ask the tool which datasets exist without reading the
source. Add a command that prints the registry with the molecule count, the
assays and the label density.

```
$ admet datasets
tox21     7823 molecules  12 assays  83% labelled
clintox   1461 molecules   2 assays  100% labelled
```

**Files:** add `command_datasets` to `src/admet/cli.py` and wire it into
`build_parser`. Copy the shape of `command_aggregate`, which is the simplest
existing command. The data comes from `DATASETS` and `load_dataset` in
`src/admet/data/loaders.py`; `Dataset.summary()` already returns everything
you need.

**Test:** a new `tests/test_cli.py` asserting `main(["datasets"])` returns 0
and that both dataset names appear in the output (use pytest's `capsys`).

---

## 7. A `tests/test_loaders.py` — done

*Done by Cas (`Chunky153/task-7-offline-loader-tests`, merged 2026-11-11).
Four offline tests in `tests/test_loaders.py`; `loaders.py` is no longer the
module without one. The original task text is kept below as the record.*

`loaders.py` was the only core module with no test file of its own, and it does
the cleaning every result depends on — canonicalisation, dropping unparseable
SMILES, merging duplicates.

**Files:** new `tests/test_loaders.py`. Copy the structure of
`tests/test_features.py`.

**Test ideas** (all offline, no download — build tiny CSVs with pytest's
`tmp_path`):

- an invalid SMILES string is dropped and counted in `Dataset.n_unparseable`
  (the same figure appears as `n_unparseable_dropped` in `summary()`);
- the same molecule written two ways (`CCO` and `OCC`) is merged into one row;
- a missing label stays missing and is **not** read as a zero — this is the
  one that protects the whole masking argument;
- `label_density` matches a hand-computed value.

**Verify it is not vacuous:** make the test fail on purpose first, by
asserting the wrong number, and confirm it really fails. A test that cannot
fail is worse than none.

---

## 8. Put pandas back — done

*Done 2026-11-11. The blocker had lifted: pandas 3.1.1 imports and round-trips
cleanly, checked over ten separate processes. `read_csv()` in `loaders.py` and
`write_csv()` in `aggregate.py` now use pandas; it is a **hard** dependency. All
four committed tables still regenerate byte-identically. The original task text
is kept below as the record.*

Proposal §5 lists pandas as a dependency. We do not use it: the loader and the
aggregation use the standard library's `csv` plus NumPy, and this is written
up as a deviation in the README, REPORT §8, `AGENT_LOG.md` and a comment in
`pyproject.toml`.

**It was never a design decision.** pandas' compiled extensions were being
blocked by the Windows Application Control policy on the development machine,
and inconsistently — two successive imports failed on *different* DLLs
(`offsets`, then `_ujson`). Building the pipeline on a dependency that fails
nondeterministically was not acceptable, so it was dropped. Restoring it
closes the only deviation that exists for an environment reason rather than a
considered one.

**The blocker may already be gone.** pandas 3.1.1 is installed in the current
`.venv` and imports cleanly, repeatedly. The environment was rebuilt when the
CUDA build of torch was installed, which is the likely explanation. Check
before you start, several times rather than once, since the original failure
was intermittent:

```bash
python -c "import pandas; print(pandas.__version__)"
```

If it fails on your machine, say so rather than working around it — that the
policy still blocks it is itself the thing worth knowing.

### Where it goes

Only two call sites, which is why this is tractable:

- `read_csv()` in `src/admet/data/loaders.py` — currently `csv.DictReader`.
- `write_csv()` in `src/admet/analysis/aggregate.py` — currently
  `csv.DictWriter` over dataclass rows.

`plots.py` and `training.py` do not touch CSV at all, so they stay as they
are.

### The constraint that makes this delicate

`results/runs.csv`, `summary.csv`, `gap.csv` and `per_task.csv` are
**committed**, and `make aggregate` on a fresh clone currently regenerates
all four with zero diff. That property is load-bearing — it is what lets anyone verify the
tables came from the run files — and a rewrite that changes one digit of
formatting breaks it silently, leaving a dirty working tree for everyone who
regenerates.

The good news is that this is achievable, and already checked: pandas
reproduces all three files **byte-identically** with

```python
df.to_csv(path, index=False, lineterminator="\r\n")
```

Float formatting matches, because pandas uses the same shortest-repr as
Python. The line terminator is the only trap — `csv.writer` opened with
`newline=""` writes `\r\n`, and pandas defaults to `os.linesep`.

On the reading side the risk is the opposite one: `read_csv` infers dtypes,
and a raw dataset CSV has label columns of `0`/`1`/blank and a SMILES column
of strings. Blanks becoming `NaN` happens to match our masking semantics, but
*happens to* is not good enough here — pass `dtype=str, keep_default_na=False`
and keep the existing `_to_label()` conversion, so the cleaning behaves
exactly as it does now. Changing how a missing label is parsed would
invalidate every committed result.

### Decide before you write code

Whether pandas becomes a hard dependency or an optional one. A hard dependency
matches the proposal, but it locks out anyone whose machine still blocks the
DLLs. An optional import with the `csv` path as fallback keeps everyone
running at the cost of two code paths. Either is defensible; pick one and say
why in the commit.

### Then

- **Test:** assert the three CSVs regenerate byte-identically. Compare
  `read_bytes()`, not text — reading as text normalises line endings and will
  tell you the files match when they do not.
- Remove the deviation from `README.md`, `REPORT.md` §8 and the `pyproject.toml`
  comment, and add an `AGENT_LOG.md` entry recording that the blocker lifted.
  Four places claim we do not use pandas; all four have to stop saying it.

---

## 9. Smaller things

*Four of the five are done (2026-11-11): `admet --version`, `CITATION.cff`
(two names still need filling in by a human), the duplicate-conflict warning,
and `make figures-clean` — which turned out to also need `clean-results`
fixed, since its globs missed the per-dataset subdirectories and it had never
deleted anything. `n_valid` in the result tables is the one left.*

- **`admet --version`.** Read `__version__` from `src/admet/__init__.py`. One
  line in `build_parser`.
- **`CITATION.cff`** in the repo root so the project can be cited; GitHub
  renders it automatically. All four names.
- **Warn when merged duplicates disagree.** The loader counts
  `n_label_conflicts` — 19 on ClinTox, 0 on Tox21 — but never says so out
  loud, so the single most suspicious fact about ClinTox is invisible unless
  you open a result file. Emit a warning at load time when it is non-zero.
- **Validation-set size in the result files.** `split_sizes` has train, valid
  and test, but `aggregate.py`'s `RunRecord` only keeps `n_train` and
  `n_test`. Add `n_valid`.
- **A `make figures-clean` target** that deletes only `figures/` and redraws
  it.
- **No dummy floor on the ClinTox charts.** The Tox21 bar charts carry a
  class-prior bar; the ClinTox ones do not, because the dummy was only run on
  Tox21. Side by side, ClinTox's 0.855 reads as comparable to Tox21's 0.834
  with nothing to say what chance looks like on either. Ten ClinTox dummy runs
  cost about 30 s and would make the pair symmetric. Adding runs is allowed;
  this touches nothing existing.
- **The ClinTox similarity histogram is mostly sampling noise.** Its ~146 test
  molecules spread over the same fixed bin count as Tox21's ~783, so every bar
  stands on three or four molecules and the spikes mean nothing. Scale the bin
  count with the test-set size, or state `n` on the figure. The medians in the
  legend (0.52 random, 0.33 scaffold) are the part worth reading.

## 10. Make the GNN reproducible — done (2026-11-12)

*Done. The GNN was not reproducible on the GPU: repeating one identical run
moved its test ROC-AUC by 0.032 — three times the seed-to-seed SD — and
changed which hyperparameter configuration the tuning selected.
`src/admet/training.py` now pins deterministic kernels before every run, every
result file records a `deterministic` field, and all 146 results were
regenerated from one commit so the grid has a single provenance. It costs the
Tox21 GNN 3.0× its runtime. REPORT §5b has the account; two follow-ups fell
out of it and are below.*

- **`versions.torch_device` reports the wrong thing.** It records what torch
  could *see*, not what the model used, so it reads
  `cuda:NVIDIA GeForce RTX 5070 Ti` on the 68 baseline runs, which are
  scikit-learn and never touch the card. Recording the device the model chose
  would make the field mean what its name says. It is documented as a caveat
  in README and REPORT §5 for now, because fixing it properly means
  regenerating 146 results for a provenance field — not worth 4 h on its own,
  but worth folding into the next re-run that happens for another reason.
- **`versions` does not record pandas.** `_library_versions()` in
  `training.py` promises "everything whose behaviour could change a number"
  and lists numpy, sklearn, rdkit, torch and yaml. pandas parses the dataset
  CSV on every single run — it is the first thing that touches the data — and
  is not in the list, so no result file says which pandas produced it.
  `pyproject.toml` also allows `pandas>=2.2` with no ceiling, which spans a
  major version; pandas 3 changed string-dtype defaults, and `read_csv()`
  depends on `dtype=str, keep_default_na=False` behaving as it does today.
  One line to add to the tuple. Group it with the `torch_device` fix above so
  that a single re-run buys both.
- **Pin the CSVs in `.gitattributes`.** The four tables in `results/` are
  written with `lineterminator="\r\n"` and `core.autocrlf` is on here, so
  `make aggregate` gives a zero diff on this machine. On a Linux clone it
  would not. `results/*.csv -text` makes the zero-diff property survive the
  clone — which is the property README's *Reproducibility* section claims.

---

## A larger milestone: make the project dataset-agnostic

Everything above is an addition. This one is a change of shape, and it is
worth doing as a milestone rather than a task — probably by two people, over
several sittings, on a long-lived branch.

**The problem.** Right now a dataset is only usable if someone edits
`src/admet/data/loaders.py` and adds a `DatasetSpec` to the `DATASETS` dict
(this is what task 3 asks you to do once, by hand). That means the project
supports exactly the datasets we thought of. It also means nobody can point
the pipeline at a new file — a dataset from a paper, a newer MoleculeNet
release, an in-house CSV — without changing library code and reinstalling.

The goal is that **any appropriate dataset can be dropped in**, and that the
project can say for itself whether a given dataset *is* appropriate and how it
compares to the ones we already ran.

Three parts. They are independently useful, so a half-finished milestone still
leaves the repo better than it found it.

### Part A — load a dataset without editing the source

Let a config name a file directly instead of a registry key:

```yaml
dataset:
  path: data/my_dataset.csv        # or a URL
  smiles_column: smiles
  task_columns: [active]
```

`DatasetSpec` already has every field this needs (`name`, `url`, `filename`,
`smiles_column`, `task_columns`, `exclude_columns`, `description`) — the
registry is just a dict of them. So the work is mostly plumbing: let
`load_dataset()` accept a spec as well as a name, and let the config loader
build one from a `dataset:` block. Keep the named registry working exactly as
it does now, because every committed config and every result file depends on
it.

A good intermediate design: move the two built-in specs into
`datasets/*.yaml` files and have the registry read that directory at import.
Then adding a dataset is adding a YAML file, and task 3 becomes a two-minute
job instead of a code change.

### Part B — a function that inspects a dataset and says whether it will work

Given a raw CSV, before any training, report what it contains and what will go
wrong. Something like `admet inspect <path>`, printing:

- **Which column is the SMILES column**, guessed by trying to parse each
  candidate column with RDKit and taking the one with the highest parse rate.
  This is the single most common thing to get wrong by hand.
- **What kind of labels each column holds** — binary, continuous, or
  categorical. This matters because the pipeline does masked multitask
  *binary* classification, so a continuous column has to be refused with a
  clear message rather than silently cast to floats and trained on. Right now
  that constraint lives in a paragraph of this file; it should live in code.
- **Label density, positives per task, and class balance.** A task with three
  positives cannot be scored and should be flagged before someone waits for a
  run to find out.
- **Duplicates and conflicts** — the same structure appearing twice, and
  whether the copies disagree. ClinTox has 19 such groups and every one
  disagrees on both labels; that is the most important single fact about the
  dataset, and we found it by accident.
- **Scaffold structure** — how many Bemis-Murcko scaffolds, and **the size of
  the largest scaffold group as a fraction of the dataset**. This one is not
  cosmetic: Tox21 has a single group of 1775 acyclic molecules, which is what
  broke the first scaffold splitter and produced an empty validation set. A
  dataset whose largest group exceeds the smallest requested part is a dataset
  that will give you a degenerate split, and the inspector can say so in
  advance instead of after five seeds.

Most of the pieces exist: `murcko_scaffold()` and `scaffold_groups()` in
`src/admet/data/splits.py`, `Dataset.summary()`, `label_density` and
`positives_per_task()` in `loaders.py`. This is mostly assembly plus the
column-type inference, which is new.

### Part C — compare a dataset against the ones we have

This is the part that turns the inspector into something scientifically
useful. Given a new dataset, put its profile beside Tox21's and ClinTox's:

| | Tox21 | ClinTox | yours |
|---|---|---|---|
| Molecules | 7823 | 1461 | |
| Tasks | 12 | 1 scored | |
| Label density | 0.829 | 1.000 | |
| Positives per task | 186–942 | 103 | |
| Largest scaffold group | 1775 (23%) | | |
| Conflicting duplicates | 0 | 19 | |

Two comparisons worth computing on top of the table:

- **Structural overlap with the datasets already run** — how many canonical
  SMILES, and how many scaffolds, the new dataset shares with Tox21 and
  ClinTox. A new dataset that overlaps heavily is not independent evidence,
  and a result from it should not be reported as if it were.
- **Which existing dataset it resembles.** This is the question that actually
  matters when someone proposes a dataset: is this another Tox21 — large
  enough and hard enough to resolve a difference between two models — or
  another ClinTox, small enough that every gap will be smaller than its own
  SD? We learned that about ClinTox by running 80 jobs. A profile would have
  suggested it beforehand.

Keep the output honest: a profile is a prediction, not a result. It should
say "this dataset is ClinTox-sized, expect wide error bars", not "this
dataset will not work".

### Scope, and what must not change

- **In scope:** loading, inspecting, comparing, and the CLI around them.
- **Out of scope:** regression support. Making the pipeline handle continuous
  labels means a new loss, new metrics and new plots — a bigger change than
  this one, and a separate conversation. Part B should *detect* regression
  datasets and refuse them clearly; that is the whole job here.
- **Must not change:** the cleaning semantics in `loaders.py` —
  canonicalisation, dropping unparseable SMILES, merging duplicates, and
  missing labels staying missing. Every committed result was produced under
  those rules, so altering them silently invalidates the comparison between
  old and new runs. Refactor around them, not through them.

**Test as you go, not at the end.** Write `tests/test_loaders.py` (task 7)
first if it does not exist yet — this milestone rewrites the module that file
is supposed to be guarding, and doing it without tests in place is how the
cleaning semantics get changed by accident.

**Verify on a dataset we know.** Point the finished inspector at Tox21 and
ClinTox and check it reports the numbers already in REPORT.md §2. If it
disagrees with the table we published, one of the two is wrong and that is
worth finding out.

---

## What not to touch

These are where the science lives, and a plausible-looking agent edit can
invalidate every result without failing a test:

- `src/admet/data/splits.py` — the scaffold packing already had one bug that
  silently emptied the validation set.
- the masking in `src/admet/evaluation.py` and `masked_bce_loss` in
  `src/admet/models/gnn.py`.
- anything under `results/` — those files are the record of 4.2 h of compute,
  and every number in the report traces back to them. Adding new runs is fine;
  editing existing ones is not. If a task needs existing runs regenerated, say
  so and scope it: the determinism re-run (§10) redid all 146 when only the 68
  GNN runs could change, and the other 78 came back bit-identical.

If a task seems to need a change in one of those, ask Roman first.

---

## Working in parallel

The tasks above touch mostly separate files, so several people can work at
once without conflicts. The exceptions worth knowing:

- Tasks **1** and **2** both benefit from per-assay data — if one person takes
  each, agree first on who writes `per_task_table()`.
- Tasks **3** and **6** both touch `loaders.py`, though in different places.
- Task **5** edits `rf.py`, which task **4** copies from — copy before the
  other person's change lands, or rebase after.
- Task **8** rewrites the CSV handling in both `loaders.py` and
  `aggregate.py`, so it collides with tasks **3**, **6** and **7**. It is a
  small diff in two functions, so landing it early is easier than rebasing
  around it.

Everything else is independent. Branch from an up-to-date `master`, and pull
before you push.

**The milestone is the exception.** It rewrites `loaders.py`, which tasks
**3**, **6** and **7** all touch. If someone starts it, land those three
first — the milestone makes task 3 trivial and needs task 7's tests in place
before it begins, so the order is the useful one anyway.
