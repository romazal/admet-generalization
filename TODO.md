# TODO — things you can contribute

Self-contained additions to the codebase. Each is small enough to finish in
one sitting with a free AI agent (ChatGPT, Claude, Gemini, Copilot — any free
tier), and each ends in a commit that stands on its own.

Pick whatever interests you. None of these can break the experiment: the
results are already committed, and nothing here touches the splitter, the
masking or the metrics.

---

## Before you start

**Set up once:**

```bash
git clone <repo-url> && cd admet-generalization
python -m venv .venv
source .venv/bin/activate        # .venv\Scripts\activate on Windows
pip install -e ".[dev]"
make test                        # must print 148 passed
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

### a. How similar are the test molecules to the training molecules?

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
  model between figures.
- `SPLIT_STYLES` — split is line style, not colour.
- `_style_axes()` and `_save()` — use them rather than styling by hand, so
  the new figure matches the other twelve.

**Test:** add to `tests/test_plots.py`. Read what is there first — the tests
in it exist because a hardcoded `ax.set_ylim(0.5, 1.0)` once pushed every
PR-AUC bar off the top of its own axis and produced six blank charts that
nobody noticed for days. Matplotlib will not warn you. So: assert your figure
has the expected number of drawn artists, and assert every value you plot
falls inside the axis limits.

**Then look at the PNG.** A passing test does not mean a readable figure.

---

## 2. Per-assay results, not just the average

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
`configs/clintox_*.yaml` files, change `dataset.name`, and drop the `tasks:`
line if you want every label column.

**Test:** add a case to `tests/test_loaders.py` (task 7) or
`tests/test_aggregate.py` asserting the new spec is in `DATASETS` and that
its `task_columns` do not include the SMILES column — a real and easy mistake
that silently trains the model to predict its own input.

**Then run it:** `admet run configs/bbbp_rf_scaffold.yaml`. Report what you
find even if it is dull. A third dataset where the scaffold split costs
~0.08 ROC-AUC would be the strongest single result in the project; one where
it does not is more interesting still.

**Watch for:** a suspiciously high score. BBBP is known to be easy, and
REPORT §4 already explains why that reads as a dataset problem rather than a
model success.

---

## 4. A `dummy` baseline — the floor every score should beat

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

Add `configs/tox21_dummy_random.yaml` and `configs/tox21_dummy_scaffold.yaml`
by copying the `rf` ones and changing `model.name`.

**Test:** in a new `tests/test_models.py`, assert the dummy's ROC-AUC on a toy
set is ≈0.5 and that it is registered in `MODELS`.

**Then run it** and add the row to REPORT.md §3. A PR-AUC floor next to our
0.47 makes that number mean something.

---

## 5. MACCS keys as a second fingerprint

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

## 6. `admet datasets` — list what is available

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

## 7. A `tests/test_loaders.py`

`loaders.py` is the only core module with no test file of its own, and it does
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

## 8. Smaller things

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

---

## What not to touch

These are where the science lives, and a plausible-looking agent edit can
invalidate every result without failing a test:

- `src/admet/data/splits.py` — the scaffold packing already had one bug that
  silently emptied the validation set.
- the masking in `src/admet/evaluation.py` and `masked_bce_loss` in
  `src/admet/models/gnn.py`.
- anything under `results/` — those files are the record of 2 h of compute
  (3.8 h without a GPU), and every number in the report traces back to them.
  Adding new runs is fine; editing existing ones is not.

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

Everything else is independent. Branch from an up-to-date `master`, and pull
before you push.
