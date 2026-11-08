# TODO — small contributions

Self-contained additions to the codebase, each small enough to finish in one
sitting with a **free AI agent** (ChatGPT, Claude, Gemini, Copilot — any free
tier) and get merged. The point is that everyone has real commits in the
history, so pick one, do it, push it.

They are ordered easiest first. None of them can break the experiment: the
results are already committed and nothing here touches the splitter, the
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
3. **Add a test for what you added.** Every task below says what to test.
   An addition without a test will be asked for one in review.

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

## 1. `admet datasets` — list what is available

**Difficulty:** ★☆☆☆☆ · **~1 h** · a good first commit

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

## 2. A `dummy` baseline — the floor every score should beat

**Difficulty:** ★★☆☆☆ · **~2 h** · the most scientifically useful item here

Right now we compare a GNN against a random forest. Neither is compared
against *nothing*. A dummy model that ignores the molecule and predicts the
training-set positive rate gives the floor — ROC-AUC 0.5 by construction, and
a PR-AUC equal to the class balance. That last number is the one we actually
need: our Tox21 PR-AUC of 0.47 means little until you can see what a model
that knows nothing scores.

**Files:** new `src/admet/models/dummy.py`. Copy `src/admet/models/rf.py`
wholesale and strip it down — it already does the per-assay masked fitting
you need, and you keep the `@register_model` decorator, the `name` attribute
and the same four methods. Use scikit-learn's `DummyClassifier(strategy=
"prior")`. `search_space()` returns `{}` — there is nothing to tune, which is
correct and worth a comment. Then add
`from admet.models import dummy as _dummy` to the imports inside
`build_model` in `src/admet/models/base.py`, or it will never be registered.

Add `configs/tox21_dummy_random.yaml` and `configs/tox21_dummy_scaffold.yaml`
by copying the `rf` ones and changing `model.name`.

**Test:** in `tests/test_models.py` (new), assert the dummy's ROC-AUC on a
toy set is ≈0.5 and that it is registered in `MODELS`.

**Bonus:** run it (`admet run configs/tox21_dummy_random.yaml`, a few seconds)
and add the row to REPORT.md §3. A PR-AUC floor next to our 0.47 makes that
number mean something.

---

## 3. Add a third dataset

**Difficulty:** ★★☆☆☆ · **~2 h** · the best value for the effort

Adding a dataset is about fifteen lines, because `loaders.py` is a registry:
you describe the file and everything downstream — splitting, masking,
tuning, metrics, figures — works unchanged. H1 and H2 currently rest on one
real dataset (ClinTox was too small to resolve anything), so a third is the
cheapest way to make the conclusion more than a Tox21 anecdote.

### Hard constraint, read first

This pipeline does **masked multitask binary classification**. A dataset
whose labels are continuous numbers will not work without new code —
so ESOL, FreeSolv, Lipophilicity and QM7/QM9 are all **out**, however
tempting their ADMET relevance. Labels must be 0/1, with blanks for unknown.

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
| **HIV** | `HIV.csv` | ~41k | 1 | Large and very imbalanced (~3.5% positives), so PR-AUC finally matters more than ROC-AUC. Hours on CPU for the GNN. |
| **BACE** | `bace.csv` | ~1513 | 1 | Clean and small, but it is target binding, not ADMET. Only interesting as a contrast. |

**Recommended: BBBP.** It is the dataset the proposal itself named, it is an
ADMET endpoint, it runs in minutes, and it has a documented quirk worth
reporting — a number of its SMILES do not parse, which our loader already
counts and drops. **Second choice: SIDER**, if you want the multitask story.

### How

Add one `DatasetSpec` to `DATASETS` in `src/admet/data/loaders.py`, copying
the `clintox` entry. The base URL `_MOLECULENET` is already defined, so
`url=f"{_MOLECULENET}/BBBP.csv"`. The fields you need are `name`, `url`,
`filename`, `smiles_column` and `task_columns`.

**Get the column names right** — they differ per file and guessing wastes an
hour. Download it once and look:

```bash
curl -sL https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv | head -3
```

For BBBP this returns `num,name,p_np,smiles` — so `smiles_column="smiles"`,
`task_columns=("p_np",)`, and `num`/`name` are simply left out. (Checked:
that is the real header, and the file has 2050 rows.) Then copy the four
`configs/clintox_*.yaml` files, change `dataset.name`, and drop the
`tasks:` line if you want every label column.

**Test:** add a case to `tests/test_loaders.py` (Task 6) or
`tests/test_aggregate.py` asserting the new spec is in `DATASETS` and that
its `task_columns` do not include the SMILES column — a real and easy
mistake that silently trains the model to predict its own input.

**Then run it:** `admet run configs/bbbp_rf_scaffold.yaml` takes a couple of
minutes. Report what you find even if it is dull. A third dataset where the
scaffold split costs ~0.08 ROC-AUC would be the strongest single result in
the project; one where it does not is more interesting still.

**Watch for:** a suspiciously high score. BBBP is known to be easy, and
REPORT §4 already explains why that reads as a dataset problem rather than a
model success.

---

## 4. MACCS keys as a second fingerprint

**Difficulty:** ★★☆☆☆ · **~2 h**

The baseline only knows Morgan/ECFP fingerprints. MACCS keys are 166
predefined substructure patterns — chemically interpretable in a way Morgan
bits are not, which makes them a nice contrast for a biomedical audience.

**Files:** add a `maccs_fingerprint()` function to
`src/admet/features/fingerprints.py` next to `morgan_fingerprint()`, using
RDKit's `rdMolDescriptors.GetMACCSKeysFingerprint`. Then let
`RandomForestModel` choose between them — add a `fp_kind` entry to its
`search_space()` in `src/admet/models/rf.py` and include it in
`featurization_key()`, **which matters**: that key is what stops two
different featurizations sharing a cache.

**Test:** in `tests/test_features.py`, assert the MACCS vector is length 167
(RDKit indexes from 1, so index 0 is unused — a classic trap) and that
benzene and cyclohexane give different vectors.

---

## 5. Per-assay results, not just the average

**Difficulty:** ★★★☆☆ · **~3 h** · the most interesting one biologically

Every score we report is macro-averaged over the 12 Tox21 assays, so a single
number hides which toxicity endpoints are easy and which are impossible.
"SR-MMP is predictable, NR-PPAR-gamma is not" is a genuinely interesting
finding and nobody has looked.

The data already exists — `evaluation.py` computes per-assay `TaskScore`
objects and `training.py` writes them into every result file. Nothing
recomputes; this is purely surfacing what is already there.

**Files:** add a `per_task_table()` to `src/admet/analysis/aggregate.py`
(copy `summarize()`), writing `results/per_task.csv` with one row per
(dataset, model, split, assay). Optionally add a horizontal bar chart to
`plots.py` — copy `plot_split_comparison`, which already carries the colour
and axis conventions to follow.

**Test:** extend `tests/test_evaluation.py` — build a two-assay case where
one assay is perfectly predicted and the other is at chance, and assert the
per-assay table separates them while the macro average hides it.

**Expect to find:** the assays with the fewest positives (NR-PPAR-gamma has
186) are probably the worst. If so, that is a data-size story inside a single
dataset and belongs in REPORT.md.

---

## 6. A `tests/test_loaders.py`

**Difficulty:** ★★☆☆☆ · **~2 h**

`loaders.py` is the only core module with no test file of its own, and it does
the cleaning every result depends on — canonicalisation, dropping unparseable
SMILES, merging duplicates.

**Files:** new `tests/test_loaders.py`. Copy the structure of
`tests/test_features.py`.

**Test ideas** (all offline, no download — build tiny CSVs with pytest's
`tmp_path`):

- an invalid SMILES string is dropped and counted in `n_unparseable_dropped`;
- the same molecule written two ways (`CCO` and `OCC`) is merged into one row;
- a missing label stays missing and is **not** read as a zero — this is the
  one that protects the whole masking argument;
- `label_density` matches a hand-computed value.

**Verify it is not vacuous:** make the test fail on purpose first, by
asserting the wrong number, and confirm it really fails. A test that cannot
fail is worse than none.

---

## 7. Smaller things, if you only have an hour

- **`admet --version`.** Read `__version__` from `src/admet/__init__.py`.
  One line in `build_parser`. ★☆☆☆☆
- **`CITATION.cff`** in the repo root so the project can be cited; GitHub
  renders it automatically. All four names. ★☆☆☆☆
- **Warn when merged duplicates disagree.** The loader counts
  `n_label_conflicts` — 19 on ClinTox, 0 on Tox21 — but never says so out
  loud, so the single most suspicious fact about ClinTox is invisible unless
  you open a result file. Emit a warning at load time when it is non-zero.
  ★★☆☆☆
- **Validation-set size in the result files.** `split_sizes` has train, valid
  and test, but `aggregate.py`'s `RunRecord` only keeps `n_train` and
  `n_test`. Add `n_valid`. ★★☆☆☆
- **A `make figures-clean` target** that deletes only `figures/` and redraws
  it. ★☆☆☆☆

---

## What not to touch

These are where the science lives, and a plausible-looking agent edit can
invalidate every result without failing a test:

- `src/admet/data/splits.py` — the scaffold packing already had one bug that
  silently emptied the validation set.
- the masking in `src/admet/evaluation.py` and `masked_bce_loss` in
  `src/admet/models/gnn.py`.
- anything under `results/` — those files are the record of 3.83 h of
  compute. Adding new runs is fine; editing existing ones is not.

If a task seems to need a change in one of those, ask Roman first.

---

## Suggested split

Three people, one each, no overlapping files:

| Who | Task |
|---|---|
| — | **3** (a third dataset) — best value for the effort; H1/H2 currently rest on one dataset |
| — | **5** (per-assay results) — most interesting biologically |
| — | **2** (dummy baseline) — small, and REPORT gains a row |

Tasks **1** (datasets command) and **6** (loader tests), plus anything in
**7**, are good second commits for whoever finishes first.
