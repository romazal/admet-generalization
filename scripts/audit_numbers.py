"""Cross-check the numbers quoted in README.md and REPORT.md against the CSVs.

Run from the repository root:

    python scripts/audit_numbers.py .

Every value in the main tables of both documents is re-derived from
``results/summary.csv`` and ``results/gap.csv`` and must appear in the text
exactly as computed. A typo, a stale number left behind by a re-run, or a
table updated in one document and not the other all fail here.

**What this cannot check.** It verifies that each number is right. It says
nothing about whether comparing two of them is warranted — whether a
difference exceeds its own spread, whether two arms of the experiment agree,
whether a multiple quoted for one condition holds for the others. Four claims
in this report were wrong in exactly that way while this script passed on all
of them; REPORT §7 lists them. Treat a pass here as necessary and nowhere
near sufficient.
"""
import csv
import io
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."


def rows(name):
    return list(csv.DictReader(io.open("%s/results/%s" % (ROOT, name), encoding="utf-8")))


summary = {}
for r in rows("summary.csv"):
    k = (r["dataset"], r["model"], r["split"], float(r["train_fraction"]))
    summary[k] = r

gap = {}
for r in rows("gap.csv"):
    k = (r["dataset"], r["split"], float(r["train_fraction"]), r["metric"])
    gap[k] = r


def f3(x):
    return "%.3f" % float(x)


docs = {}
for name in ("README.md", "REPORT.md"):
    docs[name] = io.open("%s/%s" % (ROOT, name), encoding="utf-8").read()

fails = []
checks = 0


def expect(text, where="both"):
    """Assert `text` appears in the named doc(s)."""
    global checks
    checks += 1
    targets = docs.keys() if where == "both" else [where]
    for name in targets:
        if text not in docs[name]:
            fails.append("%s: missing %r" % (name, text))


# --- main Tox21 table, both docs -------------------------------------------
for metric, key in (("roc", "roc_auc"), ("pr", "pr_auc")):
    for split in ("random", "scaffold"):
        for model in ("dummy", "rf", "gnn"):
            k = ("tox21", model, split, 1.0)
            if k not in summary:
                continue
            cell = "%s ± %s" % (f3(summary[k][key + "_mean"]), f3(summary[k][key + "_std"]))
            expect(cell)
        g = gap[("tox21", split, 1.0, key)]
        sign = "+" if float(g["gap"]) >= 0 else "−"
        cell = "%s%s ± %s" % (sign, f3(abs(float(g["gap"]))), f3(g["gap_std"]))
        expect(cell)

# --- Tox21 learning-curve gap table ----------------------------------------
for frac in (0.1, 0.25, 0.5):
    for split in ("random", "scaffold"):
        for key in ("roc_auc", "pr_auc"):
            g = gap[("tox21", split, frac, key)]
            sign = "+" if float(g["gap"]) >= 0 else "−"
            cell = "%s%s ± %s" % (sign, f3(abs(float(g["gap"]))), f3(g["gap_std"]))
            expect(cell)

# --- every secondary dataset needs its 100% table in the report -------------
# Not just its name: "BBBP" appears the moment a make target is added, so
# naming alone would pass on a dataset nobody wrote up. Requiring the actual
# cells means the audit fails until the table is there.
for name in sorted({k[0] for k in summary} - {"tox21"}):
    for split in ("random", "scaffold"):
        for model in ("rf", "gnn"):
            k = (name, model, split, 1.0)
            if k not in summary:
                continue
            cell = "%s ± %s" % (f3(summary[k]["roc_auc_mean"]), f3(summary[k]["roc_auc_std"]))
            expect(cell, "REPORT.md")

# --- every dataset that was run must be written up ---------------------------
# A dataset can be added to the registry, run, aggregated and plotted without
# anyone mentioning it in the report. That is a silent omission rather than a
# wrong number, so nothing else here would catch it.
LABELS = {"tox21": "Tox21", "clintox": "ClinTox", "bbbp": "BBBP"}
for name in sorted({k[0] for k in summary}):
    checks += 1
    label = LABELS.get(name, name)
    for doc in ("README.md", "REPORT.md"):
        if label not in docs[doc]:
            fails.append("%s: dataset %r was run but is never named" % (doc, label))

# --- scaffold penalties -----------------------------------------------------
for ds, places in (("tox21", 3), ("clintox", 3)):
    for model in ("rf", "gnn"):
        pen = float(summary[(ds, model, "random", 1.0)]["roc_auc_mean"]) - float(
            summary[(ds, model, "scaffold", 1.0)]["roc_auc_mean"]
        )
        checks += 1
        txt = "%.3f" % pen
        if txt not in docs["REPORT.md"]:
            fails.append("REPORT.md: scaffold penalty %s/%s = %s not found" % (ds, model, txt))

print("checked %d quoted values" % checks)
if fails:
    print("FAILURES (%d):" % len(fails))
    for f in fails:
        print("  " + f)
    sys.exit(1)
print("all quoted values match results/*.csv")
