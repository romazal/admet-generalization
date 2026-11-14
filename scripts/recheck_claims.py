"""Re-derive every COMPARATIVE claim in README.md and REPORT.md.

Run from the repository root:

    python scripts/recheck_claims.py

Not "does the number match the table" — that is what scripts/audit_numbers.py
does. This checks the claims that compare two quantities, which is the class
of error that four earlier claims in this report died of.
"""
import csv
import math
import statistics as st
import sys

runs = list(csv.DictReader(open("results/runs.csv", encoding="utf-8")))
gaps = list(csv.DictReader(open("results/gap.csv", encoding="utf-8")))
summ = {(r["dataset"], r["model"], r["split"], float(r["train_fraction"])): r
        for r in csv.DictReader(open("results/summary.csv", encoding="utf-8"))}

ok, bad = [], []


def chk(label, got, want, tol):
    good = abs(got - want) <= tol
    (ok if good else bad).append(label)
    print(("  PASS  " if good else "  FAIL  ") + f"{label}: computed {got:.4f}, claimed {want}")


def seeds(ds, m, sp, f=1.0, metric="test_roc_auc"):
    return {int(r["seed"]): float(r[metric]) for r in runs
            if r["dataset"] == ds and r["model"] == m and r["split"] == sp
            and float(r["train_fraction"]) == f and r[metric]}


def paired(ds, m, f=1.0, metric="test_roc_auc"):
    """Random minus scaffold, per seed."""
    r, s = seeds(ds, m, "random", f, metric), seeds(ds, m, "scaffold", f, metric)
    return [r[i] - s[i] for i in sorted(set(r) & set(s))]


def model_gap(ds, sp, f=1.0, metric="test_roc_auc"):
    rf, gnn = seeds(ds, "rf", sp, f, metric), seeds(ds, "gnn", sp, f, metric)
    return [gnn[i] - rf[i] for i in sorted(set(rf) & set(gnn))]


print("=== BBBP, the new claims ===")
g = model_gap("bbbp", "random")
chk("BBBP random GNN-RF gap", st.mean(g), -0.027, 0.0006)
chk("  its SEM", st.stdev(g) / math.sqrt(len(g)), 0.008, 0.0006)
chk("  in SE units", abs(st.mean(g)) / (st.stdev(g) / math.sqrt(len(g))), 3.3, 0.06)
g = model_gap("bbbp", "scaffold")
chk("BBBP scaffold GNN-RF gap", st.mean(g), -0.031, 0.0006)
chk("  its SEM", st.stdev(g) / math.sqrt(len(g)), 0.003, 0.0006)
chk("  in SE units", abs(st.mean(g)) / (st.stdev(g) / math.sqrt(len(g))), 10.7, 0.1)
chk("  spread of the five seeds", st.stdev(g), 0.0065, 0.0002)
chk("  all five seeds negative", float(all(x < 0 for x in g)), 1.0, 0.0)

for m, want_pen, want_sd in (("rf", 0.021, 0.033), ("gnn", 0.025, 0.027)):
    p = paired("bbbp", m)
    chk(f"BBBP {m} scaffold penalty", st.mean(p), want_pen, 0.0006)
    chk(f"  its SD", st.stdev(p), want_sd, 0.0006)
    a = paired("tox21", m)
    d = st.mean(a) - st.mean(p)
    se = math.sqrt(st.stdev(a) ** 2 / len(a) + st.stdev(p) ** 2 / len(p))
    chk(f"  vs Tox21, in SE", d / se, {"rf": 3.3, "gnn": 3.1}.get(m), 0.06)

chk("BBBP dummy PR random", float(summ[("bbbp", "dummy", "random", 1.0)]["pr_auc_mean"]), 0.764, 0.0006)
chk("BBBP dummy PR scaffold", float(summ[("bbbp", "dummy", "scaffold", 1.0)]["pr_auc_mean"]), 0.753, 0.0006)
chk("BBBP n_train at 100%", float(summ[("bbbp", "rf", "random", 1.0)]["n_train"]), 1580, 0)

curve = [st.mean(model_gap("bbbp", "random", f)) for f in (0.1, 0.25, 0.5, 1.0)]
for got, want in zip(curve, (-0.033, -0.040, -0.020, -0.027)):
    chk(f"BBBP random curve point", got, want, 0.0006)

print("\n=== claims BBBP changes ===")
best = None
for ds in ("tox21", "clintox", "bbbp"):
    for sp in ("random", "scaffold"):
        d = model_gap(ds, sp)
        se = abs(st.mean(d)) / (st.stdev(d) / math.sqrt(len(d)))
        if best is None or se > best[1]:
            best = ((ds, sp), se)
print(f"  strongest model difference anywhere: {best[0]} at {best[1]:.1f} SE")
ok.append("strongest is BBBP scaffold") if best[0] == ("bbbp", "scaffold") else bad.append(
    "strongest model difference is NOT BBBP scaffold")

cross = sum(1 for r in gaps if r["gap"] and abs(float(r["gap"])) <= float(r["gap_std"] or 0))
print(f"  gap intervals including zero: {cross} of {len([r for r in gaps if r['gap']])}")

print("\n=== previously corrected claims still hold ===")
p = paired("tox21", "rf")
chk("Tox21 rf penalty", st.mean(p), 0.073, 0.0006)
chk("  in SEM units (claimed 13)", st.mean(p) / (st.stdev(p) / math.sqrt(5)), 13, 0.2)
d3 = model_gap("tox21", "scaffold")[:3]
chk("3-seed Tox21 scaffold gap", st.mean(d3), -0.008, 0.0006)
chk("  its paired SD", st.stdev(d3), 0.002, 0.0006)
for sp, want in (("random", 2.4), ("scaffold", 1.4)):
    lo, hi = model_gap("tox21", sp, 0.1, "test_pr_auc"), model_gap("tox21", sp, 1.0, "test_pr_auc")
    se = math.sqrt(st.stdev(lo) ** 2 / len(lo) + st.stdev(hi) ** 2 / len(hi))
    chk(f"H2 narrowing {sp} in SE", (st.mean(hi) - st.mean(lo)) / se, want, 0.06)
c = paired("clintox", "rf")
chk("ClinTox rf penalty (withdrawn claim)", st.mean(c), 0.021, 0.0006)
chk("  its SD", st.stdev(c), 0.084, 0.0006)
chk("  negative seeds", float(sum(1 for x in c if x < 0)), 2, 0)

print(f"\n{len(ok)} passed, {len(bad)} failed")
for f in bad:
    print("  FAILED: " + f)
sys.exit(1 if bad else 0)
