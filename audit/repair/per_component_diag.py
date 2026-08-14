"""Per-edit-component diagnostic: where does the 3x t7 ensemble gain/lose vs nuisance?"""
import json
import numpy as np
from collections import defaultdict
from audit.evaluation.metrics import row_nll

R = "/mnt/cunyuliu/rna_junction_repair_20260811T090000Z"
P = f"{R}/r24_t7_seed7/combined_r20_r21_r23_r24_preds.jsonl"

MEMBERS = [
    "nonlinear_mlp_extended_hybrid_reg_deep_t7",
    "nonlinear_mlp_extended_hybrid_reg_deep_t7_s99",
    "nonlinear_mlp_extended_hybrid_reg_deep_t7_s2026",
]
NUIS = "motif_topology_hierarchy"

rows = [json.loads(l) for l in open(P)]
# index: (model_id, source_row_id) -> row
idx = {}
for q in rows:
    if q["support"] and not q["abstain"]:
        idx[(q["model_id"], q["source_row_id"])] = q

# fold -> edit component map (from first t7 row)
comp_of_fold = {}
for q in rows:
    if q["model_id"] == MEMBERS[0] and q["support"] and not q["abstain"]:
        comp_of_fold[q["fold"]] = q["fold"].split(":")[1] if ":" in q["fold"] else q["fold"]

# gather ensemble-eligible rows: present in all members + nuisance
rids = set()
for q in rows:
    if q["model_id"] == MEMBERS[0] and q["support"] and not q["abstain"]:
        rid = q["source_row_id"]
        ok = all((m, rid) in idx for m in MEMBERS) and (NUIS, rid) in idx
        if ok:
            rids.add(rid)

by_comp = defaultdict(list)
for rid in rids:
    ref = idx[(MEMBERS[0], rid)]
    comp = comp_of_fold.get(ref["fold"], "?")
    mu = np.mean([idx[(m, rid)]["mu"] for m in MEMBERS])
    y = ref["y"]
    cens = ref["cens"]
    by_comp[comp].append((mu, y, cens, rid))

print("=== Per-edit-component: 3x t7 ensemble vs nuisance ===")
results = []
for comp, lst in sorted(by_comp.items(), key=lambda kv: -len(kv[1])):
    n = len(lst)
    mu = np.array([x[0] for x in lst])
    y = np.array([x[1] for x in lst])
    cens = np.array([x[2] for x in lst], dtype=bool)
    ens_nll = float(np.mean([row_nll([y[i]], [cens[i]], [mu[i]], [0.7])[0] for i in range(n)]))
    nuis_nll = float(np.mean([row_nll([y[i]], [cens[i]], [idx[(NUIS, lst[i][3])]["mu"]], [0.7])[0] for i in range(n)]))
    gain = 100 * (nuis_nll - ens_nll) / nuis_nll if nuis_nll else 0.0
    results.append((comp, n, ens_nll, nuis_nll, gain))
    print(f"  {comp:20s} n={n:4d}  ens={ens_nll:.4f}  nuis={nuis_nll:.4f}  gain={gain:+.2f}%")

worst = sorted(results, key=lambda r: r[4])
print("\n=== WORST 5 components (ensemble underperforms) ===")
for r in worst[:5]:
    print(f"  {r[0]:20s} n={r[1]:4d} gain={r[4]:+.2f}%")
print("\n=== BEST 5 components ===")
for r in worst[-5:]:
    print(f"  {r[0]:20s} n={r[1]:4d} gain={r[4]:+.2f}%")
print(f"\ntotal components: {len(results)}, total rows: {sum(r[1] for r in results)}")
