"""Analyze r40 smoke: training-time per-scaffold sigma vs frozen t7."""
import json
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "/home/cunyuliu/rna_junction_repair_20260811")
from audit.evaluation.metrics import row_nll

R = "/mnt/cunyuliu/rna_junction_repair_20260811T090000Z/r40_scaf_sigma_smoke"
rows = [json.loads(l) for l in open(R + "/Predictions_v3.jsonl")]

T7 = "nonlinear_mlp_extended_hybrid_reg_deep_t7"
SCAF = "nonlinear_mlp_extended_hybrid_reg_deep_t7_scaf"

for m in [T7, SCAF]:
    print("===", m)
    byf = defaultdict(list)
    for p in rows:
        if p["model_id"] == m and p["support"] and not p["abstain"]:
            byf[p["fold"]].append(p)
    for f in sorted(byf):
        jd = defaultdict(list)
        for p in byf[f]:
            nll = float(row_nll([p["y"]], [p["cens"]], [p["mu"]], [p["sigma"]])[0])
            jd[p["jid"]].append(nll)
        nll = float(np.mean([np.mean(v) for v in jd.values()]))
        sigs = [p["sigma"] for p in byf[f]]
        print(f"  {f}: nll={nll:.4f} sigma_mean={np.mean(sigs):.3f}")

# learned sigma per scaffold
print("\n=== learned scaf sigma by fold ===")
by_scaf = defaultdict(lambda: defaultdict(list))
for p in rows:
    if p["model_id"] == SCAF and p["support"] and not p["abstain"]:
        by_scaf[p["fold"]][int(p["scaf"])].append(p["sigma"])
for f in sorted(by_scaf):
    s = {str(k): round(float(np.mean(v)), 3) for k, v in sorted(by_scaf[f].items())}
    print(f"  {f}: {s}")
