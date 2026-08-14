"""Quick smoke analysis: pooled junction-macro NLL per model (r27 smoke)."""
import json
from collections import defaultdict
import numpy as np
from audit.evaluation.metrics import row_nll

P = "/mnt/cunyuliu/rna_junction_repair_20260811T090000Z/r27_latent_operator_smoke/Predictions_v3.jsonl"
rows = [json.loads(l) for l in open(P)]

folds = {}
for p in rows:
    folds.setdefault(p["model_id"], set()).add(p["fold"])

by = defaultdict(list)
for p in rows:
    if p["support"] and not p["abstain"]:
        nll = float(row_nll([p["y"]], [p["cens"]], [p["mu"]], [p["sigma"]])[0])
        by[p["model_id"]].append((p["jid"], nll))

for m, lst in sorted(by.items(), key=lambda kv: -len(kv[1])):
    jd = defaultdict(list)
    for j, n in lst:
        jd[j].append(n)
    pooled = float(np.mean([np.mean(v) for v in jd.values()]))
    nf = len(folds.get(m, set()))
    print(f"{m:55s} folds={nf} rows={len(lst):5d} junc={len(jd):4d} pooled_jm={pooled:.4f}")
