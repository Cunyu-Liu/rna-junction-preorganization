"""Build the definitive ablation ladder table (pooled junction-macro NLL)."""
import json
import numpy as np
from collections import defaultdict
from audit.evaluation.metrics import row_nll

R = "/mnt/cunyuliu/rna_junction_repair_20260811T090000Z"
FILES = [
    f"{R}/r14_extended_mlp_scan/Predictions_v3.jsonl",
    f"{R}/r20_robust_t_df_sweep/Predictions_v3.jsonl",
    f"{R}/r21_seed99_replication/Predictions_v3.jsonl",
    f"{R}/r23_seed2026_replication/Predictions_v3.jsonl",
    f"{R}/r24_t7_seed7/Predictions_v3.jsonl",
]

rows = []
for f in FILES:
    for line in open(f):
        q = json.loads(line)
        q["nll"] = float(row_nll([q["y"]], [q["cens"]], [q["mu"]], [q["sigma"]])[0])
        rows.append(q)


def jm(model):
    jd = defaultdict(list)
    for q in rows:
        if q["model_id"] == model and q["support"] and not q["abstain"]:
            jd[q["jid"]].append(q["nll"])
    return float(np.mean([np.mean(v) for v in jd.values()]))


LADDER = [
    ("motif_topology_hierarchy", "nuisance baseline"),
    ("nonlinear_mlp_extended_hybrid_reg", "+Vienna21 nonlinear 2-layer reg"),
    ("nonlinear_mlp_extended_hybrid_reg_deep", "reg_deep 3-layer"),
    ("nonlinear_mlp_extended_hybrid_reg_deep_t", "+robust Student-t df=5"),
    ("nonlinear_mlp_extended_hybrid_reg_deep_t7", "+df=7 best single"),
]

NUIS = jm("motif_topology_hierarchy")
print("=== ABLATION LADDER (pooled junction-macro NLL) ===")
prev = None
for m, note in LADDER:
    v = jm(m)
    gain = (NUIS - v) / NUIS * 100 if v == v else float("nan")
    vs_prev = (prev - v) / prev * 100 if prev else 0.0
    print("  %-40s NLL=%.4f  vs_nuis=%+.2f%%" % (note, v, gain))
    prev = v

d = json.load(open(f"{R}/r24_t7_seed7/t7_3seed_final_ensemble.json"))
ens_nll = d["pooled_junction_macro_nll"]["ENSEMBLE_MLP_MU"]
ens_gain = d["contrasts"]["ENSEMBLE_MLP_MU_vs_motif_topology_hierarchy"]["pooled"]["relative_gain_pct"]
ens_ci = d["contrasts"]["ENSEMBLE_MLP_MU_vs_motif_topology_hierarchy"]["edit_cluster"]["edit_cluster_boot_95ci"]
print("  %-40s NLL=%.4f  vs_nuis=%+.2f%%  CI=%s" % ("3x t7 ensemble (seed 0/99/2026)", ens_nll, ens_gain, ens_ci))
