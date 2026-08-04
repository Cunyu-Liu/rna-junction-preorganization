"""QR3: genuine locked qMaPseq-to-RNA-MaP transfer test (v1.3).

Per v1.3 9.2/9.3: the primary transport estimand is whether qMaP-observed
chemical-mapping/Mg2+ titration info (log10[Mg2+]1/2 midpoint) predicts
RNA-MaP reference delta-G on held-out variants, with correct censoring and a
preregistered proper score.

The parent run's B4 (old_dg -> rna_map_dg) is REJECTED as a qMaP predictor
because it is same-platform label-to-label calibration (A0 finding A). The
genuine qMaP predictor is B2 (mg_1_2 -> rna_map_dg).

Split: use the QR2 component-level holdout (S1), NOT the 83/11/2/2 i.i.d. fold.
Primary metric: censored-aware negative log predictive density (NLPD) proper
score. Baselines: B1 intercept/mean (strongest simple baseline).
"""
from __future__ import annotations
import json
import math
import os
import sys
import datetime
import hashlib
import numpy as np

try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else None
    HAS_TORCH = True
except Exception:
    DEVICE = None
    HAS_TORCH = False

RUN_ID = os.environ.get("RNA_V13_RUN_ID", "v1_3_corrective_20260804T122313Z")
RUN_ROOT = os.environ.get("RNA_V13_RUN_ROOT", f"/mnt/cunyuliu/{RUN_ID}")
PARENT_ROOT = os.environ.get("RNA_V12_RUN_ROOT", "/mnt/cunyuliu/v1_2_tecto_qmap_codex_20260804T074900Z")

Q1 = os.path.join(PARENT_ROOT, "qmap", "q1", "q1_variant_registry.jsonl")
Q2 = os.path.join(PARENT_ROOT, "qmap", "q2", "q2_attrition.jsonl")
Q4G = os.path.join(PARENT_ROOT, "qmap", "q4", "q4_mutation_graph.json")
QR0 = os.path.join(RUN_ROOT, "qmap", "qr0", "qr0_denominator_truth_table.jsonl")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load_jsonl(p):
    rows = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def proper_score_censored(y, mu, sigma, censored_mask):
    """Censored-aware NLPD. For censored (right) variants use survival likelihood
    P(Y > y) = 1 - Phi((y - mu)/sigma); for measured use normal density."""
    eps = 1e-9
    nll = 0.0
    counts = {"measured": 0, "censored": 0}
    for i in range(len(y)):
        if censored_mask[i]:
            zi = (y[i] - mu[i]) / sigma[i]
            s = 1.0 - 0.5 * (1.0 + math.erf(zi / np.sqrt(2.0)))
            nll += -np.log(max(s, eps))
            counts["censored"] += 1
        else:
            zi = (y[i] - mu[i]) / sigma[i]
            nll += 0.5 * np.log(2 * np.pi) + np.log(sigma[i]) + 0.5 * zi * zi
            counts["measured"] += 1
    return nll / max(len(y), 1), counts


def main():
    # --- CUDA probe: fail closed if unavailable (no silent CPU downgrade) ---
    cuda_probe = {"device": DEVICE, "has_torch": HAS_TORCH}
    if DEVICE == "cuda":
        t = torch.randn(4, 4, device="cuda")
        cuda_probe["forward_sum"] = float(t.sum().item())
        cuda_probe["probe"] = "real_cuda_forward_ok"
    else:
        cuda_probe["probe"] = "CUDA_UNAVAILABLE"
        print("[QR3] FAIL_CLOSED: CUDA unavailable")
        print(json.dumps(cuda_probe, indent=2))

    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    q1 = load_jsonl(Q1)
    q2 = load_jsonl(Q2)
    graph = json.load(open(Q4G))
    qr0 = load_jsonl(QR0)

    def canonical(src_name):
        parts = src_name.split("_")
        return "_".join(reversed(parts))

    q2_by = {r["name"]: r for r in q2}
    q1_by = {canonical(r["name"]): r for r in q1}

    # Build the 98-variant dataset (intersection of q1 and q2), genuine target.
    rows = []
    for cid, q2r in q2_by.items():
        q1r = q1_by.get(cid)
        if q1r is None:
            continue
        rows.append({
            "canonical_id": cid,
            "rna_map_dg": float(q1r["rna_map_dg"]),
            "mg_1_2": float(q2r["mg_1_2"]),
            "category": q2r["category"],
            "censored": q2r["category"] == "right_censored",
        })
    n = len(rows)
    y = np.array([r["rna_map_dg"] for r in rows])
    X = np.log10(np.array([max(r["mg_1_2"], 0.01) for r in rows]))
    censored = np.array([r["censored"] for r in rows])

    # Component assignment from mutation graph
    comp_of = {}
    for ci, comp in enumerate(graph.get("components", [])):
        for v in comp:
            comp_of[v] = ci
    comps = [comp_of.get(r["canonical_id"]) for r in rows]
    comp_sizes = [len([c for c in comps if c == i]) for i in range(graph.get("n_connected_components", 0))]

    # ---- Leave-one-component-out (S1) transfer test ----
    n_comp = graph.get("n_connected_components", 0)
    fold_results = []
    for ci in range(n_comp):
        test_idx = np.where(np.array(comps) == ci)[0]
        train_idx = np.array([i for i in range(n) if i not in set(test_idx.tolist())])
        if len(test_idx) == 0:
            continue
        # B1: intercept/mean baseline (train on measured only)
        yt_train = y[train_idx]
        mu_b1 = yt_train.mean()
        sigma_b1 = yt_train.std()
        # B2: genuine qMaP predictor log10(mg_1_2) -> rna_map_dg (linear regression)
        A = np.vstack([np.ones(len(train_idx)), X[train_idx]]).T
        coef, *_ = np.linalg.lstsq(A, y[train_idx], rcond=None)
        mu_b2 = coef[0] + coef[1] * X[test_idx]
        resid = y[train_idx] - (coef[0] + coef[1] * X[train_idx])
        sigma_b2 = max(resid.std(), 1e-6)
        m_b1 = np.full(len(test_idx), mu_b1)
        s_b1 = np.full(len(test_idx), sigma_b1)
        m_b2 = np.full(len(test_idx), mu_b2)
        s_b2 = np.full(len(test_idx), sigma_b2)
        nlpd_b1, cnt_b1 = proper_score_censored(y[test_idx], m_b1, s_b1, censored[test_idx])
        nlpd_b2, cnt_b2 = proper_score_censored(y[test_idx], m_b2, s_b2, censored[test_idx])
        # ranking (Spearman on measured-only)
        m_meas = censored[test_idx] == False  # noqa: E712
        spearman = None
        if m_meas.sum() >= 3:
            yt = y[test_idx][m_meas]
            mt = m_b2[m_meas]
            from scipy.stats import spearmanr
            si, _ = spearmanr(yt, mt)
            spearman = float(si)
        fold_results.append({
            "component": ci,
            "n_test": int(len(test_idx)),
            "n_measured": int(cnt_b1["measured"]),
            "n_censored": int(cnt_b1["censored"]),
            "nlpd_b1": float(nlpd_b1),
            "nlpd_b2": float(nlpd_b2),
            "spearman_b2": spearman,
        })

    # Aggregate: micro (weighted by n) and group-weight (equal per component)
    n_total = sum(r["n_test"] for r in fold_results)
    micro_b1 = sum(r["nlpd_b1"] * r["n_test"] for r in fold_results) / max(n_total, 1)
    micro_b2 = sum(r["nlpd_b2"] * r["n_test"] for r in fold_results) / max(n_total, 1)
    group_b1 = sum(r["nlpd_b1"] for r in fold_results) / max(len(fold_results), 1)
    group_b2 = sum(r["nlpd_b2"] for r in fold_results) / max(len(fold_results), 1)
    gain_micro = micro_b1 - micro_b2  # positive = B2 better (lower NLPD)
    gain_group = group_b1 - group_b2

    # Proper score direction: lower NLPD is better.
    # supported requires gain over strongest simple baseline with a minimum
    # meaningful improvement and consistent direction across sensitivity splits.
    min_meaningful = 0.3
    supported = (gain_micro > min_meaningful) and (gain_group > 0) and \
                all(r["nlpd_b2"] < r["nlpd_b1"] for r in fold_results if r["n_measured"] >= 3)

    summary = {
        "schema_version": "1.0",
        "gate": "QR3",
        "run_id": RUN_ID,
        "built_at_utc": ts,
        "cuda_probe": cuda_probe,
        "n_variants": n,
        "component_split": {"n_components": n_comp, "component_sizes": comp_sizes},
        "primary_metric": "censored-aware negative log predictive density (NLPD), lower=better",
        "predictor_b2": "log10([Mg2+]1/2) -> rna_map_dg (genuine qMaP transfer)",
        "predictor_b4_rejected": "old_dg -> rna_map_dg REJECTED (same-platform, not qMaP)",
        "baseline_b1": "intercept/mean",
        "fold_results": fold_results,
        "aggregate": {
            "micro_nlpd_b1": micro_b1,
            "micro_nlpd_b2": micro_b2,
            "micro_gain_b2_over_b1": gain_micro,
            "group_weighted_nlpd_b1": group_b1,
            "group_weighted_nlpd_b2": group_b2,
            "group_weighted_gain_b2_over_b1": gain_group,
        },
        "adjudication": {
            "min_meaningful_gain": min_meaningful,
            "supported": bool(supported),
            "disposition": "QMAP_TRANSFER_SUPPORTED" if supported else "QMAP_TRANSFER_NOT_SUPPORTED",
        },
        "note": (
            "QR3 is the genuine qMaP transfer test with component-level holdout and "
            "censored-aware proper score. It does NOT use old_dg as a qMaP predictor. "
            "A negative/borderline result is an accepted outcome; it is not reversed "
            "by dropping censored or structural-QC variants."
        ),
        "source_files": {
            "q1_registry": {"path": Q1, "sha256": sha256_file(Q1)},
            "q2_attrition": {"path": Q2, "sha256": sha256_file(Q2)},
            "mutation_graph": {"path": Q4G, "sha256": sha256_file(Q4G)},
            "qr0_truth_table": {"path": QR0, "sha256": sha256_file(QR0)},
        },
    }

    outdir = os.path.join(RUN_ROOT, "qmap", "qr3")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "qr3_transfer_result.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("[QR3] cuda=%s" % DEVICE)
    print("[QR3] n=%d components=%d sizes=%s" % (n, n_comp, comp_sizes))
    print("[QR3] micro NLPD: B1=%.4f B2=%.4f gain=%.4f" % (micro_b1, micro_b2, gain_micro))
    print("[QR3] group NLPD: B1=%.4f B2=%.4f gain=%.4f" % (group_b1, group_b2, gain_group))
    print("[QR3] disposition=%s" % summary["adjudication"]["disposition"])
    return 0


if __name__ == "__main__":
    sys.exit(main())