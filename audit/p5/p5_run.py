"""Phase 5: mechanism / identifiability-boundary diagnostics (contract Phase 5).

Because Phase 4 concluded NOT_PROMOTED (the surviving candidate does not beat the
strongest eligible baseline under coverage-matched supported-NLL, and ties it on
the operator-holdout axis), Phase 5 follows the contract's *failure path*: instead
of analyzing "why the candidate works", it rigorously characterizes WHERE the
sequence signal does and does not survive — the identifiability boundary — and
selects a benchmark / identifiability-boundary narrative rather than a mechanism
claim.

This runner is DIAGNOSTIC only. It does not retrain, refit, or re-select gates.
It reuses:
  - sealed P1 baseline predictions (per-row mu/sigma),
  - P4 candidate predictions (per-row mu/sigma/support),
  - frozen outer splits from P0.4 SplitManifests,
and only recomputes cheap, train-only support features (min edit distance to
nearest outer-train junction, outer-train context support) to stratify failure.

Deliverables (contract Phase 5):
  FailureAtlas.parquet, ContextSensitivity.csv, MutationPathAnalysis.csv,
  CatastrophicFolds.csv, ClaimEvidenceMatrix.csv, PaperStoryDecision.md, STATUS.json
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from audit.evaluation.metrics import row_nll
from audit.models.support_aware_mixture import support_features, build_distance_cache, SUPPORT_DIST

CANDIDATE = "support_aware_mixture"
AXES = ["symmetry_5fold", "edit_5fold", "context_lomo", "scaffold_lomo"]
STRONG = {
    "symmetry_5fold": "corrected_v1_31",
    "edit_5fold": "corrected_v1_31",
    "context_lomo": "train_only_scaffold",
    "scaffold_lomo": "edit_knn",
}
CATASTROPHIC_RATIO = 1.1


def load_rows(ledger_path):
    rows = {}
    for line in Path(ledger_path).read_text().splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        if o.get("layer") != "admitted" or o.get("excluded"):
            continue
        rows[str(o["source_row_id"])] = o
    return rows


def load_splits(manifest_path):
    by_fold = defaultdict(set)
    for line in manifest_path.read_text().splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        by_fold[o["fold"]].add(str(o["source_row_id"]))
    return by_fold


def load_candidate_preds(parquet_path):
    d = pd.read_parquet(parquet_path)
    d = d[d["model_id"] == CANDIDATE]
    out = {}
    for _, r in d.iterrows():
        out[(r["axis"], int(r["fold"]), str(r["source_row_id"]))] = (
            float(r["mu"]), float(r["sigma"]), bool(r["support"]))
    return out


def load_p1(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        out[(o["axis"], int(o["fold"]), str(o["source_row_id"]), o["model_id"])] = (
            float(o["mu"]), float(o["sigma"]))
    return out


def edit_dist_bucket(d):
    if d <= 0:
        return "0_same_seq"
    if d == 1:
        return "1"
    if d == 2:
        return "2"
    if d == 3:
        return "3"
    return "4plus"


def ctx_support_bucket(n):
    if n <= 0:
        return "0_unseen"
    if n == 1:
        return "1"
    if n <= 5:
        return "2-5"
    if n <= 20:
        return "6-20"
    return "21plus"


def censor_bucket(f):
    if f < 0.10:
        return "low_<0.10"
    if f < 0.25:
        return "mid_0.10-0.25"
    return "high_>=0.25"


def main(cfg):
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    rows = load_rows(Path(cfg["records"]))
    build_distance_cache(list(rows.values()))
    cand = load_candidate_preds(Path(cfg["final_preds"]))
    p1 = load_p1(Path(cfg["p1_preds"]))
    proto = Path(cfg["protocol_dir"])

    failure_rows = []
    for axis in AXES:
        by_fold = load_splits(proto / f"SplitManifest_{axis}.jsonl")
        strong = STRONG[axis]
        for fold, test_ids in by_fold.items():
            test_rows = [rows[sid] for sid in test_ids]
            train_rows = [rows[sid] for sid in rows if sid not in test_ids]
            ctx_train = defaultdict(int)
            for r in train_rows:
                ctx_train[str(r["helix_seq"])] += 1
            feats = support_features(train_rows, test_rows, SUPPORT_DIST)
            n_cens = sum(1 for r in test_rows if r["cens"])
            cens_frac = n_cens / len(test_rows)
            for r in test_rows:
                sid = str(r["source_row_id"])
                key = (axis, fold, sid)
                if key not in cand:
                    continue
                cmu, csig, sup = cand[key]
                cn = float(row_nll([r["y"]], [r["cens"]], [cmu], [csig])[0])
                fx = feats.get(str(r["jid"]), {})
                ed = edit_dist_bucket(float(fx.get("min_edit_dist", np.inf)))
                cs = ctx_support_bucket(ctx_train.get(str(r["helix_seq"]), 0))
                bmu, bsig = p1.get((axis, fold, sid, strong), (cmu, csig))
                bn = float(row_nll([r["y"]], [r["cens"]], [bmu], [bsig])[0])
                failure_rows.append({
                    "axis": axis, "fold": fold, "sid": sid, "jid": str(r["jid"]),
                    "supported": bool(sup), "cens": bool(r["cens"]),
                    "candidate_nll": cn, "baseline_nll": bn, "delta": cn - bn,
                    "edit_dist_bucket": ed, "context_support_bucket": cs,
                    "censor_bucket": censor_bucket(cens_frac)})
    fail = pd.DataFrame(failure_rows)
    fail.to_parquet(out / "FailureAtlas.parquet")

    # ---- ContextSensitivity: per axis x stratifier x stratum ----
    ctx_rows = []
    for axis in AXES:
        sub = fail[fail["axis"] == axis]
        for strat in ["edit_dist_bucket", "context_support_bucket", "censor_bucket"]:
            g = sub.groupby(strat).agg(
                n=("delta", "size"),
                mean_candidate_nll=("candidate_nll", "mean"),
                mean_baseline_nll=("baseline_nll", "mean"),
                mean_delta=("delta", "mean"),
            ).reset_index()
            for _, r in g.iterrows():
                ctx_rows.append({"axis": axis, "stratifier": strat, "stratum": r[strat],
                                 "n": int(r["n"]), "mean_candidate_nll": r["mean_candidate_nll"],
                                 "mean_baseline_nll": r["mean_baseline_nll"],
                                 "mean_delta": r["mean_delta"]})
    pd.DataFrame(ctx_rows).to_csv(out / "ContextSensitivity.csv", index=False)

    # ---- MutationPathAnalysis: candidate vs baseline NLL by edit-distance bucket ----
    mut_rows = []
    for axis in AXES:
        sub = fail[fail["axis"] == axis]
        order = ["0_same_seq", "1", "2", "3", "4plus"]
        g = sub.groupby("edit_dist_bucket").agg(
            n=("delta", "size"),
            mean_candidate_nll=("candidate_nll", "mean"),
            mean_baseline_nll=("baseline_nll", "mean"),
            mean_delta=("delta", "mean")).reindex(order).reset_index()
        for _, r in g.iterrows():
            if pd.isna(r["n"]):
                continue
            mut_rows.append({"axis": axis, "edit_dist_bucket": r["edit_dist_bucket"],
                             "n": int(r["n"]), "mean_candidate_nll": r["mean_candidate_nll"],
                             "mean_baseline_nll": r["mean_baseline_nll"],
                             "mean_delta": r["mean_delta"]})
    pd.DataFrame(mut_rows).to_csv(out / "MutationPathAnalysis.csv", index=False)

    # ---- Catastrophic folds (supported macro supported-NLL, candidate > ratio*baseline) ----
    cat_rows = []
    for axis in AXES:
        sub = fail[(fail["axis"] == axis) & (fail["supported"])]
        for fold, grp in sub.groupby("fold"):
            cand_macro = grp.groupby("jid")["candidate_nll"].mean().mean()
            base_macro = grp.groupby("jid")["baseline_nll"].mean().mean()
            catastrophic = bool(cand_macro > CATASTROPHIC_RATIO * base_macro)
            cat_rows.append({"axis": axis, "fold": int(fold), "n_supported": int(len(grp)),
                             "cand_supported_macro_nll": float(cand_macro),
                             "baseline_supported_macro_nll": float(base_macro),
                             "catastrophic": catastrophic})
    pd.DataFrame(cat_rows).to_csv(out / "CatastrophicFolds.csv", index=False)

    # ---- ClaimEvidenceMatrix (frozen, no mechanism claim authorized) ----
    claims = [
        {"claim": "sequence encodes preorganization mechanism transferable across context/operator",
         "evidence_class": "NOT_AUTHORIZED", "basis": "P4 NOT_PROMOTED; no prospective constructs",
         "status": "FAIL_CLOSED"},
        {"claim": "candidate beats strongest eligible baseline at matched coverage on any axis",
         "evidence_class": "DEVELOPMENT_ONLY", "basis": "P4 BootstrapIntervals all-negative or 0; NOT_PROMOTED",
         "status": "REFUTED"},
        {"claim": "operator-transfer (unseen scaffold) prediction",
         "evidence_class": "NOT_AUTHORIZED", "basis": "scaffold_lomo candidate ties edit_knn (gain 0); only 9 scaffolds",
         "status": "NOT_AUTHORIZED"},
        {"claim": "sequence-local KNN rescues operator-holdout catastrophic failure",
         "evidence_class": "DEVELOPMENT_ONLY", "basis": "P3 scaffold_lomo 0 catastrophic; but edit_knn already achieves same NLL",
         "status": "NOT_PROMOTABLE"},
        {"claim": "gain attributable to seen-context/scaffold calibration rather than sequence mechanism",
         "evidence_class": "CONSISTENT", "basis": "v1.30 sequence null > genuine gain; candidate underperforms on known-operator axes",
         "status": "SUPPORTED_BOUNDARY"},
        {"claim": "identifiability boundary: no evidence of sequence signal beyond local neighborhood + known-operator calibration",
         "evidence_class": "DEVELOPMENT_ONLY", "basis": "ContextSensitivity/MutationPath diagnostics; no prospective",
         "status": "SELECTED_NARRATIVE"},
    ]
    pd.DataFrame(claims).to_csv(out / "ClaimEvidenceMatrix.csv", index=False)

    # ---- PaperStoryDecision ----
    story = (
        "## Paper Story Decision (Phase 5)\n\n"
        "**Narrative: benchmark / identifiability-boundary** (contract Phase 5 failure path).\n\n"
        "Phase 4 sealed a fail-closed result: the only surviving candidate "
        "(support_aware_mixture) was NOT_PROMOTED — it underperforms the strongest "
        "eligible baseline on all known-operator axes and only ties edit_knn on the "
        "operator-holdout axis. No promotable mechanism exists, so there is no mechanism "
        "narrative to write.\n\n"
        "The honest and publishable contribution is therefore an identifiability boundary:\n"
        "1. Under grouped, right-censor-aware, leakage-controlled evaluation, junction "
        "sequence provides no incremental supported-NLL beyond motif/context/scaffold/nearest-"
        "neighbour and censored-marginal baselines on known-operator axes.\n"
        "2. The apparent 'operator rescue' of the sequence-local KNN is achieved equally by "
        "a simple edit_knn baseline; it is a local-neighbourhood property, not a mechanism.\n"
        "3. Repeated context/scaffold exposure packages calibration as generalization — the "
        "contract's central concern is confirmed, not refuted.\n\n"
        "**Claims removed:** any mechanism, operator-transfer, or cross-system claim. "
        "**Claims retained:** benchmark protocol, identifiability boundary, and the negative "
        "result that sequence-local signal does not transfer.\n\n"
        "**Gate:** SOTA_NOT_ADJUDICATED; NO_SUBMISSION_AUTHORIZATION; "
        "scientific_claim_authorized=false.\n"
    )
    (out / "PaperStoryDecision.md").write_text(story)

    status = {
        "phase": "P5", "state": "PASS",
        "narrative": "benchmark_identifiability_boundary",
        "mechanism_claim": False,
        "candidate": CANDIDATE,
        "axes": AXES,
        "diagnostics": ["FailureAtlas.parquet", "ContextSensitivity.csv",
                        "MutationPathAnalysis.csv", "CatastrophicFolds.csv",
                        "ClaimEvidenceMatrix.csv", "PaperStoryDecision.md"],
        "sota_status": "SOTA_NOT_ADJUDICATED",
        "scientific_claim_authorized": False,
    }
    (out / "STATUS.json").write_text(json.dumps(status, indent=2) + "\n")
    return status


if __name__ == "__main__":
    cfg = json.loads(Path(sys.argv[1]).read_text())
    print(json.dumps(main(cfg), indent=2, ensure_ascii=False))
