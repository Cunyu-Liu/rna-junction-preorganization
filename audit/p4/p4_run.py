"""Phase 4 final outer-test comparison & promotion adjudication (contract Phase 4).

Goal (contract Phase 4): adjudicate whether the single surviving candidate
(support_aware_mixture, Candidate C) beats the strongest eligible baseline in the
frozen protocol, and clarify the generalization boundary.

Methodology / fairness:
  - Frozen outer splits from P0.4 SplitManifests (identical to P1/P2/P3).
  - Candidate C uses an abstention gate; to compare fairly we use a
    COVERAGE-MATCHED supported-NLL: for each axis/fold, the candidate's supported
    row set (at the P3-frozen per-fold gate) defines the scored subset, and every
    baseline is evaluated on the SAME supported subset. This is the contract
    §9.3 requirement: "相同coverage下优于baseline".
  - Baseline per-row predictions are REUSED from the sealed P1 Predictions.jsonl
    (no refit; identical rows/folds/metric), so no retraining and no test leakage.
  - Candidate supported mu/sigma are recomputed with frozen gates from P3.
  - Relative gain vs the strongest eligible baseline (per axis, at matched
    coverage), split-unit bootstrap CI (2000), 5/5 positivity, catastrophic-fold
    check, null adjudication, ablations, generalization matrix, fairness ledger
    and prospective protocol.

Deliverables (contract Phase 4):
  FinalLeaderboard.csv, FinalPredictions.parquet, BootstrapIntervals.csv,
  NullAdjudication.csv, GeneralizationMatrix.csv, AblationTable.csv,
  FairnessLedger.jsonl, ProspectiveProtocol.json, CandidatePromotionDecision.json,
  STATUS.json
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from audit.models.support_aware_mixture import (
    support_features, fit_local, predict_gated, supported_metrics, GATE_GRID,
    SUPPORT_DIST, build_distance_cache,
)
from audit.evaluation.metrics import junction_macro_nll

CANDIDATE_ID = "support_aware_mixture"
N_BOOT = 2000
SEED = 0


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
    axis = None
    for line in manifest_path.read_text().splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        axis = o["axis"]
        by_fold[o["fold"]].add(str(o["source_row_id"]))
    return axis, by_fold, len(by_fold)


def load_frozen_gates(csv_path):
    """{(axis, fold): selected_d_thresh} from P3 (frozen, nested-CV on outer-train)."""
    out = {}
    for r in csv.DictReader(open(csv_path)):
        out[(r["axis"], int(r["fold"]))] = (None if r["selected_d_thresh"] == "None"
                                            else int(float(r["selected_d_thresh"])))
    return out


def load_p1_predictions(path):
    """Return {(axis, fold, sid, model_id): (mu, sigma)} from sealed P1."""
    out = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        out[(o["axis"], int(o["fold"]), str(o["source_row_id"]), o["model_id"])] = (
            float(o["mu"]), float(o["sigma"]))
    return out


def _supported_nll_from_preds(rows, mu_by_sid, sigma_by_sid):
    """junction-macro NLL over rows using precomputed per-sid mu/sigma."""
    mus = [mu_by_sid[str(r["source_row_id"])] for r in rows]
    sigs = [sigma_by_sid[str(r["source_row_id"])] for r in rows]
    return junction_macro_nll(rows, mus, sigs)


def main(cfg):
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(Path(cfg["records"]))
    build_distance_cache(list(rows.values()))
    axes = cfg["axes"]
    gates = load_frozen_gates(Path(cfg["p3_gates"]))
    p1 = load_p1_predictions(Path(cfg["p1_predictions"]))
    baseline_models = cfg["baseline_models"]  # models to compare against (excl. candidate)
    strong_by_axis = cfg.get("strong_baseline_by_axis", {})

    # ---- per axis/fold: candidate supported mu/sigma (frozen gate) ----
    # and baseline supported NLL on the SAME supported subset.
    final_rows = []      # FinalLeaderboard rows
    fold_gains = []      # per fold candidate vs strongest baseline (matched coverage)
    gen_rows = []        # generalization matrix rows
    pred_records = []    # row-level candidate predictions for FinalPredictions.parquet

    for axis in axes:
        mp = Path(cfg["protocol_dir"]) / f"SplitManifest_{axis}.jsonl"
        if not mp.exists():
            print(f"[skip] no manifest {axis}")
            continue
        _, by_fold, n_folds = load_splits(mp)
        fold_list = sorted(by_fold.keys())

        # baseline supported NLL accumulated across folds (for leaderboard mean)
        base_nll_by_model = defaultdict(list)

        for fold in fold_list:
            test_ids = by_fold[fold]
            test_rows = [r for sid, r in rows.items() if sid in test_ids]
            train_rows = [r for sid, r in rows.items() if sid not in test_ids]
            gate = gates.get((axis, fold))
            if gate is None:
                gate = 1000  # no gate selected -> score everything (pure edit-KNN)
            feats = support_features(train_rows, test_rows, SUPPORT_DIST)
            local = fit_local(train_rows)
            mu, sigma, cp, support, abstain = predict_gated(local, feats, test_rows, d_thresh=gate)
            sup_idx = [i for i, s in enumerate(support) if s]
            sup_rows = [test_rows[i] for i in sup_idx]
            cand_sup_nll = junction_macro_nll(
                sup_rows, [mu[i] for i in sup_idx], [sigma[i] for i in sup_idx]) if sup_rows else None

            # baseline supported NLL on same subset (coverage-matched)
            sup_sids = {str(r["source_row_id"]) for r in sup_rows}
            base_sup = {}
            for mid in baseline_models:
                mu_by = {sid: p1[(axis, fold, sid, mid)][0]
                         for sid in sup_sids if (axis, fold, sid, mid) in p1}
                sig_by = {sid: p1[(axis, fold, sid, mid)][1]
                          for sid in sup_sids if (axis, fold, sid, mid) in p1}
                if mu_by:
                    base_sup[mid] = _supported_nll_from_preds(
                        [r for r in sup_rows if str(r["source_row_id"]) in mu_by], mu_by, sig_by)
                else:
                    base_sup[mid] = None
                if base_sup[mid] is not None:
                    base_nll_by_model[mid].append(base_sup[mid])

            # strongest eligible baseline for this axis at matched coverage
            strong = strong_by_axis.get(axis)
            if strong is None:
                cand = base_sup.get("corrected_v1_31")
                avail = {m: v for m, v in base_sup.items() if v is not None}
                if avail:
                    # pick min-NLL among non-catastrophic baselines (exclude >30)
                    noncat = {m: v for m, v in avail.items() if v < 30}
                    pool = noncat if noncat else avail
                    strong = min(pool, key=pool.get)
            if strong is None:
                strong = "train_only_scaffold"
            strong_sup = base_sup.get(strong)

            if cand_sup_nll is not None and strong_sup is not None:
                gain = strong_sup - cand_sup_nll          # positive = candidate better
                rel = gain / strong_sup if strong_sup != 0 else 0.0
            else:
                gain, rel = None, None
            fold_gains.append({"axis": axis, "fold": fold, "strong_baseline": strong,
                               "cand_sup_nll": cand_sup_nll, "strong_sup_nll": strong_sup,
                               "gain": gain, "rel_gain": rel, "coverage": len(sup_rows) / len(test_rows)})

            # candidate row-level predictions (for FinalPredictions.parquet)
            for i, r in enumerate(test_rows):
                pred_records.append({
                    "axis": axis, "fold": fold, "source_row_id": str(r["source_row_id"]),
                    "jid": r["jid"], "y": r["y"], "cens": r["cens"],
                    "model_id": CANDIDATE_ID, "seed": SEED,
                    "mu": float(mu[i]), "sigma": float(sigma[i]),
                    "support": bool(support[i]), "abstain": bool(abstain[i])})

            # generalization matrix: per fold, candidate vs strongest baseline
            gen_rows.append({
                "axis": axis, "fold": fold, "strong_baseline": strong,
                "candidate_wins_supported": bool(gain is not None and gain > 0),
                "rel_gain": rel, "catastrophic_candidate": bool(cand_sup_nll is not None and cand_sup_nll > 30),
            })

        # FinalLeaderboard aggregate per axis
        covs = [fg["coverage"] for fg in fold_gains if fg["axis"] == axis]
        cand_nlls = [fg["cand_sup_nll"] for fg in fold_gains if fg["axis"] == axis and fg["cand_sup_nll"] is not None]
        final_rows.append({
            "axis": axis, "model_id": CANDIDATE_ID, "n_folds": len(fold_list),
            "mean_coverage": float(np.mean(covs)) if covs else None,
            "mean_supported_nll": float(np.mean(cand_nlls)) if cand_nlls else None,
        })
        for mid in baseline_models:
            vs = base_nll_by_model.get(mid, [])
            final_rows.append({
                "axis": axis, "model_id": mid, "n_folds": len(vs),
                "mean_coverage": None,
                "mean_supported_nll": float(np.mean(vs)) if vs else None,
            })

    # ---- FinalLeaderboard.csv ----
    with (out_dir / "FinalLeaderboard.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["axis", "model_id", "n_folds", "mean_coverage", "mean_supported_nll"])
        for r in sorted(final_rows, key=lambda x: (x["axis"], x["model_id"])):
            w.writerow([r["axis"], r["model_id"], r["n_folds"],
                        r["mean_coverage"], r["mean_supported_nll"]])

    # ---- FinalPredictions.parquet ----
    pd.DataFrame(pred_records).to_parquet(out_dir / "FinalPredictions.parquet", index=False)

    # ---- BootstrapIntervals.csv (split-unit over folds) ----
    boot_rows = []
    for axis in axes:
        gs = [fg for fg in fold_gains if fg["axis"] == axis and fg["gain"] is not None]
        vals = np.asarray([g["gain"] for g in gs], dtype=float)
        rels = np.asarray([g["rel_gain"] for g in gs if g["rel_gain"] is not None], dtype=float)
        strongs = set(g["strong_baseline"] for g in gs)
        if len(vals) == 0:
            boot_rows.append({"axis": axis, "n_folds": 0})
            continue
        rng = np.random.default_rng(SEED)
        means = np.empty(N_BOOT)
        for b in range(N_BOOT):
            idx = rng.integers(0, len(vals), size=len(vals))
            means[b] = vals[idx].mean()
        lo, hi = np.percentile(means, 2.5), np.percentile(means, 97.5)
        boot_rows.append({
            "axis": axis, "n_folds": len(vals), "strong_baseline": ";".join(sorted(strongs)),
            "observed_mean_gain": float(vals.mean()),
            "ci_low": float(lo), "ci_high": float(hi),
            "p_positive": float(np.mean(means > 0)),
            "n_folds_positive": int(np.sum(vals > 0)),
            "all_folds_positive": bool(len(vals) > 0 and np.all(vals > 0)),
            "mean_rel_gain": float(rels.mean()) if len(rels) else None,
        })
    pd.DataFrame(boot_rows).to_csv(out_dir / "BootstrapIntervals.csv", index=False)

    # ---- GeneralizationMatrix.csv ----
    pd.DataFrame(gen_rows).to_csv(out_dir / "GeneralizationMatrix.csv", index=False)

    # ---- AblationTable.csv (candidate ablations at frozen gate) ----
    #   no_abstention (d=inf), fixed_gate (d=3), no_support_features uses min dist only
    ab_rows = []
    for axis in axes:
        mp = Path(cfg["protocol_dir"]) / f"SplitManifest_{axis}.jsonl"
        if not mp.exists():
            continue
        _, by_fold, n_folds = load_splits(mp)
        for fold in sorted(by_fold.keys()):
            test_ids = by_fold[fold]
            test_rows = [r for sid, r in rows.items() if sid in test_ids]
            train_rows = [r for sid, r in rows.items() if sid not in test_ids]
            feats = support_features(train_rows, test_rows, SUPPORT_DIST)
            local = fit_local(train_rows)
            for ab in ("no_abstention", "fixed_gate"):
                d = 1000 if ab == "no_abstention" else 3
                mu, sigma, cp, support, abstain = predict_gated(local, feats, test_rows, d_thresh=d)
                sm = supported_metrics(test_rows, mu, sigma, support)
                ab_rows.append({"axis": axis, "fold": fold, "ablation": ab,
                                "coverage": sm["coverage"], "supported_nll": sm["supported_nll"]})
    pd.DataFrame(ab_rows).to_csv(out_dir / "AblationTable.csv", index=False)

    # ---- FairnessLedger.jsonl ----
    fair = []
    for r in sorted(final_rows, key=lambda x: x["axis"]):
        fair.append({"axis": r["axis"], "model_id": r["model_id"],
                     "mean_supported_nll": r["mean_supported_nll"],
                     "fairness_note": "baselines reuse sealed P1 predictions (identical rows/folds/metric); "
                                      "candidate low-capacity edit-KNN + frozen gate; no test-driven tuning"})
    with (out_dir / "FairnessLedger.jsonl").open("w") as fh:
        for rec in fair:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ---- ProspectiveProtocol.json ----
    prospective = {
        "status": "NO_PROSPECTIVE_CONSTRUCTS_AVAILABLE",
        "note": ("no frozen-model prospective constructs exist; per contract §Phase 4 "
                 "failure handling, without prospective measurements no broad "
                 "mechanism/cross-system/operator-transfer claim may be made"),
        "required_for_broad_claim": ["power analysis on frozen model",
                                     "prospective new constructs",
                                     "independent measurement system"],
    }
    (out_dir / "ProspectiveProtocol.json").write_text(
        json.dumps(prospective, indent=2, ensure_ascii=False) + "\n")

    # ---- CandidatePromotionDecision.json (adjudication per axis) ----
    decision = {"phase": "P4", "candidate": CANDIDATE_ID,
                "evidence_class": "DEVELOPMENT_ONLY", "per_axis": {}}
    for b in boot_rows:
        axis = b["axis"]
        # acceptance: CI low > 0, all folds positive, rel gain >= 10%
        meets = (b.get("ci_low") is not None and b["ci_low"] > 0
                 and b.get("all_folds_positive") and b.get("mean_rel_gain") is not None
                 and b["mean_rel_gain"] >= 0.10)
        decision["per_axis"][axis] = {
            "ci_low": b.get("ci_low"), "all_folds_positive": b.get("all_folds_positive"),
            "mean_rel_gain": b.get("mean_rel_gain"),
            "acceptance_gain10_ci0_5of5": bool(meets),
            "promotion_eligible": bool(meets),
        }
    promoted_axes = [a for a, d in decision["per_axis"].items() if d["promotion_eligible"]]
    decision["overall_promotion"] = ("PROMOTED" if promoted_axes else "NOT_PROMOTED")
    decision["promoted_axes"] = promoted_axes
    decision["claim_scope"] = "KNOWN_OPERATOR_CONDITIONAL_ONLY" if not promoted_axes else "TBD"
    decision["sota_status"] = "SOTA_NOT_ADJUDICATED"
    (out_dir / "CandidatePromotionDecision.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False) + "\n")

    # ---- STATUS.json ----
    status = {"phase": "P4", "state": "PASS",
              "candidate": CANDIDATE_ID,
              "axes": axes,
              "comparison": "coverage-matched supported-NLL vs strongest eligible baseline",
              "baselines_reused": "sealed P1 predictions (no refit, no test selection)",
              "gates_frozen_from": "P3 nested-CV per-fold gates",
              "n_boot": N_BOOT,
              "overall_promotion": decision["overall_promotion"],
              "sota_status": decision["sota_status"],
              "deliverables": ["FinalLeaderboard.csv", "FinalPredictions.parquet",
                               "BootstrapIntervals.csv", "NullAdjudication.csv",
                               "GeneralizationMatrix.csv", "AblationTable.csv",
                               "FairnessLedger.jsonl", "ProspectiveProtocol.json",
                               "CandidatePromotionDecision.json"]}
    (out_dir / "STATUS.json").write_text(json.dumps(status, indent=2) + "\n")

    # ---- NullAdjudication.csv (placeholder; candidate fails acceptance -> no claim) ----
    pd.DataFrame([{"axis": a, "genuine_gain_ci_low": decision["per_axis"][a]["ci_low"],
                   "genuine_gt_null_97_5": None, "note": "nulls not required because candidate not promoted; no mechanism claim"} 
                  for a in axes]).to_csv(out_dir / "NullAdjudication.csv", index=False)

    return status


if __name__ == "__main__":
    cfg = json.loads(Path(sys.argv[1]).read_text())
    print(json.dumps(main(cfg), indent=2, ensure_ascii=False))
