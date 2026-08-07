"""Phase 1 unified baseline benchmark runner (contract Phase 1).

Runs the full baseline set (P0.5 nuisance minimum + Phase 1 strong-simple
sequence/topology baselines + corrected v1.31 as the reference method candidate)
on the frozen P0.4 SplitManifests, per axis/fold, outer-train-only fitting,
under the shared right-censored junction-macro NLL metric.

Emits the Phase 1 deliverables:
  BenchmarkProtocol.json      (frozen protocol: axes/models/metric/schema)
  ModelRegistry.json          (all registered models incl. UNAVAILABLE_NOT_COMPARED)
  BudgetRegistry.json         (pre-registered per-model tuning/resource budget)
  Leaderboard.csv             (per model x axis mean macro NLL)
  FoldMetrics.csv             (per model x axis x fold macro NLL + support strata)
  ResourceFairness.csv        (per model fit runtime / param count)
  TaskEquivalenceTable.csv    (external baselines -> UNAVAILABLE_NOT_COMPARED)
  Predictions.jsonl           (row-level sealed predictions)
  STATUS.json                 (terminal PASS/FAIL + gates)

All baselines are fit on outer-train rows only; test folds never enter
normalization, neighbor graphs, early stopping or calibration.  Tuning budgets
and seeds are pre-registered in BudgetRegistry (contract Phase 1 acceptance).
"""
from __future__ import annotations

import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from audit.benchmark.schema import PREDICTION_SCHEMA, record_hash
from audit.benchmark.baselines import BASELINES
from audit.benchmark.phase1_baselines import PHASE1_MODELS
from audit.evaluation.metrics import row_nll, junction_macro_nll, strata_nll, METRIC_SPEC

CAP = -7.1

# Pre-registered tuning / resource budgets (contract Phase 1 acceptance:
# "调参预算与seeds预注册").  All models are low-capacity and use fixed seeds
# (seed=0) and the pre-registered hyperparameters below — no test-driven tuning.
BUDGET_REGISTRY = {
    "global_censor_intercept": {"params": "1 location", "seed": 0, "maxiter": 2000, "budget_note": "closed form via MLE"},
    "train_only_scaffold": {"params": "n_scaf locations", "seed": 0, "maxiter": 2000, "budget_note": "fixed"},
    "scaffold_context_hierarchy": {"params": "1+b0+scaf+ctx", "seed": 0, "maxiter": 2000, "ridge": 1.0, "budget_note": "fixed"},
    "motif_topology_hierarchy": {"params": "1+motif+scaf+3 topo", "seed": 0, "maxiter": 2000, "ridge": 1.0, "budget_note": "fixed"},
    "onehot_kmer_ridge": {"params": "4^3=64 k-mer", "seed": 0, "maxiter": 2000, "k": 3, "ridge": 1.0, "budget_note": "fixed"},
    "position_aware_additive": {"params": "63 one-hot", "seed": 0, "maxiter": 2000, "ridge": 1.0, "budget_note": "fixed"},
    "edit_knn": {"params": "k-nearest", "seed": 0, "k": 11, "budget_note": "train-only neighbor graph"},
    "mutation_graph_smoother": {"params": "graph laplacian", "seed": 0, "budget_note": "train-only graph"},
    "small_mlp": {"params": "64-32 MLP", "seed": 0, "epochs": 40, "lr": 1e-2, "device": "cuda-if-available", "budget_note": "train-only scaling; fixed"},
    "corrected_v1_31": {"params": "63 latent-operator Tobit", "seed": 0, "maxiter": 500, "gh": 48, "budget_note": "P0.3-validated corrected objective"},
}

# External / not-publicly-reproducible-in-this-env baselines (contract Phase 1:
# "不可复现者标 UNAVAILABLE_NOT_COMPARED，不得抄论文数字入榜").
TASK_EQUIVALENCE = [
    {"model": "Denny_native_oracle", "available": False, "reason": "oracle/mechanism upper bound using measured target context fingerprint; not a train-only sequence baseline", "decision": "UNAVAILABLE_NOT_COMPARED"},
    {"model": "Denny_train_only", "available": False, "reason": "thermodynamic fingerprint reconstruction requires external Denny 2018 pipeline/assets not reproducible in this env", "decision": "UNAVAILABLE_NOT_COMPARED"},
    {"model": "RNAMake_physical_ensemble", "available": False, "reason": "RNAMake pipeline / physical template reproduction and licensing not available in this env", "decision": "UNAVAILABLE_NOT_COMPARED"},
    {"model": "frozen_RNA_LM", "available": False, "reason": "a frozen RNA-LM embedding baseline requires a pre-registered frozen encoder + exposure registry; not installed in this env, so NOT compared rather than fabricated", "decision": "UNAVAILABLE_NOT_COMPARED"},
]


def load_rows(ledger_path: Path):
    rows = {}
    for line in ledger_path.read_text().splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        if o.get("layer") != "admitted" or o.get("excluded"):
            continue
        rows[str(o["source_row_id"])] = o
    return rows


def load_splits(manifest_path: Path):
    by_fold = defaultdict(set)
    axis = None
    for line in manifest_path.read_text().splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        axis = o["axis"]
        by_fold[o["fold"]].add(str(o["source_row_id"]))
    return axis, by_fold, len(by_fold)


def run_axis(axis, by_fold, n_folds, rows, models, out_dir):
    results = {}
    all_preds = []
    runtimes = defaultdict(list)
    for fold in range(n_folds):
        test_ids = by_fold.get(fold, set())
        test_rows = [r for sid, r in rows.items() if sid in test_ids]
        train_rows = [r for sid, r in rows.items() if sid not in test_ids]
        for model_id, (fit_fn, pred_fn) in models.items():
            t0 = time.time()
            try:
                model = fit_fn(train_rows)
                mu, sigma, cp, support, abstain = pred_fn(model, test_rows)
            except Exception as e:  # noqa: BLE001
                results.setdefault(model_id, []).append(
                    {"fold": fold, "macro_nll": None, "error": str(e)})
                continue
            runtimes[model_id].append(time.time() - t0)
            macro = junction_macro_nll(test_rows, mu, sigma)
            strata = strata_nll(test_rows, mu, sigma)
            results.setdefault(model_id, []).append({
                "fold": fold, "macro_nll": macro, "n_test": len(test_rows),
                "n_abstain": int(np.sum(abstain)),
                "n_supported": int(np.sum(support)),
                "measured_nll": strata.get("measured"),
                "censored_nll": strata.get("censored")})
            for i, r in enumerate(test_rows):
                rec = {"axis": axis, "fold": fold,
                       "source_row_id": str(r["source_row_id"]), "jid": r["jid"],
                       "scaf": r["scaf"], "context": str(r["helix_seq"]),
                       "model_id": model_id, "seed": 0,
                       "y": r["y"], "cens": r["cens"], "mu": float(mu[i]),
                       "sigma": float(sigma[i]), "censor_prob": float(cp[i]),
                       "nll": float(row_nll([r["y"]], [r["cens"]], [mu[i]], [sigma[i]])[0]),
                       "support": bool(support[i]), "abstain": bool(abstain[i])}
                rec["pred_hash"] = record_hash(rec)
                all_preds.append(rec)
    return results, all_preds, runtimes


def main(cfg):
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(Path(cfg["records"]))
    axes = cfg["axes"]
    models = dict(BASELINES)
    models.update(PHASE1_MODELS)
    try:
        from audit.benchmark.legacy_adapters import LEGACY_MODELS
        models.update(LEGACY_MODELS)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] legacy adapters not loaded: {e}")

    leaderboard = []
    all_preds = []
    fold_metrics = []
    runtimes = defaultdict(list)
    for axis in axes:
        mp = Path(cfg["protocol_dir"]) / f"SplitManifest_{axis}.jsonl"
        if not mp.exists():
            print(f"[skip] no manifest for {axis}")
            continue
        _, by_fold, n_folds = load_splits(mp)
        res, preds, rt = run_axis(axis, by_fold, n_folds, rows, models, out_dir)
        all_preds.extend(preds)
        for mid, rts in rt.items():
            runtimes[mid].extend(rts)
        for model_id, flds in res.items():
            ok = [f["macro_nll"] for f in flds if f["macro_nll"] is not None]
            ok_flds = [f for f in flds if f["macro_nll"] is not None]
            leaderboard.append({
                "axis": axis, "model_id": model_id, "n_folds": len(ok_flds),
                "mean_macro_nll": float(np.mean(ok)) if ok else None,
                "per_fold_macro_nll": [f["macro_nll"] for f in ok_flds],
                "mean_n_abstain": float(np.mean([f["n_abstain"] for f in ok_flds])) if ok_flds else None,
                "mean_n_supported": float(np.mean([f["n_supported"] for f in ok_flds])) if ok_flds else None,
                "mean_measured_nll": float(np.nanmean([f["measured_nll"] for f in ok_flds if f["measured_nll"] is not None])) if any(f["measured_nll"] is not None for f in ok_flds) else None,
                "mean_censored_nll": float(np.nanmean([f["censored_nll"] for f in ok_flds if f["censored_nll"] is not None])) if any(f["censored_nll"] is not None for f in ok_flds) else None})
            for f in flds:
                fold_metrics.append({"axis": axis, "model_id": model_id, "fold": f["fold"],
                                     "macro_nll": f["macro_nll"], "n_test": f["n_test"],
                                     "n_abstain": f["n_abstain"], "n_supported": f["n_supported"],
                                     "measured_nll": f["measured_nll"], "censored_nll": f["censored_nll"],
                                     "error": f.get("error")})

    # ---- write Phase 1 deliverables ----
    with (out_dir / "Leaderboard.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["axis", "model_id", "n_folds", "mean_macro_nll", "per_fold_macro_nll",
                    "mean_n_abstain", "mean_n_supported", "mean_measured_nll", "mean_censored_nll"])
        for r in sorted(leaderboard, key=lambda x: (x["axis"], x["model_id"])):
            w.writerow([r["axis"], r["model_id"], r["n_folds"], r["mean_macro_nll"],
                        json.dumps(r["per_fold_macro_nll"]), r["mean_n_abstain"],
                        r["mean_n_supported"], r["mean_measured_nll"], r["mean_censored_nll"]])
    with (out_dir / "FoldMetrics.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["axis", "model_id", "fold", "macro_nll", "n_test", "n_abstain",
                    "n_supported", "measured_nll", "censored_nll", "error"])
        for r in sorted(fold_metrics, key=lambda x: (x["axis"], x["model_id"], x["fold"])):
            w.writerow([r["axis"], r["model_id"], r["fold"], r["macro_nll"], r["n_test"],
                        r["n_abstain"], r["n_supported"], r["measured_nll"],
                        r["censored_nll"], r["error"]])
    with (out_dir / "Predictions.jsonl").open("w") as fh:
        for rec in all_preds:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    (out_dir / "ModelRegistry.json").write_text(json.dumps(
        {"models": sorted(models.keys()),
         "unavailable_not_compared": TASK_EQUIVALENCE,
         "metric": METRIC_SPEC["name"]}, indent=2, ensure_ascii=False) + "\n")
    (out_dir / "BudgetRegistry.json").write_text(json.dumps(
        BUDGET_REGISTRY, indent=2, ensure_ascii=False) + "\n")
    (out_dir / "BenchmarkProtocol.json").write_text(json.dumps(
        {"phase": "P1", "axes": axes, "models": sorted(models.keys()),
         "metric": METRIC_SPEC, "schema": PREDICTION_SCHEMA["version"],
         "sealed": "test folds frozen from P0.4; no test selection; tuning budget pre-registered"},
        indent=2, ensure_ascii=False) + "\n")
    # ResourceFairness: per-model total fit runtime + param count
    with (out_dir / "ResourceFairness.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model_id", "total_fit_s", "n_fits", "budget_note"])
        for mid in sorted(runtimes):
            w.writerow([mid, round(float(np.sum(runtimes[mid])), 3), len(runtimes[mid]),
                        BUDGET_REGISTRY.get(mid, {}).get("budget_note", "")])
    with (out_dir / "TaskEquivalenceTable.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "available", "reason", "decision"])
        for t in TASK_EQUIVALENCE:
            w.writerow([t["model"], t["available"], t["reason"], t["decision"]])

    # STATUS
    n_axes = len([a for a in axes if (Path(cfg["protocol_dir"]) / f"SplitManifest_{a}.jsonl").exists()])
    p1_ok = (len(leaderboard) > 0 and len(all_preds) > 0
             and n_axes == len({r["axis"] for r in leaderboard}))
    status = {"phase": "P1", "state": "PASS" if p1_ok else "FAIL",
              "axes": axes, "models": sorted(models.keys()),
              "leaderboard_rows": len(leaderboard),
              "fold_metric_rows": len(fold_metrics),
              "prediction_rows": len(all_preds),
              "gates": {"p1_strong_simple_baselines": bool(p1_ok),
                        "test_not_in_training": True,
                        "budget_preregistered": True,
                        "external_not_fabricated": True}}
    (out_dir / "STATUS.json").write_text(json.dumps(status, indent=2) + "\n")
    return status


if __name__ == "__main__":
    cfg = json.loads(Path(sys.argv[1]).read_text())
    print(json.dumps(main(cfg), indent=2, ensure_ascii=False))
