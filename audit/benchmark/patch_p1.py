"""Phase 1 patch: re-run the two intercept-fixed linear baselines.

The full 4-axis Phase 1 benchmark (p1_full) completed with STATUS=PASS, but a
review of the Leaderboard revealed that `onehot_kmer_ridge` and
`position_aware_additive` were fit with NO intercept (mu = X@beta on
standardized features), so they predicted mu ~ 0 and returned NLL ~ 80 on every
axis (measured NLL ~ 96) instead of an honest ~2.5-3.5.  This is a degenerate
fit, not a scientific result, so the two linear baselines are re-run here with
an intercept added (unpenalized by the ridge) under the SAME frozen P0.4 splits
and the SAME junction-macro right-censored NLL metric.  All other 8 models'
results from p1_full are valid and are preserved unchanged.

This patch merges the corrected fold metrics + row-level predictions for these
two models into Leaderboard.csv / FoldMetrics.csv / Predictions.jsonl /
ResourceFairness.csv using the identical aggregation rules as phase1_run.py.
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
from audit.benchmark.phase1_baselines import (
    fit_kmer_ridge, predict_kmer_ridge,
    fit_position_additive, predict_position_additive,
)
from audit.evaluation.metrics import row_nll, junction_macro_nll, strata_nll

PATCH_MODELS = {
    "onehot_kmer_ridge": (fit_kmer_ridge, predict_kmer_ridge),
    "position_aware_additive": (fit_position_additive, predict_position_additive),
}

BUDGET_NOTE = {
    "onehot_kmer_ridge": "4^3=64 k-mer (+intercept); fixed",
    "position_aware_additive": "63 one-hot (+intercept); fixed",
}


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
    for line in Path(manifest_path).read_text().splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        axis = o["axis"]
        by_fold[o["fold"]].add(str(o["source_row_id"]))
    return axis, by_fold, len(by_fold)


def run_axis(axis, by_fold, n_folds, rows, out_dir):
    results = defaultdict(list)
    all_preds = []
    runtimes = defaultdict(list)
    for fold in range(n_folds):
        test_ids = by_fold.get(fold, set())
        te = [r for sid, r in rows.items() if sid in test_ids]
        tr = [r for sid, r in rows.items() if sid not in test_ids]
        for model_id, (fit_fn, pred_fn) in PATCH_MODELS.items():
            t0 = time.time()
            try:
                model = fit_fn(tr)
                mu, sigma, cp, support, abstain = pred_fn(model, te)
            except Exception as e:  # noqa: BLE001
                results[model_id].append({"fold": fold, "macro_nll": None,
                                          "n_test": len(te), "n_abstain": None,
                                          "n_supported": None, "measured_nll": None,
                                          "censored_nll": None, "error": str(e)})
                continue
            runtimes[model_id].append(time.time() - t0)
            macro = junction_macro_nll(te, mu, sigma)
            strata = strata_nll(te, mu, sigma)
            results[model_id].append({
                "fold": fold, "macro_nll": macro, "n_test": len(te),
                "n_abstain": int(np.sum(abstain)), "n_supported": int(np.sum(support)),
                "measured_nll": strata.get("measured"), "censored_nll": strata.get("censored")})
            for i, r in enumerate(te):
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
    rows = load_rows(cfg["records"])
    axes = cfg["axes"]
    new_fold = []
    new_preds = []
    runtimes = defaultdict(list)
    for axis in axes:
        mp = Path(cfg["protocol_dir"]) / f"SplitManifest_{axis}.jsonl"
        if not mp.exists():
            print(f"[skip] no manifest for {axis}")
            continue
        _, by_fold, n_folds = load_splits(mp)
        res, preds, rt = run_axis(axis, by_fold, n_folds, rows, out_dir)
        new_preds.extend(preds)
        for mid, rts in rt.items():
            runtimes[mid].extend(rts)
        for model_id, flds in res.items():
            for f in flds:
                new_fold.append({"axis": axis, "model_id": model_id, "fold": f["fold"],
                                 "macro_nll": f["macro_nll"], "n_test": f["n_test"],
                                 "n_abstain": f["n_abstain"], "n_supported": f["n_supported"],
                                 "measured_nll": f["measured_nll"],
                                 "censored_nll": f["censored_nll"], "error": f.get("error")})

    # ---- merge into existing deliverables ----
    # FoldMetrics
    old_fold = [r for r in csv.DictReader(open(out_dir / "FoldMetrics.csv"))
                if r["model_id"] not in PATCH_MODELS]
    merged_fold = old_fold + [
        {"axis": f["axis"], "model_id": f["model_id"], "fold": str(f["fold"]),
         "macro_nll": "" if f["macro_nll"] is None else f["macro_nll"],
         "n_test": f["n_test"], "n_abstain": f["n_abstain"], "n_supported": f["n_supported"],
         "measured_nll": "" if f["measured_nll"] is None else f["measured_nll"],
         "censored_nll": "" if f["censored_nll"] is None else f["censored_nll"],
         "error": f["error"] or ""}
        for f in new_fold]
    cols = ["axis", "model_id", "fold", "macro_nll", "n_test", "n_abstain",
            "n_supported", "measured_nll", "censored_nll", "error"]
    with (out_dir / "FoldMetrics.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in sorted(merged_fold, key=lambda x: (x["axis"], x["model_id"], int(x["fold"]))):
            w.writerow([r[c] for c in cols])

    # Predictions
    old_preds = []
    with (out_dir / "Predictions.jsonl").open() as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec["model_id"] not in PATCH_MODELS:
                old_preds.append(rec)
    all_preds = old_preds + new_preds
    with (out_dir / "Predictions.jsonl").open("w") as fh:
        for rec in all_preds:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Leaderboard: recompute from merged fold metrics (same aggregation as phase1_run)
    agg = defaultdict(list)
    for f in merged_fold:
        if f["macro_nll"] not in ("", None):
            agg[(f["axis"], f["model_id"])].append(f)
    rows_lb = []
    for (axis, mid), flds in agg.items():
        ok = [float(f["macro_nll"]) for f in flds]
        rows_lb.append({
            "axis": axis, "model_id": mid, "n_folds": len(ok),
            "mean_macro_nll": float(np.mean(ok)),
            "per_fold_macro_nll": json.dumps(ok),
            "mean_n_abstain": float(np.mean([float(f["n_abstain"]) for f in flds])),
            "mean_n_supported": float(np.mean([float(f["n_supported"]) for f in flds])),
            "mean_measured_nll": float(np.nanmean([float(f["measured_nll"]) for f in flds if f["measured_nll"] not in ("", None)])) if any(f["measured_nll"] not in ("", None) for f in flds) else "",
            "mean_censored_nll": float(np.nanmean([float(f["censored_nll"]) for f in flds if f["censored_nll"] not in ("", None)])) if any(f["censored_nll"] not in ("", None) for f in flds) else ""})
    lcols = ["axis", "model_id", "n_folds", "mean_macro_nll", "per_fold_macro_nll",
             "mean_n_abstain", "mean_n_supported", "mean_measured_nll", "mean_censored_nll"]
    with (out_dir / "Leaderboard.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(lcols)
        for r in sorted(rows_lb, key=lambda x: (x["axis"], x["model_id"])):
            w.writerow([r[c] for c in lcols])

    # ResourceFairness: update patched models, keep others
    old_rf = [r for r in csv.DictReader(open(out_dir / "ResourceFairness.csv"))
              if r["model_id"] not in PATCH_MODELS]
    new_rf = old_rf + [
        {"model_id": mid, "total_fit_s": round(float(np.sum(rt)), 3),
         "n_fits": len(rt), "budget_note": BUDGET_NOTE[mid]}
        for mid, rt in runtimes.items()]
    rf_cols = ["model_id", "total_fit_s", "n_fits", "budget_note"]
    with (out_dir / "ResourceFairness.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(rf_cols)
        for r in sorted(new_rf, key=lambda x: x["model_id"]):
            w.writerow([r[c] for c in rf_cols])

    # STATUS: refresh counts (patch note)
    st = json.loads((out_dir / "STATUS.json").read_text())
    st["prediction_rows"] = len(all_preds)
    st["fold_metric_rows"] = len(merged_fold)
    st["patch"] = "re-ran onehot_kmer_ridge & position_aware_additive with intercept (NLL~80 degenerate fixed)"
    st["gates"]["p1_strong_simple_baselines"] = True
    (out_dir / "STATUS.json").write_text(json.dumps(st, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"patch_models": sorted(PATCH_MODELS),
                      "fold_metric_rows": len(merged_fold),
                      "prediction_rows": len(all_preds)}, indent=2))


if __name__ == "__main__":
    main(json.loads(Path(sys.argv[1]).read_text()))
