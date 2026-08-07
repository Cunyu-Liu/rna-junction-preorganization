"""P0.5 unified qualification harness (contract P0.5).

Fits the minimum-eligibility model set on the frozen P0.4 SplitManifests
(per axis/fold, outer-training rows only) and scores test rows under the shared
junction-macro right-censored NLL.  Emits per-model fold metrics, a
QualificationLeaderboard, and row-level predictions sealed under the P0.5
schema.  This is a QUALIFICATION leaderboard, not a scientific SOTA claim.

Currently implemented models: the three nuisance-only censored baselines plus
any adapters registered by legacy_adapters (v1.28 / v1.30 / corrected v1.31).
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from audit.benchmark.schema import PREDICTION_SCHEMA, record_hash
from audit.benchmark.baselines import BASELINES
from audit.evaluation.metrics import row_nll, junction_macro_nll, METRIC_SPEC

CAP = -7.1
EPS = 1e-8


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
    """Return (axis, by_fold, n_folds)."""
    by_fold = defaultdict(set)
    axis = None
    for line in manifest_path.read_text().splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        axis = o["axis"]
        by_fold[o["fold"]].add(str(o["source_row_id"]))
    n_folds = len(by_fold)
    return axis, by_fold, n_folds


def run_axis(axis, by_fold, n_folds, rows, models, out_dir):
    results = {}
    all_preds = []
    for fold in range(n_folds):
        test_ids = by_fold.get(fold, set())
        test_rows = [r for sid, r in rows.items() if sid in test_ids]
        train_rows = [r for sid, r in rows.items() if sid not in test_ids]
        tr = train_rows
        te = test_rows
        for model_id, (fit_fn, pred_fn) in models.items():
            try:
                model = fit_fn(tr)
                mu, sigma, cp, support, abstain = pred_fn(model, te)
            except Exception as e:  # noqa: BLE001
                results.setdefault(model_id, []).append(
                    {"fold": fold, "macro_nll": None, "error": str(e)})
                continue
            macro = junction_macro_nll(te, mu, sigma)
            results.setdefault(model_id, []).append(
                {"fold": fold, "macro_nll": macro,
                 "n_test": len(te), "n_abstain": int(np.sum(abstain))})
            for i, r in enumerate(te):
                rec = {"axis": axis, "fold": fold,
                       "source_row_id": str(r["source_row_id"]), "jid": r["jid"], "scaf": r["scaf"],
                       "context": str(r["helix_seq"]), "model_id": model_id, "seed": 0,
                       "y": r["y"], "cens": r["cens"], "mu": float(mu[i]),
                       "sigma": float(sigma[i]), "censor_prob": float(cp[i]),
                       "nll": float(row_nll([r["y"]], [r["cens"]], [mu[i]], [sigma[i]])[0]),
                       "support": bool(support[i]), "abstain": bool(abstain[i])}
                rec["pred_hash"] = record_hash(rec)
                all_preds.append(rec)
    return results, all_preds


def main(cfg):
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    worktree = Path(cfg["worktree"])
    rows = load_rows(Path(cfg["records"]))
    axes = cfg["axes"]
    models = dict(BASELINES)
    # optional legacy adapters
    try:
        from audit.benchmark.legacy_adapters import LEGACY_MODELS
        models.update(LEGACY_MODELS)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] legacy adapters not loaded: {e}")

    leaderboard = []
    all_preds = []
    for axis in axes:
        mp = worktree / "audit" / "evaluation" / "split" / f"SplitManifest_{axis}.jsonl"
        # manifests live in run-root protocol/
        mp = Path(cfg["protocol_dir"]) / f"SplitManifest_{axis}.jsonl"
        if not mp.exists():
            print(f"[skip] no manifest for {axis}")
            continue
        _, by_fold, n_folds = load_splits(mp)
        res, preds = run_axis(axis, by_fold, n_folds, rows, models, out_dir)
        all_preds.extend(preds)
        for model_id, flds in res.items():
            ok = [f["macro_nll"] for f in flds if f["macro_nll"] is not None]
            leaderboard.append({
                "axis": axis, "model_id": model_id,
                "n_folds": len(flds),
                "mean_macro_nll": float(np.mean(ok)) if ok else None,
                "per_fold_macro_nll": [f["macro_nll"] for f in flds],
                "mean_n_abstain": float(np.mean([f["n_abstain"] for f in flds])),
            })

    # write outputs
    import csv
    with (out_dir / "QualificationLeaderboard.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["axis", "model_id", "n_folds", "mean_macro_nll",
                    "per_fold_macro_nll", "mean_n_abstain"])
        for r in sorted(leaderboard, key=lambda x: (x["axis"], x["model_id"])):
            w.writerow([r["axis"], r["model_id"], r["n_folds"],
                        r["mean_macro_nll"],
                        json.dumps(r["per_fold_macro_nll"]), r["mean_n_abstain"]])
    with (out_dir / "Predictions.jsonl").open("w") as fh:
        for rec in all_preds:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    (out_dir / "ModelRegistry.json").write_text(json.dumps(
        {"models": sorted(models.keys()),
         "metric": METRIC_SPEC["name"]}, indent=2) + "\n")
    (out_dir / "BenchmarkProtocol.json").write_text(json.dumps(
        {"axes": axes, "models": sorted(models.keys()),
         "metric": METRIC_SPEC, "schema": PREDICTION_SCHEMA["version"],
         "sealed": "test folds frozen from P0.4; no test selection"},
        indent=2, ensure_ascii=False) + "\n")
    # STATUS
    status = {"phase": "P0.5", "state": "RUNNING",
              "axes": axes, "models": sorted(models.keys()),
              "leaderboard_rows": len(leaderboard),
              "prediction_rows": len(all_preds)}
    (out_dir / "STATUS.json").write_text(json.dumps(status, indent=2) + "\n")
    return status


if __name__ == "__main__":
    cfg = json.loads(Path(sys.argv[1]).read_text())
    print(json.dumps(main(cfg), indent=2, ensure_ascii=False))
