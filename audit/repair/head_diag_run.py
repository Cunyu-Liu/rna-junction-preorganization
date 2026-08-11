"""Head diagnosis runner (strict audit follow-up, 2026-08-11).

Question: is the *latent-operator head* the bottleneck on the decisive
``edit_x_nested_context`` axis, rather than the 63-D sequence representation?

Approach: on the IDENTICAL joint FoldSpec (same blocked train + test rows),
compare several heads over the SAME 63-D features and over scaffold-only input:

  - corrected_v1_31          : latent-operator Tobit head on 63-D
                               (GH integration + scaffold intercept/slope a_s,b_s)
  - position_aware_additive  : plain right-censored linear head on 63-D
                               (intercept + ridge, NO latent, NO scaffold)
  - no_sequence_latent_operator : latent-operator head, intercept-only location
                               (matched ablation; scaffold a_s,b_s)
  - train_only_scaffold      : plain linear scaffold one-hot (no sequence)
  - motif_topology_hierarchy : plain linear motif+scaffold+topology

This is a DIAGNOSTIC: if a simpler plain-linear head on the same 63-D features
beats the latent-operator head, the sophisticated latent machinery is not
conferring value and should be replaced before any representation shootout.

Like P0.5, every model consumes the same typed FoldSpec, the same scorer, and
the strict optimizer gate (projected-gradient for bounded latent operators,
gradient-norm for unbounded linear heads).  Outputs land in a NEW run root
subdirectory ``r05_head_diag`` so the P0.5 artifacts are untouched.
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from audit.benchmark.baselines import BASELINES
from audit.benchmark.legacy_adapters import LEGACY_MODELS
from audit.benchmark.phase1_baselines import PHASE1_MODELS
from audit.data.audit_dataset import audit_dataset
from audit.evaluation.metrics import row_nll
from audit.evaluation.scorer_v2 import full_coverage_score, validate_unique_keys
from audit.models.no_sequence_latent_operator import NO_SEQUENCE_LATENT_OPERATOR
from audit.repair.fold_loader import build_joint_edit_context_folds
from audit.repair.optimizer_gate import gate_from_fit, unbounded_fit_gate

# Head-diagnosis model set.  position_aware_additive is the plain linear head on
# the same 63-D features as corrected_v1_31.
def _universe():
    U = {}
    U.update(LEGACY_MODELS)                  # corrected_v1_31
    U.update(NO_SEQUENCE_LATENT_OPERATOR)    # no_sequence_latent_operator
    U.update({k: BASELINES[k] for k in ("train_only_scaffold",)})
    U.update({k: PHASE1_MODELS[k] for k in
              ("motif_topology_hierarchy", "position_aware_additive")})
    return U


BOUNDED = {"corrected_v1_31", "no_sequence_latent_operator"}


def _fit_gate(model: dict) -> dict:
    if model.get("gate") is not None and "eligible" not in model["gate"]:
        g = model["gate"]
        return unbounded_fit_gate(g.get("beta", []), g.get("grad", []),
                                  success=bool(g.get("converged", True)),
                                  grad_tol=1e-3)
    if model.get("gate") is not None:
        return model["gate"]
    if "grad" in model and "bounds" in model:
        return gate_from_fit(model, bounds=model["bounds"])
    return {"eligible": True, "reason": "deterministic_or_no_optimizer"}


def _param_count(model: dict) -> int:
    if "beta" in model:
        return int(np.size(model["beta"]))
    for key in ("theta", "a", "b"):
        if key in model:
            return int(np.size(model[key]))
    return 0


def _convergence_row(axis, fold, model_id, model, gate, runtime, elig):
    base = {"axis": axis, "fold": str(fold), "model_id": model_id,
            "success": bool(model.get("success", True)),
            "optimizer_message": str(model.get("optimizer_message", "")),
            "final_grad_norm": float(model.get("final_grad_norm", float("nan"))),
            "n_iter": int(model.get("nit", -1)),
            "n_param": _param_count(model),
            "runtime_s": round(runtime, 3),
            "eligible": bool(gate.get("eligible", False)),
            "eligible_full_coverage": bool(elig.get("eligible", False)),
            "elig_reason": elig.get("reason"),
            **({k: gate[k] for k in
                ("projected_grad_norm", "proj_grad_tol", "n_bound_hits",
                 "n_nan_inf_params", "grad_tol")
                if k in gate})}
    return base


def _paired_rows(all_preds, a_id, b_id):
    """Return {rowid: {a: pred, b: pred}} on rows both support & non-abstain."""
    by = defaultdict(dict)
    for p in all_preds:
        if p["model_id"] in (a_id, b_id):
            if p["support"] and not p["abstain"]:
                by[p["source_row_id"]][p["model_id"]] = p
    return {rid: d for rid, d in by.items() if a_id in d and b_id in d}


def _pooled_contrast(all_preds, a_id, b_id, a_name, b_name, theta_label):
    """pooled-OOF junction-macro NLL delta = (b - a); positive means a is better."""
    pairs = _paired_rows(all_preds, a_id, b_id)
    if not pairs:
        return {"available": False, "reason": "no matched supported pairs"}
    jid_d = defaultdict(list)
    jid_base = defaultdict(list)
    for rid, d in pairs.items():
        pa, pb = d[a_id], d[b_id]
        dla = float(row_nll([pa["y"]], [pa["cens"]], [pa["mu"]], [pa["sigma"]])[0])
        dlb = float(row_nll([pb["y"]], [pb["cens"]], [pb["mu"]], [pb["sigma"]])[0])
        jid_d[pa["jid"]].append(dlb - dla)
        jid_base[pa["jid"]].append(dlb)
    theta = float(np.mean([np.mean(v) for v in jid_d.values()]))
    base = float(np.mean([np.mean(v) for v in jid_base.values()]))
    rel = theta / base if base else None
    return {
        "available": True,
        "contrast": f"{a_name} vs {b_name}",
        "theta_label": theta_label,
        "n_rows": len(pairs), "n_junctions": len(jid_d),
        "theta_abs": theta,
        "relative_gain": rel,
        "relative_gain_pct": (rel * 100.0) if rel is not None else None,
        "n_positive_junctions": int(np.sum([np.mean(v) > 0 for v in jid_d.values()])),
    }


def _edit_cluster_ci(all_preds, admitted, a_id, b_id):
    pairs = _paired_rows(all_preds, a_id, b_id)
    if not pairs:
        return {"available": False}
    jid_edit = {}
    for r in admitted:
        jid_edit.setdefault(str(r["jid"]), str(r["edit_component"]))
    jid_d = defaultdict(list)
    for rid, d in pairs.items():
        pa, pb = d[a_id], d[b_id]
        dla = float(row_nll([pa["y"]], [pa["cens"]], [pa["mu"]], [pa["sigma"]])[0])
        dlb = float(row_nll([pb["y"]], [pb["cens"]], [pb["mu"]], [pb["sigma"]])[0])
        jid_d[pa["jid"]].append(dlb - dla)
    by_edit = defaultdict(list)
    for j, vals in jid_d.items():
        by_edit[jid_edit.get(j, "?")].append(float(np.mean(vals)))
    edit_names = list(by_edit)
    rng = np.random.default_rng(17)
    boots = []
    for _ in range(1000):
        chosen = rng.choice(edit_names, size=len(edit_names), replace=True)
        vals = [v for e in chosen for v in by_edit[e]]
        boots.append(float(np.mean(vals)))
    lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    sizes = {e: len(v) for e, v in by_edit.items()}
    largest = max(sizes, key=sizes.get)
    keep = [v for e, v in by_edit.items() if e != largest]
    leave1 = float(np.mean([x for grp in keep for x in grp])) if keep else None
    return {
        "available": True,
        "n_junctions": len(jid_d), "n_edit_components": len(by_edit),
        "mean_delta": float(np.mean([float(np.mean(v)) for v in jid_d.values()])),
        "edit_cluster_boot_95ci": [lo, hi],
        "ci_lower_gt_0": bool(lo > 0),
        "largest_edit_component": largest,
        "largest_edit_size": sizes[largest],
        "leave_one_largest_mean_delta": leave1,
    }


def main(cfg):
    t0 = time.time()
    run_root = Path(cfg["run_root"])
    out = run_root / "r05_head_diag"
    out.mkdir(parents=True, exist_ok=True)

    _, admitted, profile, *_ = audit_dataset(Path(cfg["canonical_source"]))
    rows = {str(r["source_row_id"]): r for r in admitted}

    specs = build_joint_edit_context_folds(admitted)
    if cfg.get("max_folds"):
        specs = specs[: int(cfg["max_folds"])]
    U = _universe()
    model_ids = cfg.get("models") or list(U.keys())

    all_preds = []
    leaderboard = []
    conv_rows = []
    spec_lines = []

    preds_fh = (out / "Predictions_v3.jsonl").open("w")
    foldspec_fh = (out / "FoldSpec.jsonl").open("w")
    try:
        for spec in specs:
            spec.validate()
            spec_lines.append(spec.to_manifest())
            foldspec_fh.write(json.dumps(spec.to_manifest(), sort_keys=True) + "\n")
            foldspec_fh.flush()
            test_rows = [r for sid, r in rows.items() if sid in spec.test_ids]
            train_rows = [r for sid, r in rows.items() if sid in spec.train_ids]
            for model_id in model_ids:
                if model_id not in U:
                    continue
                fit_fn, pred_fn = U[model_id]
                tm = time.time()
                try:
                    model = fit_fn(train_rows)
                    mu, sigma, cp, support, abstain = pred_fn(model, test_rows)
                    gate = _fit_gate(model)
                except Exception as e:  # noqa: BLE001
                    conv_rows.append({"axis": spec.axis, "fold": spec.fold,
                                      "model_id": model_id,
                                      "error": f"{type(e).__name__}: {e}"})
                    leaderboard.append({"axis": spec.axis, "fold": spec.fold,
                                        "model_id": model_id,
                                        "error": f"{type(e).__name__}: {e}"})
                    continue
                runtime = time.time() - tm

                preds_by_rowid = {}
                for i, r in enumerate(test_rows):
                    preds_by_rowid[str(r["source_row_id"])] = {
                        "mu": float(mu[i]), "sigma": float(sigma[i]),
                        "abstain": bool(abstain[i]), "support": bool(support[i]),
                        "fallback_type": None,
                    }
                metric, elig = full_coverage_score(test_rows, preds_by_rowid)
                conv_rows.append(_convergence_row(
                    spec.axis, spec.fold, model_id, model, gate, runtime, elig))
                if preds_by_rowid:
                    for rid, p in preds_by_rowid.items():
                        r = next(x for x in test_rows if str(x["source_row_id"]) == rid)
                        rec = {
                            "axis": spec.axis, "fold": spec.fold,
                            "source_row_id": rid, "jid": r["jid"],
                            "scaf": int(r["scaf"]), "context": str(r["helix_seq"]),
                            "model_id": model_id, "y": r["y"], "cens": bool(r["cens"]),
                            "mu": p["mu"], "sigma": p["sigma"],
                            "abstain": p["abstain"], "support": p["support"],
                            "fallback_type": p["fallback_type"]}
                        all_preds.append(rec)
                        preds_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                preds_fh.flush()
                leaderboard.append({
                    "axis": spec.axis, "fold": spec.fold, "model_id": model_id,
                    "coverage": metric["coverage"],
                    "pooled_junction_macro_nll": metric["pooled_junction_macro_nll"],
                    "eligible_full_coverage": elig["eligible"],
                    "elig_reason": elig["reason"],
                    "n_eligible": metric["n_eligible"],
                    "n_abstain_no_fallback": metric["n_abstain_no_fallback"],
                    "optimizer_eligible": bool(gate.get("eligible", False)),
                })
            (out / f"_fold_{spec.fold}.status").write_text(
                json.dumps({"axis": spec.axis, "fold": spec.fold, "done": True,
                            "n_models": len(model_ids)}, sort_keys=True) + "\n")
    finally:
        preds_fh.close()
        foldspec_fh.close()

    with (out / "Predictions_v3.jsonl").open("w") as fh:
        for rec in all_preds:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with (out / "FoldSpec.jsonl").open("w") as fh:
        for s in spec_lines:
            fh.write(json.dumps(s, sort_keys=True) + "\n")

    df_lb = pd.DataFrame(leaderboard)
    df_lb.to_csv(out / "Leaderboard_full_coverage.csv", index=False)
    pd.DataFrame(conv_rows).to_parquet(out / "ConvergenceLedger_v3.parquet")
    dups = validate_unique_keys([{**p} for p in all_preds])

    # Head-diagnosis contrast: latent-operator head vs plain linear head, both on
    # the same 63-D features.  Positive delta means the plain linear head wins.
    head_contrast = _pooled_contrast(
        all_preds, "position_aware_additive", "corrected_v1_31",
        "position_aware_additive(plain-linear-head)",
        "corrected_v1_31(latent-operator-head)",
        "pooled-OOF junction-macro NLL delta (corrected_v1_31 - position_aware_additive)")
    head_cluster = _edit_cluster_ci(
        all_preds, admitted, "position_aware_additive", "corrected_v1_31")

    # Reference: sequence increment under the latent head (corrected vs no-seq).
    seq_contrast = _pooled_contrast(
        all_preds, "corrected_v1_31", "no_sequence_latent_operator",
        "corrected_v1_31", "no_sequence_latent_operator",
        "pooled-OOF junction-macro NLL delta (no_sequence - corrected_v1_31)")

    report = {
        "axis": "edit_x_nested_context",
        "purpose": "HEAD_DIAGNOSIS",
        "n_eligible_folds": int(df_lb[df_lb["eligible_full_coverage"] == True]["fold"].nunique()) if len(df_lb) else 0,
        "head_diagnosis": {
            "question": "Does the latent-operator head add value over a plain right-censored linear head on the SAME 63-D features?",
            "note": ("positive delta = position_aware_additive (plain linear) is "
                     "BETTER than corrected_v1_31 (latent operator) on matched "
                     "supported rows; if so, the latent machinery is the bottleneck."),
            "pooled": head_contrast,
            "edit_cluster": head_cluster,
        },
        "reference_sequence_increment_latent_head": {
            "note": "corrected_v1_31 vs matched no-sequence under the latent head.",
            "pooled": seq_contrast,
        },
    }
    (out / "HeadDiagnosis.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n")

    status = {
        "phase": "HEAD_DIAGNOSIS", "state": "DONE",
        "n_models": len(model_ids), "n_folds": len(specs),
        "n_predictions": len(all_preds), "n_leaderboard_rows": len(leaderboard),
        "n_convergence_rows": len(conv_rows), "duplicate_primary_keys": len(dups),
        "models": model_ids,
        "elapsed_s": round(time.time() - t0, 1),
        "note": ("Head diagnosis on the decisive joint axis comparing the "
                 "latent-operator head vs a plain right-censored linear head on "
                 "the same 63-D features, all consuming the SAME typed FoldSpec."),
    }
    (out / "STATUS.json").write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(status, indent=2, ensure_ascii=False))
    return status


if __name__ == "__main__":
    main(json.loads(Path(sys.argv[1]).read_text()))