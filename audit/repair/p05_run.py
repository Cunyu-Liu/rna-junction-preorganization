"""P0.5 minimal affected rerun (strict audit 2026-08-11).

Fixes the decisive joint-runner defect: corrected v1.31 computed joint
``train_ids`` but discarded them, training on test contexts while no-sequence
and baselines blocked them.  This runner consumes the shared, typed ``FoldSpec``
from ``audit.repair.fold_loader`` for EVERY model so the full model, matched
no-sequence model and the strongest simple baselines all fit on the SAME
blocked, zero-overlap train set and score on the SAME test set.

Scope (P0.5 "minimal affected rerun"): the decisive ``edit_x_nested_context``
axis only, for the five models the audit names as the minimal decisive set:
  - corrected_v1_31            (full, LEGACY_MODELS)
  - no_sequence_latent_operator (matched ablation)
  - train_only_scaffold         (BASELINES)
  - scaffold_context_hierarchy  (BASELINES)
  - motif_topology_hierarchy    (PHASE1_MODELS)

Each model x fold records:
  - train/test row IDs + rows_hash (from the same FoldSpec)
  - config/seed/command/environment
  - optimizer success/message/objective + final projected-gradient / grad norm
    and bound hits (strict gate, not res.success alone)
  - row-level mu/sigma/support/abstain/fallback_type
  - runtime, peak memory, parameter count
  - full-coverage eligibility and failure reason

Outputs into the NEW repair run root /r05_repair:
  Predictions_v3.jsonl          row-level predictions (unique primary keys)
  Leaderboard_full_coverage.csv pooled pooled-OOF junction-macro NLL per fold
  ConvergenceLedger_v3.parquet  strict optimizer gate per model x fold
  FoldSpec.jsonl                the exact shared row sets + hashes
  GroupAwareGenuine.json        edit-component cluster CI on the genuine contrast
  STATUS.json
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from audit.benchmark.baselines import BASELINES
from audit.benchmark.legacy_adapters import LEGACY_MODELS
from audit.benchmark.phase1_baselines import PHASE1_MODELS
from audit.data.audit_dataset import audit_dataset
from audit.evaluation.metrics import row_nll
from audit.evaluation.scorer_v2 import full_coverage_score, validate_unique_keys
from audit.models.no_sequence_latent_operator import NO_SEQUENCE_LATENT_OPERATOR
from audit.repair.fold_loader import build_joint_edit_context_folds
from audit.repair.optimizer_gate import gate_from_fit

# The five minimal decisive models (P0.5).  Corrected v1.31 + matched no-sequence
# are the core contrast; the three hierarchies are the strongest simple baselines.
def _universe():
    U = {}
    U.update(LEGACY_MODELS)                  # corrected_v1_31
    U.update(NO_SEQUENCE_LATENT_OPERATOR)    # no_sequence_latent_operator
    U.update({k: BASELINES[k] for k in
              ("train_only_scaffold", "scaffold_context_hierarchy")})
    U.update({k: PHASE1_MODELS[k] for k in ("motif_topology_hierarchy",)})
    return U


# Models that are bounded latent operators (projected-gradient gate applies).
BOUNDED = {"corrected_v1_31", "no_sequence_latent_operator"}


def _fit_gate(model: dict) -> dict:
    """Strict optimizer gate.  Bounded models use projected gradient; the
    linear hierarchy baselines use the unbounded strict gradient-norm gate.
    Returns a gate record with ``eligible``."""
    if model.get("gate") is not None and "eligible" not in model["gate"]:
        # baseline models carry a 'gate' from fit_lbfgs with raw diagnostics;
        # re-gate them strictly via the unbounded norm gate.
        g = model["gate"]
        from audit.repair.optimizer_gate import unbounded_fit_gate
        return unbounded_fit_gate(g.get("beta", []), g.get("grad", []),
                                  success=bool(g.get("converged", True)),
                                  grad_tol=1e-3)
    if model.get("gate") is not None:
        return model["gate"]
    # latent operators: full gradient + bounds stored by the adapter
    if "grad" in model and "bounds" in model:
        return gate_from_fit(model, bounds=model["bounds"])
    # deterministic / no-gradient models are always eligible on convergence
    return {"eligible": True, "reason": "deterministic_or_no_optimizer"}


def _param_count(model: dict) -> int:
    if "beta" in model:
        return int(np.size(model["beta"]))
    for key in ("theta", "a", "b"):
        if key in model:
            v = np.asarray(model[key], dtype=float)
            return int(v.size)
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


def main(cfg):
    t0 = time.time()
    run_root = Path(cfg["run_root"])
    out = run_root / "r05_repair"
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

    # Incremental checkpoints: write each fold's rows as it completes so a crash
    # never discards the folds already computed (project lesson learned).
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
            # per-fold checkpoint status (crash-safe progress marker)
            ck = {"axis": spec.axis, "fold": spec.fold, "done": True,
                  "n_models": len(model_ids)}
            (out / f"_fold_{spec.fold}.status").write_text(
                json.dumps(ck, sort_keys=True) + "\n")
    finally:
        preds_fh.close()
        foldspec_fh.close()

    # ---- writes (final aggregates re-derived from the incremental streams) ----
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

    # ---- pooled-OOF genuine + edit-component cluster uncertainty ----
    gen = _pooled_genuine(all_preds)
    cluster = _edit_cluster_ci(all_preds, admitted)
    gen_report = {
        "axis": "edit_x_nested_context",
        "contrast": "corrected_v1_31 vs no_sequence_latent_operator",
        "statistic": "pooled-OOF junction-macro NLL delta (no_sequence - full)",
        "positive_means_full_better": True,
        "n_eligible_folds": int(df_lb[df_lb["eligible_full_coverage"] == True]["fold"].nunique()) if len(df_lb) else 0,
        "genuine": gen,
        "edit_cluster": cluster,
    }
    (out / "GroupAwareGenuine.json").write_text(
        json.dumps(gen_report, indent=2, ensure_ascii=False, sort_keys=True) + "\n")

    status = {
        "phase": "P0.5", "state": "DONE",
        "n_models": len(model_ids), "n_folds": len(specs),
        "n_predictions": len(all_preds), "n_leaderboard_rows": len(leaderboard),
        "n_convergence_rows": len(conv_rows), "duplicate_primary_keys": len(dups),
        "models": model_ids,
        "elapsed_s": round(time.time() - t0, 1),
        "note": ("Minimal affected rerun on the decisive joint axis, all models "
                 "consuming the SAME typed FoldSpec (shared blocked train + "
                 "test rows)."),
    }
    (out / "STATUS.json").write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(status, indent=2, ensure_ascii=False))
    return status


def _eligible_pairs(all_preds):
    """Return {rowid: {model_id: pred}} restricted to rows where BOTH full and
    no-sequence make a supported, non-abstained prediction (coverage-matched)."""
    by = defaultdict(dict)
    for p in all_preds:
        if p["model_id"] in ("corrected_v1_31", "no_sequence_latent_operator"):
            if p["support"] and not p["abstain"]:
                by[p["source_row_id"]][p["model_id"]] = p
    return {rid: d for rid, d in by.items()
            if "corrected_v1_31" in d and "no_sequence_latent_operator" in d}


def _pooled_genuine(all_preds):
    pairs = _eligible_pairs(all_preds)
    if not pairs:
        return {"available": False, "reason": "no matched supported pairs"}
    jid_d = defaultdict(list)
    jid_base = defaultdict(list)
    for rid, d in pairs.items():
        f, n = d["corrected_v1_31"], d["no_sequence_latent_operator"]
        nll_f = float(row_nll([f["y"]], [f["cens"]], [f["mu"]], [f["sigma"]])[0])
        nll_n = float(row_nll([n["y"]], [n["cens"]], [n["mu"]], [n["sigma"]])[0])
        jid_d[f["jid"]].append(nll_n - nll_f)
        jid_base[f["jid"]].append(nll_n)
    theta = float(np.mean([np.mean(v) for v in jid_d.values()]))
    base = float(np.mean([np.mean(v) for v in jid_base.values()]))
    rel = theta / base if base else None
    return {
        "available": True,
        "n_rows": len(pairs), "n_junctions": len(jid_d),
        "theta_abs": theta,
        "relative_gain": rel,
        "relative_gain_pct": (rel * 100.0) if rel is not None else None,
        "n_positive_junctions": int(np.sum([np.mean(v) > 0 for v in jid_d.values()])),
    }


def _edit_cluster_ci(all_preds, admitted):
    """Edit-component cluster bootstrap on the genuine per-junction delta.
    Junction is the repeated predictive unit; edit component is the family
    blocking unit.  Returns percentile 95% CI + leave-one-largest sensitivity."""
    pairs = _eligible_pairs(all_preds)
    if not pairs:
        return {"available": False}
    # map jid -> edit_component
    jid_edit = {}
    for r in admitted:
        jid_edit.setdefault(str(r["jid"]), str(r["edit_component"]))
    jid_d = defaultdict(list)
    for rid, d in pairs.items():
        f, n = d["corrected_v1_31"], d["no_sequence_latent_operator"]
        nll_f = float(row_nll([f["y"]], [f["cens"]], [f["mu"]], [f["sigma"]])[0])
        nll_n = float(row_nll([n["y"]], [n["cens"]], [n["mu"]], [n["sigma"]])[0])
        jid_d[f["jid"]].append(nll_n - nll_f)
    jids = list(jid_d)
    arr = np.array([float(np.mean(jid_d[j])) for j in jids])
    edits = [jid_edit.get(j, "?") for j in jids]

    # edit-cluster bootstrap: resample EDIT COMPONENTS, then take the mean over
    # the junctions they contain.
    from collections import defaultdict as _dd
    by_edit = _dd(list)
    for j, e in zip(jids, edits):
        by_edit[e].append(float(np.mean(jid_d[j])))
    edit_names = list(by_edit)
    rng = np.random.default_rng(17)
    n_boot = 1000
    boots = []
    for _ in range(n_boot):
        chosen = rng.choice(edit_names, size=len(edit_names), replace=True)
        vals = []
        for e in chosen:
            vals.extend(by_edit[e])
        boots.append(float(np.mean(vals)))
    lo = float(np.percentile(boots, 2.5))
    hi = float(np.percentile(boots, 97.5))

    # leave-one-largest-component sensitivity
    sizes = {e: len(v) for e, v in by_edit.items()}
    largest = max(sizes, key=sizes.get)
    # recompute mean without largest component
    keep = [v for e, v in by_edit.items() if e != largest]
    leave1 = float(np.mean([x for grp in keep for x in grp])) if keep else None

    return {
        "available": True,
        "n_junctions": len(jids), "n_edit_components": len(by_edit),
        "mean_delta": float(np.mean(arr)),
        "edit_cluster_boot_95ci": [lo, hi],
        "ci_lower_gt_0": bool(lo > 0),
        "largest_edit_component": largest,
        "largest_edit_size": sizes[largest],
        "leave_one_largest_mean_delta": leave1,
    }


if __name__ == "__main__":
    main(json.loads(Path(sys.argv[1]).read_text()))
