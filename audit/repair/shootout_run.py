"""Representation shootout on the decisive joint axis (strict audit follow-up).

Head diagnosis established that the latent-operator head is NOT the bottleneck:
on the same 63-D features it beats a plain right-censored linear head by ~1.77
nats.  The bottleneck is the 63-D position/composition representation, which on
the latent head made pooled junction-macro NLL WORSE than the matched
no-sequence model (1.428 vs 1.147).

This runner tests whether a ViennaRNA thermodynamic/secondary-structure
representation restores a sequence-specific increment.  All models consume the
SAME joint FoldSpec (37 blocked edit x nested-context folds), the same scorer,
and the strict optimizer gate.  The latent-operator head is held fixed.

Models:
  - vienna_latent_operator      : latent-operator head on ViennaRNA features
  - no_sequence_latent_operator : latent-operator head, intercept-only location
                                  (matched ablation)
  - corrected_v1_31             : latent-operator head on 63-D (reference, known
                                  to be WORSE than no-sequence)
  - train_only_scaffold         : plain linear scaffold one-hot (no sequence)
  - motif_topology_hierarchy    : plain linear motif+scaffold+topology

Outputs into a NEW run root /r06_shootout so prior artifacts are untouched.
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
from audit.models.vienna_latent_operator import VIENNA_LATENT_OPERATOR
from audit.models.kmer_latent_operator import KMER_LATENT_OPERATOR
from audit.models.vienna_linear_hybrid import VIENNA_LINEAR_HYBRID
from audit.models.vienna_extended_linear_hybrid import VIENNA_EXTENDED_LINEAR_HYBRID
from audit.models.rnafm_linear_hybrid import make_rnafm_linear_hybrid
from audit.models.rnafm_vienna_linear_hybrid import make_rnafm_vienna_linear_hybrid
from audit.models.rnafm_pca_linear_hybrid import make_rnafm_pca_linear_hybrid
from audit.models.vienna_interaction_linear_hybrid import VIENNA_INTERACTION_LINEAR_HYBRID
from audit.models.nonlinear_mlp_hybrid import NONLINEAR_MLP_HYBRID
from audit.models.nonlinear_mlp_rich_hybrid import (
    make_nonlinear_mlp_extended_hybrid,
    make_nonlinear_mlp_extended_hybrid_reg,
    make_nonlinear_mlp_extended_hybrid_reg_strong,
    make_nonlinear_mlp_extended_hybrid_reg_light,
    make_nonlinear_mlp_extended_hybrid_reg_wider,
    make_nonlinear_mlp_extended_hybrid_reg_deep,
    make_nonlinear_mlp_extended_hybrid_reg_deep4,
    make_nonlinear_mlp_extended_hybrid_reg_deep4w,
    make_nonlinear_mlp_extended_hybrid_reg_deep5,
    make_nonlinear_mlp_extended_hybrid_het,
    make_nonlinear_mlp_extended_hybrid_localctx,
    make_nonlinear_mlp_extended_hybrid_reg_deep_t,
    make_nonlinear_mlp_rnafm_pca_hybrid,
    make_nonlinear_mlp_rnafm_only_pca_hybrid,
    make_nonlinear_mlp_rnafm_extended_reg_deep,
)
from audit.repair.fold_loader import build_joint_edit_context_folds
from audit.repair.optimizer_gate import gate_from_fit, unbounded_fit_gate

# The decisive representation-shootout model set.
def _universe(rnafm_cache=None):
    U = {}
    U.update(VIENNA_LATENT_OPERATOR)               # ViennaRNA folding features
    U.update(KMER_LATENT_OPERATOR)                 # k-mer composition
    U.update(NO_SEQUENCE_LATENT_OPERATOR)          # matched ablation
    U.update(LEGACY_MODELS)                        # corrected_v1_31 (63-D reference)
    U.update({k: BASELINES[k] for k in ("train_only_scaffold",)})
    U.update({k: PHASE1_MODELS[k] for k in ("motif_topology_hierarchy",)})
    U.update({k: PHASE1_MODELS[k] for k in ("onehot_kmer_ridge",)})  # k-mer plain linear
    U.update(VIENNA_LINEAR_HYBRID)                 # winning linear head + ViennaRNA
    U.update(VIENNA_EXTENDED_LINEAR_HYBRID)        # winning linear head + extended ViennaRNA
    U.update(VIENNA_INTERACTION_LINEAR_HYBRID)     # winning linear head + Vienna x scaffold/motif interactions
    U.update(NONLINEAR_MLP_HYBRID)                 # nonlinear MLP on the winning feature set
    U["nonlinear_mlp_extended_hybrid"] = make_nonlinear_mlp_extended_hybrid()  # MLP + 21-D ViennaRNA
    U["nonlinear_mlp_extended_hybrid_reg"] = make_nonlinear_mlp_extended_hybrid_reg()  # dropout=0.1, wd=1e-2
    U["nonlinear_mlp_extended_hybrid_reg_strong"] = make_nonlinear_mlp_extended_hybrid_reg_strong()  # do=0.2, wd=3e-2
    U["nonlinear_mlp_extended_hybrid_reg_light"] = make_nonlinear_mlp_extended_hybrid_reg_light()  # do=0.05, wd=1e-2
    U["nonlinear_mlp_extended_hybrid_reg_wider"] = make_nonlinear_mlp_extended_hybrid_reg_wider()  # (128,64)
    U["nonlinear_mlp_extended_hybrid_reg_deep"] = make_nonlinear_mlp_extended_hybrid_reg_deep()  # (96,64,32)
    U["nonlinear_mlp_extended_hybrid_reg_deep4"] = make_nonlinear_mlp_extended_hybrid_reg_deep4()  # (96,64,32,16)
    U["nonlinear_mlp_extended_hybrid_reg_deep4w"] = make_nonlinear_mlp_extended_hybrid_reg_deep4w()  # (128,96,64,32)
    U["nonlinear_mlp_extended_hybrid_reg_deep5"] = make_nonlinear_mlp_extended_hybrid_reg_deep5()  # (128,96,64,32,16)
    U["nonlinear_mlp_extended_hybrid_het"] = make_nonlinear_mlp_extended_hybrid_het()  # reg_deep + learned sigma
    U["nonlinear_mlp_extended_hybrid_localctx"] = make_nonlinear_mlp_extended_hybrid_localctx()  # reg_deep + Vienna21 + localctx24
    U["nonlinear_mlp_extended_hybrid_reg_deep_t"] = make_nonlinear_mlp_extended_hybrid_reg_deep_t(df=5.0)  # reg_deep + Student-t obj
    U["nonlinear_mlp_extended_hybrid_reg_deep_t3"] = make_nonlinear_mlp_extended_hybrid_reg_deep_t(df=3.0)   # heavier-tailed Student-t
    U["nonlinear_mlp_extended_hybrid_reg_deep_t7"] = make_nonlinear_mlp_extended_hybrid_reg_deep_t(df=7.0)   # lighter-tailed Student-t
    U["nonlinear_mlp_extended_hybrid_reg_deep_t10"] = make_nonlinear_mlp_extended_hybrid_reg_deep_t(df=10.0)  # near-Gaussian Student-t
    # independent-seed replication (seed=99) of the df-diverse ensemble members
    U["nonlinear_mlp_extended_hybrid_reg_deep_s99"] = make_nonlinear_mlp_extended_hybrid_reg_deep(seed=99)
    U["nonlinear_mlp_extended_hybrid_reg_deep_t_s99"] = make_nonlinear_mlp_extended_hybrid_reg_deep_t(df=5.0, seed=99)
    U["nonlinear_mlp_extended_hybrid_reg_deep_t7_s99"] = make_nonlinear_mlp_extended_hybrid_reg_deep_t(df=7.0, seed=99)
    U["nonlinear_mlp_extended_hybrid_reg_deep_t10_s99"] = make_nonlinear_mlp_extended_hybrid_reg_deep_t(df=10.0, seed=99)
    # third independent seed (2026) for a larger cross-seed robust ensemble
    U["nonlinear_mlp_extended_hybrid_reg_deep_t_s2026"] = make_nonlinear_mlp_extended_hybrid_reg_deep_t(df=5.0, seed=2026)
    U["nonlinear_mlp_extended_hybrid_reg_deep_t7_s2026"] = make_nonlinear_mlp_extended_hybrid_reg_deep_t(df=7.0, seed=2026)
    U["nonlinear_mlp_extended_hybrid_reg_deep_t10_s2026"] = make_nonlinear_mlp_extended_hybrid_reg_deep_t(df=10.0, seed=2026)
    # fourth independent seed (7) of the optimal t7 member only (saturation test)
    U["nonlinear_mlp_extended_hybrid_reg_deep_t7_s7"] = make_nonlinear_mlp_extended_hybrid_reg_deep_t(df=7.0, seed=7)
    # SWA (stochastic weight averaging) over the last swa_n converged epochs:
    # variance reduction at the weight level, matching the mu-ensemble theme.
    U["nonlinear_mlp_extended_hybrid_reg_deep_t_swa"] = make_nonlinear_mlp_extended_hybrid_reg_deep_t(df=5.0, swa_n=10)
    U["nonlinear_mlp_extended_hybrid_reg_deep_t7_swa"] = make_nonlinear_mlp_extended_hybrid_reg_deep_t(df=7.0, swa_n=10)
    U["nonlinear_mlp_extended_hybrid_reg_deep_t10_swa"] = make_nonlinear_mlp_extended_hybrid_reg_deep_t(df=10.0, swa_n=10)
    if rnafm_cache is not None:
        U["rnafm_linear_hybrid"] = make_rnafm_linear_hybrid(rnafm_cache)
        U["rnafm_vienna_linear_hybrid"] = make_rnafm_vienna_linear_hybrid(rnafm_cache)
        U["rnafm_pca_linear_hybrid"] = make_rnafm_pca_linear_hybrid(rnafm_cache)
        U["nonlinear_mlp_rnafm_pca_hybrid"] = make_nonlinear_mlp_rnafm_pca_hybrid(rnafm_cache)
        U["nonlinear_mlp_rnafm_only_pca_hybrid"] = make_nonlinear_mlp_rnafm_only_pca_hybrid(rnafm_cache)
        U["nonlinear_mlp_rnafm_extended_reg_deep"] = make_nonlinear_mlp_rnafm_extended_reg_deep(rnafm_cache)
    return U


BOUNDED = {"vienna_latent_operator", "kmer_latent_operator",
           "no_sequence_latent_operator", "corrected_v1_31"}


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
    # Diagnostics live on the gate record for hybrid/MLP models (which store
    # optimizer state under "gate"); fall back to top-level fields for the
    # legacy latent-operator models that keep them at the top level.
    success = bool(model.get("success", gate.get("success", True)))
    opt_msg = str(model.get("optimizer_message", gate.get("optimizer_message", "")))
    fgn = model.get("final_grad_norm", gate.get("final_grad_norm", float("nan")))
    n_iter = int(model.get("nit", gate.get("n_iter", gate.get("n_epochs", -1))))
    base = {"axis": axis, "fold": str(fold), "model_id": model_id,
            "success": success,
            "optimizer_message": opt_msg,
            "final_grad_norm": float(fgn) if fgn is not None else float("nan"),
            "n_iter": n_iter,
            "n_param": _param_count(model),
            "runtime_s": round(runtime, 3),
            "eligible": bool(gate.get("eligible", False)),
            "eligible_full_coverage": bool(elig.get("eligible", False)),
            "elig_reason": elig.get("reason"),
            **({k: gate[k] for k in
                ("projected_grad_norm", "proj_grad_tol", "n_bound_hits",
                 "n_nan_inf_params", "grad_tol", "converged", "n_epochs",
                 "max_epochs", "final_train_nll", "best_train_nll",
                 "plateau_reached")
                if k in gate})}
    return base


def _pooled_nll_by_model(all_preds):
    """junction-macro pooled NLL per model over supported non-abstain rows."""
    by = defaultdict(list)
    for p in all_preds:
        if p["support"] and not p["abstain"]:
            nll = float(row_nll([p["y"]], [p["cens"]], [p["mu"]], [p["sigma"]])[0])
            by[p["model_id"]].append((p["jid"], nll))
    out = {}
    for m, lst in by.items():
        jd = defaultdict(list)
        for j, n in lst:
            jd[j].append(n)
        out[m] = float(np.mean([np.mean(v) for v in jd.values()]))
    return out


def _paired_rows(all_preds, a_id, b_id):
    by = defaultdict(dict)
    for p in all_preds:
        if p["model_id"] in (a_id, b_id):
            if p["support"] and not p["abstain"]:
                by[p["source_row_id"]][p["model_id"]] = p
    return {rid: d for rid, d in by.items() if a_id in d and b_id in d}


def _pooled_contrast(all_preds, a_id, b_id, a_name, b_name, theta_label):
    """delta = (b - a); positive means a is better."""
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
        "hits_10pct_gate": bool(rel >= 0.10) if rel is not None else False,
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
    out = run_root / cfg.get("out_subdir", "r06_shootout")
    out.mkdir(parents=True, exist_ok=True)

    _, admitted, profile, *_ = audit_dataset(Path(cfg["canonical_source"]))
    rows = {str(r["source_row_id"]): r for r in admitted}

    specs = build_joint_edit_context_folds(admitted)
    if cfg.get("max_folds"):
        specs = specs[: int(cfg["max_folds"])]
    rnafm_cache = None
    if cfg.get("rnafm_cache"):
        from audit.benchmark.rnafm_features import load_cache
        rnafm_cache = load_cache(Path(cfg["rnafm_cache"]))
    U = _universe(rnafm_cache)
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

    # Pooled junction-macro NLL per model (authoritative estimand).
    pooled = _pooled_nll_by_model(all_preds)

    # Core contrast: does ViennaRNA representation beat matched no-sequence?
    vienna_vs_noseq = _pooled_contrast(
        all_preds, "vienna_latent_operator", "no_sequence_latent_operator",
        "vienna_latent_operator", "no_sequence_latent_operator",
        "pooled-OOF junction-macro NLL delta (no_sequence - vienna_latent_operator)")
    vienna_cluster = _edit_cluster_ci(
        all_preds, admitted, "vienna_latent_operator", "no_sequence_latent_operator")

    # Reference: does ViennaRNA beat the 63-D representation?
    vienna_vs_63d = _pooled_contrast(
        all_preds, "vienna_latent_operator", "corrected_v1_31",
        "vienna_latent_operator", "corrected_v1_31(63D)",
        "pooled-OOF junction-macro NLL delta (corrected_v1_31 - vienna_latent_operator)")

    # Reference: does k-mer beat the 63-D representation and matched no-seq?
    kmer_vs_63d = _pooled_contrast(
        all_preds, "kmer_latent_operator", "corrected_v1_31",
        "kmer_latent_operator", "corrected_v1_31(63D)",
        "pooled-OOF junction-macro NLL delta (corrected_v1_31 - kmer_latent_operator)")
    kmer_vs_noseq = _pooled_contrast(
        all_preds, "kmer_latent_operator", "no_sequence_latent_operator",
        "kmer_latent_operator", "no_sequence_latent_operator",
        "pooled-OOF junction-macro NLL delta (no_sequence - kmer_latent_operator)")
    kmer_cluster = _edit_cluster_ci(
        all_preds, admitted, "kmer_latent_operator", "no_sequence_latent_operator")

    # Reference: does ViennaRNA beat the strongest simple baseline?
    vienna_vs_scaffold = _pooled_contrast(
        all_preds, "vienna_latent_operator", "train_only_scaffold",
        "vienna_latent_operator", "train_only_scaffold",
        "pooled-OOF junction-macro NLL delta (train_only_scaffold - vienna_latent_operator)")

    # DECISIVE increment test: winning plain-linear nuisance + ViennaRNA sequence
    # block vs the same nuisance-only model (identical head class). Positive theta
    # = the sequence block adds predictive increment over the strongest simple model.
    hybrid_vs_nuisance = _pooled_contrast(
        all_preds, "vienna_linear_hybrid", "motif_topology_hierarchy",
        "vienna_linear_hybrid", "motif_topology_hierarchy",
        "pooled-OOF junction-macro NLL delta (motif_topology_hierarchy - vienna_linear_hybrid)")
    hybrid_cluster = _edit_cluster_ci(
        all_preds, admitted, "vienna_linear_hybrid", "motif_topology_hierarchy")

    # Enrichment test: extended ViennaRNA vs base 11-D ViennaRNA (same winning head).
    ext_vs_base = _pooled_contrast(
        all_preds, "vienna_extended_linear_hybrid", "vienna_linear_hybrid",
        "vienna_extended_linear_hybrid", "vienna_linear_hybrid",
        "pooled-OOF junction-macro NLL delta (vienna_linear_hybrid - vienna_extended_linear_hybrid)")
    ext_vs_nuisance = _pooled_contrast(
        all_preds, "vienna_extended_linear_hybrid", "motif_topology_hierarchy",
        "vienna_extended_linear_hybrid", "motif_topology_hierarchy",
        "pooled-OOF junction-macro NLL delta (motif_topology_hierarchy - vienna_extended_linear_hybrid)")
    ext_cluster = _edit_cluster_ci(
        all_preds, admitted, "vienna_extended_linear_hybrid", "motif_topology_hierarchy")

    # RNA-FM: does a frozen learned sequence representation beat the folding
    # proxy and the nuisance-only model (same winning head)?
    rnafm_vs_nuisance = _pooled_contrast(
        all_preds, "rnafm_linear_hybrid", "motif_topology_hierarchy",
        "rnafm_linear_hybrid", "motif_topology_hierarchy",
        "pooled-OOF junction-macro NLL delta (motif_topology_hierarchy - rnafm_linear_hybrid)")
    rnafm_vs_vienna = _pooled_contrast(
        all_preds, "rnafm_linear_hybrid", "vienna_linear_hybrid",
        "rnafm_linear_hybrid", "vienna_linear_hybrid",
        "pooled-OOF junction-macro NLL delta (vienna_linear_hybrid - rnafm_linear_hybrid)")
    rnafm_cluster = _edit_cluster_ci(
        all_preds, admitted, "rnafm_linear_hybrid", "motif_topology_hierarchy")
    rnafmvienna_vs_vienna = _pooled_contrast(
        all_preds, "rnafm_vienna_linear_hybrid", "vienna_linear_hybrid",
        "rnafm_vienna_linear_hybrid", "vienna_linear_hybrid",
        "pooled-OOF junction-macro NLL delta (vienna_linear_hybrid - rnafm_vienna_linear_hybrid)")
    rnafmvienna_vs_nuisance = _pooled_contrast(
        all_preds, "rnafm_vienna_linear_hybrid", "motif_topology_hierarchy",
        "rnafm_vienna_linear_hybrid", "motif_topology_hierarchy",
        "pooled-OOF junction-macro NLL delta (motif_topology_hierarchy - rnafm_vienna_linear_hybrid)")
    rnafmvienna_cluster = _edit_cluster_ci(
        all_preds, admitted, "rnafm_vienna_linear_hybrid", "motif_topology_hierarchy")
    rnafm_pca_vs_vienna = _pooled_contrast(
        all_preds, "rnafm_pca_linear_hybrid", "vienna_linear_hybrid",
        "rnafm_pca_linear_hybrid", "vienna_linear_hybrid",
        "pooled-OOF junction-macro NLL delta (vienna_linear_hybrid - rnafm_pca_linear_hybrid)")
    rnafm_pca_vs_nuisance = _pooled_contrast(
        all_preds, "rnafm_pca_linear_hybrid", "motif_topology_hierarchy",
        "rnafm_pca_linear_hybrid", "motif_topology_hierarchy",
        "pooled-OOF junction-macro NLL delta (motif_topology_hierarchy - rnafm_pca_linear_hybrid)")
    rnafm_pca_cluster = _edit_cluster_ci(
        all_preds, admitted, "rnafm_pca_linear_hybrid", "motif_topology_hierarchy")

    # NONLINEAR/INTERACTION: does adding Vienna x scaffold/motif interactions or
    # replacing the linear head with an MLP capture the residual sequence signal?
    interact_vs_nuisance = _pooled_contrast(
        all_preds, "vienna_interaction_linear_hybrid", "motif_topology_hierarchy",
        "vienna_interaction_linear_hybrid", "motif_topology_hierarchy",
        "pooled-OOF junction-macro NLL delta (motif_topology_hierarchy - vienna_interaction_linear_hybrid)")
    interact_vs_vienna = _pooled_contrast(
        all_preds, "vienna_interaction_linear_hybrid", "vienna_linear_hybrid",
        "vienna_interaction_linear_hybrid", "vienna_linear_hybrid",
        "pooled-OOF junction-macro NLL delta (vienna_linear_hybrid - vienna_interaction_linear_hybrid)")
    interact_cluster = _edit_cluster_ci(
        all_preds, admitted, "vienna_interaction_linear_hybrid", "motif_topology_hierarchy")

    mlp_vs_nuisance = _pooled_contrast(
        all_preds, "nonlinear_mlp_hybrid", "motif_topology_hierarchy",
        "nonlinear_mlp_hybrid", "motif_topology_hierarchy",
        "pooled-OOF junction-macro NLL delta (motif_topology_hierarchy - nonlinear_mlp_hybrid)")
    mlp_vs_vienna = _pooled_contrast(
        all_preds, "nonlinear_mlp_hybrid", "vienna_linear_hybrid",
        "nonlinear_mlp_hybrid", "vienna_linear_hybrid",
        "pooled-OOF junction-macro NLL delta (vienna_linear_hybrid - nonlinear_mlp_hybrid)")
    mlp_cluster = _edit_cluster_ci(
        all_preds, admitted, "nonlinear_mlp_hybrid", "motif_topology_hierarchy")

    # RICHER-FEATURE NONLINEAR step: does the nonlinear head finally unlock the
    # richer representations (21-D ViennaRNA, RNA-FM-PCA) that saturated the
    # linear head?  Contrast each against the base nonlinear MLP and nuisance.
    ext_mlp_vs_nuisance = _pooled_contrast(
        all_preds, "nonlinear_mlp_extended_hybrid", "motif_topology_hierarchy",
        "nonlinear_mlp_extended_hybrid", "motif_topology_hierarchy",
        "pooled-OOF junction-macro NLL delta (motif_topology_hierarchy - nonlinear_mlp_extended_hybrid)")
    ext_mlp_vs_mlp = _pooled_contrast(
        all_preds, "nonlinear_mlp_extended_hybrid", "nonlinear_mlp_hybrid",
        "nonlinear_mlp_extended_hybrid", "nonlinear_mlp_hybrid",
        "pooled-OOF junction-macro NLL delta (nonlinear_mlp_hybrid - nonlinear_mlp_extended_hybrid)")
    ext_mlp_cluster = _edit_cluster_ci(
        all_preds, admitted, "nonlinear_mlp_extended_hybrid", "motif_topology_hierarchy")

    rnafm_mlp_vs_nuisance = _pooled_contrast(
        all_preds, "nonlinear_mlp_rnafm_pca_hybrid", "motif_topology_hierarchy",
        "nonlinear_mlp_rnafm_pca_hybrid", "motif_topology_hierarchy",
        "pooled-OOF junction-macro NLL delta (motif_topology_hierarchy - nonlinear_mlp_rnafm_pca_hybrid)")
    rnafm_mlp_vs_mlp = _pooled_contrast(
        all_preds, "nonlinear_mlp_rnafm_pca_hybrid", "nonlinear_mlp_hybrid",
        "nonlinear_mlp_rnafm_pca_hybrid", "nonlinear_mlp_hybrid",
        "pooled-OOF junction-macro NLL delta (nonlinear_mlp_hybrid - nonlinear_mlp_rnafm_pca_hybrid)")
    rnafm_mlp_vs_rnafm_lin = _pooled_contrast(
        all_preds, "nonlinear_mlp_rnafm_pca_hybrid", "rnafm_pca_linear_hybrid",
        "nonlinear_mlp_rnafm_pca_hybrid", "rnafm_pca_linear_hybrid",
        "pooled-OOF junction-macro NLL delta (rnafm_pca_linear_hybrid - nonlinear_mlp_rnafm_pca_hybrid)")
    rnafm_mlp_cluster = _edit_cluster_ci(
        all_preds, admitted, "nonlinear_mlp_rnafm_pca_hybrid", "motif_topology_hierarchy")

    rnafm_only_mlp_vs_nuisance = _pooled_contrast(
        all_preds, "nonlinear_mlp_rnafm_only_pca_hybrid", "motif_topology_hierarchy",
        "nonlinear_mlp_rnafm_only_pca_hybrid", "motif_topology_hierarchy",
        "pooled-OOF junction-macro NLL delta (motif_topology_hierarchy - nonlinear_mlp_rnafm_only_pca_hybrid)")
    rnafm_only_mlp_vs_mlp = _pooled_contrast(
        all_preds, "nonlinear_mlp_rnafm_only_pca_hybrid", "nonlinear_mlp_hybrid",
        "nonlinear_mlp_rnafm_only_pca_hybrid", "nonlinear_mlp_hybrid",
        "pooled-OOF junction-macro NLL delta (nonlinear_mlp_hybrid - nonlinear_mlp_rnafm_only_pca_hybrid)")
    rnafm_only_mlp_cluster = _edit_cluster_ci(
        all_preds, admitted, "nonlinear_mlp_rnafm_only_pca_hybrid", "motif_topology_hierarchy")

    # LOCAL-CONTEXT step: does the position-anchored join-local-context one-hot
    # block add signal beyond the folding aggregates under the reg_deep arch?
    localctx_vs_nuisance = _pooled_contrast(
        all_preds, "nonlinear_mlp_extended_hybrid_localctx", "motif_topology_hierarchy",
        "nonlinear_mlp_extended_hybrid_localctx", "motif_topology_hierarchy",
        "pooled-OOF junction-macro NLL delta (motif_topology_hierarchy - nonlinear_mlp_extended_hybrid_localctx)")
    localctx_vs_reg_deep = _pooled_contrast(
        all_preds, "nonlinear_mlp_extended_hybrid_localctx", "nonlinear_mlp_extended_hybrid_reg_deep",
        "nonlinear_mlp_extended_hybrid_localctx", "nonlinear_mlp_extended_hybrid_reg_deep",
        "pooled-OOF junction-macro NLL delta (nonlinear_mlp_extended_hybrid_reg_deep - nonlinear_mlp_extended_hybrid_localctx)")
    localctx_cluster = _edit_cluster_ci(
        all_preds, admitted, "nonlinear_mlp_extended_hybrid_localctx", "motif_topology_hierarchy")

    # ROBUST-LIKELIHOOD step: does training reg_deep with a heavy-tailed
    # Student-t objective (down-weighting outlier/catastrophic folds) lower the
    # Gaussian evaluation NLL vs the Gaussian-trained reg_deep?
    robust_t_vs_nuisance = _pooled_contrast(
        all_preds, "nonlinear_mlp_extended_hybrid_reg_deep_t", "motif_topology_hierarchy",
        "nonlinear_mlp_extended_hybrid_reg_deep_t", "motif_topology_hierarchy",
        "pooled-OOF junction-macro NLL delta (motif_topology_hierarchy - nonlinear_mlp_extended_hybrid_reg_deep_t)")
    robust_t_vs_reg_deep = _pooled_contrast(
        all_preds, "nonlinear_mlp_extended_hybrid_reg_deep_t", "nonlinear_mlp_extended_hybrid_reg_deep",
        "nonlinear_mlp_extended_hybrid_reg_deep_t", "nonlinear_mlp_extended_hybrid_reg_deep",
        "pooled-OOF junction-macro NLL delta (nonlinear_mlp_extended_hybrid_reg_deep - nonlinear_mlp_extended_hybrid_reg_deep_t)")
    robust_t_cluster = _edit_cluster_ci(
        all_preds, admitted, "nonlinear_mlp_extended_hybrid_reg_deep_t", "motif_topology_hierarchy")

    report = {
        "axis": "edit_x_nested_context",
        "purpose": "REPRESENTATION_SHOOTOUT",
        "n_eligible_folds": int(df_lb[df_lb["eligible_full_coverage"] == True]["fold"].nunique()) if len(df_lb) else 0,
        "pooled_junction_macro_nll": {k: round(v, 5) for k, v in sorted(pooled.items(), key=lambda kv: kv[1])},
        "vienna_vs_no_sequence": {
            "note": "positive delta = vienna_latent_operator is BETTER than matched no-sequence.",
            "pooled": vienna_vs_noseq,
            "edit_cluster": vienna_cluster,
            "gate_10pct": 0.10,
        },
        "vienna_vs_63d": {
            "note": "positive delta = vienna_latent_operator is BETTER than corrected_v1_31 (63-D).",
            "pooled": vienna_vs_63d,
        },
        "kmer_vs_63d": {
            "note": "positive delta = kmer_latent_operator is BETTER than corrected_v1_31 (63-D).",
            "pooled": kmer_vs_63d,
        },
        "kmer_vs_no_sequence": {
            "note": "positive delta = kmer_latent_operator is BETTER than matched no-sequence.",
            "pooled": kmer_vs_noseq,
            "edit_cluster": kmer_cluster,
            "gate_10pct": 0.10,
        },
        "vienna_vs_scaffold": {
            "note": "positive delta = vienna_latent_operator is BETTER than train_only_scaffold.",
            "pooled": vienna_vs_scaffold,
        },
        "hybrid_vs_nuisance": {
            "note": ("positive delta = vienna_linear_hybrid (winning plain-linear head "
                     "+ ViennaRNA sequence block) is BETTER than motif_topology_hierarchy "
                     "(same head, nuisance-only). Decisive test of whether a sequence "
                     "representation adds increment over the strongest simple model."),
            "pooled": hybrid_vs_nuisance,
            "edit_cluster": hybrid_cluster,
            "gate_10pct": 0.10,
        },
        "extended_vs_nuisance": {
            "note": ("positive delta = vienna_extended_linear_hybrid (winning head + "
                     "21-D extended ViennaRNA) is BETTER than motif_topology_hierarchy."),
            "pooled": ext_vs_nuisance,
            "edit_cluster": ext_cluster,
            "gate_10pct": 0.10,
        },
        "extended_vs_base": {
            "note": ("positive delta = vienna_extended_linear_hybrid is BETTER than "
                     "vienna_linear_hybrid (11-D). Tests if richer folding features "
                     "push the sequence increment further."),
            "pooled": ext_vs_base,
        },
        "rnafm_vs_nuisance": {
            "note": ("positive delta = rnafm_linear_hybrid (winning head + RNA-FM "
                     "frozen 1920-D) is BETTER than motif_topology_hierarchy."),
            "pooled": rnafm_vs_nuisance,
            "edit_cluster": rnafm_cluster,
            "gate_10pct": 0.10,
        },
        "rnafm_vs_vienna": {
            "note": ("positive delta = rnafm_linear_hybrid is BETTER than "
                     "vienna_linear_hybrid (11-D folding proxy). Tests whether the "
                     "learned representation beats the folding proxy."),
            "pooled": rnafm_vs_vienna,
        },
        "rnafm_vienna_vs_vienna": {
            "note": ("positive delta = rnafm_vienna_linear_hybrid (winning head + "
                     "RNA-FM + ViennaRNA) is BETTER than vienna_linear_hybrid. Tests "
                     "whether the learned representation is complementary to the "
                     "folding proxy."),
            "pooled": rnafmvienna_vs_vienna,
        },
        "rnafm_vienna_vs_nuisance": {
            "note": ("positive delta = rnafm_vienna_linear_hybrid is BETTER than "
                     "motif_topology_hierarchy."),
            "pooled": rnafmvienna_vs_nuisance,
            "edit_cluster": rnafmvienna_cluster,
            "gate_10pct": 0.10,
        },
        "rnafm_pca_vs_vienna": {
            "note": ("positive delta = rnafm_pca_linear_hybrid (RNA-FM PCA-reduced "
                     "to 64-D, train-only) is BETTER than vienna_linear_hybrid. Tests "
                     "whether the learned representation adds increment once "
                     "dimensional overfitting is removed."),
            "pooled": rnafm_pca_vs_vienna,
        },
        "rnafm_pca_vs_nuisance": {
            "note": ("positive delta = rnafm_pca_linear_hybrid is BETTER than "
                     "motif_topology_hierarchy."),
            "pooled": rnafm_pca_vs_nuisance,
            "edit_cluster": rnafm_pca_cluster,
            "gate_10pct": 0.10,
        },
        "interaction_vs_nuisance": {
            "note": ("positive delta = vienna_interaction_linear_hybrid (winning "
                     "head + ViennaRNA x scaffold/motif interactions) is BETTER than "
                     "motif_topology_hierarchy. Tests whether the residual sequence "
                     "signal is interactive (scaffold/motif-dependent)."),
            "pooled": interact_vs_nuisance,
            "edit_cluster": interact_cluster,
            "gate_10pct": 0.10,
        },
        "interaction_vs_vienna": {
            "note": ("positive delta = vienna_interaction_linear_hybrid is BETTER "
                     "than vienna_linear_hybrid (base 11-D). Tests whether the "
                     "interaction blocks push the sequence increment further."),
            "pooled": interact_vs_vienna,
        },
        "mlp_vs_nuisance": {
            "note": ("positive delta = nonlinear_mlp_hybrid (shallow MLP on the "
                     "winning feature set, right-censored NLL) is BETTER than "
                     "motif_topology_hierarchy. Tests whether nonlinearity captures "
                     "signal a linear head cannot."),
            "pooled": mlp_vs_nuisance,
            "edit_cluster": mlp_cluster,
            "gate_10pct": 0.10,
        },
        "mlp_vs_vienna": {
            "note": ("positive delta = nonlinear_mlp_hybrid is BETTER than "
                     "vienna_linear_hybrid (linear head, same features). Tests "
                     "whether nonlinearity adds increment over the linear head."),
            "pooled": mlp_vs_vienna,
        },
        "extended_mlp_vs_nuisance": {
            "note": ("positive delta = nonlinear_mlp_extended_hybrid (nonlinear "
                     "head + 21-D ViennaRNA) is BETTER than motif_topology_hierarchy."),
            "pooled": ext_mlp_vs_nuisance,
            "edit_cluster": ext_mlp_cluster,
            "gate_10pct": 0.10,
        },
        "extended_mlp_vs_mlp": {
            "note": ("positive delta = nonlinear_mlp_extended_hybrid (21-D) is "
                     "BETTER than nonlinear_mlp_hybrid (11-D). Tests whether the "
                     "nonlinear head unlocks the richer folding representation."),
            "pooled": ext_mlp_vs_mlp,
        },
        "rnafm_mlp_vs_nuisance": {
            "note": ("positive delta = nonlinear_mlp_rnafm_pca_hybrid (nonlinear "
                     "head + 11-D ViennaRNA + RNA-FM-PCA) is BETTER than "
                     "motif_topology_hierarchy."),
            "pooled": rnafm_mlp_vs_nuisance,
            "edit_cluster": rnafm_mlp_cluster,
            "gate_10pct": 0.10,
        },
        "rnafm_mlp_vs_mlp": {
            "note": ("positive delta = nonlinear_mlp_rnafm_pca_hybrid is BETTER "
                     "than nonlinear_mlp_hybrid (base 11-D). Tests whether the "
                     "nonlinear head unlocks the learned RNA-FM representation."),
            "pooled": rnafm_mlp_vs_mlp,
        },
        "rnafm_mlp_vs_rnafm_linear": {
            "note": ("positive delta = nonlinear_mlp_rnafm_pca_hybrid is BETTER "
                     "than rnafm_pca_linear_hybrid (same features, linear head). "
                     "Tests whether nonlinearity unlocks RNA-FM that the linear "
                     "head could not."),
            "pooled": rnafm_mlp_vs_rnafm_lin,
        },
        "rnafm_only_mlp_vs_nuisance": {
            "note": ("positive delta = nonlinear_mlp_rnafm_only_pca_hybrid "
                     "(nonlinear head + RNA-FM-PCA only) is BETTER than "
                     "motif_topology_hierarchy."),
            "pooled": rnafm_only_mlp_vs_nuisance,
            "edit_cluster": rnafm_only_mlp_cluster,
            "gate_10pct": 0.10,
        },
        "rnafm_only_mlp_vs_mlp": {
            "note": ("positive delta = nonlinear_mlp_rnafm_only_pca_hybrid is "
                     "BETTER than nonlinear_mlp_hybrid (base 11-D). Isolates the "
                     "learned representation under the nonlinear head."),
            "pooled": rnafm_only_mlp_vs_mlp,
        },
        "localctx_vs_nuisance": {
            "note": ("positive delta = nonlinear_mlp_extended_hybrid_localctx "
                     "(reg_deep + 21-D ViennaRNA + 24-D join-local-context) is "
                     "BETTER than motif_topology_hierarchy. Tests whether the "
                     "position-anchored edit-site local-context block adds "
                     "sequence signal beyond folding aggregates."),
            "pooled": localctx_vs_nuisance,
            "edit_cluster": localctx_cluster,
            "gate_10pct": 0.10,
        },
        "localctx_vs_reg_deep": {
            "note": ("positive delta = nonlinear_mlp_extended_hybrid_localctx is "
                     "BETTER than nonlinear_mlp_extended_hybrid_reg_deep (same "
                     "arch, without local-context block). Tests whether the "
                     "local-context features add increment over the folding-only "
                     "reg_deep reference."),
            "pooled": localctx_vs_reg_deep,
        },
        "robust_t_vs_nuisance": {
            "note": ("positive delta = nonlinear_mlp_extended_hybrid_reg_deep_t "
                     "(reg_deep trained with a heavy-tailed Student-t objective, "
                     "df=5) is BETTER than motif_topology_hierarchy. Tests whether "
                     "robust training against outlier/catastrophic folds improves "
                     "the Gaussian evaluation NLL."),
            "pooled": robust_t_vs_nuisance,
            "edit_cluster": robust_t_cluster,
            "gate_10pct": 0.10,
        },
        "robust_t_vs_reg_deep": {
            "note": ("positive delta = nonlinear_mlp_extended_hybrid_reg_deep_t "
                     "is BETTER than nonlinear_mlp_extended_hybrid_reg_deep (same "
                     "arch + features, Gaussian-trained). Tests whether the "
                     "Student-t training objective improves on the Gaussian "
                     "objective at equal capacity."),
            "pooled": robust_t_vs_reg_deep,
        },
    }
    (out / "ShootoutReport.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n")

    status = {
        "phase": "REPRESENTATION_SHOOTOUT", "state": "DONE",
        "n_models": len(model_ids), "n_folds": len(specs),
        "n_predictions": len(all_preds), "n_leaderboard_rows": len(leaderboard),
        "n_convergence_rows": len(conv_rows), "duplicate_primary_keys": len(dups),
        "models": model_ids,
        "elapsed_s": round(time.time() - t0, 1),
        "note": ("Representation shootout on the decisive joint axis holding the "
                 "latent-operator head fixed, testing ViennaRNA features vs "
                 "matched no-sequence, 63-D, and the strongest simple baseline."),
    }
    (out / "STATUS.json").write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(status, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return status


if __name__ == "__main__":
    main(json.loads(Path(sys.argv[1]).read_text()))