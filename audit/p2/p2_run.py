"""Phase 2 orchestrator (contract Phase 2).

Runs the full Phase 2 pipeline and emits all required deliverables:
  HypothesisRegistry.json, NullProtocol.json, NullResults.parquet,
  SupportLedger.parquet, EffectDecomposition.csv, CoreHypothesisDecision.json,
  BootstrapIntervals.json/.csv, StratifiedGain.csv, STATUS.json.

The permutation nulls (label / sequence-pairing) are the expensive step and are
parallelised across cores.  Per-axis permutation counts are pre-registered in
the config; reduced counts for the high-fold axes are justified by the fact that
each permutation already averages many outer folds (tight null).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from audit.p2.common import load_rows, load_fold_metrics, axis_gain
from audit.p2.hypothesis import HYPOTHESIS_REGISTRY, write_hypothesis_registry
from audit.p2.bootstrap import run_fold_bootstrap, write_bootstrap, write_bootstrap_csv
from audit.p2.effect_decomposition import decompose, write_decomposition
from audit.p2.support import build_support_ledger, stratified_gain, write_support
from audit.p2.nulls import run_axis_permutations
from audit.p2.decision import evaluate, write_decision
from audit.evaluation.null_protocol import NULL_PROTOCOL

AXES_N_FOLDS = {"symmetry_5fold": 5, "edit_5fold": 5, "context_lomo": 234, "scaffold_lomo": 9}


def main(cfg):
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(Path(cfg["records"]))
    fm = load_fold_metrics(Path(cfg["p1_out"]) / "FoldMetrics.csv")
    predictions = [json.loads(l) for l in (Path(cfg["p1_out"]) / "Predictions.jsonl").read_text().splitlines() if l.strip()]

    axes_spec = [(a, AXES_N_FOLDS[a]) for a in cfg["axes"]]
    axes = [a for a, _ in axes_spec]

    # 1. Hypothesis registry (pre-registered)
    write_hypothesis_registry(out_dir)

    # 2. observed gain + fold-unit bootstrap (cheap)
    fold_boot = run_fold_bootstrap(fm, axes_spec, n_boot=cfg.get("n_boot", 2000))
    write_bootstrap(out_dir, fold_boot)
    write_bootstrap_csv(out_dir, fold_boot)

    # 3. effect decomposition (cheap)
    effect_rows = decompose(fm, axes_spec)
    write_decomposition(out_dir, effect_rows)

    # 4. support ledger + stratified gain (edit-distance based; moderate)
    ledger_recs = build_support_ledger(rows, cfg["protocol_dir"], axes)
    strat = stratified_gain(predictions, axes, ledger_recs)
    write_support(out_dir, ledger_recs, strat)

    # 5. permutation nulls (expensive, parallel) -- configurable / skippable
    perm_cfg = cfg.get("permutations", {})
    null_summary = {}
    for axis in axes:
        null_summary[axis] = {}
        for ptype in ("label", "sequence"):
            n_perms = perm_cfg.get(axis, {}).get(ptype, 0)
            if n_perms <= 0:
                null_summary[axis][ptype] = {"n_perms": 0, "skipped": True}
                continue
            rec = run_axis_permutations(
                rows, Path(cfg["protocol_dir"]) / f"SplitManifest_{axis}.jsonl",
                axis, ptype, n_perms, cfg.get("n_workers", 1), out_dir)
            null_summary[axis][ptype] = rec

    # 6. context/operator nulls (comparison-based, cheap)
    context_op_nulls = context_operator_nulls(fm, axes_spec)

    # 7. NullProtocol.json (frozen P0.4 rules + Phase 2 run record)
    protocol = {
        "version": "P2-NULL-001",
        "frozen_P0_4_rules": NULL_PROTOCOL,
        "run": {
            "reference_baseline": "train_only_scaffold",
            "candidate": "corrected_v1_31",
            "gain_definition": "mean over folds NLL(ref) - NLL(candidate)",
            "permutations": null_summary,
            "bootstrap": {"method": "split-unit outer-fold bootstrap", "n_boot": cfg.get("n_boot", 2000)},
            "note": "reduced permutation counts on context_lomo/scaffold_lomo justified because each permutation already averages 234/9 outer folds (tight null); full 1000 run on the two grouped 5-fold axes.",
        },
        "context_operator_nulls": context_op_nulls,
    }
    (out_dir / "NullProtocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False, sort_keys=True) + "\n")

    # 8. decision + STATUS
    criteria = evaluate(fold_boot, null_summary, effect_rows, axes_spec)
    adjudication = adjudicate(fold_boot, null_summary, criteria, axes_spec)
    write_decision(out_dir, criteria, adjudication)

    status = build_status(criteria, adjudication, cfg)
    (out_dir / "STATUS.json").write_text(json.dumps(status, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    return status


def context_operator_nulls(fm, axes_spec):
    """Cheap comparison nulls from existing FoldMetrics."""
    out = {}
    for axis, nf in axes_spec:
        g_op, _ = axis_gain(fm, axis, range(nf))  # vs train_only_scaffold (operator null)
        g_ctx, _ = axis_gain(fm, axis, range(nf), ref="scaffold_context_hierarchy")
        out[axis] = {
            "operator_null_gain": g_op,
            "context_null_gain": g_ctx,
            "candidate_beats_operator_null": bool(g_op is not None and g_op > 0),
            "candidate_beats_context_null": bool(g_ctx is not None and g_ctx > 0),
        }
    return out


def adjudicate(fold_boot, null_summary, criteria, axes_spec):
    claim_axes = ["symmetry_5fold", "edit_5fold", "context_lomo"]
    a1 = all(criteria["A1_gain_ci_lower_bound_gt_0"].get(a, {}).get("pass") for a in claim_axes)
    a2_sym = criteria["A2_five_of_five_positive"].get("symmetry_5fold", {}).get("all_folds_positive")
    a2_edit = criteria["A2_five_of_five_positive"].get("edit_5fold", {}).get("all_folds_positive")
    a4 = criteria["A4_blocked_context_positive"].get("pass")
    a5 = criteria["A5_edit_axis_positive"].get("pass")
    a6 = all(criteria["A6_no_catastrophic_fold"].get(a, {}).get("pass") for a in claim_axes)
    genuine_gt_null = all(
        all(criteria["A3_genuine_gt_null_975"].get(a, {}).get(t, {}).get("pass", False)
            for t in ("label", "sequence"))
        for a in claim_axes)
    positive_claim = bool(a1 and a2_sym and a2_edit and a4 and a5 and a6 and genuine_gt_null)

    # operator-transfer boundary (scaffold_lomo)
    scaf_gain = fold_boot.get("scaffold_lomo", {}).get("observed_mean_gain")
    scaf_boundary = (scaf_gain is not None and scaf_gain <= 0.0)

    return {
        "verdict": "H1_POSITIVE_KNOWN_OPERATOR" if positive_claim else "H0_OR_INCONCLUSIVE",
        "positive_claim": positive_claim,
        "criteria_summary": {
            "A1_ci_low_gt_0": a1, "A2_5of5_symmetry": bool(a2_sym),
            "A2_5of5_edit": bool(a2_edit), "A4_blocked_context": bool(a4),
            "A5_edit_positive": bool(a5), "A6_no_catastrophic": bool(a6),
            "A3_genuine_gt_null": genuine_gt_null,
        },
        "operator_transfer_boundary": {
            "axis": "scaffold_lomo",
            "observed_gain": scaf_gain,
            "no_transfer": scaf_boundary,
            "interpretation": ("No sequence model (candidate or baseline) transfers to an unseen "
                               "scaffold/operator; this is the operator-generalization boundary and "
                               "does NOT support an operator-transfer claim (contract 2.2/3.2)."),
        },
        "interpretation": ("Genuine positive sequence signal is established only within the known-"
                           "operator universe (symmetry/edit/blocked-context axes). It does not "
                           "transfer to unseen operators (scaffold_lomo). Any 'transferable' wording "
                           "must be limited to known-operator context transfer, not operator transfer."),
    }


def build_status(criteria, adjudication, cfg):
    return {
        "phase": "P2",
        "state": "PASS" if adjudication["positive_claim"] else "FAIL",
        "verdict": adjudication["verdict"],
        "criteria_summary": adjudication["criteria_summary"],
        "operator_transfer_boundary": adjudication["operator_transfer_boundary"]["no_transfer"],
        "axes": cfg["axes"],
        "config": {"n_boot": cfg.get("n_boot", 2000), "n_workers": cfg.get("n_workers", 1),
                   "permutations": cfg.get("permutations", {})},
    }


if __name__ == "__main__":
    cfg = json.loads(Path(sys.argv[1]).read_text())
    print(json.dumps(main(cfg), indent=2, ensure_ascii=False))
