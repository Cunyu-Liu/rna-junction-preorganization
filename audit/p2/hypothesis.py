"""Phase 2 pre-registered hypotheses and primary contrasts (contract Phase 2).

The scientific question (contract 3.2): does junction sequence provide
incremental, transferable information beyond motif, helix context,
scaffold/operator, local neighborhood and the censoring margin?

We pre-register H0/H1 and the primary contrasts BEFORE any Phase 2 run so the
acceptance criteria are fixed a priori.  All statistics are computed from the
frozen Phase 1 FoldMetrics (full-config corrected_v1_31 vs train_only_scaffold)
and the Phase 2 permutation/bootstraps.
"""
from __future__ import annotations

import json
from pathlib import Path

# Reference (nuisance) baseline for the gain: train_only_scaffold is the
# strongest sequence-free baseline on every axis (Phase 1 Leaderboard), i.e. the
# operator-null.  gain = NLL(ref) - NLL(candidate); positive means the candidate
# (corrected_v1_31) is better (lower NLL).
REFERENCE_BASELINE = "train_only_scaffold"
CANDIDATE = "corrected_v1_31"
CONTEXT_NULL_BASELINE = "scaffold_context_hierarchy"

HYPOTHESIS_REGISTRY = {
    "version": "P2-HYP-001",
    "scientific_question": "Does junction sequence encode transferable preorganization info beyond scaffold/context/motif/neighborhood/censor margin?",
    "estimand": "junction-macro right-censored NLL (audit.evaluation.metrics)",
    "gain_definition": "mean over outer folds of [NLL(train_only_scaffold, fold) - NLL(corrected_v1_31, fold)]",
    "reference_baseline": REFERENCE_BASELINE,
    "candidate": CANDIDATE,
    "hypotheses": {
        "H0": {
            "statement": "After controlling scaffold/context/motif/neighborhood/censor margin, junction sequence provides NO additional transferable information (gain <= 0).",
            "prediction": "observed gain has CI upper bound <= 0 and/or is reproducible by permutation nulls."
        },
        "H1": {
            "statement": "Junction sequence provides non-zero incremental information (gain > 0) that is NOT reproducible by any null.",
            "prediction": "gain CI lower bound > 0; gain positive on 5/5 outer folds; genuine gain > null 97.5th percentile upper bound; blocked-context and edit axes positive; no catastrophic fold."
        }
    },
    "contrasts": {
        "C1_sequence_vs_operator_null": {
            "description": "candidate beats the operator-null (train_only_scaffold) -> the sequence/operator-transfer gain",
            "rule": "gain CI lower bound > 0 and gain > 0 on 5/5 outer folds"
        },
        "C2_sequence_vs_context_null": {
            "description": "candidate beats scaffold_context_hierarchy on the blocked-context (context_lomo) axis",
            "rule": "context_lomo gain vs context-null > 0"
        },
        "C3_genuine_vs_permutation_nulls": {
            "description": "observed genuine gain exceeds the label/sequence-pairing permutation null 97.5th percentile upper bound",
            "rule": "genuine gain > null 97.5% upper bound on the claimed axes"
        },
        "C4_fold_positivity": {
            "description": "gain positive on all outer folds of the claimed axes",
            "rule": "5/5 outer folds positive (symmetry, edit); context_lomo aggregated positive"
        },
        "C5_no_catastrophic_fold": {
            "description": "no outer fold with relative gain < -10% vs the reference baseline",
            "rule": "all folds gain_relative >= -10%"
        }
    },
    "axes_of_claim": ["symmetry_5fold", "edit_5fold", "context_lomo"],
    "operator_transfer_boundary_axis": "scaffold_lomo",
    "pre_registered": "2026-08-08",
    "note": "scaffold_lomo (leave-one-scaffold-out) is expected to be the operator-transfer boundary; a null-like/catastrophic result there is a documented negative boundary, not a positive claim axis."
}


def write_hypothesis_registry(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "HypothesisRegistry.json").write_text(
        json.dumps(HYPOTHESIS_REGISTRY, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    return out_dir / "HypothesisRegistry.json"


if __name__ == "__main__":
    import sys
    print(write_hypothesis_registry(Path(sys.argv[1])))
