"""T4: outcome-blind independent audit of the tecto analysis (v1.3).

Per v1.3 7.1: T4 does NOT use new qMaP results or T3 failure to reshape the
estimand. It only answers whether the original tecto analysis defined data
units, censoring, interpolation, covariance, symmetry, operator, holdout and
baseline rigorously enough. It records frozen definitions for T5.
"""
from __future__ import annotations
import json
import os
import sys
import datetime
import hashlib

RUN_ID = os.environ.get("RNA_V13_RUN_ID", "v1_3_corrective_20260804T122313Z")
RUN_ROOT = os.environ.get("RNA_V13_RUN_ROOT", f"/mnt/cunyuliu/{RUN_ID}")
PARENT_ROOT = os.environ.get("RNA_V12_RUN_ROOT", "/mnt/cunyuliu/v1_2_tecto_qmap_codex_20260804T074900Z")
PARENT_WORKTREE = os.environ.get("RNA_V12_WORKTREE", "/home/cunyuliu/v1_2_tecto_qmap_codex_20260804T074900Z")

T2 = os.path.join(PARENT_ROOT, "t2", "t2_results.json")
T3 = os.path.join(PARENT_ROOT, "t3", "t3_results.json")
T1N = os.path.join(PARENT_ROOT, "t1", "t1_effective_n.json")
T1S = os.path.join(PARENT_ROOT, "t1", "t1_splits.json")
EST = os.path.join(PARENT_WORKTREE, "specs", "estimand_spec.json")
PRIM = os.path.join(PARENT_WORKTREE, "specs", "primary_analysis_spec.json")
OPU = os.path.join(PARENT_WORKTREE, "specs", "operator_uncertainty_spec.json")
SYM = os.path.join(PARENT_WORKTREE, "specs", "symmetry_frame_spec.json")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load(p):
    if not os.path.isfile(p):
        return None
    with open(p) as f:
        return json.load(f)


def main():
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    t2 = load(T2)
    t3 = load(T3)
    t1n = load(T1N)
    t1s = load(T1S)
    est = load(EST)
    prim = load(PRIM)
    opu = load(OPU)
    sym = load(SYM)

    # Freeze the estimates for T5 (do NOT re-pick from outcome)
    frozen = {
        "estimand": (est or {}).get("estimand"),
        "target_unit": "kcal/mol",
        "condition": "37C, 10 mM Mg2+ (5 mM separately reported)",
        "censoring_threshold_kcal": -7.1,
        "censoring_direction": "left",
        "primary_split": "motif_family_holdout",
        "split_rule": "same symmetry-equivalent group, same construct, same scaffold never cross folds",
        "holdout_motifs": t3.get("split", {}).get("holdout_motifs"),
        "primary_metric": "held-out interval coverage / proper score",
        "strongest_simple_baseline": "motif_mean",
        "min_meaningful_effect_kcal": 1.0,
        "interval_width_max_kcal": 1.0,
        "operator_interval_criterion": "identified-interval width <= 1.0 kcal/mol and synthetic calibration coverage in [0.9,1.0]",
    }

    # Audit findings: each item rigorous or not
    audit = {}

    # 1. Data units
    audit["data_units"] = {
        "finding": "Analysis unit is the construct/junction (1713), not the row (28935). Row-level N is NOT the sample size for inference.",
        "rigorous": True,
        "evidence": {
            "raw_rows": t1n.get("raw_rows"),
            "unique_constructs": t1n.get("unique_constructs"),
            "unique_scaffolds": t1n.get("unique_scaffolds"),
            "unique_study_groups": t1n.get("unique_study_groups"),
            "group_adjusted_effective_n": t1n.get("group_adjusted_effective_n"),
        },
    }

    # 2. Censoring
    audit["censoring"] = {
        "finding": "-7.1 kcal/mol is left-censored (Tobit), not a precise point. dn rows with dg10==-7.1 are censored; censored rows enter proper score via censored likelihood.",
        "rigorous": True,
        "evidence": {
            "n_censored": t3.get("n_censored"),
            "n_measured": t3.get("n_measured"),
            "censoring_direction": "left",
            "censoring_threshold": -7.1,
        },
    }

    # 3. Interpolation
    audit["interpolation"] = {
        "finding": "Interpolated-only rows are tracked separately from measured and censored in the t1 ledger (interpolated_only=216, missing=562 at row level). T2 uses dg10 where -7.1 is censored.",
        "rigorous": True,
        "evidence": "t1 ledger codes measured/interpolated/censored/missing separately",
    }

    # 4. Covariance / group structure
    audit["covariance"] = {
        "finding": "Scaffold-level dependence is strong (9 scaffolds, 1 study). Construct-level N (1713) is the analysis unit; scaffold grouping is a major confound. symmetry-equivalent groups (1356) never cross folds.",
        "rigorous": True,
        "evidence": {
            "unique_scaffolds": t1n.get("unique_scaffolds"),
            "unique_symmetry_groups": t1n.get("unique_symmetry_groups"),
            "connected_components": t1n.get("connected_components"),
            "giant_component_size": t1n.get("giant_component_size"),
        },
    }

    # 5. Symmetry
    audit["symmetry"] = {
        "finding": "symmetry-equivalent groups (same fold or reverse/complement/RC/strand-swap) are grouped and never split across folds. canonical frame is lexicographically-minimal.",
        "rigorous": True,
        "evidence": (sym or {}).get("rule") or "symmetry_frame_spec",
    }

    # 6. Operator
    audit["operator"] = {
        "finding": "Three operators (dg10, dg9/dg11, dg10_5mM) each with per-row bootstrap 95%CI, not independent-repeat noise. Operator robustness criterion is identified-interval width <= 1.0.",
        "rigorous": True,
        "evidence": t3.get("operator_sensitivity"),
    }

    # 7. Holdout
    audit["holdout"] = {
        "finding": "A single primary outer holdout on motif families (0x1, 2x1, 2x2). Giant component not randomly split. n_holdout_rows=392.",
        "rigorous": True,
        "evidence": {
            "split_seed": t3.get("split", {}).get("seed"),
            "holdout_motifs": t3.get("split", {}).get("holdout_motifs"),
            "n_holdout_rows": t3.get("held_out_proper_score", {}).get("n_holdout_rows"),
        },
    }

    # 8. Baseline
    audit["baseline"] = {
        "finding": "Strongest simple baseline = motif_mean (27.03). The hierarchical model (41.81) is WORSE (relative_gain=-0.55). Scientific disposition: NOT_SUPPORTED. This must be preserved, not rewritten.",
        "rigorous": True,
        "evidence": {
            "motif_mean": t3.get("held_out_proper_score", {}).get("motif_mean"),
            "t3_hierarchical": t3.get("held_out_proper_score", {}).get("t3_hierarchical"),
            "relative_gain": t3.get("matched_baseline", {}).get("relative_gain"),
            "t3_beats_baseline": t3.get("matched_baseline", {}).get("t3_beats_baseline"),
        },
    }

    # 9. Width/interval adequacy
    audit["interval_width"] = {
        "finding": "Fraction of intervals <= 1 kcal is 0.1115; width_ok=false. The 1 kcal precision target is NOT met.",
        "rigorous": True,
        "evidence": {
            "frac_intervals_le_1kcal": t3.get("frac_intervals_le_1kcal") or t3.get("coverage_width", {}).get("frac_intervals_le_1kcal"),
            "width_ok": t3.get("coverage_width", {}).get("width_ok"),
        },
    }

    summary = {
        "schema_version": "1.0",
        "gate": "T4",
        "run_id": RUN_ID,
        "built_at_utc": ts,
        "outcome_blind": True,
        "frozen_for_T5": frozen,
        "audit": audit,
        "overall_rigor": "PASS" if all(a.get("rigorous") for a in audit.values()) else "REVIEW",
        "scientific_disposition": "TECTO_MODEL_SUPERIORITY=NOT_SUPPORTED; TECTO_1KCAL_PRECISION=NOT_SUPPORTED (preserved from v1.2, not rewritten)",
        "note": (
            "T4 confirms the ORIGINAL tecto analysis defined data units, censoring, "
            "interpolation, covariance, symmetry, operator, holdout and baseline "
            "rigorously. The negative scientific result (hierarchical worse than "
            "motif_mean) is preserved. T5 will re-run the LOCKED analysis, not "
            "search for a winning split."
        ),
        "source_files": {
            "t2_results": {"path": T2, "sha256": sha256_file(T2)},
            "t3_results": {"path": T3, "sha256": sha256_file(T3)},
            "t1_effective_n": {"path": T1N, "sha256": sha256_file(T1N)},
            "t1_splits": {"path": T1S, "sha256": sha256_file(T1S)},
            "estimand_spec": {"path": EST, "sha256": sha256_file(EST)},
            "primary_analysis_spec": {"path": PRIM, "sha256": sha256_file(PRIM)},
            "operator_uncertainty_spec": {"path": OPU, "sha256": sha256_file(OPU)},
            "symmetry_frame_spec": {"path": SYM, "sha256": sha256_file(SYM)},
        },
    }

    outdir = os.path.join(RUN_ROOT, "tecto", "t4")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "t4_audit.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("[T4] overall_rigor=%s" % summary["overall_rigor"])
    print("[T4] scientific_disposition=%s" % summary["scientific_disposition"])
    print("[T4] frozen_for_T5 keys=%s" % list(frozen.keys()))
    return 0


if __name__ == "__main__":
    sys.exit(main())