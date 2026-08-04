#!/usr/bin/env python3
# Q4 finalize.
from __future__ import annotations
import json, hashlib
from pathlib import Path
from datetime import datetime, timezone

WT = Path("/home/cunyuliu/rna_junction_preorganization_v1_2_20260803")
QDATA = Path("/mnt/cunyuliu/rna_junction_preorganization_v1_2_20260803/qmap")
Q4DIR = QDATA / "q4"
MANIFEST = WT / "manifests" / "canonical_manifest_v1_2_20260803.json"

summary = json.loads((Q4DIR / "q4_freeze_summary.json").read_text())
checks = {
    "frozen_before_viewing_transfer_outcome": summary["frozen_before_viewing_transfer_outcome"],
    "n_variants_is_98": summary["n_variants"] == 98,
    "leakage_violations_zero": summary["leakage_violations"] == 0,
    "same_variant_all_rows_same_fold": summary["same_variant_all_rows_same_fold"],
    "selection_boundary_locked": summary["selection_boundary_locked"],
    "mutation_graph_locked": summary["mutation_graph_locked"],
    "outer_groups_locked": summary["outer_groups_locked"],
    "baseline_locked": summary["baseline_locked"],
    "primary_metric_locked": summary["primary_metric_locked"],
    "secondary_metrics_locked": summary["secondary_metrics_locked"],
    "minimum_meaningful_effect_locked": summary["minimum_meaningful_effect_locked"],
    "power_rule_locked": summary["power_rule_locked"],
    "negative_controls_locked": summary["negative_controls_locked"],
    "calibration_rule_locked": summary["calibration_rule_locked"],
    "interval_rule_locked": summary["interval_rule_locked"],
    "outcome_adjudication_rule_locked": summary["outcome_adjudication_rule_locked"],
}
all_pass = all(checks.values())
gate_result = "PASS" if all_pass else "FAIL"
ts = datetime.now(timezone.utc).isoformat()

q4_manifest = {
    "gate": "Q4", "title": "Selection, split and analysis freeze",
    "gate_result": gate_result, "timestamp_utc": ts,
    "n_variants": summary["n_variants"], "k_folds": summary["k_folds"],
    "fold_sizes": summary["fold_sizes"], "n_mutation_graph_edges": summary["n_mutation_graph_edges"],
    "n_connected_components": summary["n_connected_components"],
    "leakage_violations": summary["leakage_violations"],
    "checks": checks,
    "spec_path": "specs/q4_selection_split_freeze_spec.json",
    "build_script": "scripts/q4_build.py", "finalize_script": "scripts/finalize_q4.py",
    "spec_sha256": hashlib.sha256((WT/"specs"/"q4_selection_split_freeze_spec.json").read_bytes()).hexdigest(),
    "build_script_sha256": hashlib.sha256((WT/"scripts"/"q4_build.py").read_bytes()).hexdigest(),
    "contract_ref": "提示词/rna 三级.md §18 Q4 (lines 1068-1091)",
}
(Q4DIR / "q4_manifest.json").write_text(json.dumps(q4_manifest, indent=2))

sentinel = {"gate": "Q4", "gate_result": gate_result, "timestamp_utc": ts,
            "n_variants": summary["n_variants"], "k_folds": summary["k_folds"],
            "leakage_violations": summary["leakage_violations"],
            "all_freeze_items_locked": all_pass}
(WT / "Sentinel_Q4.txt").write_text(json.dumps(sentinel, indent=2))

m = json.loads(MANIFEST.read_text())
m["gate_statuses"]["Q4"] = gate_result
m["current_operational_state"] = "RUNNING" if gate_result == "PASS" else "RUNNING"
m["last_updated_utc"] = ts
MANIFEST.write_text(json.dumps(m, indent=2))
print("[Q4-finalize] gate_result=" + gate_result)
print("[Q4-finalize] checks=" + json.dumps(checks, indent=2))
