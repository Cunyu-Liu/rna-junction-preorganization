"""R0.3 runner: freeze nesting, estimands, and joint split feasibility.

Contract §13.4 / §2.1.1 / §7.2.  Loads admitted rows from the canonical
source (faithful reconstruction via audit_dataset), then writes:
  ContextOperatorNestingManifest.json
  SplitFeasibility.json
  MetricSpec_v2.json
  STATUS.json
into RUN_ROOT/splits/ (under the new isolated R0 run root).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from audit.data.audit_dataset import audit_dataset
from audit.splits.context_operator_nesting import write_nesting
from audit.splits.joint_blocked import write_feasibility

METRIC_SPEC_V2 = {
    "version": "v2",
    "frozen": True,
    "primary": {
        "name": "pooled_OOF_junction_macro_right_censored_nll",
        "aggregation": "pooled OOF: collect per-junction mean NLL over test rows, then macro-mean",
        "units": "nats",
    },
    "estimands": {
        "pooled_junction_macro": "junction-mean NLL macro-averaged over test junctions",
        "nested_context_macro": "junction-mean NLL macro-averaged over nested helix contexts",
        "scaffold_bundle_macro": "bundle-mean NLL macro-averaged over scaffold+context bundles",
    },
    "naming_rule": "the same string 'NLL' must NEVER mix pooled / nested-context / scaffold-bundle aggregation",
    "selective": {
        "coverage_floor": "pre-registered per claim",
        "comparator": "coverage-matched",
        "metrics": ["supported_junction_macro_nll", "risk_coverage", "AURC", "abstention_cost"],
        "never_replaces_primary": True,
    },
}


def utc_now():
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main(cfg: dict):
    run_root = Path(cfg["run_root"])
    run_id = cfg["run_id"]
    utc = utc_now()
    out = run_root / "splits"
    out.mkdir(parents=True, exist_ok=True)

    _, admitted, profile, eff_n, dep, exposure = audit_dataset(Path(cfg["canonical_source"]))

    nesting = write_nesting(admitted, out / "ContextOperatorNestingManifest.json")
    feas = write_feasibility(admitted, out)

    (out / "MetricSpec_v2.json").write_text(
        json.dumps(METRIC_SPEC_V2, indent=2, ensure_ascii=False, sort_keys=True) + "\n")

    status = {
        "run_id": run_id, "phase": "R0.3", "generated_at_utc": utc,
        "admitted_rows": len(admitted),
        "profile": profile,
        "nesting": {
            "strict_nested": nesting["strict_nested"],
            "n_contexts": nesting["n_contexts"],
            "n_scaffolds": nesting["n_scaffolds"],
            "contexts_per_scaffold": nesting["contexts_per_scaffold"],
        },
        "joint_feasibility": {
            "edit_x_nested_context": feas["joint_edit_x_nested_context"]["feasible"],
            "edit_x_nested_context_n_feasible_folds": feas["joint_edit_x_nested_context"]["n_feasible"],
            "seq_x_scaffold_bundle": feas["joint_seq_x_scaffold_bundle"]["feasible"],
            "seq_x_scaffold_bundle_n_feasible_folds": feas["joint_seq_x_scaffold_bundle"]["n_feasible"],
        },
        "artifacts": {
            "ContextOperatorNestingManifest.json": str(out / "ContextOperatorNestingManifest.json"),
            "SplitFeasibility.json": str(out / "SplitFeasibility.json"),
            "MetricSpec_v2.json": str(out / "MetricSpec_v2.json"),
        },
    }
    (out / "STATUS.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(status, indent=2, ensure_ascii=False))
    return status


if __name__ == "__main__":
    main(json.loads(Path(sys.argv[1]).read_text()))
