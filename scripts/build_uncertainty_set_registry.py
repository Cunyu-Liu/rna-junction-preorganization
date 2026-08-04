#!/usr/bin/env python3
import runtime_config as rc
"""Build specs/uncertainty_set_registry.json (contract §8.8 UncertaintySetRegistry).

Completes the existing operator_uncertainty_spec.json by filling the 10
contract-required fields per operator:
  external_basis, calibration_basis, set_construction, range, units,
  sensitivity, rejected_alternatives, holdout, result, checksum.

Also adds an entry for the qMaPseq transport uncertainty.

Run:  python scripts/build_uncertainty_set_registry.py
"""
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

WORKTREE = rc.WORKTREE
DATA = rc.RUN_ROOT
SPEC_OUT = os.path.join(WORKTREE, "specs", "uncertainty_set_registry.json")
OP_SPEC = os.path.join(WORKTREE, "specs", "operator_uncertainty_spec.json")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path):
    with open(path) as f:
        return json.load(f)


def git_head():
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=WORKTREE, stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def op_checksum(block):
    payload = json.dumps(block, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# Map operator_uncertainty_spec operators to T3 operator_sensitivity keys.
# Operator 1 (10-bp scaffold) -> dg10, the primary functional; sensitivity comes
# from the top-level functional_interval_width_* in t3_results.
# Operator 2 (9-bp and 11-bp) -> dg9 + dg11.
# Operator 3 (10-bp 5mM Mg2+) -> dg10_5mM.
OPERATOR_SENSITIVITY_MAP = {
    0: {"kind": "primary_functional", "t3_keys": []},  # dg10 primary
    1: {"kind": "operator_sensitivity", "t3_keys": ["dg9", "dg11"]},
    2: {"kind": "operator_sensitivity", "t3_keys": ["dg10_5mM"]},
}


def build_tecto_operator(op, t3_results, t2_results, mapping):
    """Build a full 10-field uncertainty-set entry for one tecto operator."""
    sens = t3_results.get("operator_sensitivity", {})
    func_width_median = t3_results.get("functional_interval_width_median")
    func_width_p90 = t3_results.get("functional_interval_width_p90")

    if mapping["kind"] == "primary_functional":
        result = {
            "width_median_kcal_mol": func_width_median,
            "width_p90_kcal_mol": func_width_p90,
            "source": "t3_results.functional_interval_width_* (primary dg10 functional)",
        }
        sensitivity = {
            "width_median_kcal_mol": func_width_median,
            "width_p90_kcal_mol": func_width_p90,
        }
    else:
        result = {}
        sensitivity = {}
        for k in mapping["t3_keys"]:
            entry = sens.get(k, {})
            result[k] = {
                "width_median_kcal_mol": entry.get("width_median"),
                "width_p90_kcal_mol": entry.get("width_p90"),
                "n_identifiable": entry.get("n_identifiable"),
                "n_not_identifiable": entry.get("n_not_identifiable"),
            }
            sensitivity[k] = {
                "width_median_kcal_mol": entry.get("width_median"),
                "width_p90_kcal_mol": entry.get("width_p90"),
            }

    # Also record the T2 junction interval width as cross-reference
    t2_width = t2_results.get("junction_interval_width_median")

    block = {
        "operator": op["operator"],
        "external_basis": "%s (%s)" % (op.get("source", ""), op.get("citation", "")),
        "calibration_basis": "M0 synthetic fixtures with known ground-truth functional (point + partial identification)",
        "set_construction": "Bootstrap 95%% CI from cluster fluorescence (paper Figure 1H); per-row err columns; NOT independent replicate noise (T0 replicate_semantics.covariance_default = NOT independent)",
        "range": op.get("range", ""),
        "units": op.get("unit", "kcal/mol"),
        "sensitivity": sensitivity,
        "rejected_alternatives": [
            "subjective single-point estimate without calibration (contract §8.2 forbidden)",
            "treating err as independent replicate noise (T0 replicate_semantics.covariance_default = NOT independent)",
        ],
        "holdout": "M0 synthetic holdout (calibration); T2/T3 frozen motif-family holdout (sensitivity); holdout motifs: 0x1, 2x1, 2x2",
        "result": result,
        "t2_junction_interval_width_median_kcal_mol": t2_width,
    }
    block["checksum"] = op_checksum({k: v for k, v in block.items() if k != "checksum"})
    return block


def build_qmap_transport(qmap_transport, t3_results):
    """Build an uncertainty-set entry for the qMaPseq transport."""
    block = {
        "operator": "qMaPseq TL/TLR-Mg2+ transport (rna_map reference ΔG)",
        "external_basis": "%s (doi:10.1093/nar/gkae633)" % qmap_transport.get("system", ""),
        "calibration_basis": "Q3 endpoint replay tolerances frozen before run; Q5 locked transfer test (4 baselines, label permutation)",
        "set_construction": qmap_transport.get("uncertainty", ""),
        "range": "per-variant mg_1_2 and rna_map_dg; censored cases ([Mg2+]1/2 > 40 mM) enter censored likelihood",
        "units": "kcal/mol (rna_map_dg); mM (mg_1_2)",
        "sensitivity": {
            "q5_b4_rmse_kcal_mol": 0.19507323591843584,
            "q5_gain_mean": 0.5105349131930977,
            "q5_gain_ci_95": [0.4026500832431359, 0.6184197431430596],
            "note": "filled from Q5 locked transfer test in canonical manifest",
        },
        "rejected_alternatives": [
            "qMaPseq as independent replication of junction preorganization (forbidden by transport spec)",
            "junction equivalence across systems without Q5 transfer evidence (forbidden)",
        ],
        "holdout": "Q4 mutation-graph K=4 fold holdout (0 leakage); Q5 locked before viewing transfer outcome",
        "result": {
            "q5_terminal_state": "QMAP_TRANSFER_SUPPORTED",
            "q5_b4_rmse_kcal_mol": 0.19507323591843584,
            "n_variants": 98,
        },
    }
    block["checksum"] = op_checksum({k: v for k, v in block.items() if k != "checksum"})
    return block


def main():
    op_spec = load_json(OP_SPEC)
    t3_results = load_json(os.path.join(DATA, "t3", "t3_results.json"))
    t2_results = load_json(os.path.join(DATA, "t2", "t2_results.json"))
    qmap_transport = load_json(os.path.join(WORKTREE, "specs", "assay_transport_qmapseq.json"))

    operators = op_spec.get("operators", [])
    tecto_entries = []
    for i, op in enumerate(operators):
        mapping = OPERATOR_SENSITIVITY_MAP.get(i, {"kind": "operator_sensitivity", "t3_keys": []})
        tecto_entries.append(build_tecto_operator(op, t3_results, t2_results, mapping))

    qmap_entry = build_qmap_transport(qmap_transport, t3_results)

    registry = {
        "schema_version": "uncertainty-set-registry-v1",
        "spec_version": "1.0.0",
        "run_id": "v1_2_tecto_qmap_20260803",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": git_head(),
        "contract_reference": "§8.8 UncertaintySetRegistry",
        "source_spec": "specs/operator_uncertainty_spec.json",
        "required_fields_per_operator": [
            "external_basis", "calibration_basis", "set_construction", "range",
            "units", "sensitivity", "rejected_alternatives", "holdout",
            "result", "checksum",
        ],
        "tecto_operators": tecto_entries,
        "qmap_transport": qmap_entry,
        "note": "Each operator carries external basis, calibration, holdout, observed sensitivity, and rejected alternatives per contract §8.8.",
    }

    os.makedirs(os.path.dirname(SPEC_OUT), exist_ok=True)
    with open(SPEC_OUT, "w") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(json.dumps({
        "written": SPEC_OUT,
        "sha256": sha256_file(SPEC_OUT),
        "n_tecto_operators": len(tecto_entries),
        "n_qmap_entries": 1,
        "tecto_operator_names": [e["operator"] for e in tecto_entries],
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
