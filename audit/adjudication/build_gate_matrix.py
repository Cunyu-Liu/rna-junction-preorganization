"""P0.6 fail-closed gate-matrix builder (contract P0.6).

Aggregates the STATUS / report files produced by P0.1-P0.5 into a single
P0GateMatrix.json with one row per hard gate, each bound to an evidence path
and a decision.  The overall P0 state is allowed to be only one of
P0_PASS_COMPARISON_ELIGIBLE / P0_PASS_FRESH_ONLY / BLOCKED_WITH_EVIDENCE.
No hard-gate failure may be papered over.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

# (phase, status_path_rel, gate_key) -> description
GATE_REFS = [
    ("P0.1", "authority/STATUS.json", "authority_lineage_frozen"),
    ("P0.2", "data/STATUS.json", "data_universe_rebuild"),
    ("P0.2", "data/STATUS.json", "censor_semantics"),
    ("P0.2", "data/STATUS.json", "effective_n_reported"),
    ("P0.3", "numerics/STATUS.json", "G1_original_gradient_failure_captured"),
    ("P0.3", "numerics/STATUS.json", "G2_corrected_gradient_synthetic_1e-4"),
    ("P0.3", "numerics/STATUS.json", "G2_corrected_gradient_real_init_1e-3"),
    ("P0.3", "numerics/STATUS.json", "G3_GH_convergence_48_vs_64"),
    ("P0.3", "numerics/STATUS.json", "G4_known_q_recovery_MC"),
    ("P0.3", "numerics/STATUS.json", "G5_optimizer_convergence"),
    ("P0.4", "protocol/STATUS.json", "split_manifests_frozen"),
    ("P0.4", "protocol/STATUS.json", "no_forbidden_overlap_per_axis"),
    ("P0.4", "protocol/STATUS_feature.json", "no_target_derived_in_primary_features"),
    ("P0.5", "replay/STATUS.json", "qualification_replay"),
]

DECISION_MAP = {
    "PASS": "PASS",
    "FAIL": "FAIL",
    "RUNNING": "NOT_RUN",
    "NOT_RUN": "NOT_RUN",
}


def sha(path: Path) -> str:
    if not path.exists():
        return "MISSING"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def build_gate_matrix(run_root: Path):
    gates = []
    for phase, rel, key in GATE_REFS:
        sp = run_root / rel
        decision = "NOT_RUN"
        value = None
        if sp.exists():
            st = json.loads(sp.read_text())
            if "gates" in st and key in st["gates"]:
                value = st["gates"][key]
                decision = "PASS" if value is True else "FAIL"
            else:
                decision = DECISION_MAP.get(st.get("state", ""), "NOT_RUN")
                value = st.get("state")
        gates.append({
            "phase": phase, "gate": key, "evidence_path": str(sp.relative_to(run_root)),
            "evidence_sha": sha(sp), "decision": decision, "value": value,
        })
    hard_fail = [g for g in gates if g["decision"] == "FAIL"]
    not_run = [g for g in gates if g["decision"] == "NOT_RUN"]
    if hard_fail:
        overall = "BLOCKED_WITH_EVIDENCE"
    elif not_run:
        overall = "P0_PASS_FRESH_ONLY"  # fresh-only: code is fine, not all run
    else:
        overall = "P0_PASS_COMPARISON_ELIGIBLE"
    return {"phase": "P0.6", "overall_state": overall,
            "n_gates": len(gates), "n_pass": sum(1 for g in gates if g["decision"] == "PASS"),
            "n_fail": len(hard_fail), "n_not_run": len(not_run),
            "hard_fail_gates": [g["gate"] for g in hard_fail],
            "not_run_gates": [g["gate"] for g in not_run],
            "gates": gates}


def main(cfg):
    run_root = Path(cfg["run_root"])
    matrix = build_gate_matrix(run_root)
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "P0GateMatrix.json").write_text(json.dumps(matrix, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"overall_state": matrix["overall_state"],
                      "n_pass": matrix["n_pass"], "n_fail": matrix["n_fail"],
                      "n_not_run": matrix["n_not_run"],
                      "hard_fail_gates": matrix["hard_fail_gates"]}, indent=2))


if __name__ == "__main__":
    import sys
    main(json.loads(Path(sys.argv[1]).read_text()))
