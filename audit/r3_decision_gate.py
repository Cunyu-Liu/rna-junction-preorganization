"""R3 D1 go/no-go decision gate (contract §12.4).

Reads the frozen R2 CoreHypothesisDecision_v3.json and issues a single,
machine-readable, irreversible go/no-go on the method track.  It also
permanently registers Candidate C as REJECTED.

Candidate C registration (fail-closed, no post-hoc rescue):
  - Candidate C is isomorphic to edit-KNN (K=11) plus a distance-threshold
    abstention policy: on supported rows its mu/sigma match edit-KNN
    pointwise (R0.3 isomorphism regression).  It is therefore NOT an
    independent method family.
  - Its best inner gain is <= 0 (P3 adjudicated fact), so per contract
    §12.4 it MUST be REJECTED regardless of any downstream result.

D1 branching:
  - If R2 shows SEQUENCE_INCREMENT_SUPPORTED on the joint axis
    (edit_x_nested_context): allow Track A (corrected latent-operator) AND
    Track B (physical ensemble) — prospective only, pending owner/legal.
  - Otherwise lock Track A (benchmark / failure-boundary negative result)
    and forbid further model search on the current data.

Outputs into RUN_ROOT/r3/:
  DecisionGateD1.json
  CandidateRegistry_v2.json
  STATUS.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

AXES = ["symmetry_5fold", "edit_5fold", "context_lomo",
        "scaffold_lomo", "edit_x_nested_context"]
JOINT_AXIS = "edit_x_nested_context"


def utc_now():
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main(cfg):
    run_root = Path(cfg["run_root"])
    out = run_root / "r3"
    out.mkdir(parents=True, exist_ok=True)
    utc = utc_now()

    r2_path = run_root / "r2" / "CoreHypothesisDecision_v3.json"
    if not r2_path.exists():
        status = {"phase": "R3", "state": "BLOCKED_PENDING_R2",
                  "reason": "R2 CoreHypothesisDecision_v3.json missing"}
        (out / "STATUS.json").write_text(json.dumps(status, indent=2) + "\n")
        print(json.dumps(status, indent=2))
        return status

    r2 = json.loads(r2_path.read_text())
    joint = next((a for a in r2.get("axes", [])
                  if a.get("axis") == JOINT_AXIS), {})
    joint_supported = bool(joint.get("verdict") == "SEQUENCE_INCREMENT_SUPPORTED")

    # Candidate C permanent registration
    candidates = [
        {
            "candidate_id": "support_aware_mixture",
            "status": "REJECTED",
            "reason": ("isomorphic to edit-KNN + distance-gate abstention; "
                       "best inner gain <= 0 (P3); not an independent method "
                       "family; rejected per contract §12.4 (no post-hoc rescue)"),
            "evidence": ["edit_knn_isomorphism_regression_R0.3",
                         "p3_best_inner_gain_le_0"],
        },
        {
            "candidate_id": "corrected_v1_31_latent_operator",
            "status": "ALLOWED" if joint_supported else "TRACK_A_ONLY",
            "reason": ("sequence-aware latent-operator reference; allowed as a "
                       "Track A benchmark always; prospective Track B only if "
                       "joint signal supported"),
            "evidence": ["r05_v131_fresh_rerun"],
        },
        {
            "candidate_id": "physical_ensemble",
            "status": "PROSPECTIVE_PENDING_OWNER_LEGAL" if joint_supported
                       else "TRACK_A_ONLY",
            "reason": ("physical/thermodynamic ensemble baseline; requires "
                       "owner/legal approval and new measurements if promoted"),
            "evidence": [],
        },
    ]

    decision = {
        "run_id": cfg["run_id"], "phase": "R3", "generated_at_utc": utc,
        "gate": "D1",
        "joint_axis": JOINT_AXIS,
        "joint_sequence_increment_supported": bool(joint_supported),
        "decision": ("DUAL_TRACK" if joint_supported
                     else "TRACK_A_LOCKED"),
        "track_a": ("benchmark / failure-boundary negative result: "
                    "apparent sequence gain sensitivity to exposure, support "
                    "policy, censoring implementation and estimand"),
        "track_b": ("prospective mechanism route: requires factorial data "
                    "crossing context x operator AND owner/legal approval"),
        "candidates": candidates,
        "note": ("D1 is irreversible per contract §12.4.  No candidate is "
                 "revived post-hoc; any best_inner_gain<=0 candidate is "
                 "permanently REJECTED.  authority gate blocks R4 if this "
                 "decision is violated."),
    }
    (out / "DecisionGateD1.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    (out / "CandidateRegistry_v2.json").write_text(
        json.dumps(candidates, indent=2, ensure_ascii=False, sort_keys=True) + "\n")

    status = {
        "phase": "R3", "state": "D1_DECIDED", "generated_at_utc": utc,
        "decision": decision["decision"],
        "joint_supported": bool(joint_supported),
        "n_candidates_registered": len(candidates),
        "n_rejected": sum(1 for c in candidates if c["status"] == "REJECTED"),
    }
    (out / "STATUS.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(status, indent=2, ensure_ascii=False))
    return status


if __name__ == "__main__":
    main(json.loads(Path(sys.argv[1]).read_text()))
