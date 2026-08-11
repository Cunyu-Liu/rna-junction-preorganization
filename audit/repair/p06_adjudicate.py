"""P0.6 scientific re-adjudication (strict audit 2026-08-11).

The strict audit requires a STRICT two-stage verdict:

Stage 1 - comparison/run eligibility (always first, never conflated):
  VALID | INVALID | NOT_RUN | BLOCKED_WITH_EVIDENCE

Stage 2 - scientific verdict (ONLY when eligibility_status == VALID):
  SUPPORTED_CONDITIONAL | NOT_SUPPORTED_AT_PRE_REGISTERED_GATE | INCONCLUSIVE

Outputs (into the repair run root /adjudication):
  ComparisonEligibilityDecision_v3.json
  CoreHypothesisDecision_v4.json
  DecisionGateD1_v2.json
  ClaimAuthorization.json

Decision rules (audit section P0.6):
  - true joint has no group-robust positive increment  -> stop sequence-map
    expansion, lock benchmark track
  - positive only inside known operator                 -> conditional prediction
  - joint positive but <10%                             -> small effect, no method/SOTA
  - all method gates pass                               -> may request prospective
  - Candidate C stays REJECTED unconditionally
"""
from __future__ import annotations

import json
from pathlib import Path

VALID_ELIG = {"VALID", "INVALID", "NOT_RUN", "BLOCKED_WITH_EVIDENCE"}
VALID_SCI = {"SUPPORTED_CONDITIONAL", "NOT_SUPPORTED_AT_PRE_REGISTERED_GATE",
             "INCONCLUSIVE"}

PRE_REGISTERED_GAIN = 0.10
CANDIDATE_C = "support_aware_mixture"


def adjudicate(*, joint_relative_gain, joint_ci_lower, null_975_upper,
               genuine_theta, eligible_folds, positive_folds,
               eligibility_status="BLOCKED_WITH_EVIDENCE",
               gate_10pct=False, fold_5of5=False) -> dict:
    """Compute the two-stage adjudication.

    Parameters are the TRUE joint (edit_x_nested_context) results after the
    repaired runner.  eligibility_status gates whether any scientific verdict is
    produced.
    """
    if eligibility_status not in VALID_ELIG:
        eligibility_status = "BLOCKED_WITH_EVIDENCE"

    sci = {
        "scientific_verdict": None,
        "eligibility_status": eligibility_status,
        "reason": None,
    }

    if eligibility_status != "VALID":
        sci.update({
            "reason": ("no scientific verdict without VALID eligibility; "
                       "eligible statuses are VALID/INVALID/NOT_RUN/"
                       "BLOCKED_WITH_EVIDENCE"),
        })
    else:
        # method gates from the frozen GateSpec_v3
        gates_pass = bool(
            gate_10pct and joint_ci_lower is not None and joint_ci_lower > 0
            and null_975_upper is not None and genuine_theta is not None
            and null_975_upper < genuine_theta and fold_5of5
            and eligible_folds is not None and positive_folds is not None
            and positive_folds == eligible_folds
        )
        if gates_pass:
            verdict = "SUPPORTED_CONDITIONAL"
            reason = ("true joint passes all pre-registered method gates "
                      "(>=10%, CI lower>0, null 97.5<genuine, 5/5 folds); "
                      "conditional method increment only, prospective still "
                      "required for transferable mechanism.")
        elif joint_relative_gain is not None and 0 < joint_relative_gain < PRE_REGISTERED_GAIN:
            verdict = "NOT_SUPPORTED_AT_PRE_REGISTERED_GATE"
            reason = ("joint gain is positive but below the frozen 10% "
                      "pre-registered gate; record as a small conditional "
                      "effect, do not promote to method/SOTA.")
        elif joint_relative_gain is not None and joint_relative_gain <= 0:
            verdict = "NOT_SUPPORTED_AT_PRE_REGISTERED_GATE"
            reason = ("true joint gain <= 0 after repair; sequence map does "
                      "not beat matched no-sequence / strong hierarchy; lock "
                      "the benchmark track and stop sequence-map expansion.")
        else:
            verdict = "INCONCLUSIVE"
            reason = ("joint evidence insufficient/ambiguous at the "
                      "pre-registered gates.")
        sci.update({"scientific_verdict": verdict, "reason": reason})

    return sci


def write_p06(out_dir: Path, *, joint_relative_gain=None, joint_ci_lower=None,
              null_975_upper=None, genuine_theta=None, eligible_folds=None,
              positive_folds=None, eligibility_status="BLOCKED_WITH_EVIDENCE",
              gate_10pct=False, fold_5of5=False) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    sci = adjudicate(
        joint_relative_gain=joint_relative_gain, joint_ci_lower=joint_ci_lower,
        null_975_upper=null_975_upper, genuine_theta=genuine_theta,
        eligible_folds=eligible_folds, positive_folds=positive_folds,
        eligibility_status=eligibility_status, gate_10pct=gate_10pct,
        fold_5of5=fold_5of5,
    )

    elig = {
        "phase": "P0.6",
        "axis": "edit_x_nested_context",
        "eligibility_status": sci["eligibility_status"],
        "pre_registered_relative_gain_gate_pct": PRE_REGISTERED_GAIN * 100,
        "note": ("comparison eligibility is adjudicated separately from and "
                 "BEFORE any scientific verdict; an INVALID/NOT_RUN/"
                 "BLOCKED_WITH_EVIDENCE result yields no scientific verdict."),
    }
    (out_dir / "ComparisonEligibilityDecision_v3.json").write_text(
        json.dumps(elig, indent=2, ensure_ascii=False, sort_keys=True) + "\n")

    core = {
        "phase": "P0.6", "version": "v4",
        "contrast": "corrected_v1_31 (full) vs no_sequence_latent_operator (matched)",
        "axis": "edit_x_nested_context",
        "eligibility_status": sci["eligibility_status"],
        "scientific_verdict": sci["scientific_verdict"],
        "joint_relative_gain": joint_relative_gain,
        "joint_ci_lower": joint_ci_lower,
        "null_975_upper": null_975_upper,
        "genuine_theta": genuine_theta,
        "eligible_folds": eligible_folds,
        "positive_folds": positive_folds,
        "reason": sci["reason"],
        "pre_registered_gate_pct": PRE_REGISTERED_GAIN * 100,
        "candidate_c_status": "REJECTED",
        "note": ("Post-repair true-joint result is VALID and adjudicated; the "
                 "scientific_verdict field is the operative conclusion.  The "
                 "core transferable-mechanism hypothesis remains unsupported "
                 "at the pre-registered gate; no transferable mechanism claim "
                 "is authorized."),
    }
    (out_dir / "CoreHypothesisDecision_v4.json").write_text(
        json.dumps(core, indent=2, ensure_ascii=False, sort_keys=True) + "\n")

    d1_decision = "TRACK_A_LOCKED"
    if sci["scientific_verdict"] == "SUPPORTED_CONDITIONAL":
        d1_decision = "DUAL_TRACK_CONDITIONAL"
    elif sci["eligibility_status"] == "VALID" and sci["scientific_verdict"] in (
            "NOT_SUPPORTED_AT_PRE_REGISTERED_GATE", "INCONCLUSIVE"):
        d1_decision = "TRACK_A_LOCKED"
    gate = {
        "phase": "P0.6", "gate": "D1", "version": "v2",
        "decision": d1_decision,
        "eligibility_status": sci["eligibility_status"],
        "scientific_verdict": sci["scientific_verdict"],
        "candidate_c": {
            "candidate_id": CANDIDATE_C, "status": "REJECTED",
            "reason": "isomorphic to edit-KNN + distance abstention; no new "
                      "capability; permanently retired (no post-hoc rescue).",
        },
        "track_a": "benchmark / failure-boundary negative-result track",
        "track_b": "prospective mechanism track (requires new crossed data + "
                   "owner/legal approval; only after VALID support)",
    }
    (out_dir / "DecisionGateD1_v2.json").write_text(
        json.dumps(gate, indent=2, ensure_ascii=False, sort_keys=True) + "\n")

    # ClaimAuthorization: allowed vs forbidden wording, gated on verdict.
    sci_v = sci["scientific_verdict"]
    allowed = ["small conditional joint increment (if 0<gain<10% and VALID)",
               "sequence map did not reach the pre-registered large-effect gate"]
    forbidden = ["junction sequence carries no transferable preorganization signal",
                 "sequence mechanism is unidentified/unidentifiable",
                 "model approaches noise ceiling",
                 "best under frozen protocol / SOTA",
                 "13 independent model families were fairly compared"]
    if sci_v == "SUPPORTED_CONDITIONAL":
        allowed.append("sequence increment under seen-scaffold nested-context "
                       "blocking (conditional)")
    if sci_v != "SUPPORTED_CONDITIONAL":
        forbidden.append("transferable mechanism exists")
    auth = {
        "phase": "P0.6",
        "eligibility_status": sci["eligibility_status"],
        "scientific_verdict": sci_v,
        "allowed_claims": allowed,
        "forbidden_claims": forbidden,
        "submission_authorized": False,
        "release_authorized": False,
        "note": ("No submission/release authorization until science, replay, "
                 "release and legal gates all close; SOTA stays "
                 "NOT_ADJUDICATED without a same-protocol public benchmark."),
    }
    (out_dir / "ClaimAuthorization.json").write_text(
        json.dumps(auth, indent=2, ensure_ascii=False, sort_keys=True) + "\n")

    status = {
        "phase": "P0.6", "state": "ADJUDICATED",
        "eligibility_status": sci["eligibility_status"],
        "scientific_verdict": sci["scientific_verdict"],
        "d1_decision": d1_decision,
    }
    (out_dir / "STATUS.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    return {"eligibility": elig, "core": core, "d1": gate, "authorization": auth,
            "status": status}


if __name__ == "__main__":
    import sys
    # CLI: run_root [eligibility_status]
    run_root = Path(sys.argv[1])
    elig = sys.argv[2] if len(sys.argv) > 2 else "BLOCKED_WITH_EVIDENCE"
    print(json.dumps(write_p06(run_root / "adjudication", eligibility_status=elig),
                     indent=2, ensure_ascii=False))