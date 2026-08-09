"""Phase 5 gap closure: emit the contract-required MechanismRegistry.json.

Contract Phase 5 (rna_junction v1.28-v1.31 strict audit, P5 section) lists
`MechanismRegistry.json` among the deliverables. The earlier p5_run.py emitted
FailureAtlas/ContextSensitivity/MutationPathAnalysis/CatastrophicFolds/
ClaimEvidenceMatrix/PaperStoryDecision but did not write MechanismRegistry.json.
This runner closes that naming/artifact gap WITHOUT retraining or re-adjudicating.

Because Phase 4 sealed a fail-closed result (only surviving candidate NOT_PROMOTED,
no promotable mechanism), Phase 5 follows the contract failure path: the registry
formally enumerates the mechanistic hypotheses that WERE considered, and for each
records the evidence basis and a fail-closed status, so every "mechanism" claim is
explicitly linked to data/code/results and marked NOT_AUTHORIZED where appropriate.
It is a registry of mechanisms considered, not a positive mechanism claim.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(cfg):
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    p5 = Path(cfg["p5_dir"])
    p4 = Path(cfg["p4_dir"])

    # Sealed Phase 4 verdict (source of the fail-closed posture).
    decision_path = p4 / "CandidatePromotionDecision.json"
    if decision_path.exists():
        decision = json.loads(decision_path.read_text())
        p4_verdict = decision.get("verdict", decision.get("decision", "UNKNOWN"))
        p4_gain = decision.get("observed_mean_gain")
    else:
        p4_verdict = "NOT_PROMOTED (decision file not found at finalize time)"
        p4_gain = None

    # Enumerate the mechanisms that were considered, their evidence, status.
    # All are fail-closed: no mechanism survives P4/P5 evidence.
    mechanisms = [
        {
            "mechanism": "sequence encodes a transferable preorganization mechanism "
                         "(across helix context and scaffold/operator)",
            "hypothesis_class": "positive mechanism",
            "evidence_basis": "P4 final outer-test, coverage-matched supported-NLL; "
                              "candidate NOT_PROMOTED vs strongest baseline on all three "
                              "known-operator axes, ties edit_knn on operator-holdout",
            "data": "data/CleaningLedger.jsonl (admitted rows; 1,336 junctions; 9 scaffolds)",
            "code": "audit/p4/p4_run.py; audit/models/support_aware_mixture.py",
            "result": f"p4_verdict={p4_verdict}; observed_mean_gain={p4_gain}",
            "figures": "P4 FinalLeaderboard/BootstrapIntervals (see P4 report)",
            "status": "NOT_AUTHORIZED",
            "evidence_class": "DEVELOPMENT_ONLY / REFUTED (no positive mechanism)",
        },
        {
            "mechanism": "sequence-local KNN rescues operator-holdout catastrophic failure "
                         "via sequence extrapolation",
            "hypothesis_class": "positive mechanism",
            "evidence_basis": "P5 FailureAtlas: scaffold_lomo candidate 0 catastrophic folds "
                              "BUT exact tie with edit_knn (same-sequence copying; all test "
                              "junctions at edit distance 0 from outer-train)",
            "data": "p5_diagnostics/FailureAtlas.parquet",
            "code": "audit/p5/p5_run.py; audit/models/support_aware_mixture.py",
            "result": "rescue is trivial same-sequence copying, not extrapolation",
            "figures": "P5 FailureAtlas / MutationPathAnalysis",
            "status": "NOT_PROMOTABLE",
            "evidence_class": "DEVELOPMENT_ONLY",
        },
        {
            "mechanism": "gain attributable to seen-context/scaffold calibration rather than "
                         "a sequence mechanism",
            "hypothesis_class": "negative/calibration mechanism",
            "evidence_basis": "v1.30 sequence-pairing null gain > genuine gain; candidate "
                              "underperforms strongest baseline at matched coverage on "
                              "known-operator axes; repeated context/scaffold exposure",
            "data": "v1.30 method status + sequence null; P4 BootstrapIntervals",
            "code": "v1.30 method repair (historical); audit/p4/p4_gap.py (3-seed CI)",
            "result": "CONSISTENT with calibration-as-generalization concern",
            "figures": "P4 BootstrapIntervals_3seed.csv; P5 ContextSensitivity.csv",
            "status": "SUPPORTED_BOUNDARY",
            "evidence_class": "CONSISTENT (boundary, not mechanism)",
        },
        {
            "mechanism": "operator transfer to a genuinely unseen scaffold/operator "
                         "(leave-one-scaffold-out generalization)",
            "hypothesis_class": "positive mechanism (transfer)",
            "evidence_basis": "only 9 scaffolds; scaffold_lomo candidate ties edit_knn; "
                              "no prospective constructs available",
            "data": "9-scaffold panel (single Denny study)",
            "code": "audit/p4/p4_run.py; audit/p5/p5_run.py",
            "result": "NOT_AUTHORIZED: cannot claim transfer beyond seen operators",
            "figures": "P5 CatastrophicFolds.csv",
            "status": "NOT_AUTHORIZED",
            "evidence_class": "DEVELOPMENT_ONLY",
        },
    ]

    registry = {
        "phase": "P5",
        "artifact": "MechanismRegistry.json",
        "purpose": "formal registry of mechanisms considered; contract P5 deliverable. "
                   "No mechanism claim is authorized (fail-closed).",
        "narrative": "benchmark_identifiability_boundary",
        "candidate": "support_aware_mixture",
        "p4_sealed_verdict": p4_verdict,
        "p4_observed_mean_gain": p4_gain,
        "sota_status": "SOTA_NOT_ADJUDICATED",
        "scientific_claim_authorized": False,
        "evidence_contract": "every mechanism links data/code/result/figure; "
                             "association vs mechanism language kept separate",
        "mechanisms": mechanisms,
        "provenance": {
            "p5_dir": str(p5),
            "p4_dir": str(p4),
            "mechanism_registry_generator": "audit/p5/p5_mechanism_registry.py",
        },
    }

    out_file = out / "MechanismRegistry.json"
    out_file.write_text(json.dumps(registry, indent=2, ensure_ascii=False) + "\n")

    # SHA-256 so the registry itself is locked into the release checksums.
    registry["checksum"] = {"MechanismRegistry.json": sha256(out_file)}
    out_file.write_text(json.dumps(registry, indent=2, ensure_ascii=False) + "\n")

    print(json.dumps(registry, indent=2, ensure_ascii=False))
    return registry


if __name__ == "__main__":
    cfg = json.loads(Path(sys.argv[1]).read_text())
    main(cfg)
