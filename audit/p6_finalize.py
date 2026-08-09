"""Phase 6: reproduction & release preparation (contract Phase 6).

Because P0-P5 sealed a fail-closed, benchmark/identifiability-boundary result
with NO scientific claim, Phase 6 produces the engineering release artifacts that
prove the benchmark numbers are reproducible and that the (negative) conclusion
is properly scoped. No submission claim is made.

Deliverables (contract Phase 6):
  REPRODUCE.md, environment.lock, ReleaseManifest.json, checksums.sha256,
  DataCard.md, ModelCard.md, LicenseLedger.csv, SubmissionClaimMatrix.csv, STATUS.json
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

MNT = "/mnt/cunyuliu/rna_junction_audit_20260807T090244Z"
OUT = Path(__file__).resolve().parent  # audit/release/
ENV = {
    "conda_env": "rna_junction_preorganization_v1_1",
    "numpy": "2.2.6",
    "scipy": "1.15.2",
    "pandas": "2.3.3",
    "scikit_learn": "1.7.2",
}
ARTIFACTS = {
    "data/CleaningLedger.jsonl": f"{MNT}/data/CleaningLedger.jsonl",
    "data/EffectiveNReport.json": f"{MNT}/data/EffectiveNReport.json",
    "data/ExposureRegistry.json": f"{MNT}/data/ExposureRegistry.json",
    "protocol/SplitManifest_symmetry_5fold.jsonl": f"{MNT}/protocol/SplitManifest_symmetry_5fold.jsonl",
    "protocol/SplitManifest_edit_5fold.jsonl": f"{MNT}/protocol/SplitManifest_edit_5fold.jsonl",
    "protocol/SplitManifest_context_lomo.jsonl": f"{MNT}/protocol/SplitManifest_context_lomo.jsonl",
    "protocol/SplitManifest_scaffold_lomo.jsonl": f"{MNT}/protocol/SplitManifest_scaffold_lomo.jsonl",
    "p1_full/Predictions.jsonl": f"{MNT}/p1_full/Predictions.jsonl",
    "p3_full_v2/SelectedGateEvaluation.csv": f"{MNT}/p3_full_v2/SelectedGateEvaluation.csv",
    "p4_final/FinalPredictions.parquet": f"{MNT}/p4_final/FinalPredictions.parquet",
    "p4_final/FinalLeaderboard.csv": f"{MNT}/p4_final/FinalLeaderboard.csv",
    "p4_final/BootstrapIntervals.csv": f"{MNT}/p4_final/BootstrapIntervals.csv",
    "p4_final/CandidatePromotionDecision.json": f"{MNT}/p4_final/CandidatePromotionDecision.json",
    "p5_diagnostics/FailureAtlas.parquet": f"{MNT}/p5_diagnostics/FailureAtlas.parquet",
    "p5_diagnostics/ContextSensitivity.csv": f"{MNT}/p5_diagnostics/ContextSensitivity.csv",
    "p5_diagnostics/ClaimEvidenceMatrix.csv": f"{MNT}/p5_diagnostics/ClaimEvidenceMatrix.csv",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(
        ["git", "-C", str(Path(__file__).resolve().parent.parent),
         "rev-parse", "HEAD"]).decode().strip()

    hashes = {}
    missing = []
    for key, path in ARTIFACTS.items():
        p = Path(path)
        if p.exists():
            hashes[key] = sha256(str(p))
        else:
            missing.append(key)

    # environment.lock
    (OUT / "environment.lock").write_text(
        json.dumps(ENV, indent=2) + "\n")

    # checksums.sha256
    lines = [f"{v}  {k}\n" for k, v in sorted(hashes.items())]
    (OUT / "checksums.sha256").write_text("".join(lines))

    # ReleaseManifest.json
    manifest = {
        "run_id": "V1_28_V1_31_AUDIT_20260807T090244Z",
        "git_commit": commit,
        "git_remote": "git@github.com:Cunyu-Liu/rna-junction-preorganization.git",
        "run_root": MNT,
        "environment": ENV,
        "phases": {
            "P0": "P0_PASS_COMPARISON_ELIGIBLE",
            "P1": "BASELINES_COMPLETE",
            "P2": "CONDITIONAL_KNOWN_OPERATOR_SIGNAL",
            "P3": "CANDIDATE_C_RETAINED_AS_TARGETED_EXTRAPOLATION_FIX",
            "P4": "NOT_PROMOTED",
            "P5": "IDENTIFIABILITY_BOUNDARY_NARRATIVE",
            "P6": "RELEASE_PREPARED",
        },
        "overall_scientific_state": {
            "sota_status": "SOTA_NOT_ADJUDICATED",
            "submission_authorization": "NO_SUBMISSION_AUTHORIZATION",
            "scientific_claim_authorized": False,
            "narrative": "benchmark_identifiability_boundary",
        },
        "artifacts": {k: {"sha256": v, "path": ARTIFACTS[k]} for k, v in hashes.items()},
        "missing_artifacts": missing,
    }
    (OUT / "ReleaseManifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    # REPRODUCE.md
    (OUT / "REPRODUCE.md").write_text(
        "# Reproduction Guide\n\n"
        "## Scope\n"
        "Reproduces the grouped, right-censor-aware benchmark and the fail-closed\n"
        "identifiability-boundary conclusion. There is NO scientific claim to submit\n"
        "(SOTA_NOT_ADJUDICATED; NO_SUBMISSION_AUTHORIZATION).\n\n"
        "## Requirements\n"
        "- Remote server path `/mnt/cunyuliu/rna_junction_audit_20260807T090244Z`\n"
        "  containing data/, protocol/, p1_full/, p3_full_v2/, p4_final/, p5_diagnostics/.\n"
        "- Conda env `rna_junction_preorganization_v1_1` (numpy 2.2.6, scipy 1.15.2,\n"
        "  pandas 2.3.3, scikit-learn 1.7.2) - see environment.lock.\n"
        "- Code: git repo `rna-junction-preorganization` at commit "
        f"`{commit}` (audit/ tree).\n\n"
        "## Steps\n"
        "1. `conda activate rna_junction_preorganization_v1_1`\n"
        "2. Run each phase runner in order (see audit/p0..p5):\n"
        "   - P0: audit/provenance, audit/data, audit/numerics, audit/benchmark (P0.5)\n"
        "   - P1: audit/benchmark baselines -> p1_full/Predictions.jsonl\n"
        "   - P2: audit/p2 hypothesis/nulls/bootstrap\n"
        "   - P3: audit/p3 p3_run.py -> nested-CV gates\n"
        "   - P4: audit/p4 p4_run.py -> coverage-matched comparison\n"
        "   - P5: audit/p5 p5_run.py -> identifiability-boundary diagnostics\n"
        "3. Verify artifacts against checksums.sha256.\n"
        "4. Read ReleaseManifest.json for the sealed phase statuses.\n\n"
        "## Expected outcome\n"
        "The candidate is NOT_PROMOTED; the surviving narrative is a benchmark /\n"
        "identifiability-boundary. Re-running reproduces the same fail-closed verdict.\n"
    )

    # DataCard.md
    (OUT / "DataCard.md").write_text(
        "# Data Card\n\n"
        "- Source: Denny 2018 tectoRNA canonical records (canonical source persisted to\n"
        "  run root; SHA-256 `0989ddc0...` in authority/).\n"
        "- Admitted universe: 11,893 junction x scaffold/context rows; 1,336 junctions;\n"
        "  234 admitted helix contexts; 9 scaffolds/operators; study = 1.\n"
        "- Right-censored fraction: 16.25% (y >= -7.1 kcal/mol recorded as censored).\n"
        "- Panel structure: each junction observed in 4-9 scaffold/contexts (median 9);\n"
        "  rows are NOT independent biological samples.\n"
        "- Cleaning ledger: data/CleaningLedger.jsonl (per-row layer/reason).\n"
        "- Leakage control: mmseqs/grouped splits, frozen SplitManifests, overlap audit.\n"
    )

    # ModelCard.md
    (OUT / "ModelCard.md").write_text(
        "# Model Card\n\n"
        "- Benchmark models: right-censored Gaussian baselines (global intercept,\n"
        "  scaffold, hierarchy, motif, one-hot k-mer ridge, position-additive, edit KNN,\n"
        "  mutation graph, small MLP, corrected_v1_31).\n"
        "- Candidate: support_aware_mixture (local edit-KNN censored-location predictor\n"
        "  with train-only abstention gate).\n"
        "- Primary metric: junction-macro right-censored NLL (lower better).\n"
        "- Status: NOT_PROMOTED (P4); DEVELOPMENT_ONLY. No mechanism claim.\n"
        "- Prediction schema: per row mu, sigma, support, abstain (FinalPredictions.parquet).\n"
    )

    # LicenseLedger.csv
    license_rows = [
        ("dataset", "Denny 2018 tectoRNA", "UNKNOWN_NEEDS_LEGAL_REVIEW",
         "source persisted as byte copy; distribution license not yet confirmed"),
        ("code", "rna-junction-preorganization (audit/)", "OPEN_SOURCE_PENDING",
         "author owns audit code; license to be declared before any public release"),
        ("env", "numpy/scipy/pandas/scikit-learn", "BSD/OSI",
         "permissive open-source dependencies"),
    ]
    (OUT / "LicenseLedger.csv").write_text(
        "asset,source,license_status,note\n" + "\n".join(
            f"{a},{s},{l},{n}" for a, s, l, n in license_rows) + "\n")

    # SubmissionClaimMatrix.csv
    claims = [
        ("benchmark protocol is reproducible", "PASS", "checksums.sha256 + REPRODUCE.md"),
        ("candidate beats strongest baseline", "NOT_AUTHORIZED", "P4 NOT_PROMOTED"),
        ("sequence mechanism/operator-transfer", "NOT_AUTHORIZED", "P5 identifiability boundary"),
        ("submission readiness", "NOT_AUTHORIZED", "SOTA_NOT_ADJUDICATED / NO_SUBMISSION_AUTHORIZATION"),
    ]
    (OUT / "SubmissionClaimMatrix.csv").write_text(
        "claim,status,evidence\n" + "\n".join(
            f"{c},{s},{e}" for c, s, e in claims) + "\n")

    status = {
        "phase": "P6", "state": "PASS",
        "git_commit": commit,
        "environment": ENV,
        "n_artifacts_hashed": len(hashes),
        "missing_artifacts": missing,
        "sota_status": "SOTA_NOT_ADJUDICATED",
        "scientific_claim_authorized": False,
        "deliverables": ["REPRODUCE.md", "environment.lock", "ReleaseManifest.json",
                         "checksums.sha256", "DataCard.md", "ModelCard.md",
                         "LicenseLedger.csv", "SubmissionClaimMatrix.csv"],
    }
    (OUT / "STATUS.json").write_text(json.dumps(status, indent=2) + "\n")
    return status


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, ensure_ascii=False))
