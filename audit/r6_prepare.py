"""R6 engineering-side preparation (contract §12.7, first pass).

R6 full acceptance needs a SECOND compute environment (raw->final clean replay)
plus owner/legal authorization for dataset redistribution and submission.  That
part is out of scope here.  This module prepares everything that CAN be sealed
on the current environment without external authorization:

  ReleaseManifest.json  bound to current git HEAD (branch/commit)
  checksums.sha256      SHA-256 of the final R0-lineage artifacts
  environment.lock      conda package pins
  REPRODUCE.md          exact R0->R5 reproduction commands
  LicenseLedger.csv     data/code license status (legal fields marked PENDING)

Outputs into RUN_ROOT/r6/.
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

# final artifacts to seal (relative to run_root)
FINAL_ARTIFACTS = [
    "r05/ConvergenceLedger.parquet",
    "r1/Leaderboard_v2.csv",
    "r1/ConvergenceLedger.parquet",
    "r2/CoreHypothesisDecision_v3.json",
    "r2/MultiwayCluster.json",
    "r2/MatchedAblationContrast.json",
    "r4/EffectiveN.json",
    "r4/NoiseCeiling.json",
    "r4/CensoringSensitivity.json",
    "r4/PowerAnalysis.json",
    "r4/ModelCoverage.json",
    "r5/ClaimEvidenceMatrix.json",
    "r5/FailureAtlas.json",
]

PHASES = {
    "R0.5": "GRADIENT_CORRECTED_BASELINES_RERUN",
    "R0.6": "COMPARISON_ELIGIBILITY_READJUDICATED",
    "R1": "UNIFIED_LEADERBOARD_v2",
    "R2": "CORE_HYPOTHESIS_NOT_SUPPORTED_OR_INCONCLUSIVE",
    "R3": "D1_TRACK_A_LOCKED",
    "R4": "TRACK_A_EVIDENCE_CLOSURE_DONE",
    "R5": "CLAIM_MATRIX_AND_NARRATIVE_DONE",
}


def _sha256(path: Path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_info(repo: Path):
    def run(*args):
        r = subprocess.run(["git", "-C", str(repo), *args],
                           capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None
    return {
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "commit": run("rev-parse", "HEAD"),
        "remote": run("config", "--get", "remote.origin.url"),
        "dirty": bool(run("status", "--porcelain")),
    }


def env_lock(conda_env: str):
    """Best-effort conda pin capture; non-fatal on failure."""
    try:
        r = subprocess.run(["conda", "list", "-n", conda_env, "--json"],
                           capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            return {"conda_env": conda_env, "note": "conda list unavailable", "packages": []}
        pkgs = json.loads(r.stdout)
        return {"conda_env": conda_env,
                "packages": [{"name": p["name"], "version": p["version"],
                              "channel": p.get("channel")} for p in pkgs]}
    except Exception as e:  # noqa: BLE001
        return {"conda_env": conda_env, "note": f"env capture failed: {e}", "packages": []}


def reproduce_md(cfg, git):
    return (
        "# Reproduction Guide (R0 lineage)\n\n"
        f"## Scope\n"
        "Reproduces the corrected right-censor-aware benchmark, joint-blocked\n"
        "rerun, matched no-sequence ablation, and Track A evidence closure.\n"
        "NO scientific claim is authorized (SOTA_NOT_ADJUDICATED;\n"
        "NO_SUBMISSION_AUTHORIZATION).\n\n"
        "## Pinned code & environment\n"
        f"- Branch: `{git['branch']}`  Commit: `{git['commit']}`\n"
        f"- Remote: `{git['remote']}`\n"
        "- Conda env: rna_junction_preorganization_v1_1 (see environment.lock)\n\n"
        "## Steps (each phase must hit its acceptance gate before the next)\n"
        "```bash\n"
        "conda activate rna_junction_preorganization_v1_1\n"
        f"cd {cfg.get('repo', cfg['run_root'])}\n"
        "export PYTHONPATH=.\n"
        "# R0.5 gradient-corrected baselines + edit_knn (needs protocol splits)\n"
        "python audit/r05_run.py audit/provenance/r05_cfg.json\n"
        "python audit/r05_v131_run.py audit/provenance/r05_v131_cfg.json\n"
        "# R0.6 re-adjudicate comparison eligibility\n"
        "python audit/r06_adjudicate.py audit/provenance/r06_cfg.json\n"
        "# R1 unified leaderboard (adds matched no-sequence latent operator + joint axis)\n"
        "python audit/r1_run.py audit/provenance/r1_cfg.json\n"
        "# R2 matched ablation + multiway cluster + decision\n"
        "python audit/r2_null_and_decision.py audit/provenance/r2_cfg.json\n"
        "# R3 D1 decision gate\n"
        "python audit/r3_decision_gate.py audit/provenance/r3_cfg.json\n"
        "# R4 Track A evidence closure\n"
        "python audit/r4_track_a.py audit/provenance/r4_cfg.json\n"
        "# R5 mechanism / claim matrix / narrative\n"
        "python audit/r5_mechanism.py audit/provenance/r5_cfg.json\n"
        "# verify sealed checksums (checksum paths are run_root-relative)\n"
        f"cd {cfg['run_root']}\n"
        "sha256sum -c r6/checksums.sha256\n"
        "```\n\n"
        "## Expected outcome\n"
        "All axes NOT_SUPPORTED_OR_INCONCLUSIVE with adequate power to exclude\n"
        "the 10% target; Track A locked; claim matrix scoped to tested model\n"
        "class / data / power.\n"
    )


def license_ledger():
    return [
        {"asset": "dataset (tecto_v111 canonical records)", "type": "data",
         "status": "UNKNOWN_NEEDS_LEGAL_REVIEW", "owner": "PENDING_LEGAL",
         "note": "acquisition chain / redistribution right not closed"},
        {"asset": "code (audit/)", "type": "code",
         "status": "OPEN_SOURCE_PENDING", "owner": "PENDING_LEGAL",
         "note": "repo has no LICENSE file; release blocked until resolved"},
        {"asset": "results/artifacts (r1-r5)", "type": "derivative",
         "status": "PENDING_DATA_LICENSE", "owner": "PENDING_LEGAL",
         "note": "derivative redistribution gated by source data license"},
    ]


def run(cfg):
    run_root = Path(cfg["run_root"])
    repo = Path(cfg.get("repo", str(run_root)))
    out = run_root / "r6"
    out.mkdir(parents=True, exist_ok=True)

    git = git_info(repo)
    env = env_lock(cfg.get("conda_env", "rna_junction_preorganization_v1_1"))
    (out / "environment.lock").write_text(
        json.dumps(env, indent=2, ensure_ascii=False, sort_keys=True) + "\n")

    # checksums of final artifacts
    checksum_lines = []
    artifacts = {}
    missing = []
    for rel in FINAL_ARTIFACTS:
        p = run_root / rel
        if p.exists():
            c = _sha256(p)
            checksum_lines.append(f"{c}  {rel}")
            artifacts[rel] = {"sha256": c, "path": str(p)}
        else:
            missing.append(rel)
    (out / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n")

    (out / "REPRODUCE.md").write_text(reproduce_md(cfg, git))

    with (out / "LicenseLedger.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["asset", "type", "status", "owner", "note"])
        for r in license_ledger():
            w.writerow([r["asset"], r["type"], r["status"], r["owner"], r["note"]])

    manifest = {
        "run_id": cfg["run_id"],
        "phase": "R6_ENGINEERING_PREP",
        "git": git,
        "conda_env": cfg.get("conda_env", "rna_junction_preorganization_v1_1"),
        "phases": PHASES,
        "overall_scientific_state": {
            "sota_status": "SOTA_NOT_ADJUDICATED",
            "submission_authorization": "NO_SUBMISSION_AUTHORIZATION",
            "scientific_claim_authorized": False,
        },
        "artifacts": artifacts,
        "missing_artifacts": missing,
        "license": license_ledger(),
        "note": "Engineering prep only; two-environment clean replay and legal "
                "review remain for full R6 acceptance.",
    }
    (out / "ReleaseManifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n")

    status = {
        "phase": "R6", "substate": "ENGINEERING_PREP_DONE",
        "git_commit": git["commit"], "git_dirty": git["dirty"],
        "n_artifacts_sealed": len(artifacts),
        "n_artifacts_missing": len(missing),
        "legal_closed": False,
        "generated_at_utc": _utc(),
    }
    (out / "STATUS.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(status, indent=2, ensure_ascii=False))
    return status


def _utc():
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


if __name__ == "__main__":
    import sys
    run(json.loads(Path(sys.argv[1]).read_text()))
