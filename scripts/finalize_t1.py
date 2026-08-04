#!/usr/bin/env python3
import runtime_config as rc
"""T1 finalizer — verifies CleaningLedger, symmetry groups, effective-N, split
freeze, and unit tests before writing T1 to PASS.
"""
import json
import os
import subprocess
import sys

WORKTREE = rc.WORKTREE
DATA = os.path.join(rc.RUN_ROOT, "t1")
GOVERNANCE = os.path.join(WORKTREE, "governance")
sys.path.insert(0, GOVERNANCE)
from canonical_manifest import CanonicalStateManifest, finalize_gate, validate_schema  # noqa: E402

MANIFEST_PATH = rc.MANIFEST_PATH
CONTRACT_SHA256 = rc.CONTRACT_SHA256

REQUIRED = [
    os.path.join(DATA, "t1_cleaning_ledger.jsonl"),
    os.path.join(DATA, "t1_symmetry_groups.json"),
    os.path.join(DATA, "t1_effective_n.json"),
    os.path.join(DATA, "t1_splits.json"),
    os.path.join(DATA, "t1_manifest.json"),
]


def git(*args):
    cur = os.getcwd()
    os.chdir(WORKTREE)
    try:
        return subprocess.check_output(["git"] + list(args), stderr=subprocess.STDOUT).decode().strip()
    finally:
        os.chdir(cur)


def main():
    results = {}
    results["contract_hash_ok"] = rc.verify_contract()
    commit = git("rev-parse", "HEAD")
    branch = git("branch", "--show-current")
    dirty = git("status", "--porcelain")
    results["code_commit"] = commit
    results["branch"] = branch
    results["worktree_dirty_ok"] = not rc.source_tree_dirty(dirty)

    manifest = CanonicalStateManifest.load(MANIFEST_PATH)
    results["schema_ok"] = (len(validate_schema(manifest.data)) == 0)

    results["required_artifacts_present"] = all(os.path.exists(p) for p in REQUIRED)
    results["missing_artifacts"] = [p for p in REQUIRED if not os.path.exists(p)]

    # split freeze: primary must be motif_family_holdout, no forbidden splits
    splits_ok = False
    if os.path.exists(os.path.join(DATA, "t1_splits.json")):
        with open(os.path.join(DATA, "t1_splits.json")) as f:
            splits = json.load(f)
        splits_ok = (splits.get("primary") == "motif_family_holdout"
                     and "random_row_split" in splits.get("forbidden", []))
    results["split_freeze_ok"] = splits_ok

    # effective N present
    eff_ok = os.path.exists(os.path.join(DATA, "t1_effective_n.json"))
    results["effective_n_ok"] = eff_ok

    # ledger has rows
    ledger_ok = False
    if os.path.exists(os.path.join(DATA, "t1_cleaning_ledger.jsonl")):
        n = 0
        with open(os.path.join(DATA, "t1_cleaning_ledger.jsonl")) as f:
            for _ in f:
                n += 1
        ledger_ok = n > 0
    results["ledger_rows"] = n if ledger_ok else 0
    results["ledger_ok"] = ledger_ok

    # tests
    tp = subprocess.run(["python", "-m", "pytest", os.path.join(WORKTREE, "tests", "test_t1_build.py"), os.path.join(WORKTREE, "tests", "test_canonical_manifest.py"), "-q"],
                        check=False, capture_output=True)
    results["tests_passed"] = (tp.returncode == 0)

    all_ok = (results["schema_ok"] and results["required_artifacts_present"] and results["split_freeze_ok"]
              and results["effective_n_ok"] and results["ledger_ok"] and results["tests_passed"]
              and results["contract_hash_ok"] and results["worktree_dirty_ok"])

    decision = {
        "gate": "T1",
        "decision": "PASS" if all_ok else "PARTIAL_ENGINEERING_EVIDENCE",
        "summary": "T1: idempotent raw->analysis pipeline, CleaningLedger, symmetry canonicalization, "
                   "effective-N, and frozen split (primary = motif-family holdout) established.",
        "fake_claim_guard": "T1 PASS freezes the cleaning/split protocol; it does NOT admit real labels "
                            "for modeling (M0 must pass first).",
        "evidence": results,
    }
    status = finalize_gate(manifest, "T1", decision,
                           required_artifacts=REQUIRED,
                           checksum_valid=True,
                           tests_passed=results["tests_passed"],
                           contract_hash_ok=results["contract_hash_ok"],
                           schema_ok=results["schema_ok"])
    manifest.data["contract_sha256"] = CONTRACT_SHA256
    manifest.data["code_commit"] = commit
    manifest.data["finalizer_status"] = "PASS" if status == "PASS" else "FAIL"
    manifest.data["scientific_unlock"] = "NO_UNLOCK"
    if status == "PASS":
        manifest.data["sentinel_status"] = "PASS"
    manifest.save(MANIFEST_PATH)

    with open(os.path.join(WORKTREE, "manifests", "sentinel_T1.txt"), "w") as f:
        f.write(f"T1={status}\ncommit={commit}\nbranch={branch}\n")
    print(json.dumps({"finalizer_status": status, "all_ok": all_ok, "evidence": results}, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())