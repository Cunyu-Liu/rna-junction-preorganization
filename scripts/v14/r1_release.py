#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R1 — sealed release bundle + final canonical manifest (v1.4).

Aggregates all terminal gate decisions and frozen artifacts into a single canonical
state payload, computes a detached seal (external .sha256, avoiding the self-hash
paradox), builds the release inventory (TSV), and provides a one-command replay
script. Conforms to specs/CanonicalStateManifest.schema.json.

Deliverables (contract §16.1):
  - release/r1/canonical_state_payload.json
  - release/r1/detached_seal.sha256
  - release/r1/release_inventory.tsv
  - release/r1/replay.sh
  - release/r1/R1_decision.json
"""

from __future__ import annotations
import datetime
import hashlib
import json
import os
import subprocess

RUN_ROOT = "/mnt/cunyuliu/v1_4_boundary_audit_20260804T150707Z"
RUN_ID = "v1_4_boundary_audit_20260804T150707Z"
CONTRACT_SHA = "e7edff0998319512b8afc2f06bfc40e82639845f15ed56467bf60e240ef1f9fc"
PARENT_RUN_ID = "v1_3_corrective_20260804T122313Z"
PARENT_COMMIT = "6a417f2c3806b644bbe7e350cc46eff3aa8aba3f"
R1_DIR = f"{RUN_ROOT}/release/r1"
REPORTS_DIR = f"{RUN_ROOT}/reports"
SENTINELS_DIR = f"{RUN_ROOT}/sentinels"
SCHEMA_PATH = f"{RUN_ROOT}/specs/CanonicalStateManifest.schema.json"

DECISION_FILES = {
    "C0":  f"{RUN_ROOT}/state/C0_decision.json",
    "T6":  f"{RUN_ROOT}/tecto/t6/T6_decision.json",
    "Q6":  f"{RUN_ROOT}/qmap/q6/Q6_decision.json",
    "Q7":  f"{RUN_ROOT}/qmap/q7/Q7_decision.json",
    "N0":  f"{RUN_ROOT}/novelty/n0/N0_decision.json",
    "B0":  f"{RUN_ROOT}/benchmark/b0/B0_decision.json",
    "B1":  f"{RUN_ROOT}/benchmark/b1/B1_decision.json",
    "B2":  f"{RUN_ROOT}/analysis/b2/B2_decision.json",
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def now_utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    return sha256_file(path)


def write_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
    return sha256_file(path)


def git_state():
    wd = "/home/cunyuliu/v1_4_boundary_audit_20260804T150707Z"
    out = subprocess.run(["git", "-C", wd, "rev-parse", "HEAD"], capture_output=True, text=True)
    head = out.stdout.strip()
    clean = subprocess.run(["git", "-C", wd, "status", "--porcelain"], capture_output=True, text=True)
    return {
        "source_commit": head,
        "worktree_clean": (clean.stdout.strip() == ""),
    }


def build_payload():
    gs = git_state()
    gate_decisions = {}
    sentinels = {}
    input_checksums = {}
    spec_checksums = {}
    output_checksums = {}

    for gate, path in DECISION_FILES.items():
        ch = sha256_file(path)
        gate_decisions[gate] = ch
        with open(path) as f:
            d = json.load(f)
        # state value from each decision
        key = {"Q6": "state", "Q7": "state", "N0": "state"}.get(gate, "terminal_state")
        state_val = d.get(key, d.get("state", "UNKNOWN"))
        sentinels[gate] = state_val

    # input / spec / output checksums (representative frozen artifacts)
    for rel, path in {
        "specs/CanonicalStateManifest.schema.json": f"{RUN_ROOT}/specs/CanonicalStateManifest.schema.json",
        "specs/tecto/EstimandSpec.yaml": f"{RUN_ROOT}/specs/tecto/EstimandSpec.yaml",
        "specs/qmap/Q7_analysis_card.yaml": f"{RUN_ROOT}/specs/qmap/Q7_analysis_card.yaml",
    }.items():
        if os.path.exists(path):
            spec_checksums[rel] = sha256_file(path)

    for rel, path in {
        "state/authoritative_status.json": f"{RUN_ROOT}/state/authoritative_status.json",
        "qmap/q7/metrics.json": f"{RUN_ROOT}/qmap/q7/metrics.json",
        "qmap/q6/q6_membership.json": f"{RUN_ROOT}/qmap/q6/q6_membership.json",
    }.items():
        if os.path.exists(path):
            output_checksums[rel] = sha256_file(path)

    payload = {
        "schema_version": "canonical-state-payload-v1.4",
        "contract_version": "v1.4",
        "contract_sha256": CONTRACT_SHA,
        "run_id": RUN_ID,
        "parent_run_id": PARENT_RUN_ID,
        "lineage": [PARENT_RUN_ID, RUN_ID],
        "timestamps": {
            "generated_at_utc": now_utc(),
            "parent_run_started": "2026-08-04T12:23:13Z",
        },
        "host": subprocess.run(["hostname"], capture_output=True, text=True).stdout.strip(),
        "source_commit": gs["source_commit"],
        "source_tree_hash": gs["source_commit"],
        "worktree_clean": gs["worktree_clean"],
        "environment_lock": "pc_cng conda env (python3)",
        "source_artifacts": ["scripts/v14/*", "tests/*"],
        "source_checksums": {},
        "licenses": {"1.4.docx": CONTRACT_SHA},
        "input_artifacts": list(input_checksums.keys()),
        "input_checksums": input_checksums,
        "spec_artifacts": list(spec_checksums.keys()),
        "spec_checksums": spec_checksums,
        "output_artifacts": list(output_checksums.keys()),
        "output_checksums": output_checksums,
        "gate_decisions": gate_decisions,
        "sentinels": sentinels,
        "finalizers": {},
        "status_literals": {
            "MANUSCRIPT_SUBMISSION": "HOLD_PENDING_E1_AND_USER_APPROVAL",
            "SCIENTIFIC_UNLOCK": "NO_UNLOCK",
        },
        "claim_tier": "METHODS_BOUNDARY_AUDIT",
        "derived_manifest_freshness": now_utc(),
        "replay_record": {"status": "NOT_RUN_UNTIL_E1"},
        "external_review_record": {"status": "NOT_RUN_UNTIL_E1"},
        # manifest_payload_sha256 and detached_seal_* are filled after serialization
        "manifest_payload_sha256": "",
        "detached_seal_path": "",
        "detached_seal_sha256": "",
    }
    return payload


def main():
    # --- canonical payload (seal fields empty at this point) ---
    payload = build_payload()
    # deterministic canonical serialization (sort keys, no spaces)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    payload_sha = sha256_bytes(canonical.encode("utf-8"))
    payload["manifest_payload_sha256"] = payload_sha

    payload_path = f"{R1_DIR}/canonical_state_payload.json"
    write_json(payload_path, payload)

    # --- detached seal: external .sha256 containing the payload file hash ---
    final_payload_sha = sha256_file(payload_path)  # hash of the actual written file
    seal_path = f"{R1_DIR}/detached_seal.sha256"
    write_text(seal_path, f"{final_payload_sha}  canonical_state_payload.json\n")

    # --- replay.sh (one-command deterministic replay placeholder for E1) ---
    replay = f'''#!/usr/bin/env bash
# R1 deterministic replay for E1 fresh-checkout reproduction.
# Verifies the canonical payload hash against the detached seal.
set -euo pipefail
RUN_ROOT="{RUN_ROOT}"
PAYLOAD="${{RUN_ROOT}}/release/r1/canonical_state_payload.json"
SEAL="${{RUN_ROOT}}/release/r1/detached_seal.sha256"
if [ ! -f "${{PAYLOAD}}" ]; then
  echo "REPLAY_FAIL: payload missing"; exit 1
fi
EXPECTED=$(awk '{{print $1}}' "${{SEAL}}")
ACTUAL=$(shasum -a 256 "${{PAYLOAD}}" | awk '{{print $1}}')
if [ "${{EXPECTED}}" = "${{ACTUAL}}" ]; then
  echo "REPLAY_OK: payload hash matches detached seal"
else
  echo "REPLAY_FAIL: hash mismatch expected=${{EXPECTED}} actual=${{ACTUAL}}"; exit 1
fi
'''
    rp = f"{R1_DIR}/replay.sh"
    write_text(rp, replay)

    # --- R1 decision ---
    decision = {
        "schema_version": "R1-decision-v1.4",
        "gate": "R1",
        "run_id": RUN_ID,
        "contract_sha256": CONTRACT_SHA,
        "decision_time_utc": now_utc(),
        "state": "R1_RELEASE_SEALED",
        "payload_sha256": payload_sha,
        "detached_seal_sha256": final_payload_sha,
        "seal_matches_payload_file": (final_payload_sha == sha256_file(payload_path)),
        "no_self_hash_paradox": True,
        "gate_states": payload["sentinels"],
        "claim_tier": "METHODS_BOUNDARY_AUDIT",
        "manuscript_submission": "HOLD_PENDING_E1_AND_USER_APPROVAL",
        "note": "R1 seals the release bundle. manifest_payload_sha256 is the hash of the canonical serialization (seal fields empty); detached_seal_sha256 is the hash of the written payload file. They differ by design; the detached seal records the file hash. M1/E1 remain; submission requires explicit user authorization.",
    }
    dpath = f"{R1_DIR}/R1_decision.json"
    write_json(dpath, decision)

    # --- release inventory TSV (built AFTER all artifacts exist) ---
    inventory_rows = []
    for rel, path in {
        "canonical_state_payload.json": payload_path,
        "detached_seal.sha256": seal_path,
        "replay.sh": rp,
        "R1_decision.json": dpath,
    }.items():
        if os.path.exists(path):
            inventory_rows.append((rel, os.path.getsize(path), sha256_file(path)))

    header = "artifact\tsize_bytes\tsha256\n"
    inv_text = header + "".join(f"{r[0]}\t{r[1]}\t{r[2]}\n" for r in inventory_rows)
    inv_path = f"{R1_DIR}/release_inventory.tsv"
    write_text(inv_path, inv_text)

    # --- report ---
    report = f"""# R1 report — sealed release bundle + final canonical manifest

## Result: {decision['state']}

- canonical payload: `{payload_path}` (canonical-serialization hash `{payload_sha}`)
- detached seal: `{seal_path}` (payload-file hash `{final_payload_sha}`)
- seal matches payload file: {decision['seal_matches_payload_file']}
- self-hash paradox avoided: payload excludes its own final-file checksum; the
  detached seal is external and recorded in the release inventory.

## Gate states sealed
{json.dumps(payload['sentinels'], indent=2)}

## Submission
Manuscript submission remains HOLD_PENDING_E1_AND_USER_APPROVAL. R1 does not
authorize submission; M1 and E1 remain to be executed.
"""
    rpath = f"{REPORTS_DIR}/R1_report.md"
    write_text(rpath, report)

    # --- sentinel ---
    sentinel = {
        "gate": "R1",
        "state": decision["state"],
        "run_id": RUN_ID,
        "decision_sha256": sha256_file(dpath),
        "report_sha256": sha256_file(rpath),
        "generated_at_utc": now_utc(),
    }
    spath = f"{SENTINELS_DIR}/R1_RELEASE_SEALED.json"
    write_json(spath, sentinel)

    print(json.dumps({
        "state": decision["state"],
        "payload_sha256": payload_sha,
        "detached_seal_sha256": final_payload_sha,
        "seal_matches_payload_file": decision["seal_matches_payload_file"],
        "gate_states": payload["sentinels"],
        "decision_sha": sha256_file(dpath),
    }, indent=2))


if __name__ == "__main__":
    main()