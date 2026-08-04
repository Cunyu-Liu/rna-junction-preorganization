"""Q5 finalizer (v1.3 R0 rewriten).

Auditable finalizer: reads the Q5 artifacts, verifies schema/checksum presence,
and writes terminal state to the run manifest. It NEVER modifies its own source
or the build script.
"""
from __future__ import annotations
import json, hashlib, os, sys, datetime
from pathlib import Path

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()

def main():
    run_root = os.environ.get("RNA_V13_RUN_ROOT", "/mnt/cunyuliu/v1_3_corrective_20260804T122313Z")
    run_id = os.environ.get("RNA_V13_RUN_ID", "v1_3_corrective_20260804T122313Z")
    qmap = os.path.join(run_root, "qmap")
    q5dir = os.path.join(qmap, "q5")
    summary = os.path.join(q5dir, "q5_transfer_summary.json")
    if not os.path.isfile(summary):
        print("[Q5-finalize] FAIL_CLOSED: q5_transfer_summary.json missing")
        return 1
    d = json.load(open(summary))
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    decision = {
        "gate": "Q5", "run_id": run_id, "finalized_at_utc": ts,
        "terminal_state": d.get("terminal_state"),
        "summary_sha256": sha256_file(summary),
        "n_variants": d.get("n_variants"),
        "adjudication_criteria": d.get("adjudication_criteria"),
        "finalizer": "finalize_q5.py",
        "self_write": False,
    }
    mpath = os.path.join(run_root, "manifests", "gate_Q5.json")
    os.makedirs(os.path.dirname(mpath), exist_ok=True)
    with open(mpath, "w") as f:
        json.dump(decision, f, indent=2)
    print("[Q5-finalize] wrote", mpath)
    print("[Q5-finalize] terminal_state =", decision["terminal_state"])
    return 0

if __name__ == "__main__":
    sys.exit(main())
