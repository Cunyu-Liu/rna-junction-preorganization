"""QR4 finalizer: record independent fresh-checkout replay result (v1.3)."""
from __future__ import annotations
import json
import os
import sys
import datetime
import hashlib

RUN_ID = os.environ.get("RNA_V13_RUN_ID", "v1_3_corrective_20260804T122313Z")
RUN_ROOT = os.environ.get("RNA_V13_RUN_ROOT", f"/mnt/cunyuliu/{RUN_ID}")
QR3 = os.path.join(RUN_ROOT, "qmap", "qr3", "qr3_transfer_result.json")
REPLAY_QR3 = "/tmp/qr4_replay_root/qmap/qr3/qr3_transfer_result.json"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def check_match(a, b):
    if not (os.path.isfile(a) and os.path.isfile(b)):
        return None
    json_a = json.load(open(a))
    json_b = json.load(open(b))
    same = (
        json_a.get("aggregate") == json_b.get("aggregate")
        and json_a.get("fold_results") == json_b.get("fold_results")
        and json_a.get("adjudication") == json_b.get("adjudication")
    )
    return same


def main():
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    matched = check_match(QR3, REPLAY_QR3)
    if matched is None:
        print("[QR4-finalize] FAIL_CLOSED: replay artifact missing")
        return 1
    state = "REPLAY_MATCH" if matched else "REPLAY_MISMATCH"
    decision = {
        "gate": "QR4",
        "run_id": RUN_ID,
        "finalized_at_utc": ts,
        "state": state,
        "original_qr3_sha256": sha256_file(QR3) if os.path.isfile(QR3) else None,
        "replay_qr3_sha256": sha256_file(REPLAY_QR3) if os.path.isfile(REPLAY_QR3) else None,
        "aggregate_match": matched,
        "note": "QR4 replayed QR3/QR0/QR1/QR2 from a fresh detached checkout at the locked commit; exact match confirms reproducibility.",
    }
    outdir = os.path.join(RUN_ROOT, "qmap", "qr4")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "qr4_replay_decision.json"), "w") as f:
        json.dump(decision, f, indent=2)
    sentinel = os.path.join(RUN_ROOT, "state", "QR4_CLOSED.yaml")
    os.makedirs(os.path.dirname(sentinel), exist_ok=True)
    with open(sentinel, "w") as f:
        f.write(f"gate: QR4\nstate: {state}\nfinalized_at_utc: {ts}\n")
    print("[QR4-finalize] state=%s" % state)
    return 0 if matched else 1


if __name__ == "__main__":
    sys.exit(main())