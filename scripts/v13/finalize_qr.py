"""QR0-QR3 finalizer: record gate closure for qMaP corrective stages (v1.3)."""
from __future__ import annotations
import json
import os
import sys
import datetime
import hashlib

RUN_ID = os.environ.get("RNA_V13_RUN_ID", "v1_3_corrective_20260804T122313Z")
RUN_ROOT = os.environ.get("RNA_V13_RUN_ROOT", f"/mnt/cunyuliu/{RUN_ID}")

ARTIFACTS = {
    "QR0": os.path.join(RUN_ROOT, "qmap", "qr0", "qr0_summary.json"),
    "QR1": os.path.join(RUN_ROOT, "qmap", "qr1", "qr1_category_reconstruction.json"),
    "QR2": os.path.join(RUN_ROOT, "qmap", "qr2", "qr2_split_feasibility.json"),
    "QR3": os.path.join(RUN_ROOT, "qmap", "qr3", "qr3_transfer_result.json"),
}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    decisions = {}
    ok = True
    for gate, path in ARTIFACTS.items():
        if not os.path.isfile(path):
            decisions[gate] = {"terminal_state": "FAIL_CLOSED_MISSING", "present": False}
            ok = False
            continue
        d = json.load(open(path))
        info = {"present": True, "sha256": sha256_file(path)}
        if gate == "QR0":
            info["n_q1_99"] = d.get("n_q1_99")
            info["n_q2_98"] = d.get("n_q2_98")
            info["excluded"] = d.get("excluded_99_to_98", {}).get("canonical_id")
        elif gate == "QR1":
            info["category_counts"] = d.get("q2_category_counts")
        elif gate == "QR2":
            info["recommended_primary"] = d.get("recommended_primary")
        elif gate == "QR3":
            info["disposition"] = d.get("adjudication", {}).get("disposition")
        decisions[gate] = info

    summary = {
        "run_id": RUN_ID,
        "finalized_at_utc": ts,
        "gates": decisions,
        "proceed_to_P0": ok,
    }
    outdir = os.path.join(RUN_ROOT, "qmap", "qr_finalize")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "qr_gates_closed.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("[QR-finalize] decisions:", json.dumps({k: v.get("disposition", v.get("sha256", "")) for k, v in decisions.items()}, indent=2))
    print("[QR-finalize] proceed_to_P0=%s" % ok)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())