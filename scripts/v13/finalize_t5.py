"""T5 finalizer: record locked tecto reanalysis result (v1.3)."""
from __future__ import annotations
import json
import os
import sys
import datetime
import hashlib

RUN_ID = os.environ.get("RNA_V13_RUN_ID", "v1_3_corrective_20260804T122313Z")
RUN_ROOT = os.environ.get("RNA_V13_RUN_ROOT", f"/mnt/cunyuliu/{RUN_ID}")
T5 = os.path.join(RUN_ROOT, "tecto", "t5", "t3", "t3_results.json")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if not os.path.isfile(T5):
        print("[T5-finalize] FAIL_CLOSED: t3_results.json missing")
        return 1
    d = json.load(open(T5))
    mb = d.get("matched_baseline", {})
    gb = d.get("group_bootstrap", {})
    decision = {
        "gate": "T5",
        "run_id": RUN_ID,
        "finalized_at_utc": ts,
        "device": d.get("device"),
        "result_sha256": sha256_file(T5),
        "n_rows": d.get("n_rows"),
        "n_censored": d.get("n_censored"),
        "n_measured": d.get("n_measured"),
        "t3_score": mb.get("t3_score"),
        "motif_mean": mb.get("strongest_baseline_score"),
        "relative_gain": mb.get("relative_gain"),
        "t3_beats_baseline": mb.get("t3_beats_baseline"),
        "bootstrap_gain_ci": gb.get("gain_ci"),
        "bootstrap_gain_positive_frac": gb.get("gain_positive_frac"),
        "width_ok": d.get("coverage_width", {}).get("width_ok"),
        "frac_intervals_le_1kcal": d.get("coverage_width", {}).get("frac_intervals_le_1kcal"),
        "scientific_disposition": d.get("scientific_disposition"),
        "terminal_state": "T5_LOCKED_REANALYSIS_NOT_SUPPORTED",
        "note": (
            "T5 re-ran the LOCKED analysis (frozen split/estimand/baseline per T4). "
            "The hierarchical model does NOT beat the strongest simple baseline. "
            "Per v1.3 12.2: stop architecture escalation; preserve the negative result."
        ),
    }
    outdir = os.path.join(RUN_ROOT, "tecto", "t5")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "t5_decision.json"), "w") as f:
        json.dump(decision, f, indent=2)
    sentinel = os.path.join(RUN_ROOT, "state", "T5_CLOSED.yaml")
    os.makedirs(os.path.dirname(sentinel), exist_ok=True)
    with open(sentinel, "w") as f:
        f.write(f"gate: T5\nterminal_state: {decision['terminal_state']}\nfinalized_at_utc: {ts}\n")
    print("[T5-finalize] terminal_state=%s" % decision["terminal_state"])
    print("[T5-finalize] t3_score=%.4f motif_mean=%.4f gain=%.4f" % (
        mb.get("t3_score"), mb.get("strongest_baseline_score"), mb.get("relative_gain")))
    return 0


if __name__ == "__main__":
    sys.exit(main())