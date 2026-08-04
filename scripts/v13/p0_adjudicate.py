"""P0: publication-route and claim adjudication (v1.3).

P0 is the ONLY unlock point for submission and claim. It does NOT predict
journal acceptance; it only decides which tier of claim the evidence
authorizes, based on the real results of T5, QR3, QR4 (and preserved T4).

The final route is determined ONLY by real results, not by engineering PASS.
"""
from __future__ import annotations
import json
import os
import sys
import datetime
import hashlib

RUN_ID = os.environ.get("RNA_V13_RUN_ID", "v1_3_corrective_20260804T122313Z")
RUN_ROOT = os.environ.get("RNA_V13_RUN_ROOT", f"/mnt/cunyuliu/{RUN_ID}")

T5 = os.path.join(RUN_ROOT, "tecto", "t5", "t5_decision.json")
QR3 = os.path.join(RUN_ROOT, "qmap", "qr3", "qr3_transfer_result.json")
QR4 = os.path.join(RUN_ROOT, "qmap", "qr4", "qr4_replay_decision.json")
T4 = os.path.join(RUN_ROOT, "tecto", "t4", "t4_audit.json")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load(p):
    if not os.path.isfile(p):
        return None
    with open(p) as f:
        return json.load(f)


def main():
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    t5 = load(T5)
    qr3 = load(QR3)
    qr4 = load(QR4)
    t4 = load(T4)

    missing = [n for n, d in [("T5", t5), ("QR3", qr3), ("QR4", qr4), ("T4", t4)] if d is None]
    if missing:
        print("[P0] FAIL_CLOSED: missing evidence for %s" % missing)
        return 1

    # Extract real results
    t5_result = {
        "t3_score": t5.get("t3_score"),
        "motif_mean": t5.get("motif_mean"),
        "relative_gain": t5.get("relative_gain"),
        "t3_beats_baseline": t5.get("t3_beats_baseline"),
        "bootstrap_gain_ci": t5.get("bootstrap_gain_ci"),
        "bootstrap_gain_positive_frac": t5.get("bootstrap_gain_positive_frac"),
        "width_ok": t5.get("width_ok"),
        "frac_intervals_le_1kcal": t5.get("frac_intervals_le_1kcal"),
    }
    qr3_result = {
        "disposition": qr3.get("adjudication", {}).get("disposition"),
        "micro_gain_b2_over_b1": qr3.get("aggregate", {}).get("micro_gain_b2_over_b1"),
        "group_weighted_gain_b2_over_b1": qr3.get("aggregate", {}).get("group_weighted_gain_b2_over_b1"),
    }
    qr4_state = qr4.get("state")

    # ---- Adjudication logic (real results only) ----
    # Route selection per v1.3 13.2 / 15.1-7:
    #   strong method route: requires T5 positive AND QR3 supported AND QR4 replay match
    #   tecto-specific: requires T5 positive
    #   qMaP benchmark/boundary: QR3 supported but T5 negative
    #   boundary/audit route: QR3 NOT supported and T5 negative (both negative)
    #   STOP: QR4 mismatch
    t5_positive = bool(t5_result["t3_beats_baseline"])
    qr3_supported = qr3_result["disposition"] == "QMAP_TRANSFER_SUPPORTED"
    qr4_match = qr4_state == "REPLAY_MATCH"

    routes = []
    if not qr4_match:
        routes.append("STOP_REPRODUCIBILITY")
    elif t5_positive and qr3_supported:
        routes.append("STRONG_METHOD")
    elif t5_positive:
        routes.append("TECTO_SPECIFIC")
    elif qr3_supported:
        routes.append("QMAP_BENCHMARK")
    else:
        routes.append("BOUNDARY_AUDIT")

    claim_tier = "NO_STRONG_CROSS_SYSTEM_CLAIM"
    if routes[0] == "STRONG_METHOD":
        claim_tier = "STRONG_METHOD_CLAIM"
    elif routes[0] == "TECTO_SPECIFIC":
        claim_tier = "TECTO_SPECIFIC_BOUNDED"
    elif routes[0] == "QMAP_BENCHMARK":
        claim_tier = "QMAP_BENCHMARK_CLASS"
    elif routes[0] == "BOUNDARY_AUDIT":
        claim_tier = "METHODS_BOUNDARY_AUDIT"
    else:
        claim_tier = "HOLD_REPRODUCIBILITY"

    decision = {
        "gate": "P0",
        "run_id": RUN_ID,
        "finalized_at_utc": ts,
        "real_results": {
            "T5": t5_result,
            "QR3": qr3_result,
            "QR4_state": qr4_state,
            "T4_rigor": t4.get("overall_rigor"),
        },
        "adjudication": {
            "t5_positive": t5_positive,
            "qr3_supported": qr3_supported,
            "qr4_replay_match": qr4_match,
        },
        "selected_route": routes[0],
        "claim_tier": claim_tier,
        "manuscript_submission": "HOLD" if routes[0] == "STOP_REPRODUCIBILITY" else "AUTHORIZED_UNDER_CLAIM_TIER",
        "note": (
            "P0 does not evaluate journal hit probability; it only decides which "
            "tier of claim the evidence authorizes. With T5 negative and QR3 "
            "NOT_SUPPORTED, the authorized route is a methods-boundary / audit / "
            "partial-ID manuscript, NOT a strong cross-system or strong method claim."
        ),
        "evidence_sha256": {
            "T5": sha256_file(T5),
            "QR3": sha256_file(QR3),
            "QR4": sha256_file(QR4),
            "T4": sha256_file(T4),
        },
    }

    outdir = os.path.join(RUN_ROOT, "p0")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "p0_adjudication.json"), "w") as f:
        json.dump(decision, f, indent=2)
    sentinel = os.path.join(RUN_ROOT, "state", "P0_CLOSED.yaml")
    os.makedirs(os.path.dirname(sentinel), exist_ok=True)
    with open(sentinel, "w") as f:
        f.write(f"gate: P0\nselected_route: {routes[0]}\nclaim_tier: {claim_tier}\nfinalized_at_utc: {ts}\n")

    print("[P0] t5_positive=%s qr3_supported=%s qr4_match=%s" % (t5_positive, qr3_supported, qr4_match))
    print("[P0] selected_route=%s" % routes[0])
    print("[P0] claim_tier=%s" % claim_tier)
    return 0


if __name__ == "__main__":
    sys.exit(main())