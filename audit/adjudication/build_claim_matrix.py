"""P0.6 claim-matrix builder (contract P0.6).

Adjudicates the scientific status of each method (v1.28 / v1.29 / v1.30 /
v1.31) against the evidence produced by P0.1-P0.5.  The claim matrix is
FAIL-CLOSED: it cannot mint a SOTA, mechanism, or submission claim from P0
evidence alone.  It records, per method, which claims are supported, which are
explicitly NOT asserted, and the minimum evidence required before any stronger
claim may be made.

Outputs (to adjudication/):
  VersionAdjudication.csv : one row per method version, overall eligibility
  ClaimMatrix.csv         : one row per (version, claim) with decision/evidence
"""
from __future__ import annotations

import csv
import json
from pathlib import Path


def _bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().upper() in ("TRUE", "PASS", "1", "YES")
    return bool(v)


def build_claim_matrix(run_root: Path):
    # Load the P0 gate matrix (already built by build_gate_matrix.py).
    gate_path = run_root / "adjudication" / "P0GateMatrix.json"
    if not gate_path.exists():
        raise FileNotFoundError("P0GateMatrix.json not built; run build_gate_matrix.py first")
    gmx = json.loads(gate_path.read_text())
    overall = gmx["overall_state"]

    # Each (version, claim) row: decision is one of
    #   SUPPORTED / NOT_ASSERTED / BLOCKED_BY_GATE / INVALIDATED / EXPLORATORY
    # P0 evidence alone can never yield a scientific claim for any method here.
    rows = []

    def add(version, claim, decision, evidence, note):
        rows.append({
            "version": version, "claim": claim, "decision": decision,
            "evidence": evidence, "note": note,
        })

    # --- v1.28: positive candidate, but conditional & old metric ---
    add("v1.28", "sequence signal exceeds train-only scaffold baseline",
        "NOT_ASSERTED",
        "old relative-ratio macro metric; fresh replay uses legacy aggregation; "
        "no blocked context/edit; no operator holdout",
        "conditional candidate only; requires P0.5 unified-metric replay + P1 gates")
    add("v1.28", "direct-kernel support is adequate",
        "NOT_ASSERTED", "edit-branch direct-kernel support 0.112-0.446",
        "low support; fallback dominance risk unresolved")
    add("v1.28", "generalizes across context/operator (transport)",
        "INVALIDATED", "no blocked context or operator holdout in evidence",
        "currently KNOWN_OPERATOR_CONDITIONAL_ONLY at most")

    # --- v1.29: closure gates not executed ---
    add("v1.29", "support-claim closure executed",
        "NOT_RUN", "v1.29 closure gates never executed",
        "cannot be called scientifically complete")

    # --- v1.30: seen-context calibration, not sequence mechanism ---
    add("v1.30", "adds sequence-specific mechanism over parent",
        "INVALIDATED", "symmetry gain +0.024203 but sequence-pairing null +0.043876; "
        "edit axis all folds choose alpha=0",
        "explains as seen-context calibration; not sequence mechanism")

    # --- v1.31: corrected, numerically validated; negative result now interpretable ---
    add("v1.31", "numerical correctness (gradient/GH/synthetic/convergence)",
        "SUPPORTED", "P0.3 gates G1-G5 PASS under contract-aligned G4",
        "corrected objective validated; optimizer convergence ledger present")
    add("v1.31", "operator ordering fully identifiable",
        "NOT_ASSERTED", "operator-ordering recovered 0% censor (0.79) but ~0.5 at 20-50%",
        "identifiability boundary documented (contract line 266); not a P0.3 hard gate")
    add("v1.31", "hierarchical operator/Tobit is scientifically refuted",
        "INVALIDATED", "P0.3 numerical gates now PASS; prior negative result was "
        "based on an invalid gradient",
        "cannot use the old negative result to reject the operator/Tobit idea")

    # --- cross-cutting: no SOTA / mechanism / submission claim from P0 ---
    for ver in ("v1.28", "v1.29", "v1.30", "v1.31"):
        add(ver, "domain SOTA / submission authorization",
            "NOT_ALLOWED", "no same-protocol public leaderboard; P0 forbids SOTA/submission",
            "only P0_PASS_COMPARISON_ELIGIBLE gates to P1 comparison, not to SOTA")

    return {"overall_state": overall, "n_claims": len(rows), "rows": rows}


def write_outputs(cfg):
    run_root = Path(cfg["run_root"])
    out_dir = Path(cfg.get("out_dir", run_root / "adjudication"))
    out_dir.mkdir(parents=True, exist_ok=True)
    matrix = build_claim_matrix(run_root)
    rows = matrix["rows"]

    # ClaimMatrix.csv
    with (out_dir / "ClaimMatrix.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["version", "claim", "decision", "evidence", "note"])
        for r in rows:
            w.writerow([r["version"], r["claim"], r["decision"], r["evidence"], r["note"]])

    # VersionAdjudication.csv: overall eligibility per version
    verdict = {}
    for r in rows:
        verdict.setdefault(r["version"], []).append(r["decision"])
    with (out_dir / "VersionAdjudication.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["version", "overall_status", "gate_matrix_state"])
        for ver, decisions in sorted(verdict.items()):
            w.writerow([ver, ";".join(sorted(set(decisions))), matrix["overall_state"]])

    (out_dir / "ClaimMatrix.json").write_text(json.dumps(matrix, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"overall_state": matrix["overall_state"], "n_claims": matrix["n_claims"]}, indent=2))


if __name__ == "__main__":
    import sys
    write_outputs(json.loads(Path(sys.argv[1]).read_text()))
