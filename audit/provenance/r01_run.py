"""R0.1 orchestrator: bind authority, contract, source, and freeze GateSpec.

Contract §13.2 (serial first).  Creates, in RUN_ROOT/authority/:
  CanonicalStateManifest_v2.json
  RunDAG_v2.json
  GateStatus_v2.json
  GateSpec_v2.json
  STATUS.json

The run root is a NEW, isolated, hash-bound root.  Old P1-P6 artifacts are
never modified.  Any authority condition that is FALSE makes the gate FAIL
(fail-closed) with evidence.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np  # noqa: F401  (kept to match env assumptions)

from audit.provenance.authority_v2 import (
    run_authority_gate, write_authority_v2, AUTHORIZED_CONTRACT_SHA,
    CANONICAL_SOURCE_SHA, sha256,
)
from audit.provenance.gate_spec_v2 import GATE_SPEC, write_gate_spec, gate_spec_hash

# Relative path (inside worktree) to the tracked strict contract.
CONTRACT_REL = "contract/rna_junction_post_execution_strict_audit_2026-08-09.md"


def utc_now():
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main(cfg: dict):
    run_root = Path(cfg["run_root"])
    worktree = Path(cfg["worktree"])
    run_id = cfg["run_id"]
    utc = utc_now()
    auth = run_root / "authority"
    auth.mkdir(parents=True, exist_ok=True)

    contract_path = worktree / CONTRACT_REL
    if not contract_path.exists():
        print(json.dumps({"phase": "R0.1", "state": "FAIL",
                          "reason": f"contract missing: {contract_path}"}))
        (auth / "STATUS.json").write_text(json.dumps(
            {"run_id": run_id, "phase": "R0.1", "state": "FAIL",
             "reason": "contract missing", "contract_path": str(contract_path)},
            indent=2) + "\n")
        return {"state": "FAIL"}

    node_ids = cfg["dag_nodes"]
    edges = [tuple(e) for e in cfg["dag_edges"]]

    # 1. authority gate + manifests
    status = write_authority_v2(
        worktree, auth, contract_path,
        Path(cfg["canonical_source"]), node_ids, edges, run_id, utc)

    # 2. freeze GateSpec_v2 (hash-bound, before any predictions)
    gh = write_gate_spec(auth, GATE_SPEC)

    # 3. R0.1 STATUS
    r01 = {
        "run_id": run_id, "phase": "R0.1", "generated_at_utc": utc,
        "state": status["state"],
        "authority": {
            "strict_contract_sha": AUTHORIZED_CONTRACT_SHA,
            "canonical_source_sha": CANONICAL_SOURCE_SHA,
            "gate_spec_v2_hash": gh,
            "contract_path_in_worktree": str(contract_path),
        },
        "run_dag": {
            "nodes": node_ids, "edges": edges,
            "n_dangling": len(status["run_dag"]["dangling_parent_edges"]),
            "n_cycles": len(status["run_dag"]["cycles"]),
        },
        "artifacts": {
            "CanonicalStateManifest_v2.json": str(auth / "CanonicalStateManifest_v2.json"),
            "RunDAG_v2.json": str(auth / "RunDAG_v2.json"),
            "GateStatus_v2.json": str(auth / "GateStatus_v2.json"),
            "GateSpec_v2.json": str(auth / "GateSpec_v2.json"),
        },
        "gates": {
            "authority_bound_fail_closed": status["state"] == "PASS",
            "gate_spec_frozen_before_predictions": True,
        },
    }
    (auth / "STATUS.json").write_text(
        json.dumps(r01, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(r01, indent=2, ensure_ascii=False))
    return r01


if __name__ == "__main__":
    main(json.loads(Path(sys.argv[1]).read_text()))
