"""CanonicalStateManifest — the single authoritative execution-state source for v1.2.

Only the finalizer may write a formal Gate to PASS. Training/report scripts have no
right to write PASS. The manifest fails closed if it is inconsistent with the current
source or inputs. Old reports/registries conflicting with the canonical manifest are
marked STALE_NOT_AUTHORITATIVE.

This module is intentionally dependency-light (stdlib json + hashlib + jsonschema if
available) so it can run in any environment.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

GATE_STATES = {
    "NOT_STARTED", "RUNNING", "PASS", "FAIL", "BLOCKED",
    "CLOSED", "NOT_APPLICABLE", "STALE_NOT_AUTHORITATIVE",
}
OPERATIONAL_STATES = {
    "BLOCKED_AT_TECTO_DATA_ADMISSION", "RUNNING", "TECTO_SPECIFIC",
    "STOP_MECHANISM_ROUTE", "IMPLEMENTATION_COMPLETE",
}
SCIENTIFIC_DISPOSITIONS = {"CONDITIONAL_CANDIDATE", "NOT_ADJUDICATED", "ADJUDICATED"}
SCIENTIFIC_UNLOCKS = {"NO_UNLOCK", "UNLOCKED"}
QMAP_DISPOSITIONS = {
    "NOT_STARTED", "NOT_ADJUDICATED", "QMAP_TRANSFER_SUPPORTED",
    "QMAP_TRANSFER_NOT_SUPPORTED", "QMAP_INCONCLUSIVE", "QMAP_NOT_ADMITTED",
}
CLAIM_CLASSES = {"NOT_ADJUDICATED", "TECTO_SPECIFIC", "STRONG_CROSS_SYSTEM", "STOP_MECHANISM_ROUTE"}
FINALIZER_STATUSES = {"NOT_RUN", "PASS", "FAIL", "BLOCKED"}

SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "..", "schemas", "canonical_manifest.schema.json")

DEFAULT_STATE = {
    "CURRENT_OPERATIONAL_STATE": "BLOCKED_AT_TECTO_DATA_ADMISSION",
    "CURRENT_SCIENTIFIC_DISPOSITION": "CONDITIONAL_CANDIDATE",
    "CURRENT_DMS_CROSSWALK": "ASSUMED_PERMANENTLY_UNAVAILABLE_V1_2",
    "CURRENT_DMS_PRIMARY_LABELS": "NOT_ADMITTED_FINAL_V1_2",
    "CURRENT_DMS_REPLAY": "ENGINEERING_EVIDENCE_ONLY",
    "CURRENT_DMS_JOINT_TRANSPORT": "CLOSED_NOT_AUTHORIZED",
    "QMAPSEQ_ROLE": "MANDATORY_COMPLETION_GATE_FOR_STRONG_MANUSCRIPT",
    "SCIENTIFIC_UNLOCK": "NO_UNLOCK",
}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_schema() -> Dict[str, Any]:
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_schema(manifest: Dict[str, Any]) -> List[str]:
    """Validate the manifest against the canonical schema (best-effort jsonschema)."""
    errors: List[str] = []
    try:
        import jsonschema  # type: ignore
        schema = _load_schema()
        try:
            jsonschema.validate(instance=manifest, schema=schema)
            return []
        except jsonschema.ValidationError as e:
            return [f"jsonschema: {e.message}"]
    except ImportError:
        pass
    # Fallback minimal checks
    for field in _load_schema()["required"]:
        if field not in manifest:
            errors.append(f"missing required field: {field}")
    gs = manifest.get("gate_statuses", {})
    for k, v in gs.items():
        if v not in GATE_STATES:
            errors.append(f"gate_statuses[{k}] invalid state: {v}")
    if manifest.get("current_operational_state") not in OPERATIONAL_STATES:
        errors.append("invalid current_operational_state")
    if manifest.get("current_scientific_disposition") not in SCIENTIFIC_DISPOSITIONS:
        errors.append("invalid current_scientific_disposition")
    if manifest.get("scientific_unlock") not in SCIENTIFIC_UNLOCKS:
        errors.append("invalid scientific_unlock")
    if manifest.get("qmap_terminal_disposition") not in QMAP_DISPOSITIONS:
        errors.append("invalid qmap_terminal_disposition")
    if manifest.get("claim_class") not in CLAIM_CLASSES:
        errors.append("invalid claim_class")
    if manifest.get("finalizer_status") not in FINALIZER_STATUSES:
        errors.append("invalid finalizer_status")
    return errors


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class CanonicalStateManifest:
    def __init__(self, data: Dict[str, Any]):
        self.data = data

    @classmethod
    def new(cls, *, contract_version: str, contract_sha256: str, code_commit: str,
            worktree_path: str, run_id: str, parent_run_id: str, host: str,
            environment_lock_hash: str = "") -> "CanonicalStateManifest":
        now = utcnow()
        data: Dict[str, Any] = {
            "schema_version": "1.0.0",
            "contract_version": contract_version,
            "contract_sha256": contract_sha256,
            "code_commit": code_commit,
            "worktree_path": worktree_path,
            "run_id": run_id,
            "parent_run_id": parent_run_id,
            "created_at_utc": now,
            "updated_at_utc": now,
            "host": host,
            "environment_lock_hash": environment_lock_hash,
            "source_manifests": [],
            "input_artifacts": [],
            "input_checksums": {},
            "output_artifacts": [],
            "output_checksums": {},
            "gate_statuses": {
                "T0": "RUNNING", "S0": "NOT_STARTED", "T1": "NOT_STARTED",
                "M0": "NOT_STARTED", "T2": "NOT_STARTED", "T3": "NOT_STARTED",
                "Q0": "NOT_STARTED", "Q1": "NOT_STARTED", "Q2": "NOT_STARTED",
                "Q3": "NOT_STARTED", "Q4": "NOT_STARTED", "Q5": "NOT_STARTED",
            },
            "gate_decisions": {},
            "current_operational_state": "BLOCKED_AT_TECTO_DATA_ADMISSION",
            "current_scientific_disposition": "CONDITIONAL_CANDIDATE",
            "scientific_unlock": "NO_UNLOCK",
            "qmap_terminal_disposition": "NOT_STARTED",
            "claim_class": "NOT_ADJUDICATED",
            "finalizer_status": "NOT_RUN",
            "sentinel_status": "RUNNING",
            "derived_manifest_freshness": {},
        }
        return cls(data)

    @classmethod
    def load(cls, path: str) -> "CanonicalStateManifest":
        with open(path, "r", encoding="utf-8") as f:
            return cls(json.load(f))

    def save(self, path: str) -> str:
        self.data["updated_at_utc"] = utcnow()
        errors = validate_schema(self.data)
        if errors:
            raise ValueError("manifest schema invalid: " + "; ".join(errors))
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
        return sha256_file(path)

    # ---- finalizer-only gate write ----
    def write_gate(self, gate: str, status: str, decision: Optional[Dict[str, Any]] = None,
                   finalizer: bool = False) -> None:
        if status not in GATE_STATES:
            raise ValueError(f"invalid gate state: {status}")
        if status == "PASS" and not finalizer:
            raise ValueError(
                "finalizer-only: a non-finalizer cannot write a Gate to PASS. "
                "Use PARTIAL_ENGINEERING_EVIDENCE or FAIL/BLOCKED instead."
            )
        self.data["gate_statuses"][gate] = status
        if decision is not None:
            self.data["gate_decisions"][gate] = decision

    def mark_partial(self, gate: str, decision: Optional[Dict[str, Any]] = None) -> None:
        """Partial engineering success is NOT scientific PASS."""
        self.write_gate(gate, "RUNNING", decision=decision, finalizer=False)

    def mark_stale(self, gate: str) -> None:
        self.write_gate(gate, "STALE_NOT_AUTHORITATIVE", finalizer=False)


def finalize_gate(manifest: CanonicalStateManifest, gate: str, decision: Dict[str, Any],
                  required_artifacts: List[str], checksum_valid: bool,
                  tests_passed: bool, contract_hash_ok: bool, schema_ok: bool) -> str:
    """Finalizer: only sets PASS when ALL required criteria are true."""
    criteria = {
        "required_artifacts_present": all(os.path.exists(a) for a in required_artifacts),
        "checksums_valid": checksum_valid,
        "tests_passed": tests_passed,
        "contract_hash_ok": contract_hash_ok,
        "schema_ok": schema_ok,
    }
    decision["finalizer_criteria"] = criteria
    decision["finalized_at_utc"] = utcnow()
    if all(criteria.values()):
        manifest.write_gate(gate, "PASS", decision=decision, finalizer=True)
        return "PASS"
    decision["decision"] = "PARTIAL_ENGINEERING_EVIDENCE"
    manifest.write_gate(gate, "RUNNING", decision=decision, finalizer=True)
    return "PARTIAL_ENGINEERING_EVIDENCE"