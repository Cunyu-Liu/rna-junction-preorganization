"""v1.5 final acceptance report + handoff tests (§26)."""
import csv
import json
import os

RUN_ROOT = "/mnt/cunyuliu/v1_5_manuscript_readiness_20260805T052052Z"


def test_final_status():
    with open(os.path.join(RUN_ROOT, "state", "final_status.json")) as f:
        d = json.load(f)
    assert d["CURRENT_OPERATIONAL_STATE"] == "X1_AWAITING_INDEPENDENT_REVIEW"
    assert d["MANUSCRIPT_SUBMISSION"] == "HOLD_PENDING_X1_R2_S1_AND_USER_AUTHORIZATION"
    assert d["PUBLIC_RELEASE"] == "HOLD_PENDING_USER_AUTHORIZATION"
    assert d["SCIENTIFIC_UNLOCK"] == "NO_UNLOCK"
    assert "X1_AWAITING_INDEPENDENT_REVIEW" in d["gate_states"]["X1"]


def test_acceptance_report():
    p = os.path.join(RUN_ROOT, "reports", "v1_5_acceptance_report.md")
    assert os.path.exists(p)
    with open(p) as f:
        c = f.read()
    assert "X1_AWAITING_INDEPENDENT_REVIEW" in c
    assert "RNA_THERMODYNAMIC_AUDIT_RESOURCE_NOTE" in c
    assert "No writes to the parent v1.4 run root" in c


def test_canonical_manifest():
    with open(os.path.join(RUN_ROOT, "manifests", "canonical_manifest.json")) as f:
        c = json.load(f)
    assert c["gate_states"]["R2"] == "R2_RELEASE_SEALED_FINAL"
    assert c["gate_states"]["S1"] == "S1_INTERNAL_PACKAGE_READY_HOLD_USER_AUTHORIZATION"
    assert c["submission_authorization"] == "HOLD_PENDING_USER_AUTHORIZATION"


def test_inventory_and_checksums():
    assert os.path.exists(os.path.join(RUN_ROOT, "manifests", "artifact_inventory.tsv"))
    with open(os.path.join(RUN_ROOT, "manifests", "artifact_inventory.tsv")) as f:
        rows = list(csv.reader(f, delimiter="\t"))
    assert rows[0] == ["relative_path", "size_bytes", "sha256"]
    assert len(rows) > 20
    checksums = os.path.join(RUN_ROOT, "manifests", "checksums.sha256")
    assert os.path.exists(checksums)
    with open(checksums) as f:
        n = sum(1 for _ in f)
    assert n == len(rows) - 1


def test_sentinel_registry():
    with open(os.path.join(RUN_ROOT, "manifests", "sentinel_registry.tsv")) as f:
        rows = list(csv.reader(f, delimiter="\t"))
    assert rows[0] == ["sentinel", "state", "size_bytes", "sha256"]
    states = [r[1] for r in rows[1:]]
    assert "R2_RELEASE_SEALED_FINAL" in states
    assert "X1_AWAITING_INDEPENDENT_REVIEW" in states


def test_residual_risk_and_runbook():
    with open(os.path.join(RUN_ROOT, "reports", "residual_risk_register.tsv")) as f:
        rows = list(csv.reader(f, delimiter="\t"))
    assert rows[0] == ["risk_id", "severity", "description", "status", "mitigation_or_recovery"]
    assert any("X1-independent-recomputation" == r[0] for r in rows[1:])
    assert os.path.exists(os.path.join(RUN_ROOT, "reports", "next_action_runbook.md"))
    with open(os.path.join(RUN_ROOT, "reports", "next_action_runbook.md")) as f:
        assert "X1" in f.read()