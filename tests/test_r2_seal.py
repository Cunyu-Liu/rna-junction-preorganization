"""R2 final clean-commit recursive seal tests (v1.5 §20)."""
import csv
import json
import os

RUN_ROOT = "/mnt/cunyuliu/v1_5_manuscript_readiness_20260805T052052Z"
R2 = os.path.join(RUN_ROOT, "release", "r2")


def test_decision_state():
    with open(os.path.join(R2, "R2_decision.json")) as f:
        d = json.load(f)
    assert d["state"] == "R2_RELEASE_SEALED_FINAL"
    assert d["gate"] == "R2"
    assert d["detached_verification"] is True


def test_m3_precondition():
    assert os.path.exists(os.path.join(
        RUN_ROOT, "sentinels", "M3_CORRECTIONS_CLOSED_CARRIED_X1_R2.sentinel"))


def test_inventory_schema_and_self_exclusion():
    with open(os.path.join(R2, "release_inventory.tsv")) as f:
        rows = list(csv.reader(f, delimiter="\t"))
    assert rows[0] == ["relative_path", "size_bytes", "sha256"]
    assert len(rows) > 100
    rels = [r[0] for r in rows[1:]]
    # self-hash paradox avoided: manifest/seal files must not cover themselves
    assert "release/r2/canonical_manifest.json" not in rels
    assert "release/r2/detached_seal.json" not in rels
    assert "release/r2/release_inventory.tsv" not in rels


def test_canonical_manifest_authority():
    with open(os.path.join(R2, "canonical_manifest.json")) as f:
        c = json.load(f)
    assert c["state"] == "R2_RELEASE_SEALED_FINAL"
    assert c["final_commit"]
    assert c["old_manifests"].startswith("derived/stale")
    assert "self_hash_exclusion" in c


def test_detached_verification():
    with open(os.path.join(R2, "detached_verification.json")) as f:
        d = json.load(f)
    assert d["verified"] is True
    assert d["mismatch_count"] == 0


def test_seal_hashes_match_inventory():
    with open(os.path.join(R2, "detached_seal.json")) as f:
        seal = json.load(f)
    with open(os.path.join(R2, "release_inventory.tsv")) as f:
        rows = list(csv.reader(f, delimiter="\t"))[1:]
    assert len(seal["sha256"]) == len(rows)
    for r in rows:
        assert seal["sha256"][r[0]] == r[2]


def test_report_and_sentinel():
    assert os.path.exists(os.path.join(RUN_ROOT, "reports", "R2_report.md"))
    sent = os.path.join(RUN_ROOT, "sentinels", "R2_RELEASE_SEALED_FINAL.sentinel")
    assert os.path.exists(sent)
    with open(sent) as f:
        c = f.read()
    assert "state=R2_RELEASE_SEALED_FINAL" in c
    assert "detached_verification=PASS" in c