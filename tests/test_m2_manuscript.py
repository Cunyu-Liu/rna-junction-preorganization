"""M2 manuscript generation tests (v1.5)."""
import json
import os

RUN_ROOT = "/mnt/cunyuliu/v1_5_manuscript_readiness_20260805T052052Z"
M2 = os.path.join(RUN_ROOT, "manuscript", "m2")


def test_decision_state():
    with open(os.path.join(M2, "M2_decision.json")) as f:
        d = json.load(f)
    assert d["state"] == "M2_MANUSCRIPT_DRAFT_READY"


def test_hard_boundary_compliance():
    with open(os.path.join(M2, "M2_decision.json")) as f:
        d = json.load(f)
    hb = d["hard_boundary_compliance"]
    for k, v in hb.items():
        assert v is True, f"boundary not enforced: {k}"


def test_documents_exist():
    for name in ("manuscript.docx", "manuscript.pdf", "supplement.docx",
                 "supplement.pdf", "cover_letter_draft.docx", "cover_letter_draft.pdf"):
        assert os.path.exists(os.path.join(M2, name))


def test_claim_map_and_checklist():
    assert os.path.exists(os.path.join(M2, "claim_evidence_citation_map.tsv"))
    assert os.path.exists(os.path.join(M2, "reporting_checklist.tsv"))


def test_manuscript_nonempty():
    assert os.path.getsize(os.path.join(M2, "manuscript.pdf")) > 5000