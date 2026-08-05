"""N1 novelty + manuscript-route gate tests (v1.5)."""
import json
import os

RUN_ROOT = "/mnt/cunyuliu/v1_5_manuscript_readiness_20260805T052052Z"


def _load(name):
    with open(os.path.join(RUN_ROOT, "novelty", "n1", name)) as f:
        return json.load(f)


def test_route_is_resource_note():
    d = _load("N1_decision.json")
    assert d["route"] == "RNA_THERMODYNAMIC_AUDIT_RESOURCE_NOTE"
    assert d["state"] == "N1_ROUTE_RNA_THERMODYNAMIC_AUDIT_RESOURCE_NOTE"


def test_route_consistent_with_b3_x0():
    d = _load("N1_decision.json")
    r = _load("manuscript_route.json")
    assert d["b3_state"] == "B3_VALIDATED"
    assert d["x0_state"] == "X0_INCONCLUSIVE_LOW_N_OR_OPERATOR_AMBIGUITY"
    assert r["x0_qualified"] is False


def test_no_strong_cross_case_claim():
    t = _load("claim_tier.json")
    assert t["authorized_claims"]["x0_strong_cross_case"] is False
    assert t["authorized_claims"]["dms_validates_tecto"] is False


def test_authorized_claims_present():
    t = _load("claim_tier.json")
    assert t["authorized_claims"]["qmap_signal_present_gain_met"] is True
    assert t["authorized_claims"]["qmap_full_criterion_not_met"] is True


def test_manuscript_route_has_title():
    r = _load("manuscript_route.json")
    assert "Predictive signal is not sufficient" in r["recommended_title"]


def test_outputs_present():
    for p in ("claim_prior_art_matrix.tsv", "contribution_evidence_map.tsv",
              "manuscript_route.json", "claim_tier.json"):
        assert os.path.exists(os.path.join(RUN_ROOT, "novelty", "n1", p))