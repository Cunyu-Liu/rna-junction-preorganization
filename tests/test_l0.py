#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent tests for L0 prior-art + venue freeze (v1.5)."""

import json
import os

RUN_ROOT = "/mnt/cunyuliu/v1_5_manuscript_readiness_20260805T052052Z"
L0_DIR = f"{RUN_ROOT}/literature/l0"


def load_json(path):
    with open(path) as f:
        return json.load(f)


def test_decision_present():
    d = load_json(f"{L0_DIR}/L0_decision.json")
    assert d["gate"] == "L0"
    assert d["state"] == "L0_PRIOR_ART_AND_VENUE_FROZEN"


def test_prior_art_matrix_covers_required():
    """§13.1 requires qMaPseq/PRIME, DataSAIL, DIONYSUS, benchmark guidelines, OSIRIS."""
    with open(f"{L0_DIR}/prior_art_matrix.tsv") as f:
        header = f.readline().strip().split("\t")
        rows = [dict(zip(header, line.strip().split("\t"))) for line in f if line.strip()]
    ids = {r["prior_art_id"] for r in rows}
    assert {"qMaP2024", "PRIME2026", "DataSAIL", "DIONYSUS", "benchmark_guidelines", "OSIRIS"} <= ids
    assert len(rows) >= 6


def test_no_retracted_gain_claim():
    """V15-01: must NOT restate 'gain below 0.3' as the qMaP result."""
    d = load_json(f"{L0_DIR}/L0_decision.json")
    assert "RETRACTED_STALE_CLAIM" in d["retracted_claim_handling"]
    assert "gain MET" in d["retracted_claim_handling"]
    assert "coverage rule FAILED" in d["retracted_claim_handling"]
    # scan the report for forbidden phrasing
    rep = open(f"{L0_DIR}/l0_report.md").read()
    assert "below the predeclared meaningful threshold" not in rep


def test_venue_freeze_routes():
    v = load_json(f"{L0_DIR}/venue_freeze.json")
    venues = {r["venue"] for r in v["routes"]}
    assert "Bioinformatics Advances" in venues
    assert "PLOS Computational Biology" in venues
    assert "NAR Genomics and Bioinformatics" in venues
    assert "Application Note / Resource" in venues
    assert "NAR main journal (mechanism paper)" in venues
    # mechanism paper must be closed for v1.5
    nar_main = [r for r in v["routes"] if r["venue"].startswith("NAR main")][0]
    assert "CLOSED" in nar_main["judgement"]


def test_decision_links_outputs():
    d = load_json(f"{L0_DIR}/L0_decision.json")
    for key in ("prior_art_matrix", "venue_freeze"):
        rel = d["outputs"][key]
        abs_path = f"{RUN_ROOT}/{rel}"
        assert os.path.exists(abs_path), f"missing {abs_path}"


def test_prior_art_has_differential_not_just_citation():
    with open(f"{L0_DIR}/prior_art_matrix.tsv") as f:
        header = f.readline().strip().split("\t")
        rows = [dict(zip(header, line.strip().split("\t"))) for line in f if line.strip()]
    for r in rows:
        assert len(r["differential"]) > 20, f"{r['prior_art_id']}: differential too short"
        assert len(r["project_does_not_claim"]) > 0


def run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        t()
        passed += 1
        print(f"PASS {t.__name__}")
    print(f"\n{passed} L0 tests passed")


if __name__ == "__main__":
    run_all()