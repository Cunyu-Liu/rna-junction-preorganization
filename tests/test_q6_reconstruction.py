#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent tests for Q6 source-authoritative reconstruction."""

import json
import os
import sys

RUN_ROOT = "/mnt/cunyuliu/v1_4_boundary_audit_20260804T150707Z"
Q6_DIR = f"{RUN_ROOT}/qmap/q6"


def load_registry():
    rows = []
    with open(f"{Q6_DIR}/q6_source_registry.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_membership():
    with open(f"{Q6_DIR}/q6_membership.json") as f:
        return json.load(f)


def test_denominator_99_to_98():
    rows = load_registry()
    assert len(rows) == 99, f"expected 99 source rows, got {len(rows)}"
    excluded = [r for r in rows if r["source_category"] == "excluded_99_to_98"]
    assert len(excluded) == 1, f"expected 1 excluded, got {len(excluded)}"
    assert excluded[0]["canonical_id"] == "GCUAAG_UACGG", excluded[0]["canonical_id"]
    in_s1 = [r for r in rows if r["in_s1_98"]]
    assert len(in_s1) == 98, f"expected 98 in S1, got {len(in_s1)}"


def test_partition_counts():
    m = load_membership()
    cnt = m["counts"]
    assert cnt["fitted"] == 84, cnt
    assert cnt["beyond_40mM"] == 11, cnt
    assert cnt["closing_pair_abnormal"] == 2, cnt
    assert cnt["alternate_structure"] == 1, cnt
    total = cnt["fitted"] + cnt["beyond_40mM"] + cnt["closing_pair_abnormal"] + cnt["alternate_structure"]
    assert total == 98, f"partition must sum to 98, got {total}"


def test_no_overlap_and_disjoint():
    m = load_membership()
    sets = {
        "fitted": set(m["fitted"]),
        "beyond_40mM": set(m["beyond_40mM"]),
        "closing_pair_abnormal": set(m["closing_pair_abnormal"]),
        "alternate_structure": set(m["alternate_structure"]),
    }
    names = set()
    for k, s in sets.items():
        assert not (names & s), f"overlap between {k} and previous sets"
        names |= s
    assert len(names) == 98, f"expected 98 distinct members, got {len(names)}"


def test_named_structural_variants():
    m = load_membership()
    assert set(m["closing_pair_abnormal"]) == {"UCUAAA_CAUGA", "CCUACA_UACGG"}
    assert set(m["alternate_structure"]) == {"CUUAAC_UAUGG"}


def test_explicit_beyond_contained():
    m = load_membership()
    explicit = set(m["beyond_40mM_explicit_source_evidence"])
    beyond = set(m["beyond_40mM"])
    assert explicit <= beyond, "explicit beyond rows must be subset of beyond_40mM"
    assert len(explicit) == 10, f"expected 10 explicit, got {len(explicit)}"
    assert len(beyond - explicit) == 1, f"expected 1 fit-identified, got {len(beyond - explicit)}"


def test_registry_matches_membership():
    rows = load_registry()
    m = load_membership()
    by_cat = {
        "fitted": [r["canonical_id"] for r in rows if r["source_category"] == "fitted"],
        "beyond_40mM": [r["canonical_id"] for r in rows if r["source_category"] == "beyond_40mM"],
        "closing_pair_abnormal": [r["canonical_id"] for r in rows if r["source_category"] == "closing_pair_abnormal"],
        "alternate_structure": [r["canonical_id"] for r in rows if r["source_category"] == "alternate_structure"],
    }
    assert len(by_cat["fitted"]) == 84, len(by_cat["fitted"])
    assert set(by_cat["fitted"]) == set(m["fitted"])
    assert set(by_cat["beyond_40mM"]) == set(m["beyond_40mM"])
    assert set(by_cat["closing_pair_abnormal"]) == set(m["closing_pair_abnormal"])
    assert set(by_cat["alternate_structure"]) == set(m["alternate_structure"])


def test_decision_state():
    with open(f"{Q6_DIR}/Q6_decision.json") as f:
        d = json.load(f)
    assert d["gate"] == "Q6"
    assert d["state"] == "QMAP_SOURCE_RECONSTRUCTED"


def run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        t()
        passed += 1
        print(f"PASS {t.__name__}")
    print(f"\n{passed} Q6 tests passed")


if __name__ == "__main__":
    run_all()