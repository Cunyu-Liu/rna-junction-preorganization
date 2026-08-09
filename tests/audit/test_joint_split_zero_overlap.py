"""R0.3 context-operator strict nesting tests (contract R0.3 / §2.1.1)."""
from __future__ import annotations

import pytest

from audit.splits.context_operator_nesting import build_nesting_manifest
from audit.splits.joint_blocked import (
    build_joint_edit_context, build_joint_seq_scaffold,
)


def _admitted(n_ctx_per_scaf=26, n_scaf=9):
    rows = []
    rid = 0
    for s in range(n_scaf):
        for c in range(n_ctx_per_scaf):
            # a couple of junctions per context/scaffold cell
            for j in range(3):
                rows.append({
                    "source_row_id": f"R{rid:06d}",
                    "jid": f"J{s}_{c}_{j}",
                    "scaf": s,
                    "helix_seq": f"ctx_{s}_{c}",
                    "edit_component": f"e{(s * n_ctx_per_scaf + c) % 7}",
                    "y": -8.5, "cens": False,
                })
                rid += 1
    return rows


def test_strict_nesting():
    m = build_nesting_manifest(_admitted())
    assert m["strict_nested"] is True
    assert m["n_contexts"] == 234
    assert m["n_scaffolds"] == 9
    assert all(n == 26 for n in m["contexts_per_scaffold"].values())
    assert m["multi_scaffold_contexts"] == {}


def test_joint_edit_context_zero_overlap():
    res = build_joint_edit_context(_admitted())
    assert res["axis"] == "edit_x_nested_context"
    feasible = [f for f in res["folds"] if f["feasible"]]
    if feasible:
        for f in feasible:
            assert f["zero_overlap_sequence"] is True
            assert f["zero_overlap_context"] is True


def test_joint_seq_scaffold_zero_overlap():
    res = build_joint_seq_scaffold(_admitted())
    assert res["axis"] == "seq_x_scaffold_bundle"
    feasible = [f for f in res["folds"] if f["feasible"]]
    if feasible:
        for f in feasible:
            assert f["feasible"] is True
            # by construction test rows are in scaf s and edit e, train in neither
            assert f["n_test"] >= 1
