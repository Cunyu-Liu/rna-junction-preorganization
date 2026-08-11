"""P0.2 repair tests for the sorted audit (2026-08-11).

Covers the decisive joint-runner defect and the strict optimizer gate:

1. FoldSpec consumption -- corrected v1.31 must fit on the SAME blocked
   ``train_ids`` as no-sequence and every baseline.  The OLD buggy runner
   computed ``train_ids`` but trained full on "all rows not in test", which
   leaks the held-out nested contexts into the full model's train set.  These
   tests prove the old behavior is detectable and the new shared FoldSpec is
   zero-overlap in BOTH the edit and context dimensions.
2. Optimizer gate -- SciPy ``success=true`` is NOT sufficient for
   comparison eligibility; a large final projected/raw gradient must make the
   fold comparison-ineligible.
3. Adjudication -- a joint CI lower bound <= 0 (crossing zero) must never
   yield SUPPORTED_CONDITIONAL.
"""
from __future__ import annotations

import numpy as np
import pytest

from audit.repair.foldspec import FoldSpec, fold_to_spec
from audit.repair.fold_loader import build_joint_edit_context_folds
from audit.repair.optimizer_gate import (
    bounded_fit_gate, unbounded_fit_gate, gate_from_fit,
)
from audit.repair.p06_adjudicate import adjudicate


def _admitted():
    """Small synthetic admitted set reproducing the real leak structure: a
    nested context (helix_seq) is shared across two edit components, so the
    OLD buggy runner (train = all non-test rows) leaks that context into the
    full model's train set.

    Structure:
      scaf0: context C0 (all edit e0); context C1 (juncts 0,1 -> e0; junct 2 -> e1)
      scaf1: context D0 (all edit e1); context D1 (all edit e1)
    Holding out edit e0 => test contexts {C0, C1}.  The buggy train keeps e1's
    row in C1 (a blocked context), which the correct FoldSpec train removes.
    """
    rows = []
    rid = 0
    plan = [
        # (scaf, helix_seq, n_junct, edit_for_junct)
        (0, "C0", 3, lambda j: "e0"),
        (0, "C1", 3, lambda j: "e0" if j < 2 else "e1"),
        (1, "D0", 3, lambda j: "e1"),
        (1, "D1", 3, lambda j: "e1"),
    ]
    for scaf, ctx, n, editfn in plan:
        for j in range(n):
            rows.append({
                "source_row_id": f"R{rid:05d}", "jid": f"J{scaf}_{ctx}_{j}",
                "scaf": scaf, "helix_seq": ctx, "edit_component": editfn(j),
                "y": -8.5, "cens": False,
            })
            rid += 1
    return rows


def _buggy_train_ids(admitted, test_ids):
    """Reproduce the OLD r05_v131 bug: train = all rows not in test (ignores the
    computed joint blocking, so held-out contexts leak into train)."""
    return {str(r["source_row_id"]) for r in admitted
            if str(r["source_row_id"]) not in {str(x) for x in test_ids}}


def test_joint_foldspec_zero_overlap_both_dimensions():
    admitted = _admitted()
    specs = build_joint_edit_context_folds(admitted)
    assert len(specs) == 2
    for s in specs:
        s.validate()
        tr_edit = {r["edit_component"] for r in admitted
                   if r["source_row_id"] in s.train_ids}
        te_edit = {r["edit_component"] for r in admitted
                   if r["source_row_id"] in s.test_ids}
        tr_ctx = {r["helix_seq"] for r in admitted
                  if r["source_row_id"] in s.train_ids}
        te_ctx = {r["helix_seq"] for r in admitted
                  if r["source_row_id"] in s.test_ids}
        assert not (tr_edit & te_edit), f"edit leak on {s.fold}"
        assert not (tr_ctx & te_ctx), f"context leak on {s.fold}"


def test_joint_foldspec_shared_across_models():
    """The SAME FoldSpec (identical train/test ids + rows_hash) must be the input
    for full and no-sequence.  A single spec object is shared; there is no
    per-model divergence."""
    admitted = _admitted()
    specs = build_joint_edit_context_folds(admitted)
    for s in specs:
        # no-sequence and full consume the identical object; assert the contract
        # fields are stable and reproduce the same hash.
        s2 = fold_to_spec(s.axis, s.fold, set(s.test_ids), set(s.train_ids),
                          blocked_seq=s.blocked_sequence_groups,
                          blocked_ctx=s.blocked_context_groups)
        assert s.rows_hash() == s2.rows_hash()
        assert s.train_ids == s2.train_ids
        assert s.test_ids == s2.test_ids


def test_old_buggy_runner_leaks_context_and_is_detectable():
    """The old runner (train = all non-test rows) leaks held-out contexts into
    train.  This must be detectable: the buggy train set differs from the
    correct FoldSpec train set."""
    admitted = _admitted()
    specs = build_joint_edit_context_folds(admitted)
    leaked = False
    for s in specs:
        buggy = _buggy_train_ids(admitted, s.test_ids)
        # The buggy train includes rows sharing a held-out context.
        extra = buggy - s.train_ids
        if extra:
            # those extra rows must include a held-out context (a leak)
            leaked_ctx = {r["helix_seq"] for r in admitted
                          if r["source_row_id"] in extra}
            assert leaked_ctx & s.blocked_context_groups, \
                f"buggy train '{extra}' should contain a blocked context"
            leaked = True
    assert leaked, "fixture too small to expose context leak; enlarge it"


def test_optimizer_success_but_large_gradient_ineligible_bounded():
    """SciPy success=true with a large projected gradient is NOT eligible."""
    gate = bounded_fit_gate(
        np.zeros(2), np.array([50.0, -30.0]),
        [(-4.0, 4.0), (-4.0, 4.0)], success=True, proj_grad_tol=1e-3)
    assert gate["eligible"] is False
    assert gate["projected_grad_norm"] > 1e-3
    assert gate["success"] is True  # success alone must not imply eligibility


def test_optimizer_success_but_large_gradient_ineligible_unbounded():
    gate = unbounded_fit_gate(np.zeros(2), np.array([10.0, 0.0]),
                              success=True, grad_tol=1e-3)
    assert gate["eligible"] is False
    assert gate["success"] is True


def test_optimizer_gate_converged_eligible():
    gate = unbounded_fit_gate(np.zeros(2), np.array([1e-6, -1e-6]),
                              success=True, grad_tol=1e-3)
    assert gate["eligible"] is True


def test_gate_from_fit_full_gradient_uses_bounded_path():
    model = {"beta": np.zeros(2), "grad": np.array([1e-6, 1e-6]),
             "success": True}
    gate = gate_from_fit(model, bounds=[(-4.0, 4.0), (-4.0, 4.0)])
    assert gate["eligible"] is True
    assert "projected_grad_norm" in gate


def test_adjudication_ci_crossing_zero_never_supported():
    """A CI lower bound <= 0 (uncertainty crosses zero) must forbid SUPPORTED,
    even if the point gain is positive and below the 10% gate."""
    sci = adjudicate(
        joint_relative_gain=0.05, joint_ci_lower=-0.01, null_975_upper=0.02,
        genuine_theta=0.05, eligible_folds=5, positive_folds=5,
        eligibility_status="VALID", gate_10pct=False, fold_5of5=True)
    assert sci["scientific_verdict"] == "NOT_SUPPORTED_AT_PRE_REGISTERED_GATE"
    assert sci["scientific_verdict"] != "SUPPORTED_CONDITIONAL"


def test_adjudication_blocked_has_no_verdict():
    sci = adjudicate(joint_relative_gain=0.05, joint_ci_lower=0.01,
                     null_975_upper=0.02, genuine_theta=0.05,
                     eligible_folds=5, positive_folds=5,
                     eligibility_status="BLOCKED_WITH_EVIDENCE")
    assert sci["scientific_verdict"] is None
    assert sci["eligibility_status"] == "BLOCKED_WITH_EVIDENCE"


def test_foldspec_validate_rejects_overlap():
    with pytest.raises(ValueError):
        FoldSpec(axis="edit_x_nested_context", fold="e:0",
                 train_ids=frozenset({"A", "B"}), test_ids=frozenset({"B"}))\
            .validate()