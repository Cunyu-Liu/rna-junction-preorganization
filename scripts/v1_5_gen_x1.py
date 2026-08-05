#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""X1 — genuine independent recomputation + independent review (v1.5 §18).

X1 has two non-interchangeable components:
  18.1 independent recomputation by an UNINVOLVED executor;
  18.2 independent review by an UNINVOLVED domain/statistical reviewer.

This execution chain is the same implementer that built the artifacts under
review. Per §18 and §7 rules ('X1 不可由同一实现者自审'), this chain MUST NOT
perform the independent recomputation, must NOT write reviewer conclusions, and
must NOT set any PASS state itself.

This script therefore:
  1. freezes the recompute candidate commit and recompute scope/tolerances
     (so an independent executor can run X1 without re-deriving the spec);
  2. records the environment lock available at the frozen commit;
  3. records reviewer independence as NOT_AVAILABLE (no fabricated reviewer);
  4. fails closed to X1_AWAITING_INDEPENDENT_REVIEW.

Only a genuinely uninvolved executor/reviewer may later flip the state to
X1_RECOMPUTATION_AND_REVIEW_PASS / FAIL / MAJOR_REVISION / BLOCKED.
"""

from __future__ import annotations
import hashlib
import json
import os
import platform
import subprocess
import sys

from datetime import datetime, timezone

RUN_ROOT = "/mnt/cunyuliu/v1_5_manuscript_readiness_20260805T052052Z"
X1_DIR = f"{RUN_ROOT}/reproducibility/x1"
WORKTREE = "/home/cunyuliu/v1_5_manuscript_readiness_20260805T052052Z"


def _utcnow():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(rel):
    with open(os.path.join(RUN_ROOT, rel)) as f:
        return json.load(f)


def main():
    os.makedirs(f"{X1_DIR}/recomputed_results", exist_ok=True)
    now = _utcnow()

    # ---- 0. frozen candidate commit (the commit an independent executor checks out) ----
    try:
        head = subprocess.check_output(
            ["git", "-C", WORKTREE, "rev-parse", "HEAD"]).decode().strip()
        branch = subprocess.check_output(
            ["git", "-C", WORKTREE, "rev-parse", "--abbrev-ref", "HEAD"]).decode().strip()
    except Exception as e:  # pragma: no cover
        head, branch = "UNKNOWN", "UNKNOWN"
        print(f"[warn] could not read git HEAD: {e}")

    # ---- 1. recompute spec: scope, frozen inputs, tolerances (pre-registered) ----
    recompute_spec = {
        "schema_version": "X1-recompute-spec-v1.5",
        "gate": "X1",
        "run_id": "v1_5_manuscript_readiness_20260805T052052Z",
        "parent_run_id": "v1_4_boundary_audit_20260804T150707Z",
        "prepared_by": "current_execution_chain (spec only; executor must be independent)",
        "prepared_time_utc": now,
        "candidate_commit": head,
        "branch": branch,
        "checkout_instruction": (
            f"git worktree add --detach /tmp/x1_recompute {head} ; "
            f"create a fresh conda env per environment_lock.json; do NOT read "
            "expected results under {RUN_ROOT} as answers."
        ),
        "recompute_targets": [
            {
                "id": "T6_LOCKED_METRICS",
                "source": "/mnt/cunyuliu/v1_4_boundary_audit_20260804T150707Z/tecto/t6/T6_decision.json",
                "recompute_from": "fixed raw inputs + registries in parent run",
                "fields_to_compare": ["n_rows", "n_measured", "n_censored",
                                      "model_score", "motif_mean_score",
                                      "relative_gain", "bootstrap_CI",
                                      "positive_fraction", "final_state"],
            },
            {
                "id": "Q8_SIX_SUBSTATES",
                "source": "qmap/q8/Q8_decision.json",
                "recompute_from": "qmap/q8/membership_sensitivity.tsv + input registries",
                "fields_to_compare": ["micro_gain", "bootstrap_ci_95",
                                      "permutation_finite_p", "observed_coverage",
                                      "observed_width", "sub_states"],
            },
            {
                "id": "B3_PRIMARY_METRICS",
                "source": "benchmark/b3/B3_decision.json",
                "recompute_from": "B3 DGP spec + frozen seed list (benchmark/b3/)",
                "fields_to_compare": ["state", "aggregate.sensitivity",
                                      "aggregate.specificity",
                                      "aggregate.false_pass_rate",
                                      "aggregate.false_fail_rate",
                                      "aggregate.per_regime"],
            },
            {
                "id": "X0_PRIMARY_RESULT",
                "source": "external_case/x0/X0_decision.json",
                "recompute_from": "PRIME public record + qualification spec (external_case/x0/)",
                "fields_to_compare": ["platform_independent", "primary_candidate",
                                      "eligibility_verdicts", "state"],
            },
            {
                "id": "MAIN_FIGURE_SOURCE_DATA",
                "source": "figures/f0/source_data/",
                "recompute_from": "frozen decision JSONs + B3 results",
                "fields_to_compare": ["row_counts", "numeric columns",
                                      "membership sets", "component sizes"],
            },
        ],
        "pre_registered_tolerances": {
            "numeric_abs": 1e-6,
            "numeric_rel": 1e-4,
            "status_strings": "exact",
            "sets_membership": "exact",
            "component_sizes": "exact",
            "row_counts": "exact",
        },
        "fail_closed_rule": (
            "Fieldwise mismatch beyond tolerance => X1_RECOMPUTATION_FAIL. "
            "Verifying existing payload hashes is NOT a recomputation PASS."
        ),
    }
    with open(f"{X1_DIR}/recompute_spec.json", "w") as f:
        json.dump(recompute_spec, f, indent=2)

    # ---- 2. environment lock (captured at the frozen commit) ----
    py = sys.version.split("\n")[0]
    env_lock = {
        "schema_version": "X1-environment-lock-v1.5",
        "gate": "X1",
        "candidate_commit": head,
        "captured_time_utc": now,
        "python": py,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "conda_env": "rna_junction_preorganization_v1_1",
        "note": "Independent executor must rebuild from this lock; a fresh env is required.",
    }
    with open(f"{X1_DIR}/environment_lock.json", "w") as f:
        json.dump(env_lock, f, indent=2)

    # ---- 3. reviewer identity & independence (fail-closed: NOT_AVAILABLE) ----
    reviewer_identity = {
        "schema_version": "X1-reviewer-identity-v1.5",
        "gate": "X1",
        "independent_recomputation_executor": {
            "status": "NOT_AVAILABLE",
            "reason": (
                "No uninvolved executor is currently available. The current "
                "execution chain built the artifacts under review and therefore "
                "cannot serve as the independent executor."
            ),
        },
        "independent_reviewer": {
            "status": "NOT_AVAILABLE",
            "reason": (
                "No uninvolved domain/statistical reviewer is currently available. "
                "Per §18, reviewer conclusions must be authored by the reviewer, "
                "not by this chain."
            ),
        },
        "fabrication_guard": "This chain never writers reviewer conclusions or sets PASS.",
        "recorded_time_utc": now,
    }
    with open(f"{X1_DIR}/reviewer_identity_and_independence.json", "w") as f:
        json.dump(reviewer_identity, f, indent=2)

    # ---- 4. decision: fail closed to AWAITING (this is the compliant terminal) ----
    decision = {
        "schema_version": "X1-decision-v1.5",
        "gate": "X1",
        "run_id": "v1_5_manuscript_readiness_20260805T052052Z",
        "parent_run_id": "v1_4_boundary_audit_20260804T150707Z",
        "decision_time_utc": now,
        "candidate_commit": head,
        "state": "X1_AWAITING_INDEPENDENT_REVIEW",
        "independent_recomputation_status": "NOT_PERFORMED",
        "independent_review_status": "NOT_PERFORMED",
        "reason": (
            "X1 requires a genuinely uninvolved executor (independent recomputation) "
            "and an uninvolved domain/statistical reviewer (independent review). "
            "Neither is currently available. This is a compliant terminal state and "
            "is NOT bypassed by this chain. The recompute spec, environment lock and "
            "constant/hard frozen tolerances are prepared so an independent party can "
            "complete X1 without re-deriving the protocol."
        ),
        "resume_condition": (
            "Assign an uninvolved executor to run recompute_spec.json from commit "
            f"{head} in a fresh environment/output root, and an uninvolved reviewer "
            "to author reviewer_comments.tsv. Only then may state be set to "
            "X1_RECOMPUTATION_AND_REVIEW_PASS / FAIL / MAJOR_REVISION / BLOCKED."
        ),
        "outputs": {
            "recompute_spec": "reproducibility/x1/recompute_spec.json",
            "environment_lock": "reproducibility/x1/environment_lock.json",
            "reviewer_identity": "reproducibility/x1/reviewer_identity_and_independence.json",
            "decision": "reproducibility/x1/X1_decision.json",
            "report": "reports/X1_report.md",
        },
    }
    with open(f"{X1_DIR}/X1_decision.json", "w") as f:
        json.dump(decision, f, indent=2)

    # ---- 5. report ----
    report = [
        "# X1 — Independent Recomputation & Review (v1.5 §18)",
        "",
        f"**State:** X1_AWAITING_INDEPENDENT_REVIEW  ({now})",
        "",
        "This gate requires two non-interchangeable components, both requiring a "
        "genuinely uninvolved party:",
        "",
        "1. **Independent recomputation** — an uninvolved executor recomputes key "
        "   results from frozen inputs in a fresh environment/output root, without "
        "   reading expected results as answers.",
        "2. **Independent review** — an uninvolved domain/statistical reviewer "
        "   authors conclusions on the scientific question, qMaP interpretation, "
        "   B3 generality, X0 independence, statistics/calibration, novelty, claim "
        "   boundaries, publishability/venue, and reproducibility/release.",
        "",
        "The current execution chain is the same implementer that built the "
        "artifacts under review and therefore cannot serve as executor or reviewer. "
        "Per §7/§18, this chain must not fabricate PASS and must stop at "
        "`X1_AWAITING_INDEPENDENT_REVIEW`.",
        "",
        "## What was prepared for the independent party",
        "",
        f"- **Candidate commit:** `{head}` (`{branch}`) — frozen at preparation time.",
        "- **recompute_spec.json** — recompute targets (T6 locked metrics, Q8 six "
        "  substates, B3 primary metrics, X0 primary result, main figure source data) "
        "  with pre-registered tolerances (numeric_abs=1e-6, numeric_rel=1e-4, "
        "  exact status/sets/sizes/row-counts).",
        "- **environment_lock.json** — Python/platform/conda environment at the "
        "  frozen commit.",
        "- **reviewer_identity_and_independence.json** — records BOTH independence "
        "  slots as NOT_AVAILABLE (no fabricated reviewer).",
        "",
        "## Resume condition",
        "",
        f"Assign an uninvolved executor and reviewer; then complete X1 from commit "
        f"`{head}`. State may then become X1_RECOMPUTATION_AND_REVIEW_PASS / "
        "X1_RECOMPUTATION_FAIL / X1_REVIEW_MAJOR_REVISION_REQUIRED / "
        "X1_BLOCKED_ENVIRONMENT_OR_LICENSE.",
        "",
    ]
    with open(f"{RUN_ROOT}/reports/X1_report.md", "w") as f:
        f.write("\n".join(report) + "\n")

    # ---- 6. sentinel (fail-closed) ----
    with open(f"{RUN_ROOT}/sentinels/X1_AWAITING_INDEPENDENT_REVIEW.sentinel", "w") as f:
        f.write(
            "gate=X1\n"
            f"state={decision['state']}\n"
            f"independent_recomputation_status=NOT_PERFORMED\n"
            f"independent_review_status=NOT_PERFORMED\n"
            f"candidate_commit={head}\n"
            f"decision_time_utc={now}\n"
        )

    print("X1 prepared and failed closed to X1_AWAITING_INDEPENDENT_REVIEW.")
    print(f"candidate_commit={head}")
    return 0


if __name__ == "__main__":
    sys.exit(main())