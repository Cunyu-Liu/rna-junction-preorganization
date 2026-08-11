"""P0.1 authority freeze + invalidation ledger (strict audit 2026-08-11).

The strict audit found the frozen authority tree has stale bindings:
- AGENTS.md:3 still points to a missing ``contract/1.1.docx``.
- authority/CanonicalStateManifest_v2.json binds start commit 5f28320, not the
  frozen HEAD b80d546.
- authority/RunDAG_v2.json is an abstract graph without R1-R6 run IDs.
- The joint/edit-context, R4 power, R5 manuscript and R6 release seal must be
  tombstoned as INVALIDATED_OR_STALE / BLOCKED_WITH_EVIDENCE.

This module writes, into the NEW repair run root:
  AuthorityMigration.json   - the sole active authority for the repair lineage
  InvalidationLedger.jsonl  - one immutable verdict per stale claim/artifact
  ModelUniverse.json        - the model universe (13 configs, honest naming)
  RunManifest.schema.json   - machine-readable run manifest schema

It is read-only with respect to the OLD run root: it never edits or rewrites
historical artifacts; it only records their invalidation verdict.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

# Fixed frozen identities from the strict audit header.
FROZEN_COMMIT = "b80d54604a6b666c48625b313b5b1537e8e87522"
FROZEN_BRANCH = "r0_comparison_eligibility"
STRICT_CONTRACT_SHA = "06dd81ccd610f91aab0d07c7980c4e898791bfa9f5c2bf0e12b60c9db3b82496"
CANONICAL_SOURCE_SHA = "0989ddc00bb230fdb00bbc65433c943a0419e35c3d0799b481e741c4a24defe2"
OLD_RUN_ROOT = "/mnt/cunyuliu/rna_junction_r0_20260809T105504Z"

# Invalidation ledger (ordered, immutable).  Each entry gives one verdict.
INVALIDATIONS = [
    {"claim_id": "joint_edit_x_nested_context_gain",
     "artifact": "r05_v131/Predictions_v1_31.jsonl",
     "verdict": "INVALIDATED_OR_STALE",
     "reason": ("corrected v1.31 computed joint train_ids but discarded them; "
                "full model trained on test contexts while no-sequence/baselines "
                "blocked them, so the matched contrast is not co-trained.")},
    {"claim_id": "r4_power",
     "artifact": "r4/PowerAnalysis.json",
     "verdict": "INVALIDATED_OR_STALE",
     "reason": ("treats 1336 junctions as independent while the joint blocking "
                "unit is 37 highly imbalanced edit components; input joint "
                "predictions themselves invalid; reported observed power.")},
    {"claim_id": "r4_noise_ceiling",
     "artifact": "r4/NoiseCeiling.json",
     "verdict": "INVALIDATED_NAME_DESCRIPTIVE_ONLY",
     "reason": ("cross-operator spread and in-sample junction-mean oracle cannot "
                "separate pure measurement noise; renamed to "
                "ObservedOperatorContextSpread + InSampleJunctionOracle.")},
    {"claim_id": "r5_manuscript_negative",
     "artifact": "r5/",
     "verdict": "BLOCKED_WITH_EVIDENCE",
     "reason": ("aggregation, joint split, null, power, baseline equivalence and "
                "causal narrative all carry conclusion-level errors.")},
    {"claim_id": "r6_release_seal",
     "artifact": "r6/ReleaseManifest.json",
     "verdict": "RELEASE_SEAL_INVALID",
     "reason": ("7/13 checksums FAILED; Manifest/REPRODUCE commit conflict; "
                "current-lineage fresh replay NOT_RUN; legal not closed.")},
    {"claim_id": "support_aware_mixture",
     "artifact": "audit/models/support_aware_mixture.py",
     "verdict": "REJECTED",
     "reason": ("isomorphic to edit-KNN + distance abstention; supported-row "
                "predictor adds no capability; censored rows treated as exact "
                "values; permanently retired.")},
    {"claim_id": "single_axis_conditional",
     "artifact": "r2/CoreHypothesisDecision_v3.json",
     "verdict": "RETAINED_AS_CONDITIONAL_DIAGNOSTIC",
     "reason": ("three single-axis matched point estimates are small positive "
                "but <10% and context two-way CI crosses 0; kept only as "
                "conditional diagnostics, not transferable mechanism.")},
    {"claim_id": "core_hypothesis",
     "artifact": "CoreHypothesisDecision",
     "verdict": "UNKNOWN_NOT_ASSERTED",
     "reason": ("transferable sequence increment not yet properly tested; "
                "blocked pending true-joint rerun.")},
    {"claim_id": "sota",
     "artifact": "leaderboard",
     "verdict": "SOTA_NOT_ADJUDICATED",
     "reason": ("no same-protocol public benchmark and no qualified frozen "
                "leaderboard.")},
    {"claim_id": "submission",
     "artifact": "release",
     "verdict": "NO_SUBMISSION_AUTHORIZATION",
     "reason": ("science, reproducibility, release and legal gates not closed.")},
]

MODEL_UNIVERSE = [
    {"model_id": "global_censor_intercept", "class": "intercept_only", "proxy": False},
    {"model_id": "train_only_scaffold", "class": "scaffold_calibration", "proxy": False},
    {"model_id": "scaffold_context_hierarchy", "class": "nested_calibration", "proxy": False},
    {"model_id": "motif_topology_hierarchy", "class": "nested_calibration", "proxy": False},
    {"model_id": "onehot_kmer_ridge", "class": "sequence_linear", "proxy": False},
    {"model_id": "position_aware_additive", "class": "sequence_linear", "proxy": False},
    {"model_id": "edit_knn", "class": "edit_distance_knn", "proxy": False},
    {"model_id": "mutation_graph_smoother", "class": "mutation_graph", "proxy": False},
    {"model_id": "no_sequence_latent_operator", "class": "latent_operator_no_sequence", "proxy": False},
    {"model_id": "corrected_v1_31", "class": "latent_operator_sequence", "proxy": False},
    {"model_id": "denny_inspired_scalar_latent_proxy",
     "class": "denny_thermodynamic_fingerprint",
     "proxy": True,
     "honest_name": "denny_inspired_scalar_latent_proxy",
     "note": "63-D sequence->scalar latent placeholder, NOT a faithful Denny fingerprint"},
    {"model_id": "viennarna_secondary_ensemble_proxy",
     "class": "physical_secondary_ensemble",
     "proxy": True,
     "honest_name": "viennarna_secondary_ensemble_proxy",
     "note": "ViennaRNA MFE/partition/defect/GC/length/BPP + linear Tobit, NOT RNAMake tertiary"},
    {"model_id": "frozen_rnafm_global_head",
     "class": "frozen_lm_embedding",
     "proxy": True,
     "honest_name": "frozen_rnafm_global_head",
     "note": "real frozen 640-D embedding with a NON-matched global censored head"},
]

RUN_MANIFEST_SCHEMA = {
    "version": "v3",
    "required": [
        "run_id", "contract_sha", "code_commit", "canonical_source_sha",
        "cleaning", "splits", "metric", "gate", "config", "environment",
        "parent", "commands", "logs", "inputs", "outputs", "row_ids_hash",
    ],
    "description": (
        "Every run manifest binds contract/code/source/cleaning/split/metric/"
        "gate/config/environment/parent/commands/logs/inputs/outputs hashes and "
        "the actual train/test row-ID hash passed to fit.  A reviewer must be "
        "able to reconstruct the exact fit rows from the manifest."),
    "row_ids_hash": {
        "kind": "sha256",
        "source": "sorted train_ids + test_ids from the typed FoldSpec",
    },
}


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def write_p01(run_root: Path, *, old_run_root: Path = Path(OLD_RUN_ROOT),
              owner: str = "AUTHOR_REVIEW_PENDING",
              commit: str = FROZEN_COMMIT) -> None:
    out = run_root / "authority"
    out.mkdir(parents=True, exist_ok=True)

    migration = {
        "version": "v1",
        "created_utc": None,  # filled by caller/generator
        "supersedes": {
            "old_run_root": str(old_run_root),
            "old_frozen_commit": FROZEN_COMMIT,
            "old_branch": FROZEN_BRANCH,
            "strict_contract_sha": STRICT_CONTRACT_SHA,
            "canonical_source_sha": CANONICAL_SOURCE_SHA,
        },
        "active_authority": {
            "run_root": str(run_root.resolve()),
            "worktree": None,  # resolved at runtime
            "commit": commit,
            "owner_activation": owner,
            "superseded_rule": (
                "The old run root is read-only.  No historical artifact is "
                "rewritten 'as-if-known'.  New results are written only under "
                "the new run root with a fresh release id."),
        },
        "write_boundary": {
            "old_run_root": "READ_ONLY",
            "new_run_root": "WRITABLE",
            "forbidden": [
                "modify/rename/delete existing results/ subdirs in old root",
                "patch historical authority/eligibility manifests",
                "rewrite old checksums",
            ],
        },
        "verdict_scope": "P0_repair_lineage_only",
    }

    invalidation = [
        {"claim_id": e["claim_id"], "artifact": e["artifact"],
         "verdict": e["verdict"], "reason": e["reason"],
         "recorded_at": None, "immutable": True}
        for e in INVALIDATIONS
    ]

    model_universe = {
        "version": "v3",
        "configs": MODEL_UNIVERSE,
        "n_configs": len(MODEL_UNIVERSE),
        "n_independent_method_families": 13,
        "note": ("13 configurations are NOT 13 independent method families; "
                 "several are proxies or non-matched heads.  Honest naming "
                 "required before any comparison claim."),
    }

    (out / "AuthorityMigration.json").write_text(
        json.dumps(migration, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    with (out / "InvalidationLedger.jsonl").open("w") as fh:
        for e in invalidation:
            fh.write(json.dumps(e, sort_keys=True) + "\n")
    (out / "ModelUniverse.json").write_text(
        json.dumps(model_universe, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    (out / "RunManifest.schema.json").write_text(
        json.dumps(RUN_MANIFEST_SCHEMA, indent=2, ensure_ascii=False, sort_keys=True) + "\n")

    # Machine-readable overall status, fail-closed on missing owner activation.
    owner_ok = bool(owner and owner not in ("", "AUTHOR_REVIEW_PENDING"))
    status = {
        "phase": "P0.1", "state": "ACTIVE" if owner_ok else "BLOCKED_PENDING_OWNER",
        "n_invalidations": len(invalidation),
        "n_model_configs": len(MODEL_UNIVERSE),
        "authority_migration": str(out / "AuthorityMigration.json"),
        "invalidation_ledger": str(out / "InvalidationLedger.jsonl"),
        "model_universe": str(out / "ModelUniverse.json"),
        "run_manifest_schema": str(out / "RunManifest.schema.json"),
        "note": ("P0.1 freezes authority and tombstones stale states.  It does "
                 "not train or modify the old root.  Owner activation is "
                 "required before P0.5 rerun."),
    }
    (out / "STATUS.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    return status


if __name__ == "__main__":
    import sys
    import datetime
    run_root = Path(sys.argv[1])
    owner = sys.argv[2] if len(sys.argv) > 2 else "AUTHOR_REVIEW_PENDING"
    st = write_p01(run_root, owner=owner)
    # stamp created_utc after the fact (kept deterministic for tests)
    print(json.dumps(st, indent=2, ensure_ascii=False))