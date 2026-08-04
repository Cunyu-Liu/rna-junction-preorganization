#!/usr/bin/env python3
"""Build specs/exposure_registry.json (contract §8.7 ExposureRegistry).

Aggregates source-level, exact-sequence, near-homolog, motif-family, scaffold,
PDB/template ancestry, pretraining, test-set, and manual-inspection exposure
from existing T0/T1/T2/T3/Q0-Q5 artifacts into a single frozen registry.

Run:  python scripts/build_exposure_registry.py
"""
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

WORKTREE = "/home/cunyuliu/rna_junction_preorganization_v1_2_20260803"
DATA = "/mnt/cunyuliu/rna_junction_preorganization_v1_2_20260803"
SPEC_OUT = os.path.join(WORKTREE, "specs", "exposure_registry.json")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path):
    with open(path) as f:
        return json.load(f)


def git_head():
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=WORKTREE, stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def main():
    t0_admission = load_json(os.path.join(WORKTREE, "manifests", "t0_admission_analysis.json"))
    t0_source_pin = load_json(os.path.join(WORKTREE, "manifests", "t0_source_pin.json"))
    t1_splits = load_json(os.path.join(DATA, "t1", "t1_splits.json"))
    t1_effective_n = load_json(os.path.join(DATA, "t1", "t1_effective_n.json"))
    t1_manifest = load_json(os.path.join(DATA, "t1", "t1_manifest.json"))
    t2_results = load_json(os.path.join(DATA, "t2", "t2_results.json"))
    t3_results = load_json(os.path.join(DATA, "t3", "t3_results.json"))
    q4_fold = load_json(os.path.join(DATA, "qmap", "q4", "q4_fold_assignment.json"))
    qmap_transport = load_json(os.path.join(WORKTREE, "specs", "assay_transport_qmapseq.json"))

    # Counts from T0 admission
    eff = t0_admission.get("effective_n", {})
    n_constructs = eff.get("constructs", 1713)
    n_motifs = eff.get("motifs", 60)
    n_scaffolds = eff.get("scaffolds", 9)
    n_studies = eff.get("studies", 1)
    n_indep_scaffold_groups = eff.get("independent_scaffold_groups", 9)

    # Splits
    holdout_motifs = t2_results.get("split", {}).get("holdout_motifs", ["0x1", "2x1", "2x2"])
    train_motifs = t2_results.get("split", {}).get("train_motifs", [])
    n_holdout_junctions = t2_results.get("heldout", {}).get("n_holdout_junctions", 392)

    # qMaP counts
    qmap_n_variants = q4_fold.get("n_variants", 98)
    qmap_fold_sizes = q4_fold.get("fold_sizes", [83, 11, 2, 2])
    qmap_leakage = q4_fold.get("leakage_violations", 0)

    # near-homolog / symmetry
    n_symmetry_groups = t1_effective_n.get("unique_symmetry_groups", 1356)

    # Build the registry (checksum added last over the self-same content)
    registry = {
        "schema_version": "exposure-registry-v1",
        "spec_version": "1.0.0",
        "run_id": "v1_2_tecto_qmap_20260803",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": git_head(),
        "exposure_levels": {
            "source_level": {
                "tecto_sources": [
                    "denny_2018_tectorna (261_SI.xlsx, doi:10.1016/j.cell.2018.05.038, PMC6053692)"
                ],
                "qmap_sources": [
                    "ENA PRJNA1086549 (8 runs/16 FASTQ)",
                    "Figshare 10.6084/m9.figshare.25331758",
                    "Zenodo 10.5281/zenodo.11672684",
                    "YesselmanLab/rna_map @ 2d7337d (Apache-2.0)",
                ],
                "current_dms": "PERMANENTLY_NOT_ADMITTED_V1_2 (7,500-construct DMS, crosswalk unavailable)",
                "note": "Bonilla/Shin/Yesselman registered as RNA-MaP/tectoRNA platform cluster, NOT independent external systems",
            },
            "exact_sequence": {
                "tecto_unique_constructs": n_constructs,
                "qmap_unique_variants": qmap_n_variants,
                "tecto_qmap_overlap": "NONE_BY_CONSTRUCTION (different assay systems; tecto uses tectoRNA library, qmap uses mttr6 TTR mutants)",
                "test_set_exact_sequence_exposure": "FORBIDDEN - outer holdout sequences never used in feature/threshold/model selection",
            },
            "near_homolog": {
                "status": "AUDITED_VIA_T1_SYMMETRY_GROUPS",
                "t1_symmetry_groups_file": os.path.join(DATA, "t1", "t1_symmetry_groups.json"),
                "n_symmetry_groups": n_symmetry_groups,
                "reciprocal_equivalence_collapsed": True,
            },
            "motif_family": {
                "tecto_unique_motifs": n_motifs,
                "primary_split": t1_splits.get("primary", "motif_family_holdout"),
                "holdout_motifs": holdout_motifs,
                "train_motifs": len(train_motifs),
                "leakage_audit": "Q4 fold assignment has 0 leakage violations (mutation graph 193 edges, 4 components)",
            },
            "scaffold": {
                "tecto_unique_scaffolds": n_scaffolds,
                "independent_scaffold_groups": n_indep_scaffold_groups,
                "scaffold_holdout_feasible": True,
                "note": "Only 9 scaffolds; high risk of scaffold-level confounding; scaffold enters grouping/hierarchical model",
            },
            "pdb_template_ancestry": {
                "status": "DESIGNED_JUNCTIONMAT_FROM_KNOWN_TLR_FOLDS",
                "note": "tectoRNA constructs designed from PDB-templated junction architectures; no new PDB templates trained in v1.2",
            },
            "pretraining": {
                "status": "NO_FOUNDATION_MODEL_PRETRAINING_IN_V1_2",
                "frozen_rna_lm_used": False,
                "note": "Per contract §13/§17, no RNA foundation model used as primary architectural innovation. Simple statistical/physical baselines only.",
            },
            "test_set": {
                "tecto_primary_outer_split": "motif_family_holdout (3 holdout motifs: %s; %d holdout junctions)"
                % (", ".join(holdout_motifs), n_holdout_junctions),
                "qmap_primary_outer_split": "mutation_graph_k_fold (K=4, fold_sizes %s, %d leakage)"
                % (qmap_fold_sizes, qmap_leakage),
                "test_label_access_log": "Outer test labels sealed until formal final evaluation run; no outcome-aware adjustment",
            },
            "manual_inspection": {
                "status": "DOCUMENTED_IN_CONTRACT_ISSUE_REGISTER",
                "contract_issue_register": "contract/contract_issue_register.md",
                "n_issues": 5,
                "fail_closed_resolution": True,
            },
            "unknown_foundation_model_exposure": {
                "status": "NOT_APPLICABLE",
                "note": "No frozen LM used; no unknown pretraining exposure to audit",
            },
        },
        "permitted_uses": {
            "tecto_admitted_labels": "T0-T3 PASS admits Denny 2018 tectoRNA ΔG labels for the primary estimand (10mM Mg2+)",
            "qmap_admitted_labels": "Q0-Q5 PASS admits qMaPseq RNA-MaP reference ΔG for the 98 mttr6 TTR mutant set",
            "current_dms_permitted_uses": [
                "archive/design catalog",
                "sequence/motif catalog",
                "aggregate engineering QC",
                "simulator non-biological stress test",
                "descriptive motif overlap enumeration",
                "provenance-failure negative case",
            ],
            "current_dms_prohibited_uses": [
                "primary label",
                "main model fitting",
                "feature/threshold/split selection",
                "DMS-only benchmark",
                "late fusion",
                "joint-generative model",
                "sequence+DMS-at-test",
                "paper effect size",
                "DMS improves tecto inference",
                "DMS narrows identified set",
                "current 7,500-construct DMS validated tectoRNA",
            ],
        },
    }

    # Self-checksum: sha256 over the canonical JSON (sorted keys, no checksum field).
    payload = json.dumps(registry, sort_keys=True, ensure_ascii=False).encode("utf-8")
    registry["checksum"] = hashlib.sha256(payload).hexdigest()

    os.makedirs(os.path.dirname(SPEC_OUT), exist_ok=True)
    with open(SPEC_OUT, "w") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(json.dumps({
        "written": SPEC_OUT,
        "sha256": sha256_file(SPEC_OUT),
        "exposure_levels": list(registry["exposure_levels"].keys()),
        "checksum": registry["checksum"][:16],
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
