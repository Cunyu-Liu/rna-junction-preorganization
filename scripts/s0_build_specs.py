#!/usr/bin/env python3
import runtime_config as rc
"""S0 — freeze estimand, operator, symmetry, assay-transport, primary-analysis,
and negative-control specifications. Registers frozen specs and their hashes.

Every spec is written as a versioned JSON file with a sha256, recorded in a
manifest. Thresholds that later gates must use are frozen BEFORE looking at any
held-out outcome, with external basis and simulation rationale recorded.
"""
import hashlib
import json
import os
import subprocess
import sys

WORKTREE = rc.WORKTREE
SPEC_DIR = os.path.join(WORKTREE, "specs")
GOVERNANCE = os.path.join(WORKTREE, "governance")
sys.path.insert(0, GOVERNANCE)
from canonical_manifest import CanonicalStateManifest, validate_schema  # noqa: E402

MANIFEST_PATH = rc.MANIFEST_PATH
CONTRACT_SHA256 = rc.CONTRACT_SHA256


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_spec(name, spec):
    os.makedirs(SPEC_DIR, exist_ok=True)
    path = os.path.join(SPEC_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2, ensure_ascii=False)
    return path, sha256_file(path)


def git(*args):
    cur = os.getcwd()
    os.chdir(WORKTREE)
    try:
        return subprocess.check_output(["git"] + list(args), stderr=subprocess.STDOUT).decode().strip()
    finally:
        os.chdir(cur)


def main():
    commit = git("rev-parse", "HEAD")
    branch = git("branch", "--show-current")
    now = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    specs = {}

    # ---- 1. EstimandSpec ----
    specs["estimand_spec.json"] = {
        "schema_version": "estimand-spec-v1",
        "spec_version": "1.0.0",
        "run_id": "v1_2_tecto_qmap_20260803",
        "frozen_at_utc": now,
        "code_commit": commit,
        "primary_estimand": "target-specific thermodynamic functional",
        "primary_target": "thermodynamic stability of forming a two-way RNA junction tertiary geometry "
                          "within a tectoRNA assembly, parameterized by the folding free energy (Delta G)",
        "condition": "in vitro, 37C, 10 mM Mg2+ (as in Denny et al. 2018 tectoRNA binding assay); "
                     "5 mM Mg2+ variant reported separately",
        "input_space": "junction sequence (two-way junction with unpaired residues), flanking base-pair "
                       "context, chip scaffold length/position, sublibrary",
        "output_functional": "Delta G(junction | scaffold, flank) in kcal/mol; the target-specific "
                             "thermodynamic functional is the conditional expected free energy of the "
                             "junction-inserted assembly relative to a reference",
        "units": "kcal/mol",
        "nuisance_variables": [
            "chip_scaffold identity (9 values)",
            "flanking sequence context",
            "helix length",
            "sublibrary / position",
            "measurement batch / cluster fluorescence",
        ],
        "identification_assumptions": [
            "conditional exchangeability given (junction, scaffold, flank) for the measured non-censored rows",
            "censoring at -7.1 kcal/mol is left-censoring (values at/more negative than floor are not "
            "point-measurable); censored rows enter a censored likelihood, not exact values",
            "no unmeasured confounder shifts Delta G differentially across within-scaffold comparisons",
        ],
        "point_identification_conditions": [
            "rows are non-censored and non-interpolated",
            "sufficient support across scaffold/flank for a given junction family",
            "no unresolved operator ambiguity for the measured value",
        ],
        "partial_identification_conditions": [
            "rows at the -7.1 cap (left-censored)",
            "rows with only interpolated values",
            "motif/scaffold cells with zero support",
            "any unresolved operator range overlap",
        ],
        "identified_set_interval": "where point identification fails, report an identified set/interval "
                                   "from the censored likelihood and operator-uncertainty propagation; "
                                   "never report a pseudo-exact point",
        "primary_uncertainty_summary": "95% credible interval (or identified interval) on the conditional "
                                       "Delta G contrast, together with the coverage of the interval on "
                                       "synthetic fixtures",
        "allowed_interpretations": [
            "conditional thermodynamic preference of a junction within the tectoRNA platform",
            "within-platform ordering of junction families under the frozen symmetric frame",
        ],
        "prohibited_interpretations": [
            "absolute free energy independent of the platform/scaffold",
            "DMS reactivity or geometric state as the same latent truth as Delta G",
            "cross-measurement-system junction equivalence without qMaP transfer evidence",
            "any sequence embedding treated as thermodynamic ground truth",
        ],
        "note": "DMS reactivity, geometric state, and sequence embedding are distinct objects, not the "
                "same latent truth.",
    }

    # ---- 2. OperatorUncertaintySpec ----
    specs["operator_uncertainty_spec.json"] = {
        "schema_version": "operator-uncertainty-spec-v1",
        "spec_version": "1.0.0",
        "run_id": "v1_2_tecto_qmap_20260803",
        "frozen_at_utc": now,
        "code_commit": commit,
        "operators": [
            {
                "operator": "Delta G 10-bp scaffold (kcal/mol)",
                "source": "Denny et al. 2018 Cell, supplementary workbook 261_SI.xlsx",
                "citation": "doi:10.1016/j.cell.2018.05.038",
                "unit": "kcal/mol",
                "range": "measured range to -7.1 kcal/mol cap (left-censored floor)",
                "uncertainty": "per-row bootstrap 95% CI reported in err10; NOT independent replicate noise",
                "applicability": "junction-inserted tectoRNA assembly, 10 mM Mg2+",
                "calibration_material": "synthetic fixtures with known Delta G (M0)",
                "calibration_procedure": "QR/interval calibration on synthetic fixtures before any real-outcome look",
                "calibration_holdout": "synthetic holdout, not real tecto labels",
                "sensitivity_result": "reserved: filled after M0 sensitivity run",
                "evidence_hash": "",
            },
            {
                "operator": "Delta G 9-bp and 11-bp scaffold (kcal/mol)",
                "source": "Denny et al. 2018 Cell, supplementary workbook 261_SI.xlsx",
                "citation": "doi:10.1016/j.cell.2018.05.038",
                "unit": "kcal/mol",
                "range": "measured range to -7.1 kcal/mol cap",
                "uncertainty": "per-row bootstrap 95% CI (err9, err11); NOT independent replicate noise",
                "applicability": "junction-inserted tectoRNA assembly, scaffold-length sensitivity",
                "calibration_material": "synthetic fixtures with known Delta G (M0)",
                "calibration_procedure": "interval calibration on synthetic fixtures",
                "calibration_holdout": "synthetic holdout",
                "sensitivity_result": "reserved",
                "evidence_hash": "",
            },
            {
                "operator": "Delta G 10-bp 5mM Mg2+ (kcal/mol)",
                "source": "Denny et al. 2018 Cell, supplementary workbook 261_SI.xlsx",
                "citation": "doi:10.1016/j.cell.2018.05.038",
                "unit": "kcal/mol",
                "range": "measured range to -7.1 kcal/mol cap",
                "uncertainty": "per-row bootstrap 95% CI (err10_5mM)",
                "applicability": "ionic-condition sensitivity operator",
                "calibration_material": "synthetic fixtures",
                "calibration_procedure": "interval calibration on synthetic fixtures",
                "calibration_holdout": "synthetic holdout",
                "sensitivity_result": "reserved",
                "evidence_hash": "",
            },
        ],
        "note": "A single subjective uncertainty set does not constitute PASS; each operator must carry "
                "external basis, calibration, holdout, and observed sensitivity.",
    }

    # ---- 3. SymmetryFrameSpec ----
    specs["symmetry_frame_spec.json"] = {
        "schema_version": "symmetry-frame-spec-v1",
        "spec_version": "1.0.0",
        "run_id": "v1_2_tecto_qmap_20260803",
        "frozen_at_utc": now,
        "code_commit": commit,
        "strand_definition": "RNA sequences are written 5'->3' for each strand of the two-way junction; "
                             "junction_seq is encoded as STRAND1_STRAND2 with '_' separator",
        "boundary": "junction unpaired residues between the two interleaved WC helices; the two flanking "
                    "base pairs are the boundary anchors",
        "flank": "the two WC base pairs adjacent to the junction on each side, part of the two-way junction",
        "sequence_orientation": "5'->3' canonical; the reciprocal orientation is handled by the "
                                "reciprocal-equivalence rule below",
        "reciprocal_equivalence": "the junction STRAND1_STRAND2 and its reverse complement / strand-swap "
                                  "are symmetry-equivalent when the physical geometry is identical; "
                                  "canonicalization maps both to the same frame",
        "symmetry_equivalence": "junctions that differ only by the exchange of the two helical directions "
                                "(mirror) or by a global sequence reversal that preserves the base-pairing "
                                "pattern are placed in the same symmetry-equivalence group",
        "canonical_frame": "lexicographically-minimal canonical representation: choose the orientation "
                           "(and strand assignment) that yields the smallest canonical string after "
                           "considering all symmetry transforms",
        "target_geometry": "two-way junction (bulge / internal loop) with the two emanating helices aligned "
                           "to the tectoRNA contact geometry",
        "canonicalization_procedure": "1) normalize strand separator; 2) generate all symmetry transforms "
                                      "(reverse, complement, reverse-complement, strand swap); 3) pick the "
                                      "lexicographically minimal canonical string; 4) assign the canonical "
                                      "frame id",
        "ambiguity_handling": "if two transforms collide (non-unique minimal), apply a deterministic tie-break "
                              "and record the ambiguity in the CleaningLedger; a symmetry-equivalent group "
                              "must never be split across train/test folds",
        "rule": "same symmetry-equivalent group must not cross folds",
    }

    # ---- 4. AssayTransportSpec-current-DMS ----
    specs["assay_transport_current_dms.json"] = {
        "schema_version": "assay-transport-spec-v1",
        "spec_version": "1.0.0",
        "run_id": "v1_2_tecto_qmap_20260803",
        "frozen_at_utc": now,
        "status": "N/A_CLOSED_NO_CROSSWALK",
        "role": "not a tecto precondition; closed and must not be reopened in v1.2",
        "allowed_use": "archive/catalog only; aggregate engineering QC; non-biological stress tests; "
                       "negative provenance case",
        "forbidden_use": "any scientific effect size, model input, split/threshold/feature selection, "
                         "or paper effect size",
    }

    # ---- 5. AssayTransportSpec-qMaPseq ----
    specs["assay_transport_qmapseq.json"] = {
        "schema_version": "assay-transport-spec-v1",
        "spec_version": "1.0.0",
        "run_id": "v1_2_tecto_qmap_20260803",
        "frozen_at_utc": now,
        "system": "qMaPseq TL/TLR-Mg2+ (NAR 2024, doi:10.1093/nar/gkae633)",
        "operational_contact_stability": "stability of the tetraloop/tetraloop-receptor contact in the "
                                         "qMaPseq scaffold as a function of [Mg2+]",
        "mg_half": "[Mg2+]1/2: the magnesium concentration at which the contact reaches half-maximal "
                   "stability; units mM",
        "rna_map_reference_dg": "RNA-MaP reference Delta G derived from the qMaP titration; a distinct "
                                "measurement from the Denny tectoRNA Delta G",
        "dms_titration_operator": "DMS signal titration vs [Mg2+] as the chemical-mapping operator",
        "uncertainty": "per-variant uncertainty from the qMaP fit; censored cases ([Mg2+]1/2 > 40 mM) "
                       "enter a censored likelihood",
        "selection_mechanism": "selection on the Mg2+ midpoint of the contact stability",
        "transport_assumptions": [
            "qMaPseq does not share a latent q with the Denny junction functional",
            "any cross-system transfer claim requires a locked Q5 transfer test",
        ],
        "allowed_cross_system_claims": [
            "restricted cross-measurement-system migration statement if QMAP_TRANSFER_SUPPORTED",
            "pre-registered negative/boundary evidence if QMAP_TRANSFER_NOT_SUPPORTED",
        ],
        "forbidden_cross_system_claims": [
            "qMaPseq as independent replication of junction preorganization",
            "junction equivalence across systems without Q5 transfer evidence",
        ],
    }

    # ---- 6. PrimaryAnalysisSpec ----
    # Thresholds frozen BEFORE outcome look, with external basis and simulation.
    specs["primary_analysis_spec.json"] = {
        "schema_version": "primary-analysis-spec-v1",
        "spec_version": "1.0.0",
        "run_id": "v1_2_tecto_qmap_20260803",
        "frozen_at_utc": now,
        "code_commit": commit,
        "primary_estimand": "target-specific thermodynamic functional",
        "primary_target": "conditional Delta G of junction within tectoRNA platform",
        "primary_generalization_split": "motif-family holdout (pre-registered; blocks leakage via shared "
                                        "motif/scaffold; see T1 for the frozen split manifest)",
        "primary_operator_robustness_criterion": "estimated identified-interval width must be <= 1.0 kcal/mol "
                                                 "for the primary contrast on the held-out axis, and interval "
                                                 "calibration must be within [0.9, 1.0] on synthetic fixtures",
        "primary_held_out_metric": "interval coverage on the held-out motif families (primary); "
                                   "point MAE as secondary, never as the primary conclusion",
        "primary_qmap_transport_test": "locked Q5 transfer test (assign only after Q0-Q3 pass)",
        "thresholds": {
            "min_meaningful_effect_kcal": {
                "value": 1.0,
                "basis": "Yesselman et al. 2019 PNAS reported >2 kcal/mol transduced effects from small "
                         "helix differences; a 1.0 kcal/mol minimum meaningful effect is a conservative "
                         "scientific floor for the tectoRNA platform",
                "frozen_before_outcome": True,
            },
            "interval_width_max_kcal": {
                "value": 1.0,
                "basis": "consistent with the minimum meaningful effect; intervals wider than this cannot "
                         "support a meaningful effect claim",
                "frozen_before_outcome": True,
            },
            "calibration_target": {
                "value": [0.9, 1.0],
                "basis": "standard 90-100% interval coverage target on synthetic fixtures",
                "frozen_before_outcome": True,
            },
        },
        "note": "Thresholds are frozen before any real held-out outcome is examined; they must not be "
                "changed because results are unfavorable.",
    }

    # ---- 7. NegativeControlSpec ----
    specs["negative_control_spec.json"] = {
        "schema_version": "negative-control-spec-v1",
        "spec_version": "1.0.0",
        "run_id": "v1_2_tecto_qmap_20260803",
        "frozen_at_utc": now,
        "code_commit": commit,
        "negative_controls": [
            "label permutation: shuffle Delta G labels before split; model must recover no signal",
            "null signal: synthetic data with no junction effect; must not reject null",
            "weak signal: synthetic data with a small effect below the minimum meaningful effect; "
            "must not claim a meaningful effect",
            "out-of-range operator: an operator outside the calibrated range; must be flagged and "
            "handled by the operator-robustness path",
            "homolog leakage: synthetic train/test near-homolog leakage; leakage detector must catch it",
            "calibration drift: synthetic drift in operator calibration; must be detected",
        ],
        "pass_rule": "a negative control must not produce a positive scientific claim; "
                     "detection of the planted artifact is success, not a biological result",
    }

    # Write and hash all specs
    spec_manifest = {"spec_version": "s0-spec-manifest-v1", "frozen_at_utc": now,
                     "code_commit": commit, "contract_sha256": CONTRACT_SHA256,
                     "specs": {}}
    for name, spec in specs.items():
        path, h = write_spec(name, spec)
        spec_manifest["specs"][name] = {"path": path, "sha256": h}
        print(f"wrote {name} sha256={h[:16]}")

    spec_manifest_path = os.path.join(SPEC_DIR, "s0_spec_manifest.json")
    with open(spec_manifest_path, "w", encoding="utf-8") as f:
        json.dump(spec_manifest, f, indent=2, ensure_ascii=False)
    spec_manifest_hash = sha256_file(spec_manifest_path)
    print(f"spec_manifest sha256={spec_manifest_hash[:16]}")

    # Update canonical manifest: record S0 artifacts + derived manifest freshness
    manifest = CanonicalStateManifest.load(MANIFEST_PATH)
    manifest.data["output_artifacts"] += [spec_manifest_path]
    manifest.data["derived_manifest_freshness"]["s0_spec_manifest"] = spec_manifest_hash
    manifest.data["gate_statuses"]["S0"] = "RUNNING"
    manifest.save(MANIFEST_PATH)
    print("manifest updated; S0 RUNNING (finalizer must confirm PASS)")
    return 0


if __name__ == "__main__":
    sys.exit(main())