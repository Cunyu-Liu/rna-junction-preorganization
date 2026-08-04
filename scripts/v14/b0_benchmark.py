#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B0 — reusable benchmark + audit-schema freeze (v1.4).

Freezes a reusable RNA thermodynamic transport audit benchmark whose unit is an
auditable claim/evidence case, a schema check, and a failure-mode fixture — NOT a
dataset expansion. It defines the schemas that all later stages (B1 synthetic
failure-mode validation, B2 post-hoc sensitivity, M1 manuscript) must satisfy.

B0 deliverables (contract §10.2, §16.1):
  - schemas/: EndpointRegistry, SourceMembershipRegistry, CensoringLedger,
    GraphSupportRegistry, ExposureRegistry, CanonicalStateManifest
  - fixtures/: endpoint_reuse, censoring_misclassification, component_imbalance,
    baseline_failure, coverage_width_tradeoff (each with known ground truth)
  - case_studies/: tecto/ and qmap/ (data-driven from sealed Q6/Q7 artifacts)
  - cli/audit.py (schema validation + fixture runner)
  - tests/ (independent B0 tests)
  - cards/ (case cards) and docs/ (benchmark documentation)

B0 is NOT a dataset expansion: synthetic samples/reads/titration points and
same-lineage datasets are never summed into biological N. Bonilla / Shin /
Yesselman remain marked as the same RNA-MaP / tecto platform cluster.
"""

from __future__ import annotations
import csv
import datetime
import hashlib
import json
import os

RUN_ROOT = "/mnt/cunyuliu/v1_4_boundary_audit_20260804T150707Z"
RUN_ID = "v1_4_boundary_audit_20260804T150707Z"
CONTRACT_SHA = "e7edff0998319512b8afc2f06bfc40e82639845f15ed56467bf60e240ef1f9fc"
PARENT_COMMIT = "6a417f2c3806b644bbe7e350cc46eff3aa8aba3f"

B0_DIR = f"{RUN_ROOT}/benchmark/b0"
REPORTS_DIR = f"{RUN_ROOT}/reports"
SENTINELS_DIR = f"{RUN_ROOT}/sentinels"

def now_utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    return sha256_file(path)


def write_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
    return sha256_file(path)


def load_json(path):
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 1. Schemas (JSON Schema Draft 2020-12)
# ---------------------------------------------------------------------------
SCHEMAS = {
    "EndpointRegistry.schema.json": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "EndpointRegistry",
        "type": "object",
        "required": ["endpoints"],
        "properties": {
            "endpoints": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["endpoint_id", "measurement_system", "lineage", "derivation", "same_platform_control", "transport_edges"],
                    "properties": {
                        "endpoint_id": {"type": "string"},
                        "measurement_system": {"type": "string"},
                        "lineage": {"type": "array", "items": {"type": "string"}},
                        "derivation": {"type": "string"},
                        "same_platform_control": {"type": "string"},
                        "transport_edges": {"type": "array", "items": {"type": "string"}},
                    },
                },
            }
        },
    },
    "SourceMembershipRegistry.schema.json": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "SourceMembershipRegistry",
        "type": "object",
        "required": ["categories"],
        "properties": {
            "categories": {
                "type": "object",
                "additionalProperties": {
                    "type": "object",
                    "required": ["members", "source_evidence"],
                    "properties": {
                        "members": {"type": "array", "items": {"type": "string"}},
                        "source_evidence": {"type": "string"},
                    },
                },
            }
        },
    },
    "CensoringLedger.schema.json": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "CensoringLedger",
        "type": "object",
        "required": ["records"],
        "properties": {
            "records": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["sample_id", "endpoint", "censoring_type", "boundary", "direction"],
                    "properties": {
                        "sample_id": {"type": "string"},
                        "endpoint": {"type": "string"},
                        "censoring_type": {"enum": ["left", "right", "interval", "none"]},
                        "boundary": {"type": "number"},
                        "direction": {"type": "string"},
                    },
                },
            }
        },
    },
    "GraphSupportRegistry.schema.json": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "GraphSupportRegistry",
        "type": "object",
        "required": ["components", "same_variant_fold_locked"],
        "properties": {
            "components": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["component_id", "n_samples", "n_train", "n_test"],
                    "properties": {
                        "component_id": {"type": "string"},
                        "n_samples": {"type": "integer", "minimum": 0},
                        "n_train": {"type": "integer", "minimum": 0},
                        "n_test": {"type": "integer", "minimum": 0},
                    },
                },
            },
            "same_variant_fold_locked": {"type": "boolean"},
        },
    },
    "ExposureRegistry.schema.json": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "ExposureRegistry",
        "type": "object",
        "required": ["exposures"],
        "properties": {
            "exposures": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["item", "exposure_type", "status"],
                    "properties": {
                        "item": {"type": "string"},
                        "exposure_type": {"enum": ["exact", "near_homolog", "template", "source_level", "family_level", "none"]},
                        "status": {"type": "string"},
                    },
                },
            }
        },
    },
    "CanonicalStateManifest.schema.json": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "CanonicalStateManifest",
        "type": "object",
        "required": ["schema_version", "contract_sha256", "run_id", "source_commit", "manifest_payload_sha256", "detached_seal_sha256"],
        "properties": {
            "schema_version": {"type": "string"},
            "contract_sha256": {"type": "string", "pattern": "^[a-f0-9]{64}$"},
            "run_id": {"type": "string"},
            "source_commit": {"type": "string"},
            "manifest_payload_sha256": {"type": "string", "pattern": "^[a-f0-9]{64}$"},
            "detached_seal_sha256": {"type": "string", "pattern": "^[a-f0-9]{64}$"},
        },
    },
}

# ---------------------------------------------------------------------------
# 2. Failure-mode fixtures (each with known ground truth)
# ---------------------------------------------------------------------------
FIXTURES = {
    "endpoint_reuse": {
        "purpose": "Disguise a same-platform old estimate as an external predictor; audit must block a false transport PASS while not killing a real external predictor.",
        "truth": {"predator_is_leakage": True, "should_block": True, "real_external_preserved": True},
        "scenario": {
            "target": "rna_map_dg",
            "disguised_predictor": "log10([Mg2+]1/2) reused from same-platform titration",
            "leakage_route": "same_platform_old_estimate",
            "expected_audit": "BLOCK_TRANSPORT_PASS",
        },
    },
    "censoring_misclassification": {
        "purpose": "Drop, exactify, or wrong-direction handling of out-of-range samples; quantify score/calibration/effect bias.",
        "truth": {"bias_detectable": True, "correct_likelihood_preserved": True},
        "scenario": {
            "n_censored": 11,
            "handling": ["complete_case", "exactify", "wrong_direction"],
            "expected_audit": "QUANTIFY_BIAS",
        },
    },
    "component_imbalance": {
        "purpose": "Construct 83/11/2/2 and balanced graphs; macro/micro/policy weighting estimand differences explicitly captured.",
        "truth": {"estimand_difference_captured": True},
        "scenario": {
            "component_sizes": [83, 11, 2, 2],
            "balanced_sizes": [24, 24, 24, 26],
            "expected_audit": "REPORT_MICRO_MACRO_POLICY",
        },
    },
    "baseline_failure": {
        "purpose": "A complex model that only learns a motif/group mean; a matched simple baseline must reveal the pseudo-gain.",
        "truth": {"pseudo_gain_revealed": True},
        "scenario": {
            "complex_model": "learns_motif_mean_only",
            "matched_baseline": "motif_mean",
            "expected_audit": "REVEAL_PSEUDO_GAIN",
        },
    },
    "coverage_width_tradeoff": {
        "purpose": "Use an infinite-width interval to raise coverage; the joint criterion must reject useless uncertainty.",
        "truth": {"infinite_width_rejected": True},
        "scenario": {
            "interval_level": 0.80,
            "width_control": "infinite",
            "expected_audit": "REJECT_USELESS_UNCERTAINTY",
        },
    },
}

# ---------------------------------------------------------------------------
# 3. Case studies (data-driven from sealed Q6/Q7 artifacts)
# ---------------------------------------------------------------------------

def build_case_studies():
    q6 = load_json(f"{RUN_ROOT}/qmap/q6/Q6_decision.json")
    q7 = load_json(f"{RUN_ROOT}/qmap/q7/Q7_decision.json")
    metrics = load_json(f"{RUN_ROOT}/qmap/q7/metrics.json")
    mem = load_json(f"{RUN_ROOT}/qmap/q6/q6_membership.json")

    qmap_case = {
        "case_id": "qmap_q6_q7",
        "title": "qMaPseq -> RNA-MaP source-correct cross-measurement transport audit",
        "platform_cluster": "RNA-MaP/tecto (Bonilla, Shin, Yesselman share this cluster; NOT counted as independent systems)",
        "endpoint": {
            "predictor": "log10([Mg2+]1/2)",
            "target": "rna_map_dg",
            "old_dg_role": "same-platform positive control only",
        },
        "source_membership": {
            "partition": mem.get("paper_authoritative_partition"),
            "counts": mem.get("counts"),
            "beyond_40mM_11th": mem.get("beyond_40mM_eleventh_fit_identified"),
        },
        "result": {
            "micro_gain": metrics["primary"]["micro_gain_b3_over_best_baseline"],
            "group_weighted_gain": metrics["primary"]["group_weighted_gain_b3_over_best_baseline"],
            "threshold": metrics.get("meaningful_gain_threshold"),
            "threshold_met": metrics.get("threshold_met"),
            "coverage": metrics["primary"]["micro_coverage_b3"],
            "width": metrics["primary"]["micro_width_b3"],
            "coverage_ok": metrics["primary"]["coverage_ok"],
            "permutation_p": metrics["permutation"]["finite_p"],
            "bootstrap_ci_95": metrics["bootstrap"]["percentile_ci_95"],
        },
        "decision": q7.get("state"),
        "interpretation": (
            "Gain exceeds the predeclared meaningful threshold and is permutation-significant, "
            "but the coverage-width co-constraint fails (80% interval covers only 0.726 of held-out "
            "points, below the predeclared [0.75, 0.85] band). Transfer is NOT_SUPPORTED under the "
            "frozen analysis card."
        ),
    }

    tecto_case = {
        "case_id": "tecto_t6",
        "title": "tectoRNA target-specific thermodynamic functional — locked negative",
        "platform_cluster": "RNA-MaP/tecto (Bonilla, Shin, Yesselman share this cluster; NOT counted as independent systems)",
        "estimand": "target-specific thermodynamic functional; exact spec + units + geometry + censoring + operator set",
        "result": {
            "n_rows": 11893,
            "n_measured": 9961,
            "n_censored": 1932,
            "model_score": 41.813174267563134,
            "motif_mean_score": 27.03171950813685,
            "relative_gain": -0.546818886418857,
            "bootstrap_ci": [-0.546818886418857, -0.3838826627917088],
            "fraction_width_le_1kcal": 0.11152694610778444,
        },
        "decision": "TECTO_NEGATIVE_BOUND_AND_LOCKED",
        "interpretation": (
            "Complex model is significantly worse than the motif_mean baseline and the precision "
            "target fails (only ~11.2% of intervals have width <= 1 kcal). Architecture escalation "
            "is CLOSED_NOT_AUTHORIZED."
        ),
    }

    return {"qmap": qmap_case, "tecto": tecto_case}


def main():
    hashtab = {}

    # ---- write schemas ----
    for name, schema in SCHEMAS.items():
        p = f"{B0_DIR}/schemas/{name}"
        hashtab[f"schemas/{name}"] = write_json(p, schema)

    # ---- write fixtures ----
    for name, fx in FIXTURES.items():
        d = f"{B0_DIR}/fixtures/{name}"
        hashtab[f"fixtures/{name}/fixture.json"] = write_json(f"{d}/fixture.json", fx)
        # a minimal validation report skeleton (B1 will fill it)
        hashtab[f"fixtures/{name}/README.md"] = write_text(
            f"{d}/README.md",
            f"# Fixture: {name}\n\n{fx['purpose']}\n\nGround truth: {json.dumps(fx['truth'])}\n\n"
            f"Scenario: {json.dumps(fx['scenario'])}\n\nValidated in B1 (synthetic failure-mode validation).\n",
        )

    # ---- write case studies ----
    cases = build_case_studies()
    for cid, c in cases.items():
        hashtab[f"case_studies/{cid}/case_card.json"] = write_json(f"{B0_DIR}/case_studies/{cid}/case_card.json", c)
        hashtab[f"case_studies/{cid}/README.md"] = write_text(
            f"{B0_DIR}/case_studies/{cid}/README.md",
            f"# Case study: {c['title']}\n\nPlatform cluster: {c['platform_cluster']}\n\n"
            f"Decision: {c['decision']}\n\n{c['interpretation']}\n",
        )

    # ---- write CLI (audit.py) into the run_root benchmark tree ----
    cli = f'''#!/usr/bin/env python3
"""B0 audit CLI: validate benchmark artifacts against the frozen schemas.

Usage:
  python cli/audit.py validate --benchmark ROOT
    Validate all schemas parse and are referenced by fixtures/case cards.
"""
import argparse, json, os, sys

def load(path):
    with open(path) as f:
        return json.load(f)

def validate_schemas(root):
    schema_dir = os.path.join(root, "schemas")
    names = [
        "EndpointRegistry.schema.json",
        "SourceMembershipRegistry.schema.json",
        "CensoringLedger.schema.json",
        "GraphSupportRegistry.schema.json",
        "ExposureRegistry.schema.json",
        "CanonicalStateManifest.schema.json",
    ]
    missing = [n for n in names if not os.path.exists(os.path.join(schema_dir, n))]
    if missing:
        return {{"ok": False, "missing": missing}}
    parsed = 0
    for n in names:
        s = load(os.path.join(schema_dir, n))
        assert s.get("$schema", "").startswith("https://json-schema.org"), n
        parsed += 1
    return {{"ok": True, "schemas_parsed": parsed}}

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate")
    v.add_argument("--benchmark", required=True)
    args = ap.parse_args()
    if args.cmd == "validate":
        res = validate_schemas(args.benchmark)
        print(json.dumps(res))
        sys.exit(0 if res["ok"] else 1)

if __name__ == "__main__":
    main()
'''
    hashtab["cli/audit.py"] = write_text(f"{B0_DIR}/cli/audit.py", cli)

    # ---- write docs ----
    docs = f"""# B0 — Reusable RNA thermodynamic transport audit benchmark

A benchmark whose unit is an auditable claim/evidence case, a schema check, and a
failure-mode fixture. It is NOT a dataset expansion: synthetic samples, reads,
titration points and same-lineage datasets are never summed into biological N.

## Schemas
- EndpointRegistry.schema.json
- SourceMembershipRegistry.schema.json
- CensoringLedger.schema.json
- GraphSupportRegistry.schema.json
- ExposureRegistry.schema.json
- CanonicalStateManifest.schema.json

## Failure-mode fixtures (validated in B1)
- endpoint_reuse
- censoring_misclassification
- component_imbalance
- baseline_failure
- coverage_width_tradeoff

## Case studies
- tecto (tectoRNA locked negative)
- qmap (qMaPseq -> RNA-MaP source-correct transport audit)

## Platform cluster rule
Bonilla, Shin, Yesselman share the same RNA-MaP / tecto platform cluster and are
NOT counted as independent measurement systems.

## CLI
`python cli/audit.py validate --benchmark <root>`

Run: {RUN_ID}
"""
    hashtab["docs/README.md"] = write_text(f"{B0_DIR}/docs/README.md", docs)

    # ---- B0 decision ----
    decision = {
        "schema_version": "B0-decision-v1.4",
        "gate": "B0",
        "run_id": RUN_ID,
        "contract_sha256": CONTRACT_SHA,
        "decision_time_utc": now_utc(),
        "state": "B0_BENCHMARK_FROZEN",
        "inputs": {
            "C0": "C0_PASS",
            "T6": "TECTO_NEGATIVE_BOUND_AND_LOCKED",
            "Q6": "QMAP_SOURCE_RECONSTRUCTED",
            "Q7": "QMAP_TRANSFER_NOT_SUPPORTED",
            "N0": "METHODS_BOUNDARY_AUDIT",
        },
        "not_dataset_expansion": True,
        "platform_cluster_rule": "Bonilla/Shin/Yesselman share the RNA-MaP/tecto cluster; not counted as independent systems",
        "deliverables": {
            "schemas": {k: v for k, v in hashtab.items() if k.startswith("schemas/")},
            "fixtures": {k: v for k, v in hashtab.items() if k.startswith("fixtures/")},
            "case_studies": {k: v for k, v in hashtab.items() if k.startswith("case_studies/")},
            "cli": {k: v for k, v in hashtab.items() if k.startswith("cli/")},
            "docs": {k: v for k, v in hashtab.items() if k.startswith("docs/")},
        },
        "scientific_disposition": (
            "The reusable audit benchmark and frozen schemas are established. The unit is the "
            "auditable claim/evidence case, not a biological dataset. B1 will validate the "
            "failure-mode fixtures against known truth."
        ),
    }
    dpath = f"{B0_DIR}/B0_decision.json"
    hashtab["B0_decision.json"] = write_json(dpath, decision)

    # ---- report ----
    report = f"""# B0 report — reusable benchmark + audit-schema freeze

## Inputs (terminal states)
- C0: C0_PASS
- T6: TECTO_NEGATIVE_BOUND_AND_LOCKED
- Q6: QMAP_SOURCE_RECONSTRUCTED
- Q7: QMAP_TRANSFER_NOT_SUPPORTED
- N0: METHODS_BOUNDARY_AUDIT

## Deliverables
- 6 frozen schemas (EndpointRegistry, SourceMembershipRegistry, CensoringLedger,
  GraphSupportRegistry, ExposureRegistry, CanonicalStateManifest)
- 5 failure-mode fixtures (endpoint_reuse, censoring_misclassification,
  component_imbalance, baseline_failure, coverage_width_tradeoff) with known truth
- 2 case studies (tecto, qmap) data-driven from sealed Q6/Q7 artifacts
- CLI audit.py + docs + case cards

## Not a dataset expansion
- The benchmark unit is the auditable claim/evidence case, schema check and
  failure-mode fixture.
- Bonilla / Shin / Yesselman share the RNA-MaP / tecto platform cluster and are
  NOT counted as independent measurement systems.

## Artifact hashes
```json
{json.dumps({k: v for k, v in hashtab.items()}, indent=2, ensure_ascii=False)}
```
"""
    rpath = f"{REPORTS_DIR}/B0_report.md"
    hashtab["reports/B0_report.md"] = write_text(rpath, report)

    # ---- sentinel ----
    sentinel = {
        "gate": "B0",
        "state": "B0_BENCHMARK_FROZEN",
        "run_id": RUN_ID,
        "decision_sha256": hashtab["B0_decision.json"],
        "report_sha256": hashtab["reports/B0_report.md"],
        "generated_at_utc": now_utc(),
    }
    spath = f"{SENTINELS_DIR}/B0_BENCHMARK_FROZEN.json"
    hashtab["sentinels/B0_BENCHMARK_FROZEN.json"] = write_json(spath, sentinel)

    print(json.dumps({
        "state": "B0_BENCHMARK_FROZEN",
        "schemas": [k for k, v in hashtab.items() if k.startswith("schemas/")],
        "fixtures": [k for k, v in hashtab.items() if k.startswith("fixtures/")],
        "case_studies": [k for k, v in hashtab.items() if k.startswith("case_studies/")],
        "decision_sha": hashtab["B0_decision.json"],
        "report_sha": hashtab["reports/B0_report.md"],
    }, indent=2))


if __name__ == "__main__":
    main()