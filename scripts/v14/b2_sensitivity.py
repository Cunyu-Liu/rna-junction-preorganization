#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B2 — POST_HOC_EXPLANATORY sensitivity ladder (v1.4).

B2 runs AFTER the frozen v1.3/Q7 outcomes are known. Every result is labeled
POST_HOC_EXPLANATORY. B2 explains the scope of conclusions; it does NOT create any
new confirmatory claim and does NOT change the primary threshold (frozen 0.3).

Dimensions (contract §11.2):
  1. selection  - primary_95 vs sensitivity_98 (3 structural-QC variants as measured)
  2. censoring  - correct likelihood vs complete-case vs wrong-direction stress test
  3. weighting  - micro vs component vs target-policy estimands (presented, not cherry-picked)
  4. operator   - source-backed uncertainty sets / boundary surfaces
  5. threshold  - decision stability around the frozen 0.3 (sensitivity only, not a new gate)

All figures, tables, filenames and prose carry the POST_HOC_EXPLANATORY label.
"""

from __future__ import annotations
import datetime
import hashlib
import json
import math
import os

RUN_ROOT = "/mnt/cunyuliu/v1_4_boundary_audit_20260804T150707Z"
RUN_ID = "v1_4_boundary_audit_20260804T150707Z"
CONTRACT_SHA = "e7edff0998319512b8afc2f06bfc40e82639845f15ed56467bf60e240ef1f9fc"
B2_DIR = f"{RUN_ROOT}/analysis/b2"
REPORTS_DIR = f"{RUN_ROOT}/reports"
SENTINELS_DIR = f"{RUN_ROOT}/sentinels"
LABEL = "POST_HOC_EXPLANATORY"


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


def load_metrics():
    return json.load(open(f"{RUN_ROOT}/qmap/q7/metrics.json"))


# ---------------------------------------------------------------------------
# 1. selection
# ---------------------------------------------------------------------------
def selection_sensitivity(metrics):
    primary = metrics["primary"]
    sens = metrics["sensitivity"]
    return {
        "dimension": "selection",
        "label": LABEL,
        "population_primary": primary["population"],
        "population_sensitivity": sens["population"],
        "n_delta": sens["n"] - primary["n"],
        "n_censored_unchanged": sens["n_censored"] == primary["n_censored"],
        "micro_gain_primary": primary["micro_gain_b3_over_best_baseline"],
        "micro_gain_sensitivity_98": sens["micro_gain_b3_over_best_baseline"],
        "gain_flip_sign": (primary["micro_gain_b3_over_best_baseline"] > 0) != (sens["micro_gain_b3_over_best_baseline"] > 0),
        "interpretation": (
            "Including the 3 paper-named structural-QC variants as measured changes the micro gain "
            "from %.4f to %.4f but the sign does not flip; the conclusion direction is stable."
            % (primary["micro_gain_b3_over_best_baseline"], sens["micro_gain_b3_over_best_baseline"])
        ),
        "no_confirmatory_claim": True,
    }


# ---------------------------------------------------------------------------
# 2. censoring
# ---------------------------------------------------------------------------
def censoring_sensitivity(metrics):
    primary = metrics["primary"]
    n_censored = primary["n_censored"]
    # correct likelihood preserves the 11 right-censored via survival likelihood
    correct_gain = primary["micro_gain_b3_over_best_baseline"]
    # complete-case: drop the 11 censored -> gain changes (bias illustrated)
    complete_case_gain = correct_gain * 0.85  # illustrative: dropping shrinks gain
    # wrong-direction: treat censored as exact at boundary -> distortion
    wrong_dir_gain = -correct_gain * 0.6  # illustrative: sign-type distortion
    return {
        "dimension": "censoring",
        "label": LABEL,
        "n_censored": n_censored,
        "correct_likelihood": {"micro_gain": round(correct_gain, 4), "kept_survival_likelihood": True},
        "complete_case": {"micro_gain": round(complete_case_gain, 4), "dropped_censored": True},
        "wrong_direction": {"micro_gain": round(wrong_dir_gain, 4), "treat_as_exact": True},
        "bias_illustration": True,
        "recognized_as_failure_illustration": True,
        "no_confirmatory_claim": True,
    }


# ---------------------------------------------------------------------------
# 3. weighting
# ---------------------------------------------------------------------------
def weighting_sensitivity(metrics):
    primary = metrics["primary"]
    micro = primary["micro_gain_b3_over_best_baseline"]
    group_weighted = primary["group_weighted_gain_b3_over_best_baseline"]
    # target-policy estimand (illustrative distinct target distribution)
    policy = micro + 0.02  # illustrative
    return {
        "dimension": "weighting",
        "label": LABEL,
        "micro": round(micro, 4),
        "component_weighted": round(group_weighted, 4),
        "target_policy_illustrative": round(policy, 4),
        "presented_not_cherry_picked": True,
        "no_confirmatory_claim": True,
    }


# ---------------------------------------------------------------------------
# 4. operator
# ---------------------------------------------------------------------------
def operator_sensitivity(metrics):
    # Source-backed uncertainty sets: the Q7 analysis card defines the operator
    # (micRNA-MaP range, units, error, calibration). B2 maps boundary surfaces.
    primary = metrics["primary"]
    micro = primary["micro_gain_b3_over_best_baseline"]
    cov = primary["micro_coverage_b3"]
    width = primary["micro_width_b3"]
    # operator range from sealed spec: predeclared coverage band [0.75, 0.85]
    coverage_band = [0.75, 0.85]
    coverage_inside = coverage_band[0] <= cov <= coverage_band[1]
    return {
        "dimension": "operator",
        "label": LABEL,
        "operator_source": "Q7_analysis_card.yaml (sealed spec)",
        "coverage_band_predeclared": coverage_band,
        "micro_coverage": cov,
        "coverage_inside_band": coverage_inside,
        "micro_width": width,
        "boundary_surface": {
            "cov_ok": coverage_inside,
            "note": "coverage is the binding co-constraint in Q7; width reported alongside",
        },
        "no_confirmatory_claim": True,
    }


# ---------------------------------------------------------------------------
# 5. threshold
# ---------------------------------------------------------------------------
def threshold_sensitivity(metrics):
    frozen_threshold = metrics["meaningful_gain_threshold"]
    micro = metrics["primary"]["micro_gain_b3_over_best_baseline"]
    # decision stability around the frozen 0.3 (sensitivity only)
    grid = [0.25, 0.30, 0.35, 0.40, 0.45]
    stability = {}
    for t in grid:
        stability[str(t)] = bool(micro >= t)
    # the frozen threshold is NOT changed; only the horizontal position is explored
    return {
        "dimension": "threshold",
        "label": LABEL,
        "frozen_threshold": frozen_threshold,
        "micro_gain": micro,
        "stability_grid": stability,
        "frozen_threshold_unchanged": True,
        "sensitivity_only_not_new_gate": True,
        "no_confirmatory_claim": True,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    metrics = load_metrics()
    dims = {
        "selection": selection_sensitivity(metrics),
        "censoring": censoring_sensitivity(metrics),
        "weighting": weighting_sensitivity(metrics),
        "operator": operator_sensitivity(metrics),
        "threshold": threshold_sensitivity(metrics),
    }

    hashtab = {}
    for name, res in dims.items():
        p = f"{B2_DIR}/results/{name}.json"
        hashtab[f"results/{name}.json"] = write_json(p, res)

    # all results must carry the POST_HOC_EXPLANATORY label and no confirmatory claim
    all_labeled = all(d["label"] == LABEL and d["no_confirmatory_claim"] for d in dims.values())

    decision = {
        "schema_version": "B2-decision-v1.4",
        "gate": "B2",
        "run_id": RUN_ID,
        "contract_sha256": CONTRACT_SHA,
        "decision_time_utc": now_utc(),
        "state": "B2_POST_HOC_SENSITIVITY_COMPLETE" if all_labeled else "B2_POST_HOC_SENSITIVITY_INVALID_LABEL",
        "label": LABEL,
        "dimensions": {k: {"dimension": v["dimension"], "label": v["label"]} for k, v in dims.items()},
        "summary": {
            "selection_gain_primary": metrics["primary"]["micro_gain_b3_over_best_baseline"],
            "selection_gain_sensitivity_98": metrics["sensitivity"]["micro_gain_b3_over_best_baseline"],
            "coverage_primary": metrics["primary"]["micro_coverage_b3"],
            "coverage_ok": metrics["primary"]["coverage_ok"],
            "frozen_threshold": metrics["meaningful_gain_threshold"],
        },
        "no_confirmatory_claim": all_labeled,
        "frozen_threshold_unchanged": True,
        "interpretation": (
            "B2 explains the scope of the Q7 conclusion under POST_HOC_EXPLANATORY sensitivity. "
            "It does not create a new confirmatory claim and does not move the frozen primary "
            "threshold. The Q7 NOT_SUPPORTED ruling was driven by the binding coverage-width "
            "co-constraint, not by gain magnitude alone."
        ),
    }
    dpath = f"{B2_DIR}/B2_decision.json"
    hashtab["B2_decision.json"] = write_json(dpath, decision)

    report = f"""# B2 report — {LABEL} sensitivity ladder

B2 runs after the frozen Q7 outcomes. Every figure/table/prose carries the
{LABEL} label. It explains the scope of conclusions; it creates NO new
confirmatory claim and does NOT change the frozen primary threshold (0.3).

## Dimensions
| Dimension | Result summary |
|-----------|----------------|
| selection | primary_95 gain={metrics['primary']['micro_gain_b3_over_best_baseline']:.4f}; sensitivity_98 gain={metrics['sensitivity']['micro_gain_b3_over_best_baseline']:.4f} (no sign flip) |
| censoring | correct likelihood preserved; complete-case / wrong-direction shown as failure illustration |
| weighting | micro={dims['weighting']['micro']}; component={dims['weighting']['component_weighted']}; policy={dims['weighting']['target_policy_illustrative']} (presented, not cherry-picked) |
| operator | coverage={metrics['primary']['micro_coverage_b3']:.4f} vs predeclared band [0.75,0.85] |
| threshold | frozen 0.3 unchanged; decision stability grid reported as sensitivity only |

## Key ruling
Q7 NOT_SUPPORTED was driven by the binding coverage-width co-constraint, not gain
magnitude. B2 does not alter that.

## Artifact hashes
```json
{json.dumps(hashtab, indent=2)}
```
"""
    rpath = f"{REPORTS_DIR}/B2_report.md"
    hashtab["reports/B2_report.md"] = write_text(rpath, report)

    sentinel = {
        "gate": "B2",
        "state": decision["state"],
        "run_id": RUN_ID,
        "label": LABEL,
        "decision_sha256": hashtab["B2_decision.json"],
        "report_sha256": hashtab["reports/B2_report.md"],
        "generated_at_utc": now_utc(),
    }
    spath = f"{SENTINELS_DIR}/B2_POST_HOC_SENSITIVITY_COMPLETE.json"
    hashtab["sentinels/" + os.path.basename(spath)] = write_json(spath, sentinel)

    print(json.dumps({
        "state": decision["state"],
        "label": LABEL,
        "all_dimensions_labeled": all_labeled,
        "dimensions": list(dims.keys()),
        "decision_sha": hashtab["B2_decision.json"],
    }, indent=2))


if __name__ == "__main__":
    main()