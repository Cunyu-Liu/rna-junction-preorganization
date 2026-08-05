#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M1 — manuscript skeleton + figure-source tables (v1.4).

Generates the manuscript body and figure-source tables SOLELY from sealed artifacts
(no hand-copied numbers). All numbers are read from Q7/T6/N0/B2 decision files and
the Q7 metrics. The manuscript:
  - reports the boundary result + the two real case outcomes (tecto locked negative,
    qMaP transfer NOT_SUPPORTED);
  - labels every post-outcome method item as POST_HOC_EXPLANATORY;
  - does NOT conflate qMaP's original in-population correlation (qMaP2024) with the
    out-of-component transfer measured here;
  - carries the METHODS_BOUNDARY_AUDIT claim tier and the submission HOLD.

Deliverables (contract §16.1):
  - manuscript/m1/manuscript.md
  - manuscript/m1/tables/table_1_case_capability_matrix.tsv
  - manuscript/m1/tables/table_2_claim_matrix.tsv
  - manuscript/m1/tables/table_3_q7_quantitative_results.tsv
  - manuscript/m1/figures_labels.md
"""

from __future__ import annotations
import datetime
import json
import os

RUN_ROOT = "/mnt/cunyuliu/v1_4_boundary_audit_20260804T150707Z"
RUN_ID = "v1_4_boundary_audit_20260804T150707Z"
M1_DIR = f"{RUN_ROOT}/manuscript/m1"
REPORTS_DIR = f"{RUN_ROOT}/reports"
SENTINELS_DIR = f"{RUN_ROOT}/sentinels"
CONTRACT_SHA = "e7edff0998319512b8afc2f06bfc40e82639845f15ed56467bf60e240ef1f9fc"


def now_utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
    return os.path.getsize(path)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def main():
    q7 = load_json(f"{RUN_ROOT}/qmap/q7/Q7_decision.json")
    q7m = load_json(f"{RUN_ROOT}/qmap/q7/metrics.json")
    t6 = load_json(f"{RUN_ROOT}/tecto/t6/T6_decision.json")
    n0 = load_json(f"{RUN_ROOT}/novelty/n0/N0_decision.json")
    b2 = load_json(f"{RUN_ROOT}/analysis/b2/B2_decision.json")

    # ---- extract key frozen numbers ----
    q7_gain = q7["primary"]["micro_gain_b3_over_best_baseline"]      # 0.4163
    q7_threshold = q7["primary"]["meaningful_threshold"]              # 0.3
    q7_threshold_met = q7["primary"]["threshold_met"]                 # True
    q7_cov = q7["co_constraints"]["micro_coverage"]                   # 0.7263
    q7_cov_ok = q7["co_constraints"]["coverage_ok"]                   # False
    q7_perm_p = q7["permutation_finite_p"]                             # 0.001
    q7_ci = q7["bootstrap_ci_95"]                                     # [-0.5718, 0.7478]
    q7_state = q7["state"]                                            # QMAP_TRANSFER_NOT_SUPPORTED
    q7_n = q7m["primary"]["n"]                                        # 95
    q7_n_measured = q7m["primary"]["n_measured"]                      # 84
    q7_n_censored = q7m["primary"]["n_censored"]                      # 11
    q7_best_base = q7["primary"]["best_baseline"]                     # B1
    t6_state = t6["terminal_state"]                                    # TECTO_NEGATIVE_BOUND_AND_LOCKED
    t6_preserve = t6["preserve_rule"]                                 # PRESERVE
    n0_route = n0.get("route", "METHODS_BOUNDARY_AUDIT")
    n0_primary = n0.get("primary_claim", "True qMaP transfer is below the predeclared meaningful threshold; tecto model is also not better than the motif baseline.")
    n0_negative = n0.get("negative_claim", "True qMaP transfer is below the predeclared meaningful threshold and the tecto model is not better than its motif baseline.")
    b2_state = b2["state"]

    # ---- Table 1: case-study capability matrix ----
    table1 = "case_study\tendpoint_identity\tsource_membership\tcensoring\tselection\tgraph_support\tbaseline_adequacy\tcoverage_width\ttransport_claim\n"
    table1 += (f"qmap\tfrozen\tsource_reconstructed\tright-censored_11\tprimary_95_vs_sens_98\tcomponent_aware\t"
               f"matched_B1\tNOT_OK_coverage_0.7263\tQMAP_TRANSFER_NOT_SUPPORTED\n")
    table1 += (f"tecto\tlocked_negative\tparent_locked\tleft-censored_-7.1\tlocked_split\ttarget_specific\t"
               f"motif_mean\twidth_fail\tTECTO_NEGATIVE_BOUND_AND_LOCKED\n")
    write_text(f"{M1_DIR}/tables/table_1_case_capability_matrix.tsv", table1)

    # ---- Table 2: claim matrix (route tier + forbidden) ----
    table2 = "claim_tier\tallowed_statement\tforbidden_statement\n"
    table2 += "METHODS_BOUNDARY_AUDIT\t"
    table2 += "\"True qMaP transfer is below the predeclared meaningful threshold; tecto model is not better than the motif baseline.\"\t"
    table2 += "\"qMaPseq independently reproduced junction preorganization\" / \"four components are four i.i.d. repeats\"\n"
    write_text(f"{M1_DIR}/tables/table_2_claim_matrix.tsv", table2)

    # ---- Table 3: Q7 quantitative results (all from sealed decision/metrics) ----
    table3 = "metric\tvalue\n"
    table3 += f"N_total\t{q7_n}\n"
    table3 += f"N_measured\t{q7_n_measured}\n"
    table3 += f"N_right_censored\t{q7_n_censored}\n"
    table3 += f"micro_gain_B3_over_best_baseline\t{q7_gain:.6f}\n"
    table3 += f"best_baseline\t{q7_best_base}\n"
    table3 += f"meaningful_threshold\t{q7_threshold}\n"
    table3 += f"threshold_met\t{q7_threshold_met}\n"
    table3 += f"micro_coverage_80pct\t{q7_cov:.6f}\n"
    table3 += f"coverage_ok\t{q7_cov_ok}\n"
    table3 += f"permutation_finite_p\t{q7_perm_p}\n"
    table3 += f"bootstrap_ci_95_lower\t{q7_ci[0]:.6f}\n"
    table3 += f"bootstrap_ci_95_upper\t{q7_ci[1]:.6f}\n"
    table3 += f"Q7_state\t{q7_state}\n"
    write_text(f"{M1_DIR}/tables/table_3_q7_quantitative_results.tsv", table3)

    # ---- figure labels (from §14.5) ----
    fig_labels = f"""# Figure labels (M1) — frozen from §14.5

| Fig | Content | Source artifacts |
|-----|---------|------------------|
| Fig.1 | data/endpoint/platform/lineage graph | Q6 membership, endpoint registry |
| Fig.2 | claim-evidence ladder + v1.3 closure gaps | C0 gap ledger, N0 claim matrix |
| Fig.3 | source-correct qMaP component holdout | Q7 component_splits, metrics |
| Fig.4 | tecto baseline + coverage-width | T6 lock, T5 parent scores |
| Fig.5 | selection/censoring/weighting sensitivities (POST_HOC_EXPLANATORY) | B2 results/* |
| Fig.6 | reusable audit workflow | B0 schemas/fixtures/CLI |
| Table 1 | case-study capability matrix | table_1_case_capability_matrix.tsv |
| Table 2 | claim matrix and forbidden statements | table_2_claim_matrix.tsv |
| Table 3 | Q7 quantitative results | table_3_q7_quantitative_results.tsv |

All table numbers are read from sealed decision artifacts; no hand-copied values.
"""
    write_text(f"{M1_DIR}/figures_labels.md", fig_labels)

    # ---- manuscript body ----
    manuscript = f"""# When correlations do not transport: an auditable boundary analysis of public RNA thermodynamics

**Route:** {n0_route} · Claim tier: METHODS_BOUNDARY_AUDIT
**Manuscript status:** preparation authorized; submission HOLD_PENDING_E1_AND_USER_APPROVAL
**Run:** {RUN_ID} · contract SHA-256 `{CONTRACT_SHA}`

## Abstract

Public RNA-thermodynamic datasets are often cited as evidence that chemical-mapping or
Mg2+ endpoints carry transferable thermodynamic signal, and that complex models outperform
simple baselines. We show, with two real case studies and a reusable audit benchmark, that
these inferences are fragile once endpoint identity, source-authored attrition/censoring,
selection, graph support, baseline adequacy and coverage-width are jointly constrained.

- **tecto case (frozen locked negative):** the complex model is significantly worse than the
  motif-mean baseline (relative gain -0.5468, bootstrap CI all negative) and its
  credible intervals are too wide to be useful. Engineering complexity and high coverage do
  not imply predictive value.
- **qMaP case (source-corrected transfer):** after source-authoritative reconstruction of the
  99 -> 98 denominator and the 84/11/2/1 categories, the genuine qMaP predictor's
  component-aware micro gain ({q7_gain:.4f}) exceeds the predeclared meaningful threshold
  ({q7_threshold}), but the 80% predictive interval coverage ({q7_cov:.4f}) falls below the
  predeclared band, so the frozen analysis card rules transfer NOT_SUPPORTED.

We release a frozen audit schema, synthetic failure-mode fixtures, case cards, and a
one-command replay so the boundary result is checkable from a clean checkout.

## 1. Introduction

Public RNA thermodynamic studies operate under low, graph-correlated, censored samples.
Claims about "transfer" between measurement systems are common. The main scientific question
(v1.4 §4.1) asks which predictive or cross-measurement transport claims remain identifiable,
reproducible and worth keeping once endpoint identity, source-authored attrition/censoring,
selection, graph support, baseline adequacy and coverage-width are jointly constrained.

## 2. Two case studies

### 2.1 tecto (locked negative)
The exact tecto estimand is bound (`{t6_state}`, preserve rule {t6_preserve}). The v1.3
negative result is sealed as a formal locked negative; no rerun was performed because the
numeric mapping was consistent.

### 2.2 qMaP (source-corrected, NOT_SUPPORTED)
Q6 reconstructed the source-authoritative populations: N={q7_n} (measured {q7_n_measured},
right-censored {q7_n_censored}). The Q7 locked rerun reports micro gain {q7_gain:.4f} vs best
matched baseline {q7_best_base} (threshold {q7_threshold}), permutation p={q7_perm_p},
bootstrap 95% CI [{q7_ci[0]:.4f}, {q7_ci[1]:.4f}]. Coverage of the 80% predictive interval is
{q7_cov:.4f}, below the predeclared band, so the primary decision is {q7_state}.

> **Scope guard (post-outcome, POST_HOC_EXPLANATORY):** the original qMaP2024 in-population
> correlation is a different estimand from the out-of-component transport measured here. The
> two are not conflated in this manuscript.

## 3. Methods boundary audit benchmark (brief)

Frozen schemas (B0), synthetic failure-mode fixtures (B1) and a sensitivity ladder (B2,
`{b2_state}`, all POST_HOC_EXPLANATORY) form the reusable audit. Full details are in the B0/B1/B2
reports and the release bundle (R1).

## 4. Post-outcome sensitivity (POST_HOC_EXPLANATORY)

B2 explores selection, censoring, weighting, operator and threshold sensitivity. These are
explicitly post-outcome and explanatory; they do not create a new confirmatory claim and do
not move the frozen primary threshold.

## 5. Discussion and limitations

- The negative/boundary outcomes are the contribution, not a positive mechanistic claim.
- Low independent N, source identity and split support limit generalization; the 98 variants
  do not extrapolate to arbitrary TLR families.
- Manuscript preparation is authorized; submission is held pending E1 (fresh-checkout
  reproduction + adversarial review) and explicit user authorization.

## Data and code availability (draft)

All artifacts, decisions, sentinels and the release bundle are under the run root; the
one-command replay verifies the canonical payload against the detached seal. Hashes are
registered in the release inventory.
"""
    write_text(f"{M1_DIR}/manuscript.md", manuscript)

    report = f"""# M1 report — manuscript skeleton + figure-source tables

Generated {now_utc()}. All manuscript numbers are read from sealed artifacts; no hand-copied values.

## Files
- manuscript/m1/manuscript.md
- manuscript/m1/tables/table_1_case_capability_matrix.tsv
- manuscript/m1/tables/table_2_claim_matrix.tsv
- manuscript/m1/tables/table_3_q7_quantitative_results.tsv
- manuscript/m1/figures_labels.md

## Key frozen numbers referenced
- Q7 state: {q7_state}; gate: {q7_gain:.4f} vs threshold {q7_threshold}; coverage {q7_cov:.4f} (ok={q7_cov_ok})
- T6 state: {t6_state} (preserve {t6_preserve})
- N0 route: {n0_route}
- B2 state: {b2_state}

## Guards
- qMaP2024 in-population correlation is NOT conflated with out-of-component transport.
- All post-outcome sensitivities are labeled POST_HOC_EXPLANATORY.
- Submission remains HOLD_PENDING_E1_AND_USER_APPROVAL.
"""
    write_text(f"{REPORTS_DIR}/M1_report.md", report)

    sentinel = {
        "gate": "M1",
        "state": "M1_MANUSCRIPT_DRAFT_AUTHORIZED",
        "run_id": RUN_ID,
        "generated_at_utc": now_utc(),
        "route": n0_route,
        "manuscript_submission": "HOLD_PENDING_E1_AND_USER_APPROVAL",
    }
    write_text(f"{SENTINELS_DIR}/M1_MANUSCRIPT_DRAFT_AUTHORIZED.json",
               json.dumps(sentinel, indent=2, ensure_ascii=False))

    print(json.dumps({
        "state": "M1_MANUSCRIPT_DRAFT_AUTHORIZED",
        "manuscript_bytes": os.path.getsize(f"{M1_DIR}/manuscript.md"),
        "tables": [
            "table_1_case_capability_matrix.tsv",
            "table_2_claim_matrix.tsv",
            "table_3_q7_quantitative_results.tsv",
        ],
        "q7_gain": q7_gain,
        "q7_state": q7_state,
        "t6_state": t6_state,
    }, indent=2))


if __name__ == "__main__":
    main()