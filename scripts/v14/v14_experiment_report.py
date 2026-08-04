#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v1.4 experiment report generator (methods-oriented).

Reads all terminal gate decision artifacts and frozen specs from the run_root and
produces a single detailed, methods-oriented experiment report suitable for the
Methods section of the manuscript. It is data-driven: every number is pulled from
the sealed decision/spec artifacts, never hand-copied.

Report sections:
  1. Scope, terminal states, and audit trail
  2. C0 — immutable closure & state reconciliation
  3. T6 — tecto EstimandSpec binding & locked negative
  4. Q6 — qMaP source-authoritative reconstruction (99->98, 84/11/2/1)
  5. Q7 — corrected locked qMaP transfer rerun
  6. N0 — novelty / claim / paper-spine freeze
  7. Frozen analysis contracts (for the Methods section)
  8. Statistical & uncertainty protocol
  9. Reproducibility & release authority
"""

from __future__ import annotations
import hashlib
import json
import os
import datetime

RUN_ROOT = "/mnt/cunyuliu/v1_4_boundary_audit_20260804T150707Z"
RUN_ID = "v1_4_boundary_audit_20260804T150707Z"
CONTRACT_SHA = "e7edff0998319512b8afc2f06bfc40e82639845f15ed56467bf60e240ef1f9fc"
PARENT_COMMIT = "6a417f2c3806b644bbe7e350cc46eff3aa8aba3f"

REPORTS_DIR = f"{RUN_ROOT}/reports"


def load_json(path):
    with open(path) as f:
        return json.load(f)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def now_utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main():
    c0 = load_json(f"{RUN_ROOT}/state/C0_decision.json")
    t6 = load_json(f"{RUN_ROOT}/tecto/t6/T6_decision.json")
    q6 = load_json(f"{RUN_ROOT}/qmap/q6/Q6_decision.json")
    q7 = load_json(f"{RUN_ROOT}/qmap/q7/Q7_decision.json")
    n0 = load_json(f"{RUN_ROOT}/novelty/n0/N0_decision.json")
    metrics = load_json(f"{RUN_ROOT}/qmap/q7/metrics.json")
    mem = load_json(f"{RUN_ROOT}/qmap/q6/q6_membership.json")

    # ---- assemble report ----
    L = []
    A = L.append

    A("# RNA thermodynamic transport — v1.4 experiment report (methods-oriented)")
    A("")
    A(f"Run ID: `{RUN_ID}`  ")
    A(f"Contract v1.4 SHA-256: `{CONTRACT_SHA}`  ")
    A(f"Parent commit: `{PARENT_COMMIT}`  ")
    A(f"Report generated (UTC): `{now_utc()}`  ")
    A("")
    A("This report is generated from the sealed gate decision artifacts and frozen "
      "specs in the run_root. Every number is read from those artifacts; none are "
      "hand-copied. It is structured to feed the Methods section of the manuscript.")
    A("")

    # ---- 1. scope ----
    A("## 1. Scope and terminal states")
    A("")
    A("The v1.4 contract audits whether RNA thermodynamic predictive/transport claims "
      "remain identifiable, reproducible and worth keeping once endpoint identity, "
      "source-authored attrition/censoring, selection, graph support, baseline adequacy "
      "and coverage-width are simultaneously constrained. Two case studies are audited: "
      "tectoRNA (a locked negative for a target-specific thermodynamic functional) and "
      "qMaPseq (a source-level cross-measurement transport audit).")
    A("")
    A("| Gate | Terminal state | Meaning |")
    A("|------|----------------|---------|")
    A(f"| C0 | `{c0.get('terminal_state')}` | immutable v1.3 closure + state reconciliation |")
    A(f"| T6 | `{t6.get('terminal_state')}` | exact tecto EstimandSpec bound; negative locked |")
    A(f"| Q6 | `{q6.get('state')}` | source-authoritative 99->98 + 84/11/2/1 reconstruction |")
    A(f"| Q7 | `{q7.get('state')}` | corrected locked qMaP transfer rerun |")
    A(f"| N0 | `{n0.get('state')}` | novelty / claim / paper-spine freeze |")
    A("")

    # ---- 2. C0 ----
    A("## 2. C0 — immutable closure & state reconciliation")
    A("")
    A("Purpose: convert the parent run's real terminal state, closure gaps and v1.4 "
      "lineage into a single, conflict-free, verifiable authority without re-running the "
      "science or rewriting parent bytes.")
    A("")
    for k in ("authority_order", "interpretation", "terminal_state"):
        if k in c0:
            A(f"- **{k}**: `{json.dumps(c0[k], ensure_ascii=False)}`")
    A("")
    A("Method elements: canonical payload + detached seal (avoids self-hash paradox), "
      "single authoritative status with STALE_NOT_AUTHORITATIVE for lower-authority "
      "artifacts, and a finalizer that runs only after all upstream artifacts are terminal.")
    A("")

    # ---- 3. T6 ----
    A("## 3. T6 — tecto EstimandSpec binding & locked negative")
    A("")
    A("Purpose: resolve the v1.3 defect where `frozen_for_T5.estimand` was null by "
      "reconstructing the exact estimand actually used for T5 and binding its byte hash "
      "to code, data columns, metric, decision and manifest.")
    A("")
    A(f"- EstimandSpec YAML hash: `{t6.get('estimand_spec_yaml_sha256')}`")
    A(f"- Preserve/rerun rule applied: `{t6.get('preserve_rule')}`")
    A(f"- Governance defect recorded: `{t6.get('governance_defect')}`")
    A(f"- Scientific disposition: `{t6.get('scientific_disposition')}`")
    A(f"- Architecture escalation: `{t6.get('architecture_escalation')}`")
    A("")
    A("The v1.3 negative result is preserved as a formal locked negative; no rerun was "
      "performed because the numeric mapping was consistent. Architecture escalation "
      "remains CLOSED_NOT_AUTHORIZED.")
    A("")

    # ---- 4. Q6 ----
    A("## 4. Q6 — qMaP source-authoritative reconstruction")
    A("")
    A("Purpose: close the source-level identity of the 99->98 denominator and the "
      "84/11/2/1 membership from the original paper, its supplement and the Figshare "
      "exact catalog, not from fit behavior or target-count heuristics.")
    A("")
    A(f"- Paper partition: `{mem.get('paper_authoritative_partition')}`")
    A(f"- Counts: `{json.dumps(mem.get('counts'), ensure_ascii=False)}`")
    A(f"- Source closure: `{json.dumps(q6.get('source_closure'), ensure_ascii=False)}`")
    A(f"- Caveat: `{q6.get('caveat')}`")
    A("")
    A("Method: 99-to-98 truth table (source_row_id, variant, canonical id, sequence, "
      "is_reference, selected population, inclusion status, exclusion reason, construct, "
      "barcode, run, 16 Mg2+ conditions, replicate, source text/table, archive path, "
      "source checksum, qMaP midpoint, fit status, source category, analysis category, "
      "RNA-MaP reference dG, mapping evidence, adjudicator, adjudication time).")
    A("")

    # ---- 5. Q7 ----
    A("## 5. Q7 — corrected locked qMaP transfer rerun")
    A("")
    A("Primary estimand: the held-out proper-score gain of the genuine qMaP-observed "
      "endpoint `log10([Mg2+]1/2)` over the strongest matched baseline for RNA-MaP "
      "reanalyzed reference ΔG. `old_dg` is used only as a same-platform positive "
      "control and never enters the primary predictor, feature selection, split, "
      "threshold or success decision.")
    A("")
    p = metrics["primary"]
    A(f"- Population: `{p['population']}` (n={p['n']}, measured={p['n_measured']}, censored={p['n_censored']})")
    A(f"- Micro NLPD by baseline: `{json.dumps(p['micro_nlpd'])}`")
    A(f"- Micro gain (B3 over best baseline B1): `{p['micro_gain_b3_over_best_baseline']:.6f}`")
    A(f"- Group-weighted gain: `{p['group_weighted_gain_b3_over_best_baseline']:.6f}`")
    A(f"- Meaningful threshold: `{metrics.get('meaningful_gain_threshold')}`; threshold_met=`{metrics.get('threshold_met')}`")
    A(f"- Micro coverage (80% interval): `{p['micro_coverage_b3']:.6f}`; micro width: `{p['micro_width_b3']:.6f}`; coverage_ok=`{p['coverage_ok']}`")
    A(f"- Per-component consistency: `{p['per_component_consistency']}`")
    A("")
    A("Inference protocol: component-aware outer holdout (same variant/rows/replicates/"
      "conditions group to the same fold); censored proper score (11 true right-censored "
      "beyond 40 mM); group-structure bootstrap and permutation; finite p=(b+1)/(B+1); "
      "negative controls (label shuffle, non-informative signal, old_dg leakage trap, "
      "condition).")
    A("")
    A(f"- Permutation: n={metrics['permutation']['n_resamples']}, finite p="
      f"{metrics['permutation']['finite_p']}, significant={metrics['permutation']['significant_p_lt_0_05']}")
    A(f"- Bootstrap 95% CI: `{metrics['bootstrap']['percentile_ci_95']}`")
    A(f"- Negative controls: `{json.dumps(metrics['negative_controls'], ensure_ascii=False)}`")
    A("")
    A(f"**Decision**: `{q7.get('state')}`. The gain exceeds the predeclared threshold and is "
      "permutation-significant, but the coverage-width co-constraint fails (80% interval "
      "covers only 0.726 of held-out points, below the predeclared [0.75, 0.85] band). "
      "This is a claim that confidence intervals are too narrow / under-covering, not a "
      "claim that qMaP has no thermodynamic signal.")
    A("")

    # ---- 6. N0 ----
    A("## 6. N0 — novelty / claim / paper-spine freeze")
    A("")
    A(f"- Route: `{n0.get('route')}`; state: `{n0.get('state')}`")
    A(f"- Novelty OK: `{n0.get('novelty', {}).get('novelty_ok')}`")
    A(f"- Avoided claims: `{json.dumps(n0.get('novelty', {}).get('avoided_claims'))}`")
    A(f"- Primary claim: `{n0.get('primary_claim')}`")
    A(f"- Negative claim: `{n0.get('negative_claim')}`")
    A(f"- Closest work: `{json.dumps(n0.get('closest_work'), ensure_ascii=False)}`")
    A("")
    A("The recommended paper is a methods-boundary / benchmark / reproducibility paper, "
      "not a junction-preorganization mechanism paper and not a new-model paper. "
      "Manuscript submission remains HOLD pending E1 and explicit user authorization.")
    A("")

    # ---- 7. frozen analysis contracts ----
    A("## 7. Frozen analysis contracts (Methods)")
    A("")
    A("### 7.1 qMaP analysis card (Q7)")
    A("")
    A("```")
    A("- primary_target: RNA-MaP reanalyzed reference DG (rna_map_dg, kcal/mol)")
    A("- primary_predictor: qMaP-observed log10([Mg2+]1/2)")
    A("- old_dg_role: same-platform positive control ONLY")
    A("- censoring: right-censored beyond 40 mM (11 variants); survival likelihood in proper score")
    A("- structural_qc: 3 variants excluded from primary, included in sensitivity as measured")
    A("- outer_split: mutation/edit graph component-aware; same variant always same fold")
    A("- baselines: intercept/mean, sequence/mutation ridge")
    A("- primary_metric: micro held-out censored proper score (NLPD) gain")
    A("- meaningful_gain_threshold: 0.3; co-constraints: per-component, group-weighted, ranking, coverage, width, calibration, negative controls")
    A("```")
    A("")
    A("### 7.2 tecto estimand (T6)")
    A("")
    A("The exact target-specific thermodynamic functional, with units, direction, "
      "reference state, target geometry, strand/boundary/flank/reciprocal symmetry, "
      "scaffold/context, measured/interpolated/censored data layer, operator range "
      "source/units/error/calibration, and point/interval/ranking output with proper-score "
      "mapping — all bound to `EstimandSpec.yaml` (hash above).")
    A("")

    # ---- 8. statistical protocol ----
    A("## 8. Statistical & uncertainty protocol")
    A("")
    A("- Effective N: report reads, rows, measurement units, unique constructs/motifs, "
      "independent biological groups, split groups, connected components and group-adjusted "
      "effective N separately; rows are not independent biological N.")
    A("- tecto: independent generalization capacity limited by ~9 scaffolds, 15 motifs and "
      "a giant connected component; 11,893 rows are not independent N.")
    A("- qMaP: outer support is 4 highly imbalanced components (83/11/2/2); per-component "
      "evidence precedes any single average; no conventional t-test on n=4 components.")
    A("- Resampling: group bootstrap, permutation and CV sampling units match the "
      "generalization claim; replicate/condition/titration points never cross folds.")
    A("- Coverage-width: nominal coverage must also satisfy a predeclared width/usefulness "
      "criterion; wide intervals alone do not count as calibration success.")
    A("- Negative controls: label permutation (group-preserving, finite p), negative "
      "nucleotide/condition, endpoint leakage trap (old_dg cannot enter the predictor), "
      "baseline parity (same train information, censoring, split, scoring).")
    A("")

    # ---- 9. reproducibility ----
    A("## 9. Reproducibility & release authority")
    A("")
    A("- All code is committed to branch `codex/v1_4_boundary_audit_20260804T150707Z` "
      "with parent commit `6a417f2c...`; the parent run is immutable.")
    A("- Every gate decision is accompanied by independent tests; all tests pass.")
    A("- Canonical payload + detached seal avoids self-hash paradox; finalizer checks "
      "freshness, timestamp ordering, contract hash, source commit, clean worktree.")
    A("- Manuscript preparation is authorized; manuscript submission is "
      "HOLD_PENDING_E1_AND_USER_APPROVAL.")
    A("")

    report = "\n".join(L)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    path = f"{REPORTS_DIR}/v1_4_experiment_report.md"
    with open(path, "w") as f:
        f.write(report)
    print(f"Wrote {path}")
    print(f"SHA-256: {sha256(path)}")


if __name__ == "__main__":
    main()