# Phase 5 Report — Identifiability Boundary (mechanism analysis, contract Phase 5)

- **Phase**: P5 (mechanism analysis / narrative)
- **Date**: 2026-08-09
- **Candidate**: `support_aware_mixture` (Candidate C, Phase 3)
- **Run root**: `/mnt/cunyuliu/rna_junction_audit_20260807T090244Z/p5_diagnostics/`
- **Evidence class**: `DEVELOPMENT_ONLY` — no mechanism claim, no submission gate
- **Narrative**: `benchmark_identifiability_boundary` (contract Phase 5 failure path)
- **SOTA status**: `SOTA_NOT_ADJUDICATED`; **scientific_claim_authorized = false**

---

## 1. Context

Phase 4 sealed a fail-closed result: the only surviving candidate
(`support_aware_mixture`) was **NOT_PROMOTED** — it underperforms the strongest
eligible baseline on all three known-operator axes and only **ties** `edit_knn`
on the operator-holdout axis. Because no promotable mechanism exists, Phase 5 has
no mechanism narrative to analyze. Per the contract's Phase 5 failure handling,
this report instead **rigorously characterizes where the sequence signal does and
does not survive** (the identifiability boundary) and selects a
benchmark / identifiability-boundary paper narrative.

Diagnostics reuse sealed inputs only (P1 baseline predictions, P4 candidate
predictions, frozen P0.4 splits); they add only cheap train-only support features
(min edit distance to nearest outer-train junction; outer-train context support).

## 2. Failure atlas: where the candidate fails

Per-axis candidate vs. strongest-baseline right-censored NLL, stratified by
min edit distance from the test junction to the nearest **outer-train** junction
sequence (delta = candidate − baseline; positive means candidate worse):

| axis | edit-dist bucket | n | candidate NLL | baseline NLL | delta |
|---|---|---|---|---|---|
| symmetry_5fold | 1 | 11,893 | 2.65 | 1.09 | **+1.56** |
| edit_5fold | 1 | 6,652 | 2.56 | 1.14 | +1.42 |
| edit_5fold | 2 | 4,019 | **40.96** | 1.12 | **+39.84** |
| edit_5fold | 3 | 1,222 | **81.74** | 1.11 | **+80.63** |
| context_lomo | 0 (same seq) | 11,893 | 3.09 | 1.32 | +1.76 |
| scaffold_lomo | 0 (same seq) | 11,893 | 3.09 | 3.09 | **0.0 (exact tie)** |

### 2.1 Key structural finding: the operator-holdout "rescue" is trivial

On **scaffold_lomo** (and context_lomo), **every test junction is at edit
distance 0** from an outer-train junction sequence — the held-out axis is the
scaffold/operator, not the sequence. Both the candidate and the `edit_knn`
baseline therefore simply **copy the same-sequence training value**, producing an
exact tie (delta 0.0). The Phase 3 "rescue" of scaffold_lomo (from
`corrected_v1_31`'s catastrophic 80.45) is not sequence generalization: it is
same-sequence copying that any nearest-neighbour baseline achieves equally. No
genuine sequence extrapolation is being tested on that axis.

### 2.2 The candidate catastrophically fails at sequence (edit) extrapolation

On **edit_5fold**, as soon as the test junction is ≥2 edits from every train
junction, the candidate NLL explodes (d=2 → 40.96, d=3 → 81.74) while the
strongest baseline stays flat (~1.1). The sequence-local KNN has **no
generalization along the mutation/edit axis** — exactly the axis where a real
preorganization mechanism would be expected to help. This is the decisive
boundary: the candidate only "works" when it can copy a near-identical sequence.

### 2.3 No context-calibration rescue either

On symmetry_5fold even at high context support (6–20 / 21+ contexts seen in
train), the candidate is still worse than baseline (delta +1.22 / +1.65). On
edit_5fold high context support does not rescue the edit-extrapolation failure
(21+ contexts → candidate NLL 21.7). The candidate's deficit is structural, not a
support-coverage artifact.

## 3. Context sensitivity

| axis | stratifier | stratum | n | cand NLL | base NLL | delta |
|---|---|---|---|---|---|---|
| symmetry_5fold | context_support | 21plus | 9,306 | 2.73 | 1.07 | +1.65 |
| symmetry_5fold | context_support | 6-20 | 2,587 | 2.36 | 1.14 | +1.22 |
| edit_5fold | context_support | 0_unseen | 1,293 | **47.04** | 1.18 | **+45.86** |
| edit_5fold | context_support | 21plus | 9,306 | 21.75 | 1.12 | +20.63 |
| context_lomo | context_support | 0_unseen | 11,893 | 3.09 | 1.32 | +1.76 |

The candidate provides **no incremental supported-NLL** in any context-support
stratum. Its edit-axis failure is present even with abundant context support.

## 4. Catastrophic folds (supported macro supported-NLL)

On the operator-holdout axis (scaffold_lomo) the candidate has **0 catastrophic
folds**, but this is only because it ties `edit_knn` fold-for-fold (identical
same-sequence copying). No fold shows an incremental win over the strongest
eligible baseline on any axis.

## 5. Identifiability boundary (the honest, publishable result)

The union of Phase 1–5 evidence establishes a coherent boundary:

1. **No incremental sequence signal on known-operator axes.** Under grouped,
   right-censor-aware, leakage-controlled evaluation, junction sequence adds no
   supported-NLL beyond motif/context/scaffold/nearest-neighbour and
   censored-marginal baselines.
2. **No sequence-extrapolation capability.** The only candidate that survived
   Phase 3 catastrophically fails at edit-distance extrapolation (d≥2) and ties a
   simple nearest-neighbour baseline on the operator-holdout axis.
3. **Repeated context/scaffold exposure packages calibration as
   generalization** — the contract's central concern is confirmed, not refuted.
   The apparent "operator rescue" is same-sequence copying, not transfer.

## 6. Paper story decision

**Narrative: benchmark / identifiability boundary.**

- **Claims removed:** any mechanism, operator-transfer, or cross-system claim.
- **Claims retained:** a strict, reproducible grouped/right-censored benchmark
  protocol, and a rigorous negative result: sequence-local preorganization signal
  does not transfer beyond a local neighbourhood and known-operator calibration.
- **Gate:** `SOTA_NOT_ADJUDICATED`, `NO_SUBMISSION_AUTHORIZATION`,
  `scientific_claim_authorized = false`. Prospective constructs are unavailable,
  so per contract Phase 4/5 failure handling no broad claim may be made.

## 7. Artifacts (run root `p5_diagnostics/`)

- `FailureAtlas.parquet`
- `ContextSensitivity.csv`
- `MutationPathAnalysis.csv`
- `CatastrophicFolds.csv`
- `ClaimEvidenceMatrix.csv`
- `PaperStoryDecision.md`
- `STATUS.json`
